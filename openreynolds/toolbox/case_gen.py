#!/usr/bin/env python3
r"""Complete, runnable OpenFOAM cases from a template name and a few numbers.

Typing out blockMeshDict, the system and constant dictionaries and a 0/ directory
by hand costs most of an hour per study, and the hour is not spent where the
thinking is. The mistakes it produces are the dull ones, and they all cost a
solver run to find: two cells in z so the "2D" case is quietly 3D and forty times
slower; a frontAndBack patch declared `empty` in the mesh and `zeroGradient` in
0/U, which stops the solver on the first time step; a patch called `inlet` in the
mesh and `Inlet` in the fields; a viscosity typed in for the Reynolds number
somebody meant to run last week.

So the shapes are parameterised instead. What comes out is a directory you can
run blockMesh in, and every file it contains is one you can open and edit --
nothing here is a black box at run time, and a generated dictionary that is 90%
of what you want is meant to be edited the other 10%.

what it derives, and says it derived

    Give it a free-stream speed, a Reynolds number and a characteristic length
    and it solves nu = U*L/Re and writes that into transportProperties, then
    prints the three numbers it used. Give it --nu instead and it prints the
    Reynolds number that implies. The one that is not stated is the one that gets
    misremembered.

    The cell count is not an estimate: blockMesh cell counts are the product of
    the divisions you wrote down, so the number printed before the mesh is built
    is the number checkMesh will report after.

2D

    Every 2D template writes exactly one cell in z, and a single patch named
    `frontAndBack` carrying both z faces with `type empty;` -- in blockMeshDict
    and in every field in 0/. That pairing is checked by the tests rather than
    trusted, because it is the failure that looks most like something else: a
    case with two cells in z runs, converges, and answers a different question.

templates

    External flow (a body in a rectangular tunnel, wrapped in an O-grid):
    `circle`, `square`, `lshape`, `vehicle`, `profile` (x,y points from a file).
    Ducts, 2D: `duct-y`, `duct-t`, `duct-z`, `duct-f`, `duct-m`.
    Bends, 2D: `bend-sharp`, `bend-mitred`, `bend-rounded`.

    Patch names are the same everywhere they can be: `inlet`, `outlet` (then
    `outlet2`, `outlet3` where a duct has more than one), `walls`, `topWall` and
    `bottomWall` where the two differ, `body` for the object in the flow, and
    `frontAndBack`.

what it will not do

    The O-grid templates need a body that is star-shaped about one interior
    point -- every ray from that point crosses the outline once. A circle, a
    square, an L and an aerofoil all are; a shape with a deep pocket in it is
    not, and you get an error naming the point that broke it rather than a mesh
    that blockMesh refuses. The `vehicle` body has its wheels as scallops in the
    underbody for exactly this reason: wheels drawn as protruding discs are not
    star-shaped from anywhere, and a car resting on the ground splits the domain,
    which blockMesh alone cannot do.

    Nothing here meshes around an STL. That is snappyHexMesh's job.

    python3 case_gen.py --list
    python3 case_gen.py circle /work/cyl --reynolds 200 --study transient
    python3 case_gen.py vehicle /work/car --moving-ground --rotating-wheels
    python3 case_gen.py profile /work/foil --profile naca0012.dat --aoa 6
    python3 case_gen.py duct-t /work/tee --duct-width 0.05 --study steady
    python3 case_gen.py bend-rounded /work/bend --bend-radius 0.1 --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state

TOLERANCE = 1e-9
"""Coordinates closer than this are the same point. blockMesh matches faces by
vertex index, not by position, so two points that ought to be one and are not
produce a mesh with a crack down it that checkMesh reports as an open cell."""


# -- small geometry helpers --------------------------------------------------------


def signed_area(points: list[tuple[float, float]]) -> float:
    """Twice the signed area of a closed polygon; positive when counter-clockwise."""
    total = 0.0
    for index, (x_here, y_here) in enumerate(points):
        x_next, y_next = points[(index + 1) % len(points)]
        total += x_here * y_next - x_next * y_here
    return total


def as_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The same loop, wound counter-clockwise.

    Aerofoil coordinate files are conventionally written the other way round
    (trailing edge, over the top, back along the bottom), and a block whose
    corners run clockwise is inside-out: blockMesh builds it and checkMesh then
    reports every cell in it as having negative volume.
    """
    return points if signed_area(points) > 0 else list(reversed(points))


def drop_repeats(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Consecutive duplicates removed, including a final point equal to the first.

    Coordinate files disagree about whether a closed loop repeats its first point
    at the end. Both spellings arrive here; only one leaves.
    """
    kept: list[tuple[float, float]] = []
    for point in points:
        if kept and abs(point[0] - kept[-1][0]) < TOLERANCE and abs(point[1] - kept[-1][1]) < TOLERANCE:
            continue
        kept.append((float(point[0]), float(point[1])))
    while len(kept) > 1 and abs(kept[0][0] - kept[-1][0]) < TOLERANCE and abs(kept[0][1] - kept[-1][1]) < TOLERANCE:
        kept.pop()
    return kept


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    area = signed_area(points)
    if abs(area) < TOLERANCE:
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
    cx = cy = 0.0
    for index, (x_here, y_here) in enumerate(points):
        x_next, y_next = points[(index + 1) % len(points)]
        cross = x_here * y_next - x_next * y_here
        cx += (x_here + x_next) * cross
        cy += (y_here + y_next) * cross
    return (cx / (3.0 * area), cy / (3.0 * area))


def unwrapped_angles(points: list[tuple[float, float]], centre: tuple[float, float]) -> list[float]:
    """Angle of every point about `centre`, in degrees, increasing without a jump.

    Starting from the first point and adding the smallest step that keeps the
    sequence going the same way, so a loop that crosses the +x axis does not read
    as a 359-degree jump backwards.
    """
    angles: list[float] = []
    previous = 0.0
    for index, (x_here, y_here) in enumerate(points):
        raw = math.degrees(math.atan2(y_here - centre[1], x_here - centre[0]))
        if index == 0:
            previous = raw
            angles.append(raw)
            continue
        step = (raw - previous + 180.0) % 360.0 - 180.0
        previous = raw
        angles.append(angles[-1] + step)
    return angles


def star_shaped_failure(
    points: list[tuple[float, float]], centre: tuple[float, float]
) -> tuple[int, float] | None:
    """The first place the outline turns back on itself as seen from `centre`.

    Returns (index, angular step in degrees) for the first non-increasing step,
    or None when the whole loop winds one way. An O-grid is a map from the body
    outline to a circle around it, and that map only exists when this is None.
    """
    angles = unwrapped_angles(points, centre)
    for index in range(1, len(angles)):
        if angles[index] <= angles[index - 1]:
            return index, angles[index] - angles[index - 1]
    return None


def arc_points(
    start: tuple[float, float],
    end: tuple[float, float],
    sagitta: float,
    count: int,
) -> list[tuple[float, float]]:
    """A circular arc from `start` to `end` bulging `sagitta` to the right of the
    chord, sampled at `count` points including both ends.

    Used for the wheel scallops in the vehicle underbody. Sagitta rather than
    radius because the thing being controlled is how deep the scallop is, and a
    radius has to be solved for anyway once the chord is fixed.
    """
    x0, y0 = start
    x1, y1 = end
    chord = math.hypot(x1 - x0, y1 - y0)
    if chord < TOLERANCE or abs(sagitta) < TOLERANCE:
        return [start, end]
    half = chord / 2.0
    radius = (half * half + sagitta * sagitta) / (2.0 * abs(sagitta))
    # Unit vectors along and to the right of the chord.
    along = ((x1 - x0) / chord, (y1 - y0) / chord)
    right = (along[1], -along[0])
    depth = radius - abs(sagitta)
    sign = 1.0 if sagitta > 0 else -1.0
    centre = (
        (x0 + x1) / 2.0 - right[0] * depth * sign,
        (y0 + y1) / 2.0 - right[1] * depth * sign,
    )
    span = 2.0 * math.asin(min(1.0, half / radius))
    start_angle = math.atan2(y0 - centre[1], x0 - centre[0])
    direction = sign
    return [
        (
            centre[0] + radius * math.cos(start_angle + direction * span * step / (count - 1)),
            centre[1] + radius * math.sin(start_angle + direction * span * step / (count - 1)),
        )
        for step in range(count)
    ]


def collinear(points: list[tuple[float, float]]) -> bool:
    """Whether every point lies on the line through the first and last.

    A polyLine edge through collinear points is what blockMesh would have drawn
    anyway, so it is left out and the dictionary stays readable.
    """
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length < TOLERANCE:
        return False
    for x_here, y_here in points[1:-1]:
        offset = abs((x1 - x0) * (y0 - y_here) - (x0 - x_here) * (y1 - y0)) / length
        if offset > 1e-7 * max(1.0, length):
            return False
    return True


def divisions(length: float, cell: float, minimum: int = 1) -> int:
    """Cells along `length` at a target size of `cell`, never fewer than `minimum`."""
    if cell <= 0:
        return minimum
    return max(minimum, int(round(abs(length) / cell)))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def shortest_cell(length: float, count: int, expansion: float) -> float:
    """The shortest of the `count` cells `simpleGrading expansion` puts along
    `length`.

    blockMesh's expansion ratio is last cell over first, so the cells are a
    geometric series of ratio r = expansion**(1/(count-1)) and the first one is
    length*(1-r)/(1-r**count). The shortest is the first when the block expands
    and the last when it contracts, and both spellings are in use here -- the far
    field is graded away from the body on one side of it and towards it on the
    other.
    """
    count = max(1, int(count))
    length = abs(float(length))
    if count == 1:
        return length
    ratio = abs(float(expansion))
    if ratio <= 0 or abs(ratio - 1.0) < 1e-12:
        return length / count
    step = ratio ** (1.0 / (count - 1))
    smallest = length * (1.0 - step) / (1.0 - step ** count)
    return min(smallest, smallest * step ** (count - 1))


def graded_divisions(length: float, first_cell: float, expansion: float) -> int:
    """Cells across `length` starting at `first_cell` and ending `expansion` times
    bigger, using the mean cell size.

    Exact would mean solving a geometric series for its term count; the mean-size
    answer is within a cell of it and is arithmetic anybody can check by eye,
    which matters more here than the last cell.
    """
    mean = first_cell * (1.0 + max(expansion, TOLERANCE)) / 2.0
    return max(3, int(round(abs(length) / max(mean, TOLERANCE))))


# -- the block mesh ----------------------------------------------------------------

SOUTH, EAST, NORTH, WEST = 0, 1, 2, 3
"""Side k of a quad runs from corner k to corner k+1. The names are only accurate
for an axis-aligned quad written corners-first from its bottom-left; for the
O-grid blocks side 3 is the body and side 1 the ring, and the numbers are what is
actually used."""

FACE_OF_SIDE = {
    SOUTH: (0, 1, 5, 4),
    EAST: (1, 2, 6, 5),
    NORTH: (3, 7, 6, 2),
    WEST: (0, 4, 7, 3),
}
"""Which of a hex block's eight vertices make each side face, wound so the normal
points out of the block. OpenFOAM's own hex vertex ordering; getting one of these
backwards gives 'face .. does not have a valid owner' out of blockMesh."""

BACK_FACE = (0, 3, 2, 1)
FRONT_FACE = (4, 5, 6, 7)


class Mesh2D:
    """Quad blocks in the x-y plane, extruded one cell thick in z.

    Points are held in 2D and each becomes two vertices, back (z=0) then front
    (z=thickness), so a 2D point index `i` is vertices `2i` and `2i+1`. Every
    block contributes its two z faces to `frontAndBack` automatically -- the
    patch is not something a template has to remember, because forgetting it is
    the classic way a 2D case turns out not to be one.
    """

    def __init__(self, thickness: float) -> None:
        self.thickness = float(thickness)
        self.points: list[tuple[float, float]] = []
        self._index: dict[tuple[int, int], int] = {}
        self.blocks: list[dict] = []
        self.edges: list[tuple[str, int, int, object]] = []
        self._edge_keys: set[tuple[int, int]] = set()
        self.patch_faces: dict[str, list[tuple[int, int, int, int]]] = {}
        self.patch_types: dict[str, str] = {}
        self.scale = 1.0

    # -- points and blocks ---------------------------------------------------------

    def point(self, x: float, y: float) -> int:
        key = (int(round(float(x) / TOLERANCE)), int(round(float(y) / TOLERANCE)))
        if key not in self._index:
            self._index[key] = len(self.points)
            self.points.append((float(x), float(y)))
        return self._index[key]

    def add_quad(
        self,
        corners: list[tuple[float, float]],
        cells: tuple[int, int],
        grading: tuple[float, float] = (1.0, 1.0),
        sides: dict[int, str] | None = None,
    ) -> dict:
        """One block. `cells[0]` counts along corner 0 -> corner 1, `cells[1]` along
        corner 0 -> corner 3. Corners must wind counter-clockwise; they are turned
        round if they do not, and `sides` follows them round."""
        corners = [(float(x), float(y)) for x, y in corners]
        if len(corners) != 4:
            raise ValueError(f"a quad has four corners, got {len(corners)}")
        sides = dict(sides or {})
        if signed_area(corners) < 0:
            # Reversing about corner 0 keeps the same four edges, renumbered: the
            # edge that was side k is side 3-k afterwards.
            corners = [corners[0], corners[3], corners[2], corners[1]]
            sides = {3 - k: v for k, v in sides.items()}
            cells = (cells[1], cells[0])
            grading = (grading[1], grading[0])
        ids = [self.point(x, y) for x, y in corners]
        if len(set(ids)) != 4:
            raise ValueError(f"degenerate block: corners {corners} collapse to {len(set(ids))} points")
        hexa = [2 * i for i in ids] + [2 * i + 1 for i in ids]
        block = {
            "hex": hexa,
            "corners": corners,
            "ids": ids,
            "cells": (int(cells[0]), int(cells[1]), 1),
            "grading": (float(grading[0]), float(grading[1]), 1.0),
        }
        self.blocks.append(block)
        for side, name in sides.items():
            self.face(len(self.blocks) - 1, side, name)
        self.face(len(self.blocks) - 1, "back", "frontAndBack")
        self.face(len(self.blocks) - 1, "front", "frontAndBack")
        self.patch_types["frontAndBack"] = "empty"
        return block

    def face(self, block_index: int, side, name: str) -> None:
        hexa = self.blocks[block_index]["hex"]
        if side == "back":
            slots = BACK_FACE
        elif side == "front":
            slots = FRONT_FACE
        else:
            slots = FACE_OF_SIDE[side]
        self.patch_faces.setdefault(name, []).append(tuple(hexa[slot] for slot in slots))

    def set_patch_type(self, name: str, kind: str) -> None:
        self.patch_types[name] = kind

    # -- curved edges --------------------------------------------------------------

    def curve(self, start: tuple[float, float], end: tuple[float, float], through: list[tuple[float, float]]) -> None:
        """A polyLine between two existing points, on both z planes.

        Registered once per pair: the same edge is shared by two blocks and
        blockMesh wants it named once, not once per user.
        """
        interior = [p for p in through]
        if len(interior) < 3 or collinear(interior):
            return
        a = self.point(*start)
        b = self.point(*end)
        key = (min(a, b), max(a, b))
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(("polyLine", a, b, interior[1:-1]))

    def arc(self, start: tuple[float, float], end: tuple[float, float], middle: tuple[float, float]) -> None:
        a = self.point(*start)
        b = self.point(*end)
        key = (min(a, b), max(a, b))
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(("arc", a, b, middle))

    # -- what it adds up to --------------------------------------------------------

    @property
    def cell_count(self) -> int:
        return sum(block["cells"][0] * block["cells"][1] * block["cells"][2] for block in self.blocks)

    @property
    def smallest_cell(self) -> float:
        """The shortest cell edge anywhere in the mesh, in x-y, grading included.

        The Courant number a time step actually produces is set by this length and
        not by an average one. A block graded 6:1 over 24 cells has a first cell a
        third of its mean, so a step sized from the mean starts the run at three
        times the Courant number that was asked for. z is left out: the one cell
        across an `empty` direction carries no flux.

        Block sides are measured straight, so a side that is really a polyLine or
        an arc reads as its chord, which is short. A cell read short gives a step
        read short, so the error is in the direction that keeps the Courant number
        under the one asked for rather than over it.
        """
        smallest = float("inf")
        for block in self.blocks:
            corners = block["corners"]
            along = min(distance(corners[0], corners[1]), distance(corners[3], corners[2]))
            across = min(distance(corners[0], corners[3]), distance(corners[1], corners[2]))
            smallest = min(
                smallest,
                shortest_cell(along, block["cells"][0], block["grading"][0]),
                shortest_cell(across, block["cells"][1], block["grading"][1]),
            )
        return smallest

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), max(xs), min(ys), max(ys))

    def patch_face_counts(self) -> dict[str, int]:
        return {name: len(faces) for name, faces in self.patch_faces.items()}

    # -- the dictionary ------------------------------------------------------------

    def block_mesh_dict(self) -> str:
        lines = [foam_header("dictionary", "blockMeshDict", "system"), f"scale   {self.scale:g};", ""]
        lines.append("vertices")
        lines.append("(")
        for index, (x, y) in enumerate(self.points):
            lines.append(f"    ({x:.10g} {y:.10g} 0)          // {2 * index}")
            lines.append(f"    ({x:.10g} {y:.10g} {self.thickness:.10g})          // {2 * index + 1}")
        lines.append(");")
        lines.append("")
        lines.append("blocks")
        lines.append("(")
        for block in self.blocks:
            hexa = " ".join(str(v) for v in block["hex"])
            nx, ny, nz = block["cells"]
            gx, gy, gz = block["grading"]
            lines.append(
                f"    hex ({hexa}) ({nx} {ny} {nz}) simpleGrading ({gx:g} {gy:g} {gz:g})"
            )
        lines.append(");")
        lines.append("")
        lines.append("edges")
        lines.append("(")
        for kind, a, b, data in self.edges:
            for plane, z in ((0, 0.0), (1, self.thickness)):
                va, vb = 2 * a + plane, 2 * b + plane
                if kind == "arc":
                    mx, my = data  # type: ignore[misc]
                    lines.append(f"    arc {va} {vb} ({mx:.10g} {my:.10g} {z:.10g})")
                else:
                    lines.append(f"    polyLine {va} {vb}")
                    lines.append("    (")
                    for px, py in data:  # type: ignore[union-attr]
                        lines.append(f"        ({px:.10g} {py:.10g} {z:.10g})")
                    lines.append("    )")
        lines.append(");")
        lines.append("")
        lines.append("boundary")
        lines.append("(")
        for name in patch_order(self.patch_faces):
            lines.append(f"    {name}")
            lines.append("    {")
            lines.append(f"        type {self.patch_types.get(name, 'patch')};")
            lines.append("        faces")
            lines.append("        (")
            for face in self.patch_faces[name]:
                lines.append("            (" + " ".join(str(v) for v in face) + ")")
            lines.append("        );")
            lines.append("    }")
        lines.append(");")
        lines.append("")
        lines.append("mergePatchPairs")
        lines.append("(")
        lines.append(");")
        lines.append("")
        lines.append(FOAM_FOOTER)
        return "\n".join(lines) + "\n"


def patch_order(patches) -> list[str]:
    """Inlets first, then outlets, then walls, then the empty one.

    Only cosmetic -- but a boundary list you can read top to bottom is a boundary
    list whose omissions you notice.
    """
    rank = {"inlet": 0, "outlet": 1, "walls": 3, "frontAndBack": 9}

    def key(name: str) -> tuple[int, str]:
        if name in rank:
            return (rank[name], name)
        if name.startswith("inlet"):
            return (0, name)
        if name.startswith("outlet"):
            return (1, name)
        if name == "frontAndBack":
            return (9, name)
        return (2, name)

    return sorted(patches, key=key)


# -- OpenFOAM file furniture -------------------------------------------------------

BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Website:  www.openfoam.com                      |
|   \\  /    A nd           | Written:  openreynolds case_gen.py              |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/"""

FOAM_FOOTER = "// ************************************************************************* //"

RULE = "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"


def foam_header(cls: str, obj: str, location: str = "") -> str:
    """The FoamFile block every OpenFOAM dictionary needs.

    `location` matters more than it looks: a field file without it still reads,
    but several utilities report the wrong path in their errors when it is
    missing, which turns a one-line typo into a hunt.
    """
    lines = [BANNER, "FoamFile", "{", "    version     2.0;", "    format      ascii;",
             f"    class       {cls};"]
    if location:
        lines.append(f'    location    "{location}";')
    lines.append(f"    object      {obj};")
    lines.append("}")
    lines.append(RULE)
    lines.append("")
    return "\n".join(lines)


def foam_file(cls: str, obj: str, body: str, location: str = "") -> str:
    return foam_header(cls, obj, location) + "\n" + body.rstrip("\n") + "\n\n" + FOAM_FOOTER + "\n"


def vector(value) -> str:
    return "(" + " ".join(f"{float(v):g}" for v in value) + ")"


# -- body outlines -----------------------------------------------------------------


class Body:
    """A closed outline, the patch each of its edges belongs to, and a point every
    part of the outline can see.

    `centre` is not the centroid. For an L the centroid falls in the notch,
    outside the metal, and an O-grid built around a point outside the body turns
    itself inside out; the templates each name a point that works and the builder
    checks it.
    """

    def __init__(self, points, edge_patches, centre, extras=None, note=""):
        self.points = list(points)
        self.edge_patches = list(edge_patches)
        self.centre = centre
        self.extras = dict(extras or {})
        self.note = note
        if len(self.edge_patches) != len(self.points):
            raise ValueError("one patch name per edge, and one edge per point")

    @property
    def bounds(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), max(xs), min(ys), max(ys))


def circle_body(size: float, samples: int = 96) -> Body:
    radius = size / 2.0
    points = [
        (radius * math.cos(2 * math.pi * i / samples), radius * math.sin(2 * math.pi * i / samples))
        for i in range(samples)
    ]
    return Body(points, ["body"] * len(points), (0.0, 0.0), note="circular cylinder, diameter = size")


def square_body(size: float, per_side: int = 12) -> Body:
    half = size / 2.0
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    points: list[tuple[float, float]] = []
    for index in range(4):
        x0, y0 = corners[index]
        x1, y1 = corners[(index + 1) % 4]
        for step in range(per_side):
            fraction = step / per_side
            points.append((x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction))
    return Body(points, ["body"] * len(points), (0.0, 0.0), note="square section, side = size")


def lshape_body(size: float, arm: float = 0.4, per_side: int = 8) -> Body:
    """An L: two arms of length `size`, each `arm * size` thick.

    The O-grid centre goes in the middle of the corner square, which is the one
    place inside an L that can see all of it.
    """
    long_side = size
    thick = arm * size
    corners = [
        (0.0, 0.0), (long_side, 0.0), (long_side, thick),
        (thick, thick), (thick, long_side), (0.0, long_side),
    ]
    points: list[tuple[float, float]] = []
    for index in range(len(corners)):
        x0, y0 = corners[index]
        x1, y1 = corners[(index + 1) % len(corners)]
        for step in range(per_side):
            fraction = step / per_side
            points.append((x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction))
    return Body(points, ["body"] * len(points), (thick / 2.0, thick / 2.0),
                note="L section, arm length = size")


def vehicle_body(
    size: float,
    height: float = 0.32,
    wheel_half_chord: float = 0.10,
    wheel_depth: float = 0.035,
    rear_wheel: float = 0.24,
    front_wheel: float = 0.76,
) -> Body:
    """A generic 2D car silhouette: flat underbody, two wheel scallops, a sloped
    screen and a tapered tail. Every fraction is of the overall length.

    The wheels are scallops rather than discs, and the body does not touch the
    ground. Both are consequences of using blockMesh: a protruding disc makes the
    outline non-star-shaped and there is then no O-grid to build, and a wheel
    resting on the road cuts the domain in two, which needs a mesher that can
    split blocks. A wheel patch that is a scallop still takes
    `rotatingWallVelocity` and still puts the right tangential velocity into the
    flow underneath it, which is the part the wake cares about.
    """
    scale = size
    tall = scale * (height / 0.32)
    upper = [
        (0.00, 0.06), (0.03, 0.16), (0.10, 0.22), (0.22, 0.24), (0.34, 0.26),
        (0.44, 0.32), (0.60, 0.32), (0.72, 0.26), (0.84, 0.20), (0.94, 0.14),
        (1.00, 0.08), (1.00, 0.00),
    ]

    points: list[tuple[float, float]] = []
    patches: list[str] = []

    def add(point, patch: str) -> None:
        """Append a point and the patch of the edge that leaves it, dropping a
        point that repeats the one before it."""
        if points and abs(point[0] - points[-1][0]) < TOLERANCE and abs(point[1] - points[-1][1]) < TOLERANCE:
            patches[-1] = patch
            return
        points.append((float(point[0]), float(point[1])))
        patches.append(patch)

    def flat_to(x_target: float, patch: str) -> None:
        x_from = points[-1][0] if points else 0.0
        steps = max(2, int(round(abs(x_target - x_from) / (0.03 * scale))))
        for step in range(1, steps + 1):
            add((x_from + (x_target - x_from) * step / steps, 0.0), patch)

    add((0.0, 0.0), "body")
    for name, centre_fraction in (("wheelRear", rear_wheel), ("wheelFront", front_wheel)):
        start_x = (centre_fraction - wheel_half_chord) * scale
        end_x = (centre_fraction + wheel_half_chord) * scale
        flat_to(start_x, "body")
        scallop = arc_points((start_x, 0.0), (end_x, 0.0), wheel_depth * scale, 13)
        patches[-1] = name
        for point in scallop[1:]:
            add(point, name)
        patches[-1] = "body"
    flat_to(scale, "body")
    for x, y in reversed(upper):
        add((x * scale, y * tall), "body")

    centre = (0.5 * scale, 0.14 * tall)
    # The scallop is an arc of a circle whose centre sits above the underbody;
    # rotatingWallVelocity wants that centre, not the lowest point, or the
    # tangential velocity comes out at the wrong angle.
    half_chord = wheel_half_chord * scale
    depth = wheel_depth * scale
    wheel_radius = (half_chord * half_chord + depth * depth) / (2.0 * depth)
    wheel_axis_y = wheel_radius - depth
    wheels = {
        "wheelRear": (rear_wheel * scale, wheel_axis_y, wheel_radius),
        "wheelFront": (front_wheel * scale, wheel_axis_y, wheel_radius),
    }
    return Body(points, patches, centre, extras={"wheels": wheels},
                note="generic car silhouette, length = size, wheels as underbody scallops")


def read_profile(path: Path) -> list[tuple[float, float]]:
    """x,y pairs from a CSV or a Selig-style .dat.

    Anything that is not two numbers on a line is skipped, which quietly disposes
    of the aerofoil name on line one of every .dat file in circulation without
    needing a flag for it.
    """
    points: list[tuple[float, float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        cleaned = line.replace(",", " ").strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        parts = cleaned.split()
        if len(parts) < 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if len(points) < 8:
        raise SystemExit(f"{path}: found {len(points)} x,y pairs; that is not an outline")
    return points


def profile_body(path: Path, size: float, aoa: float = 0.0) -> Body:
    """An imported outline, scaled to `size` across and rotated by `aoa` degrees.

    Rotation is about the quarter-chord point, and it is applied to the geometry rather
    than to the inlet vector so that the tunnel walls stay parallel to the free stream.

    `--aoa 6` is six degrees of **positive incidence**: the leading edge goes up and the
    aerofoil makes positive lift, which is the convention every aerofoil table is
    written in. This used to say "nose down" while doing the opposite -- the code was
    right and the description was backwards, so an agent trusting the docstring would
    negate the angle and report a Cl of the wrong sign.
    """
    raw = drop_repeats(read_profile(path))
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    chord = max(xs) - min(xs)
    if chord <= 0:
        raise SystemExit(f"{path}: the outline has no extent in x")
    factor = size / chord
    shifted = [((x - min(xs)) * factor, (y - (min(ys) + max(ys)) / 2.0) * factor) for x, y in raw]
    pivot = (0.25 * size, 0.0)
    angle = math.radians(-aoa)
    rotated = [
        (
            pivot[0] + (x - pivot[0]) * math.cos(angle) - (y - pivot[1]) * math.sin(angle),
            pivot[1] + (x - pivot[0]) * math.sin(angle) + (y - pivot[1]) * math.cos(angle),
        )
        for x, y in shifted
    ]
    points = as_ccw(rotated)
    return Body(points, ["body"] * len(points), polygon_centroid(points),
                note=f"imported outline from {Path(path).name}, chord = size")


# -- the O-grid around a body ------------------------------------------------------

DIAGONALS = (45.0, 135.0, 225.0, 315.0)
"""Where the ring meets the four straight lines that cut the outer region into a
3x3 grid. They are corners of the block topology, so they are always sector
boundaries whether or not the body has a corner near them."""


def reverse_loop(points, edge_patches):
    """The same closed loop traversed the other way, patches kept with their edges."""
    count = len(points)
    flipped = list(reversed(points))
    return flipped, [edge_patches[(count - 2 - k) % count] for k in range(count)]


def sector_splits(angles, edge_patches, min_separation: float = 3.0):
    """Where to cut the outline into O-grid sectors: (point index, ring angle).

    Two kinds of cut. The four diagonals have to be there because the outer
    blocks meet the ring at exactly those angles, and they are placed at the
    nearest outline point. A change of patch name has to be there too, or a
    single block would carry two patches on one face, which a block cannot.

    When the two land within `min_separation` of each other the patch change
    wins the index and the diagonal keeps its exact angle -- collapsing them
    rather than leaving a sliver sector a fraction of a degree wide, which meshes
    but produces cells no solver is happy with.
    """
    count = len(angles)
    base = angles[0]
    targets = sorted(
        diagonal + 360.0 * math.ceil((base - diagonal) / 360.0) for diagonal in DIAGONALS
    )

    chosen: dict[float, int] = {
        target: min(range(count), key=lambda i: abs(angles[i] - target)) for target in targets
    }
    extra: dict[int, float] = {}
    for index in range(count):
        if edge_patches[index - 1] == edge_patches[index]:
            continue
        near = sorted(targets, key=lambda target: abs(angles[index] - target))
        if abs(angles[index] - near[0]) < min_separation:
            chosen[near[0]] = index
        else:
            extra[index] = angles[index]

    def assemble() -> list[tuple[int, float]]:
        splits: dict[int, float] = dict(extra)
        for target, index in chosen.items():
            splits[index] = target
        return sorted(splits.items())

    # A diagonal sits at its exact angle but at whichever outline point is nearest,
    # and on a coarsely sampled outline that point can fall the far side of a
    # corner -- which reads as a sector of negative width. Give the diagonal the
    # corner's index instead and the two collapse into one cut.
    for _attempt in range(len(targets) + 1):
        ordered = assemble()
        clash = next(
            ((a, b) for a, b in zip(ordered, ordered[1:]) if b[1] <= a[1]),
            None,
        )
        if clash is None:
            return ordered
        (index_a, angle_a), (index_b, angle_b) = clash
        if angle_a in chosen and index_b in extra:
            chosen[angle_a] = index_b
            extra.pop(index_b)
        elif angle_b in chosen and index_a in extra:
            chosen[angle_b] = index_a
            extra.pop(index_a)
        else:
            break
    raise SystemExit(
        "the outline's corners and the O-grid diagonals cannot be reconciled: points "
        f"{index_a} and {index_b} give a sector spanning {angle_b - angle_a:.3f} deg. "
        "Sample the outline more finely, or move the corner."
    )


def spread(total: int, weights: list[float]) -> list[int]:
    """`total` divided among `weights`, at least one each, summing to exactly
    `total`. Largest-remainder, so the rounding lands on the widest piece."""
    if not weights:
        return []
    if total < len(weights):
        total = len(weights)
    raw = [max(1.0, total * w / sum(weights)) for w in weights]
    counts = [max(1, int(math.floor(value))) for value in raw]
    remainders = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    position = 0
    while sum(counts) < total:
        counts[remainders[position % len(remainders)]] += 1
        position += 1
    while sum(counts) > total and any(c > 1 for c in counts):
        biggest = max(range(len(counts)), key=lambda i: counts[i])
        counts[biggest] -= 1
    return counts


def quadrant_of(angle: float) -> str:
    return ("north", "west", "south", "east")[int(((angle % 360.0) - 45.0) % 360.0 // 90.0)]


def external_mesh(
    body: Body,
    *,
    upstream: float,
    downstream: float,
    above: float,
    below: float,
    ogrid_scale: float,
    ogrid_aspect: float,
    n_circ: int,
    n_radial: int,
    radial_grading: float,
    far_grading: float,
    thickness: float,
    min_separation: float = 3.0,
) -> tuple[Mesh2D, dict]:
    """A body in a rectangular tunnel: an O-grid ring on the body, a 3x3 grid of
    blocks around it.

    The ring is an ellipse rather than a circle so that a body close to a wall --
    a car over a moving ground -- can have the ring squashed vertically instead
    of the ground pushed a ring-radius away from it. When the two semi-axes are
    equal the edges are written as `arc`, which is exact; when they are not they
    are written as `polyLine` through sampled points, which is not, and the ring
    is a block edge rather than a wall so the difference is nowhere near the
    answer.
    """
    points = list(body.points)
    patches = list(body.edge_patches)
    if signed_area(points) < 0:
        points, patches = reverse_loop(points, patches)
    centre_x, centre_y = body.centre
    points = [(x - centre_x, y - centre_y) for x, y in points]

    failure = star_shaped_failure(points, (0.0, 0.0))
    if failure is not None:
        index, step = failure
        raise SystemExit(
            f"the outline is not star-shaped about {body.centre}: at point {index} "
            f"{points[index]} the outline turns back by {-step:.2f} deg. An O-grid maps the "
            "outline onto a ring around it, and that map needs every ray from the "
            "centre to cross the outline once."
        )

    count = len(points)
    angles = unwrapped_angles(points, (0.0, 0.0))
    radii = [math.hypot(x, y) for x, y in points]
    radius_x = ogrid_scale * max(radii)
    radius_y = ogrid_aspect * radius_x

    fill = max(
        (x / radius_x) ** 2 + (y / radius_y) ** 2 for x, y in points
    )
    if fill >= 0.95:
        raise SystemExit(
            f"the O-grid ring ({radius_x:.4g} x {radius_y:.4g}) does not clear the body: "
            f"it is {math.sqrt(fill) * 100:.0f}% filled. Raise --ogrid-scale, or "
            "--ogrid-aspect if the ring is too flat."
        )

    x0, x1 = -abs(upstream), abs(downstream)
    y0, y1 = -abs(below), abs(above)
    if x0 > -radius_x or x1 < radius_x or y0 > -radius_y or y1 < radius_y:
        raise SystemExit(
            f"the tunnel [{x0:.4g}, {x1:.4g}] x [{y0:.4g}, {y1:.4g}] is inside the O-grid ring "
            f"({radius_x:.4g} x {radius_y:.4g}). Enlarge the domain or shrink the ring."
        )

    def ring(angle: float) -> tuple[float, float]:
        radians = math.radians(angle)
        return (radius_x * math.cos(radians), radius_y * math.sin(radians))

    circular = abs(radius_x - radius_y) < TOLERANCE * max(1.0, radius_x)
    splits = sector_splits(angles, patches, min_separation=min_separation)

    sectors = []
    for position, (index_a, angle_a) in enumerate(splits):
        if position + 1 < len(splits):
            index_b, angle_b = splits[position + 1]
        else:
            index_b, angle_b = splits[0][0] + count, splits[0][1] + 360.0
        names = {patches[k % count] for k in range(index_a, index_b)}
        if len(names) != 1:
            raise SystemExit(f"sector {position} covers more than one patch: {sorted(names)}")
        sectors.append({
            "from": index_a, "to": index_b,
            "angle_from": angle_a, "angle_to": angle_b,
            "patch": names.pop(),
            "quadrant": quadrant_of((angle_a + angle_b) / 2.0),
        })

    by_quadrant: dict[str, list[dict]] = {}
    for sector in sectors:
        by_quadrant.setdefault(sector["quadrant"], []).append(sector)
    for quadrant, group in by_quadrant.items():
        widths = [sector["angle_to"] - sector["angle_from"] for sector in group]
        for sector, cells in zip(group, spread(int(n_circ), widths)):
            sector["cells"] = cells

    mesh = Mesh2D(thickness)

    for sector in sectors:
        index_a, index_b = sector["from"], sector["to"]
        angle_a, angle_b = sector["angle_from"], sector["angle_to"]
        inner_a, inner_b = points[index_a % count], points[index_b % count]
        outer_a, outer_b = ring(angle_a), ring(angle_b)
        mesh.add_quad(
            [inner_a, outer_a, outer_b, inner_b],
            (int(n_radial), sector["cells"]),
            (radial_grading, 1.0),
            {WEST: sector["patch"]},
        )
        mesh.set_patch_type(sector["patch"], "wall")
        mesh.curve(inner_a, inner_b, [points[k % count] for k in range(index_a, index_b + 1)])
        if circular:
            mesh.arc(outer_a, outer_b, ring((angle_a + angle_b) / 2.0))
        else:
            mesh.curve(outer_a, outer_b,
                       [ring(angle_a + (angle_b - angle_a) * step / 24.0) for step in range(25)])

    corner_x = radius_x / math.sqrt(2.0)
    corner_y = radius_y / math.sqrt(2.0)
    ring_cell = (math.pi / 2.0) * (radius_x + radius_y) / 2.0 / max(1, int(n_circ))
    nx_left = graded_divisions(-corner_x - x0, ring_cell, far_grading)
    nx_right = graded_divisions(x1 - corner_x, ring_cell, far_grading)
    ny_bottom = graded_divisions(-corner_y - y0, ring_cell, far_grading)
    ny_top = graded_divisions(y1 - corner_y, ring_cell, far_grading)
    slow, fast = 1.0 / far_grading, far_grading

    for sector in sectors:
        angle_a, angle_b = sector["angle_from"], sector["angle_to"]
        outer_a, outer_b = ring(angle_a), ring(angle_b)
        cells = sector["cells"]
        if sector["quadrant"] == "south":
            mesh.add_quad([(outer_a[0], y0), (outer_b[0], y0), outer_b, outer_a],
                          (cells, ny_bottom), (1.0, slow), {SOUTH: "bottomWall"})
        elif sector["quadrant"] == "north":
            mesh.add_quad([outer_b, outer_a, (outer_a[0], y1), (outer_b[0], y1)],
                          (cells, ny_top), (1.0, fast), {NORTH: "topWall"})
        elif sector["quadrant"] == "east":
            mesh.add_quad([outer_a, (x1, outer_a[1]), (x1, outer_b[1]), outer_b],
                          (nx_right, cells), (fast, 1.0), {EAST: "outlet"})
        else:
            mesh.add_quad([(x0, outer_b[1]), outer_b, outer_a, (x0, outer_a[1])],
                          (nx_left, cells), (slow, 1.0), {WEST: "inlet"})

    mesh.add_quad([(x0, y0), (-corner_x, y0), (-corner_x, -corner_y), (x0, -corner_y)],
                  (nx_left, ny_bottom), (slow, slow), {SOUTH: "bottomWall", WEST: "inlet"})
    mesh.add_quad([(corner_x, y0), (x1, y0), (x1, -corner_y), (corner_x, -corner_y)],
                  (nx_right, ny_bottom), (fast, slow), {SOUTH: "bottomWall", EAST: "outlet"})
    mesh.add_quad([(corner_x, corner_y), (x1, corner_y), (x1, y1), (corner_x, y1)],
                  (nx_right, ny_top), (fast, fast), {EAST: "outlet", NORTH: "topWall"})
    mesh.add_quad([(x0, corner_y), (-corner_x, corner_y), (-corner_x, y1), (x0, y1)],
                  (nx_left, ny_top), (slow, fast), {NORTH: "topWall", WEST: "inlet"})

    mesh.set_patch_type("inlet", "patch")
    mesh.set_patch_type("outlet", "patch")
    # `patch`, not `wall`, because the 0/ roles give these `slip`: they are the tunnel
    # boundary, not a surface the flow sticks to. Typed `wall` they were counted by
    # `wallDist { method meshWave; }`, so SST's F1/F2 blending near the tunnel used the
    # distance to a boundary carrying slip on k, omega and nut -- no wall function, no
    # viscous sublayer, an inconsistent near-wall state. And a `yPlus` or
    # `wallShearStress` function object defaults to *all* wall patches, so the y+
    # histogram you would check the near-wall mesh with came back diluted with
    # meaningless numbers from these.
    mesh.set_patch_type("topWall", "patch")
    mesh.set_patch_type("bottomWall", "patch")

    info = {
        "ring": (radius_x, radius_y),
        "ring_fill": math.sqrt(fill),
        "domain": (x0, x1, y0, y1),
        "sectors": len(sectors),
        "outer_divisions": (nx_left, nx_right, ny_bottom, ny_top),
        "body_patches": sorted({sector["patch"] for sector in sectors}),
    }
    return mesh, info


# -- ducts and bends ---------------------------------------------------------------


def merged(values: list[float], tolerance: float) -> list[float]:
    """Sorted unique coordinates, values within `tolerance` counted as one."""
    ordered = sorted(values)
    kept = [ordered[0]]
    for value in ordered[1:]:
        if value - kept[-1] > tolerance:
            kept.append(value)
    return kept


def rectilinear_mesh(rectangles, openings, cell: float, thickness: float, wall: str = "walls"):
    """Axis-aligned rectangles tiled into conformal blocks.

    The rectangles a duct is drawn from do not line up: the leg of a T meets the
    branch along part of one face. So every rectangle's edges become grid lines
    for all of them, each rectangle is cut into the grid cells it covers, and
    every block then meets its neighbours face to face -- which is the only thing
    blockMesh will accept without a `mergePatchPairs` that eats the mesh.

    A block face with no neighbour is a boundary. It takes the name of whichever
    opening box its midpoint falls in, and `walls` when it falls in none, so a
    template says where the flow gets in and out and never has to enumerate the
    walls.
    """
    span = max(max(r[2] for r in rectangles) - min(r[0] for r in rectangles),
               max(r[3] for r in rectangles) - min(r[1] for r in rectangles))
    tolerance = 1e-7 * max(1.0, span)
    xs = merged([value for rect in rectangles for value in (rect[0], rect[2])], tolerance)
    ys = merged([value for rect in rectangles for value in (rect[1], rect[3])], tolerance)

    occupied: set[tuple[int, int]] = set()
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mid_x = (xs[i] + xs[i + 1]) / 2.0
            mid_y = (ys[j] + ys[j + 1]) / 2.0
            for x0, y0, x1, y1 in rectangles:
                if x0 - tolerance <= mid_x <= x1 + tolerance and y0 - tolerance <= mid_y <= y1 + tolerance:
                    occupied.add((i, j))
                    break
    if not occupied:
        raise SystemExit("the rectangles given do not enclose anything")

    nx = [divisions(xs[i + 1] - xs[i], cell) for i in range(len(xs) - 1)]
    ny = [divisions(ys[j + 1] - ys[j], cell) for j in range(len(ys) - 1)]

    def opening_at(x: float, y: float) -> str:
        for name, (bx0, by0, bx1, by1), _direction in openings:
            if bx0 - tolerance <= x <= bx1 + tolerance and by0 - tolerance <= y <= by1 + tolerance:
                return name
        return wall

    mesh = Mesh2D(thickness)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            if (i, j) not in occupied:
                continue
            mid_x = (xs[i] + xs[i + 1]) / 2.0
            mid_y = (ys[j] + ys[j + 1]) / 2.0
            sides: dict[int, str] = {}
            if (i, j - 1) not in occupied or j == 0:
                sides[SOUTH] = opening_at(mid_x, ys[j])
            if (i + 1, j) not in occupied:
                sides[EAST] = opening_at(xs[i + 1], mid_y)
            if (i, j + 1) not in occupied:
                sides[NORTH] = opening_at(mid_x, ys[j + 1])
            if (i - 1, j) not in occupied or i == 0:
                sides[WEST] = opening_at(xs[i], mid_y)
            mesh.add_quad(
                [(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])],
                (nx[i], ny[j]),
                (1.0, 1.0),
                sides,
            )
    mesh.set_patch_type(wall, "wall")
    for name, _box, _direction in openings:
        mesh.set_patch_type(name, "patch")
    return mesh


class Plan:
    """A finished mesh plus what its patches are for.

    `roles` is the whole of what the 0/ writer needs: a patch is an inlet with a
    direction, an outlet, a no-slip wall, a wall that translates, or a wall that
    spins. Keeping it here rather than deriving it from the patch names means a
    renamed patch does not silently become a wall.
    """

    def __init__(self, mesh: Mesh2D, roles: dict, length: float, info: dict, notes=None):
        self.mesh = mesh
        self.roles = roles
        self.length = float(length)
        self.info = dict(info)
        self.notes = list(notes or [])


def hydraulic_diameter(width: float) -> float:
    """The length a duct's Reynolds number is actually defined on.

    For a 2D channel between parallel plates D_h = 4A/P -> 2w, not w. The templates
    passed the width, so `nu = U*L/Re` came out twice too large and every duct and bend
    ran at **twice** the Reynolds number it reported. It compounded: `LAMINAR_BELOW` is
    itself a D_h criterion, so `--reynolds 2000` on a duct was really Re_Dh = 4000 --
    turbulent -- and the tool quietly chose `laminar` for it. The mixing length was
    half what it should be for the same reason."""
    return 2.0 * float(width)


def duct_plan(name: str, opts) -> Plan:
    """Y, T, Z, F and M ducts, all 2D, all built from the same rectangle tiler."""
    width = opts["duct_width"]
    cell = width / max(1, int(opts["cells_across"]))
    leg = opts["leg_length"] * width
    branch = opts["branch_length"] * width
    openings: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float]]] = []

    if name == "duct-t":
        left = (leg - width) / 2.0
        rectangles = [(0.0, 0.0, leg, width), (left, width, left + width, width + branch)]
        openings = [
            ("inlet", (0.0, 0.0, 0.0, width), (1.0, 0.0, 0.0)),
            ("outlet", (leg, 0.0, leg, width), (1.0, 0.0, 0.0)),
            ("outlet2", (left, width + branch, left + width, width + branch), (0.0, 1.0, 0.0)),
        ]
    elif name == "duct-z":
        rise = max(2.0 * width, opts["offset"] * width)
        rectangles = [
            (0.0, 0.0, leg, width),
            (leg, 0.0, leg + width, rise + width),
            (leg + width, rise, leg + width + leg, rise + width),
        ]
        openings = [
            ("inlet", (0.0, 0.0, 0.0, width), (1.0, 0.0, 0.0)),
            ("outlet", (2 * leg + width, rise, 2 * leg + width, rise + width), (1.0, 0.0, 0.0)),
        ]
    elif name == "duct-f":
        spine = leg
        arm = branch
        first, second = 0.30 * spine, 0.62 * spine
        rectangles = [
            (0.0, 0.0, width, spine),
            (width, first, width + arm, first + width),
            (width, second, width + arm, second + width),
        ]
        openings = [
            ("inlet", (0.0, 0.0, width, 0.0), (0.0, 1.0, 0.0)),
            ("outlet", (0.0, spine, width, spine), (0.0, 1.0, 0.0)),
            ("outlet2", (width + arm, first, width + arm, first + width), (1.0, 0.0, 0.0)),
            ("outlet3", (width + arm, second, width + arm, second + width), (1.0, 0.0, 0.0)),
        ]
    elif name == "duct-m":
        height = branch
        step = opts["offset"] * width
        valley = 0.45 * height
        rectangles = [
            (0.0, 0.0, width, height),
            (width, height - width, step, height),
            (step, valley, step + width, height),
            (step + width, valley, 2 * step, valley + width),
            (2 * step, valley, 2 * step + width, height),
            (2 * step + width, height - width, 3 * step, height),
            (3 * step, 0.0, 3 * step + width, height),
        ]
        openings = [
            ("inlet", (0.0, 0.0, width, 0.0), (0.0, 1.0, 0.0)),
            ("outlet", (3 * step, 0.0, 3 * step + width, 0.0), (0.0, 1.0, 0.0)),
        ]
    elif name == "duct-y":
        return y_duct_plan(opts)
    else:
        raise SystemExit(f"no duct template called {name}")

    mesh = rectilinear_mesh(rectangles, openings, cell, opts["thickness"])
    roles = {"walls": {"kind": "wall"}, "frontAndBack": {"kind": "empty"}}
    for patch, _box, direction in openings:
        roles[patch] = ({"kind": "inlet", "direction": direction} if patch == "inlet"
                        else {"kind": "outlet"})
    info = {"rectangles": rectangles, "duct_width": width, "cell": cell}
    return Plan(mesh, roles, hydraulic_diameter(width), info,
                [f"{name}: {len(mesh.blocks)} blocks from {len(rectangles)} rectangles",
                 f"Re is on the hydraulic diameter, {hydraulic_diameter(width):g} m "
                 f"(2x the {width:g} m width), as a channel Reynolds number is defined"])


def y_duct_plan(opts) -> Plan:
    """A Y: one stem, a square junction, two arms leaving at +/- the branch angle.

    The arms are parallelograms rather than rectangles turned on their side --
    their end faces stay vertical -- because a rectangle rotated about the
    junction does not meet the junction face to face, and the mesh then needs a
    merge that costs more than the skew does.
    """
    width = opts["duct_width"]
    cell = width / max(1, int(opts["cells_across"]))
    n_cross = max(1, int(opts["cells_across"]))
    stem = opts["leg_length"] * width
    arm_length = opts["branch_length"] * width
    angle = math.radians(opts["branch_angle"])
    n_stem = divisions(stem, cell)
    n_arm = divisions(arm_length, cell)
    half = width / 2.0
    step_x, step_y = arm_length * math.cos(angle), arm_length * math.sin(angle)

    mesh = Mesh2D(opts["thickness"])
    mesh.add_quad([(0.0, -half), (stem, -half), (stem, half), (0.0, half)],
                  (n_stem, n_cross), (1.0, 1.0),
                  {SOUTH: "walls", NORTH: "walls", WEST: "inlet"})
    mesh.add_quad([(stem, -half), (stem + width, -half), (stem + width, half), (stem, half)],
                  (n_cross, n_cross), (1.0, 1.0), {EAST: "walls"})
    mesh.add_quad(
        [(stem + width, half), (stem + width + step_x, half + step_y),
         (stem + step_x, half + step_y), (stem, half)],
        (n_arm, n_cross), (1.0, 1.0),
        {SOUTH: "walls", EAST: "outlet", NORTH: "walls"},
    )
    mesh.add_quad(
        [(stem, -half), (stem + step_x, -half - step_y),
         (stem + width + step_x, -half - step_y), (stem + width, -half)],
        (n_arm, n_cross), (1.0, 1.0),
        {SOUTH: "walls", EAST: "outlet2", NORTH: "walls"},
    )
    mesh.set_patch_type("walls", "wall")
    mesh.set_patch_type("inlet", "patch")
    mesh.set_patch_type("outlet", "patch")
    mesh.set_patch_type("outlet2", "patch")
    roles = {
        "inlet": {"kind": "inlet", "direction": (1.0, 0.0, 0.0)},
        "outlet": {"kind": "outlet"},
        "outlet2": {"kind": "outlet"},
        "walls": {"kind": "wall"},
        "frontAndBack": {"kind": "empty"},
    }
    return Plan(mesh, roles, hydraulic_diameter(width),
                {"duct_width": width, "cell": cell, "branch_angle": opts["branch_angle"]},
                [f"arms leave at +/- {opts['branch_angle']:g} deg from the stem",
                 f"Re is on the hydraulic diameter, {hydraulic_diameter(width):g} m"])


def bend_plan(name: str, opts) -> Plan:
    """Three 90 degree bends of the same duct width, differing only in the corner."""
    width = opts["duct_width"]
    n_cross = max(1, int(opts["cells_across"]))
    cell = width / n_cross
    leg_in = opts["leg_length"] * width
    leg_out = opts["branch_length"] * width
    thickness = opts["thickness"]

    if name == "bend-sharp":
        rectangles = [
            (0.0, 0.0, leg_in, width),
            (leg_in, 0.0, leg_in + width, width),
            (leg_in, width, leg_in + width, width + leg_out),
        ]
        openings = [
            ("inlet", (0.0, 0.0, 0.0, width), (1.0, 0.0, 0.0)),
            ("outlet", (leg_in, width + leg_out, leg_in + width, width + leg_out), (0.0, 1.0, 0.0)),
        ]
        mesh = rectilinear_mesh(rectangles, openings, cell, thickness)
        notes = ["square outer corner and square inner corner"]
        info = {"rectangles": rectangles}
    elif name == "bend-mitred":
        mesh = Mesh2D(thickness)
        n_in = divisions(leg_in, cell)
        n_out = divisions(leg_out, cell)
        mesh.add_quad([(-leg_in, 0.0), (width, 0.0), (0.0, width), (-leg_in, width)],
                      (n_in, n_cross), (1.0, 1.0),
                      {SOUTH: "walls", NORTH: "walls", WEST: "inlet"})
        mesh.add_quad([(0.0, width), (width, 0.0), (width, leg_out), (0.0, leg_out)],
                      (n_cross, n_out), (1.0, 1.0),
                      {EAST: "walls", NORTH: "outlet", WEST: "walls"})
        mesh.set_patch_type("walls", "wall")
        mesh.set_patch_type("inlet", "patch")
        mesh.set_patch_type("outlet", "patch")
        notes = ["the two legs are butt-joined on the 45 degree plane; no corner radius"]
        info = {"mitre_plane": ((0.0, width), (width, 0.0))}
    elif name == "bend-rounded":
        inner = opts["bend_radius"]
        if inner <= 0:
            raise SystemExit("--bend-radius must be positive for bend-rounded")
        outer = inner + width
        mesh = Mesh2D(thickness)
        n_in = divisions(leg_in, cell)
        n_out = divisions(leg_out, cell)
        n_bend = max(4, divisions(0.5 * math.pi * (inner + outer) / 2.0, cell))
        mesh.add_quad([(-leg_in, -outer), (0.0, -outer), (0.0, -inner), (-leg_in, -inner)],
                      (n_in, n_cross), (1.0, 1.0),
                      {SOUTH: "walls", NORTH: "walls", WEST: "inlet"})
        mesh.add_quad([(0.0, -outer), (outer, 0.0), (inner, 0.0), (0.0, -inner)],
                      (n_bend, n_cross), (1.0, 1.0), {SOUTH: "walls", NORTH: "walls"})
        diagonal = math.sqrt(0.5)
        mesh.arc((0.0, -outer), (outer, 0.0), (outer * diagonal, -outer * diagonal))
        mesh.arc((inner, 0.0), (0.0, -inner), (inner * diagonal, -inner * diagonal))
        mesh.add_quad([(inner, 0.0), (outer, 0.0), (outer, leg_out), (inner, leg_out)],
                      (n_cross, n_out), (1.0, 1.0),
                      {EAST: "walls", NORTH: "outlet", WEST: "walls"})
        mesh.set_patch_type("walls", "wall")
        mesh.set_patch_type("inlet", "patch")
        mesh.set_patch_type("outlet", "patch")
        notes = [f"inner radius {inner:g} m, outer {outer:g} m, r/D = {inner / width:.2f}"]
        info = {"inner_radius": inner, "outer_radius": outer}
    else:
        raise SystemExit(f"no bend template called {name}")

    roles = {
        "inlet": {"kind": "inlet", "direction": (1.0, 0.0, 0.0)},
        "outlet": {"kind": "outlet"},
        "walls": {"kind": "wall"},
        "frontAndBack": {"kind": "empty"},
    }
    info["duct_width"] = width
    notes = list(notes) + [f"Re is on the hydraulic diameter, {hydraulic_diameter(width):g} m"]
    return Plan(mesh, roles, hydraulic_diameter(width), info, notes)


# -- external-flow templates -------------------------------------------------------


def external_plan(name: str, opts) -> Plan:
    """A body in a rectangular tunnel, wrapped in an O-grid.

    Everything the mesh builder needs is in `opts`, and everything the 0/ writer
    needs comes back in `roles`. `body` is one patch unless the template splits it:
    the vehicle names its two wheel scallops so they can spin independently of the
    shell they are cut into.
    """
    size = opts["size"]
    if name == "circle":
        body = circle_body(size)
    elif name == "square":
        body = square_body(size)
    elif name == "lshape":
        body = lshape_body(size)
    elif name == "vehicle":
        body = vehicle_body(size)
    elif name == "profile":
        path = opts.get("profile")
        if not path:
            raise SystemExit("the profile template needs --profile FILE (x,y points)")
        body = profile_body(Path(path), size, aoa=opts["aoa"])
    else:
        raise SystemExit(f"no external template called {name}")

    mesh, info = external_mesh(
        body,
        upstream=opts["upstream"] * size,
        downstream=opts["downstream"] * size,
        above=opts["above"] * size,
        below=opts["below"] * size,
        ogrid_scale=opts["ogrid_scale"],
        ogrid_aspect=opts["ogrid_aspect"],
        n_circ=opts["cells_around"],
        n_radial=opts["cells_radial"],
        radial_grading=opts["radial_grading"],
        far_grading=opts["far_grading"],
        thickness=opts["thickness"],
    )

    roles: dict[str, dict] = {
        "inlet": {"kind": "inlet", "direction": (1.0, 0.0, 0.0)},
        "outlet": {"kind": "outlet"},
        "topWall": {"kind": "slip"},
        "bottomWall": {"kind": "slip"},
        "frontAndBack": {"kind": "empty"},
    }
    for patch in info["body_patches"]:
        roles[patch] = {"kind": "wall"}

    # A wind-tunnel floor that moves with the stream. On a mesh that does not move,
    # the belt is a fixed tangential velocity and nothing else -- `movingWallVelocity`
    # is for a mesh that is actually moving and quietly reduces to no-slip here,
    # which is the kind of silence this file exists to avoid.
    if opts.get("moving_ground"):
        roles["bottomWall"] = {"kind": "belt"}

    # A scallop that spins. `rotatingWallVelocity` is right on a static mesh: it puts
    # the tangential velocity of the wheel surface into the flow under the car, which
    # is the part of a wheel the wake notices.
    if opts.get("rotating_wheels"):
        wheels = body.extras.get("wheels") or {}
        if not wheels:
            raise SystemExit(f"the {name} template has no wheels to rotate")
        centre_x, centre_y = body.centre
        for patch, (wheel_x, wheel_y, radius) in wheels.items():
            if patch not in roles:
                continue
            roles[patch] = {
                "kind": "spinning",
                # external_mesh works in body-centred coordinates; so must the origin.
                "origin": (wheel_x - centre_x, wheel_y - centre_y, 0.0),
                "axis": (0.0, 0.0, 1.0),
                "radius": radius,
            }

    notes = [body.note] if body.note else []
    notes.append(
        f"O-grid ring {info['ring'][0]:.4g} x {info['ring'][1]:.4g} m, "
        f"{info['ring_fill'] * 100:.0f}% filled by the body, {info['sectors']} sectors"
    )
    info["body_size"] = size
    return Plan(mesh, roles, size, info, notes)


# -- the templates -----------------------------------------------------------------

EXTERNAL = ("circle", "square", "lshape", "vehicle", "profile")

FREE_STREAM_INTENSITY = 0.001

DUCT_INTENSITY = 0.05
"""Turbulence intensity for a flow the geometry confines.

A tube bank, a duct, an exchanger: past the first row the turbulence is made
by the geometry, not brought in from outside, and the free-stream value is
both wrong and quietly so -- it starves the entry rows and the coefficient
comes out low with nothing in the output pointing at it."""

"""Turbulence intensity for a body in unbounded flow: 0.1%.

The default used to be 5% for everything, which is wind-tunnel-grid turbulence, not
free stream. Paired with a pipe mixing length it put the ambient eddy viscosity three
to four orders of magnitude above molecular on every external case."""

FREE_STREAM_VISCOSITY_RATIO = 10.0
"""Target nu_t/nu in the free stream. NASA's Turbulence Modeling Resource gives 0.1-10
as the useful band for external aerodynamics; 10 is the forgiving end of it, because
too little ambient turbulence can stall a k-omega SST solve early on."""


def is_external(name: str) -> bool:
    return name in EXTERNAL


def free_stream_intensity(name: str, opts) -> float:
    """What was asked for, else 0.1% outside a duct and 5% inside one.

    Internal flows really do carry percent-level turbulence; a body in open air does not,
    and using the duct number for both is what made every external case wrong."""
    asked = opts.get("turbulent_intensity")
    if asked is not None:
        return float(asked)
    return FREE_STREAM_INTENSITY if is_external(name) else 0.05


def viscosity_ratio(name: str, opts) -> float | None:
    """The nu_t/nu the free stream is set from, or None to use the mixing length."""
    asked = opts.get("viscosity_ratio")
    if asked is not None:
        return float(asked)
    return FREE_STREAM_VISCOSITY_RATIO if is_external(name) else None
DUCTS = ("duct-y", "duct-t", "duct-z", "duct-f", "duct-m")
BENDS = ("bend-sharp", "bend-mitred", "bend-rounded")

TEMPLATES: dict[str, dict] = {
    "circle": {"family": "external", "what": "a circular cylinder across the stream"},
    "square": {"family": "external", "what": "a square cylinder, faces normal to the stream"},
    "lshape": {"family": "external", "what": "an L-section, the re-entrant corner facing downstream"},
    "vehicle": {"family": "external", "what": "a generic car silhouette; --moving-ground and --rotating-wheels apply here"},
    "profile": {"family": "external", "what": "an outline read from a file of x,y points; --profile and --aoa"},
    "duct-y": {"family": "duct", "what": "one stem splitting into two arms at +/- --branch-angle"},
    "duct-t": {"family": "duct", "what": "a stem meeting a crossbar; two outlets"},
    "duct-z": {"family": "duct", "what": "two offset straights joined by a link"},
    "duct-f": {"family": "duct", "what": "a spine with two side branches on one side"},
    "duct-m": {"family": "duct", "what": "two peaks; four vertical legs off one floor"},
    "bend-sharp": {"family": "bend", "what": "90 degrees, square inner and outer corners"},
    "bend-mitred": {"family": "bend", "what": "90 degrees, the two legs butt-joined on the 45 degree plane"},
    "bend-rounded": {"family": "bend", "what": "90 degrees with an inner radius; --bend-radius"},
}

FAMILY_PARAMS = {
    "external": ("--size", "--upstream/--downstream/--above/--below (in body sizes)",
                 "--ogrid-scale", "--ogrid-aspect", "--cells-around", "--cells-radial"),
    "duct": ("--duct-width", "--cells-across", "--leg-length", "--branch-length",
             "--branch-angle", "--offset"),
    "bend": ("--duct-width", "--cells-across", "--leg-length", "--branch-length", "--bend-radius"),
}


EXTERNAL_ONLY_FLAGS = (
    ("moving_ground", "--moving-ground", "a ground to move"),
    ("rotating_wheels", "--rotating-wheels", "wheels to spin"),
)
"""Flags that only mean something to the external-flow family. A duct has no
ground and no wheels, and accepting the flag there and doing nothing is worse than
refusing it: the case comes out looking exactly like the one that was asked for."""


def build_plan(name: str, opts) -> Plan:
    """The mesh and its patch roles for one template name."""
    entry = TEMPLATES.get(name)
    if entry is None:
        raise SystemExit(f"no template called {name}; `--list` shows them all")
    if entry["family"] != "external":
        for key, flag, what in EXTERNAL_ONLY_FLAGS:
            if opts.get(key):
                raise SystemExit(
                    f"{flag} is for the external-flow templates ({', '.join(EXTERNAL)}); "
                    f"{name} has no {what}"
                )
    if entry["family"] == "external":
        return external_plan(name, opts)
    if entry["family"] == "duct":
        return duct_plan(name, opts)
    return bend_plan(name, opts)


# -- the numbers, and which of them was derived ------------------------------------


class Flow:
    """Speed, length and viscosity, with the one that was calculated named.

    Three numbers with one relation between them, so two of them are given and the
    third is arithmetic. Which one that was is worth carrying around: a case whose
    viscosity was solved for a Reynolds number and a case whose Reynolds number came
    out of a viscosity are the same files and different intentions, and the printed
    line is the only place the difference survives.
    """

    def __init__(self, speed: float, length: float, nu: float, reynolds: float, derived: str):
        self.speed = float(speed)
        self.length = float(length)
        self.nu = float(nu)
        self.reynolds = float(reynolds)
        self.derived = derived

    def line(self) -> str:
        what = {"nu": "nu = U*L/Re", "reynolds": "Re = U*L/nu"}[self.derived]
        return (f"U = {self.speed:g} m/s, L = {self.length:g} m, nu = {self.nu:g} m2/s, "
                f"Re = {self.reynolds:.0f}   ({what})")


def derive_flow(opts, length: float) -> Flow:
    """Solve for whichever of nu and Re was not given."""
    speed = float(opts["speed"])
    length = float(opts.get("length") or length)
    if speed <= 0 or length <= 0:
        raise SystemExit("--speed and the characteristic length must both be positive")
    nu = opts.get("nu")
    reynolds = opts.get("reynolds")
    if nu is not None and reynolds is not None:
        raise SystemExit("give --nu or --reynolds, not both: the other one follows from it")
    if nu is not None:
        nu = float(nu)
        if nu <= 0:
            raise SystemExit("--nu must be positive")
        return Flow(speed, length, nu, speed * length / nu, "reynolds")
    reynolds = float(reynolds if reynolds is not None else 100.0)
    if reynolds <= 0:
        raise SystemExit("--reynolds must be positive")
    return Flow(speed, length, speed * length / reynolds, reynolds, "nu")


LAMINAR_BELOW = 2300.0
"""Where the automatic choice of turbulence model changes. Not a law -- transition
depends on the geometry and on what is upstream -- but a case run laminar at Re=10^6
and a case run kOmegaSST at Re=40 are both wrong in ways that take a run to notice,
and `--turbulence` overrides it."""


def turbulence_model(opts, flow: Flow) -> tuple[str, str]:
    """The model and the sentence saying why it is that one."""
    asked = (opts.get("turbulence") or "auto").strip()
    if asked != "auto":
        return asked, f"--turbulence {asked}"
    if flow.reynolds < LAMINAR_BELOW:
        return "laminar", f"Re = {flow.reynolds:.0f} is below {LAMINAR_BELOW:g}"
    return "kOmegaSST", f"Re = {flow.reynolds:.0f} is above {LAMINAR_BELOW:g}"


TURBULENCE_FIELDS = {
    "kOmegaSST": ("k", "omega"),
    "kOmegaSSTLM": ("k", "omega", "gammaInt", "ReThetat"),
    "kEpsilon": ("k", "epsilon"),
    "SpalartAllmaras": ("nuTilda",),
}
"""`kOmegaSSTLM` is kOmegaSST with the Langtry-Menter transition equations bolted on,
and it exists here because below Re ~ 5e5 a fully turbulent model is not merely less
accurate, it is answering a different question. A blade section at Re 6e4 carries a
laminar separation bubble over much of its chord; assume it turbulent from the leading
edge and the lift comes out low no matter how fine the mesh. It costs two extra
transported fields."""


TRANSITION_MODELS = ("kOmegaSSTLM",)


def free_stream_re_theta(intensity: float) -> float:
    """Inlet ReThetat from turbulence intensity, Langtry & Menter's own correlation.

    Tu is in PERCENT here, which is the convention the correlation is written in and
    a factor of 100 waiting to happen -- the 1/Tu^2 term makes getting it wrong
    spectacular rather than subtle."""
    tu = max(intensity * 100.0, 0.027)
    if tu <= 1.3:
        return 1173.51 - 589.428 * tu + 0.2196 / (tu * tu)
    return 331.50 * (tu - 0.5658) ** -0.671
"""The fields each model transports, under the names it looks them up by.

Not interchangeable and not optional. kEpsilon reads `epsilon` and never reads
`omega`; SpalartAllmaras reads neither and reads `nuTilda`. Writing the wrong pair
is not a run that converges badly -- the solver stops before the first iteration
saying it cannot find a file, and the same list has to drive 0/, the divSchemes and
the linear solvers or one of the three is left describing a different case."""


def turbulence_fields(model: str) -> tuple[str, ...]:
    if model == "laminar":
        return ()
    return TURBULENCE_FIELDS.get(model, ("k", "omega"))


def nut_wall_function(model: str) -> str:
    """Spalding's law, for every model.

    `nutkWallFunction` is the plain high-Re Launder-Spalding form and is only valid for
    a first cell landing around y+ 30-300. On a generated aerofoil O-grid the first cell
    centre sits at y+ ~1200 and the boundary layer is thinner than one cell, so skin
    friction -- and therefore Cd -- came from an extrapolation the model was never valid
    for. A live study measured y+ 40-1713 on exactly this and said so itself.

    `nutUSpaldingWallFunction` blends through the viscous sublayer, the buffer layer and
    the log layer, so it is right across the whole range rather than only inside a band
    the generator cannot guarantee. It also needs no k, which is why Spalart-Allmaras
    already had it. Nothing is lost by using it everywhere: on a mesh that *is* in the
    log layer the two agree.
    """
    return "nutUSpaldingWallFunction"


# -- the 0/ directory --------------------------------------------------------------


def boundary_field(plan: Plan, entry) -> str:
    """A `boundaryField` block with one entry per patch in the mesh.

    Driven by the mesh's own patch list rather than by the roles, so a patch the
    template made and forgot to describe is a loud KeyError here instead of a
    missing entry the solver finds on the first time step.
    """
    lines = ["boundaryField", "{"]
    for name in patch_order(plan.mesh.patch_faces):
        role = plan.roles.get(name)
        if role is None:
            raise SystemExit(
                f"the mesh has a patch '{name}' that the template did not give a role; "
                "every patch needs one, or its 0/ entries are guesses"
            )
        body = entry(name, role)
        lines.append(f"    {name}")
        lines.append("    {")
        for item in body:
            lines.append(f"        {item}")
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def stream_direction(plan: Plan) -> tuple[float, float, float]:
    """The direction the inlet points, or +x when a template has no inlet.

    The internal field and the outlet's fall-back value are seeded with this. A
    duct whose inlet is on the floor (`duct-f`, `duct-m`) runs up the y axis, and
    starting every cell in it at (U 0 0) points the whole domain across the duct
    instead of along it -- which converges, eventually, from further away.
    """
    for name in patch_order(plan.roles):
        role = plan.roles[name]
        if role.get("kind") == "inlet":
            direction = tuple(float(c) for c in role.get("direction", (1.0, 0.0, 0.0)))
            length = math.sqrt(sum(c * c for c in direction))
            if length > TOLERANCE:
                return (direction[0] / length, direction[1] / length, direction[2] / length)
    return (1.0, 0.0, 0.0)


def field_U(plan: Plan, flow: Flow) -> str:
    stream = tuple(component * flow.speed for component in stream_direction(plan))
    inlet = f"uniform {vector(stream)}"

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            direction = role.get("direction", (1.0, 0.0, 0.0))
            value = tuple(component * flow.speed for component in direction)
            return ["type            fixedValue;", f"value           uniform {vector(value)};"]
        if kind == "outlet":
            return ["type            inletOutlet;",
                    "inletValue      uniform (0 0 0);",
                    f"value           {inlet};"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "belt":
            return ["type            fixedValue;", f"value           {inlet};"]
        if kind == "spinning":
            origin, axis = role["origin"], role["axis"]
            # omega is the wheel's, and a rolling wheel's is the road speed over its
            # radius: a number that follows from --speed rather than one to type in.
            omega = flow.speed / float(role["radius"])
            return ["type            rotatingWallVelocity;",
                    f"origin          {vector(origin)};",
                    f"axis            {vector(axis)};",
                    f"omega           {omega:.6g};",
                    "value           uniform (0 0 0);"]
        return ["type            noSlip;"]

    body = ("dimensions      [0 1 -1 0 0 0 0];\n\n"
            f"internalField   {inlet};\n\n" + boundary_field(plan, entry))
    return foam_file("volVectorField", "U", body, "0")


def field_p(plan: Plan) -> str:
    """Kinematic pressure: incompressible OpenFOAM solves p/rho, in m2/s2. Naming
    the units here is not decoration -- a force computed from this as if it were
    pascals is out by a factor of rho and looks plausible."""

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            # `symmetry`, not `symmetryPlane`: the latter is a constraint that
            # requires its faces to be coplanar, so it cannot carry the two
            # opposite walls of a tunnel as one patch, and blockMesh rejects it
            # outright. The physics is identical.
            # A symmetry patch takes the constraint on every field it
            # carries, p included: OpenFOAM refuses a zeroGradient there
            # rather than quietly accepting it.
            return ["type            symmetry;"]
        if kind == "outlet":
            return ["type            fixedValue;", "value           uniform 0;"]
        if kind == "inlet":
            return ["type            zeroGradient;"]
        return ["type            zeroGradient;"]

    body = ("dimensions      [0 2 -2 0 0 0 0];   // kinematic: p/rho, m2/s2\n\n"
            "internalField   uniform 0;\n\n" + boundary_field(plan, entry))
    return foam_file("volScalarField", "p", body, "0")


CMU = 0.09
"""The k-epsilon family's model constant. The omega and epsilon estimates are the same
mixing-length argument written twice."""


def wall_function(name: str) -> list[str]:
    return [f"type            {name};", "value           $internalField;"]


def field_k(plan: Plan, flow: Flow, intensity: float) -> str:
    value = 1.5 * (intensity * flow.speed) ** 2

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function("kqRWallFunction")

    body = (f"dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform {value:.6g};\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "k", body, "0")


def omega_for(k: float, nu: float, mixing: float, ratio: float | None) -> float:
    """omega for the free stream.

    `ratio` is a target eddy-viscosity ratio nu_t/nu, and when it is given omega comes
    straight from it: nu_t = k/omega, so omega = k/(ratio*nu). That is the external-aero
    way to state free-stream turbulence, and it is stated because the other way was
    silently wrong here: omega = sqrt(k)/(Cmu^0.25 * l) with l = 0.07*L is the
    fully-developed **pipe** mixing-length correlation, applied to a body length in
    unbounded flow. With the old 5% intensity default it put nu_t/nu at ~7,000 on every
    turbulent external case -- against a recommended free-stream band of 0.1 to 10.

    Two live studies confirmed it and neither noticed: a NACA 0012 came back with Cd
    seven times the published value, and an Ahmed body 41% high with its 25-degree slant
    separation diffused away. The mixing-length form is kept for the internal flows it
    is actually about, and for anyone who asks for it with --mixing-length.
    """
    if ratio and nu > 0:
        return k / (ratio * nu)
    return math.sqrt(k) / (CMU ** 0.25 * mixing)


def field_omega(plan: Plan, flow: Flow, intensity: float, mixing: float,
                nu: float = 0.0, ratio: float | None = None) -> str:
    k = 1.5 * (intensity * flow.speed) ** 2
    value = omega_for(k, nu, mixing, ratio)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function("omegaWallFunction")

    body = (f"dimensions      [0 0 -1 0 0 0 0];\n\ninternalField   uniform {value:.6g};\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "omega", body, "0")


def field_epsilon(plan: Plan, flow: Flow, intensity: float, mixing: float) -> str:
    """epsilon for kEpsilon, from the same k and mixing length omega comes from:
    epsilon = Cmu^0.75 k^1.5 / L."""
    k = 1.5 * (intensity * flow.speed) ** 2
    value = CMU ** 0.75 * k ** 1.5 / mixing

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function("epsilonWallFunction")

    body = (f"dimensions      [0 2 -3 0 0 0 0];\n\ninternalField   uniform {value:.6g};\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "epsilon", body, "0")


SA_NU_TILDA_RATIO = 4.0
"""Free-stream nuTilda as a multiple of nu. Spalart's own recommendation is between
3 and 5 times; the middle of it is a starting value, not a measurement."""


def field_nu_tilda(plan: Plan, flow: Flow) -> str:
    value = SA_NU_TILDA_RATIO * flow.nu

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "inlet":
            return ["type            fixedValue;", "value           $internalField;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        # Spalart-Allmaras carries nuTilda to zero at a wall; there is no wall
        # function for it, the wall function is on nut.
        return ["type            fixedValue;", "value           uniform 0;"]

    body = (f"dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform {value:.6g};   "
            f"// {SA_NU_TILDA_RATIO:g} * nu\n\n" + boundary_field(plan, entry))
    return foam_file("volScalarField", "nuTilda", body, "0")


def field_gamma_int(plan: Plan) -> str:
    """Intermittency: 1 in the free stream, and the model decides where it falls."""

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "inlet":
            return ["type            inletOutlet;", "inletValue      uniform 1;",
                    "value           uniform 1;"]
        if kind == "outlet":
            return ["type            inletOutlet;", "inletValue      uniform 1;",
                    "value           uniform 1;"]
        return ["type            zeroGradient;"]

    body = ("dimensions      [0 0 0 0 0 0 0];\n\ninternalField   uniform 1;\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "gammaInt", body, "0")


def field_re_theta(plan: Plan, intensity: float) -> str:
    value = free_stream_re_theta(intensity)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind in ("inlet", "outlet"):
            return ["type            inletOutlet;", "inletValue      $internalField;",
                    "value           $internalField;"]
        return ["type            zeroGradient;"]

    body = (f"dimensions      [0 0 0 0 0 0 0];\n\ninternalField   uniform "
            f"{value:.6g};   // Langtry-Menter, Tu = {intensity * 100:g}%\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "ReThetat", body, "0")


def field_nut(plan: Plan, model: str) -> str:
    wall = nut_wall_function(model)

    def entry(name: str, role: dict) -> list[str]:
        kind = role["kind"]
        if kind == "empty":
            return ["type            empty;"]
        if kind in ("inlet", "outlet"):
            return ["type            calculated;", "value           uniform 0;"]
        if kind == "slip":
            return ["type            slip;"]
        if kind == "symmetry":
            return ["type            symmetry;"]
        return wall_function(wall)

    body = ("dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
            + boundary_field(plan, entry))
    return foam_file("volScalarField", "nut", body, "0")


# -- system and constant -----------------------------------------------------------

SOLVERS = {"steady": "simpleFoam", "transient": "pimpleFoam", "mesh": ""}


def function_objects(plan: Plan, flow: Flow, opts, study: str) -> str:
    """`forceCoeffs` and a residual log, written into the case that is meant to answer
    for them.

    controlDict had no `functions` block at all, while `preflight.py` warns when a
    forceCoeffs object is missing and `results.py` reads `forceCoeffs.dat` -- both ends
    of the pipeline assumed one existed and only the generator omitted it. So every
    study hand-wrote its own, including `Aref = size * thickness`, where the thickness
    defaults to size/10: an easy factor of ten that silently rescales every coefficient.

    `lRef` and `Aref` are written from the geometry this template actually built, so the
    coefficients mean what their names say without anybody re-deriving them. Only the
    external templates get forces -- a duct has no body to take them on.
    """
    body = str(opts.get("body_patch") or "body")
    if body not in plan.mesh.patch_face_counts():
        return ""
    area = plan.length * plan.mesh.thickness
    magnitude = flow.speed
    entries = [
        "functions", "{",
        "    forceCoeffs", "    {",
        "        type            forceCoeffs;",
        "        libs            (forces);",
        f"        patches         ({body});",
        "        rho             rhoInf;",
        "        rhoInf          1;",
        "        liftDir         (0 1 0);",
        "        dragDir         (1 0 0);",
        "        CofR            (0 0 0);",
        "        pitchAxis       (0 0 1);",
        f"        magUInf         {magnitude:g};",
        f"        lRef            {plan.length:.6g};",
        f"        Aref            {area:.6g};   // {plan.length:g} m span-chord x "
        f"{plan.mesh.thickness:g} m thickness",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "    }",
        "    residuals", "    {",
        "        type            solverInfo;",
        "        libs            (utilityFunctionObjects);",
        "        fields          (U p);",
        "        writeResidualFields false;",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "    }",
        "}",
    ]
    return "\n" + "\n".join(entries)


def control_dict(plan: Plan, flow: Flow, opts, study: str) -> str:
    """The run's clock.

    A steady run counts iterations; a transient one counts seconds, and its deltaT
    is the Courant number asked for times the mesh's shortest cell over the free
    stream -- the shortest cell as the mesh actually has it, grading included,
    because that is the one the Courant number is set by.

    `--delta-t` means a fixed step, so it turns `adjustTimeStep` off. Writing the
    step the user asked for and then leaving the solver free to change it is the
    kind of quiet disagreement between what was said and what was written that
    only shows up in the time directory names.
    """
    solver = SOLVERS.get(study, "")
    if study == "transient":
        end = float(opts["end_time"])
        cell = plan.mesh.smallest_cell
        fixed = opts.get("delta_t") is not None
        delta = float(opts["delta_t"]) if fixed else float(opts["courant"]) * cell / flow.speed
        interval = end / max(1, int(opts["writes"]))
        entries = [f"application     {solver};", "startFrom       latestTime;", "startTime       0;",
                   "stopAt          endTime;", f"endTime         {end:g};",
                   f"deltaT          {delta:.6g};",
                   f"writeControl    {'runTime' if fixed else 'adjustableRunTime'};",
                   f"writeInterval   {interval:.6g};", "purgeWrite      0;", "writeFormat     ascii;",
                   "writePrecision  6;", "writeCompression off;", "timeFormat      general;",
                   "timePrecision   6;", "runTimeModifiable true;"]
        if fixed:
            entries += ["adjustTimeStep  no;",
                        f"// --delta-t {delta:.6g} s; the shortest cell is {cell:.4g} m, so "
                        f"Co = {delta * flow.speed / cell:.3g} at {flow.speed:g} m/s"]
        else:
            entries += ["adjustTimeStep  yes;",
                        f"maxCo           {float(opts['courant']):g};",
                        # 5x, not 100x. The solver grows into maxDeltaT whenever the
                        # flow lets it, and 100x the initial step is a Courant number
                        # around 90 -- a bound that bounds nothing.
                        f"maxDeltaT       {delta * 5:.6g};",
                        f"// deltaT = maxCo * {cell:.4g} m (the shortest cell) / {flow.speed:g} m/s"]
    else:
        end = float(opts["iterations"])
        entries = [f"application     {solver or 'simpleFoam'};", "startFrom       latestTime;",
                   "startTime       0;", "stopAt          endTime;", f"endTime         {end:g};",
                   "deltaT          1;", "writeControl    timeStep;",
                   f"writeInterval   {max(1, int(end / max(1, int(opts['writes'])))):d};",
                   "purgeWrite      0;", "writeFormat     ascii;", "writePrecision  6;",
                   "writeCompression off;", "timeFormat      general;", "timePrecision   6;",
                   "runTimeModifiable true;"]
    body = "\n".join(entries) + function_objects(plan, flow, opts, study)
    return foam_file("dictionary", "controlDict", body, "system")


def fv_schemes(study: str, model: str) -> str:
    time_scheme = "steadyState" if study != "transient" else "backward"
    # Steady runs keep linearUpwind: bounded and stable is what gets a SIMPLE run to a
    # converged answer. A transient one gets `Gauss linear`, because upwind dissipation
    # damps exactly the instability a transient study is usually run to observe -- a
    # Re=100 shedding case can be smoothed into a steady-looking wake by the scheme
    # rather than by the physics, which is the one error that looks like a result.
    divergence = ("bounded Gauss linearUpwind grad(U)" if study != "transient"
                  else "Gauss linear")
    prefix = "bounded " if study != "transient" else ""
    lines = ["ddtSchemes", "{", f"    default         {time_scheme};", "}", "",
             "gradSchemes", "{", "    default         Gauss linear;", "}", "",
             "divSchemes", "{", "    default         none;", f"    {'div(phi,U)':<16}{divergence};"]
    # `default none` means every convected field has to be named here. Naming the
    # ones this model does not transport is harmless; leaving out one it does
    # transport stops the run on the first iteration.
    for field in turbulence_fields(model):
        lines.append(f"    {f'div(phi,{field})':<16}{prefix}Gauss limitedLinear 1;")
    lines += ["    div((nuEff*dev2(T(grad(U))))) Gauss linear;", "}", "",
              "laplacianSchemes", "{", "    default         Gauss linear corrected;", "}", "",
              "interpolationSchemes", "{", "    default         linear;", "}", "",
              "snGradSchemes", "{", "    default         corrected;", "}", "",
              "wallDist", "{", "    method          meshWave;", "}"]
    return foam_file("dictionary", "fvSchemes", "\n".join(lines), "system")


P_SOLVER = ["solver          GAMG;", "tolerance       1e-07;", "relTol          0.01;",
            "smoother        GaussSeidel;"]
SMOOTH_SOLVER = ["solver          smoothSolver;", "smoother        symGaussSeidel;",
                 "tolerance       1e-08;", "relTol          0.1;"]


def solver_entry(name: str, settings: list[str]) -> list[str]:
    return [f"    {name}", "    {"] + [f"        {line}" for line in settings] + ["    }", ""]


def converged(settings: list[str]) -> list[str]:
    """The same solver run to its absolute tolerance instead of a relative one."""
    return [line for line in settings if not line.startswith("relTol")] + ["relTol          0;"]


RESIDUAL_CONTROL = (("p", 1e-4), ("U", 1e-5))
"""What "converged" means for a steady run, so the solver can say it reached it.

There was no residualControl anywhere in the toolbox, so `endTime = --iterations` was
the only stopping rule: every steady case ran exactly its iteration count and stopped,
converged or not, and never printed "SIMPLE solution converged". In one run of ten
studies, four would have shipped an unconverged answer if the agent had trusted the
residual tables it was shown -- it caught them instead by computing conservation checks
the product does not provide.

Ordinary values, and deliberately loose enough not to stop a run early. They decide
nothing: a solver that converges says so and stops, one that plateaus still runs to
endTime, and whether a plateau is physics or a bad mesh is a reading nothing here makes.
"""


def residual_control(fields: tuple[str, ...]) -> list[tuple[str, float]]:
    """The residualControl entries, including whichever turbulence fields this model has."""
    entries = list(RESIDUAL_CONTROL)
    if fields:
        # One field is named plainly; several become one regex. Same convention the
        # linear-solver block uses, so the two read as describing the same case.
        entries.append((f'"({"|".join(fields)})"' if len(fields) > 1 else fields[0], 1e-5))
    return entries


def fv_solution(study: str, model: str) -> str:
    """The linear solvers.

    PIMPLE asks for `pFinal` and `UFinal` by those exact literal names on the last
    inner iteration of every time step, and `p` does not stand in for `pFinal`:
    without the Final entries pimpleFoam stops on the first step with `keyword
    pFinal is undefined in dictionary solvers`. A steady SIMPLE run never asks for
    them, which is why the omission survives a steady case and kills a transient one.
    """
    fields = turbulence_fields(model)
    entries = [("p", P_SOLVER), ("U", SMOOTH_SOLVER)]
    if fields:
        entries.append((f'"({"|".join(fields)})"' if len(fields) > 1 else fields[0], SMOOTH_SOLVER))

    lines = ["solvers", "{"]
    for name, settings in entries:
        lines += solver_entry(name, settings)
        if study == "transient":
            final = name[:-1] + 'Final"' if name.endswith('"') else name + "Final"
            lines += solver_entry(final, converged(settings))
    lines += ["}", ""]
    if study == "transient":
        lines += ["PIMPLE", "{", "    nOuterCorrectors 2;", "    nCorrectors     2;",
                  "    nNonOrthogonalCorrectors 1;", "}"]
    else:
        lines += ["SIMPLE", "{", "    nNonOrthogonalCorrectors 1;", "    consistent      yes;",
                  "    residualControl", "    {"]
        lines += [f"        {field:<15} {tol:g};" for field, tol in residual_control(fields)]
        lines += ["    }", "}", "",
                  "relaxationFactors", "{", "    equations", "    {",
                  '        U               0.9;', '        ".*"            0.9;', "    }", "}"]
    return foam_file("dictionary", "fvSolution", "\n".join(lines), "system")


def transport_properties(flow: Flow) -> str:
    body = ("transportModel  Newtonian;\n\n"
            f"nu              {flow.nu:.6g};\n\n"
            f"// nu = U*L/Re with U = {flow.speed:g} m/s, L = {flow.length:g} m, "
            f"Re = {flow.reynolds:.0f}")
    return foam_file("dictionary", "transportProperties", body, "constant")


def turbulence_properties(model: str) -> str:
    if model == "laminar":
        body = "simulationType  laminar;"
    else:
        body = ("simulationType  RAS;\n\nRAS\n{\n"
                f"    RASModel        {model};\n"
                "    turbulence      on;\n"
                "    printCoeffs     on;\n}")
    return foam_file("dictionary", "turbulenceProperties", body, "constant")


# -- the case ----------------------------------------------------------------------


def case_files(plan: Plan, flow: Flow, opts, study: str, model: str) -> dict[str, str]:
    """Every file the case is made of, path relative to the case directory.

    Returned rather than written so `--dry-run` and the writer share one answer:
    a dry run that lists different files from the ones a real run writes is worse
    than no dry run.
    """
    files = {
        "system/blockMeshDict": plan.mesh.block_mesh_dict(),
        "system/controlDict": control_dict(plan, flow, opts, study),
        "system/fvSchemes": fv_schemes(study, model),
        "system/fvSolution": fv_solution(study, model),
        "constant/transportProperties": transport_properties(flow),
        "constant/turbulenceProperties": turbulence_properties(model),
    }
    if study == "mesh":
        # A mesh-only case still gets fields. blockMesh does not read 0/, but
        # checkMesh, paraFoam and the next person all behave better with a case that
        # is complete, and a mesh-only study that turns into a solve is one edit away.
        pass
    files["0/U"] = field_U(plan, flow)
    files["0/p"] = field_p(plan)
    fields = turbulence_fields(model)
    if fields:
        template = str(opts.get("template") or "")
        intensity = free_stream_intensity(template, opts)
        mixing = float(opts.get("mixing_length") or 0.07 * plan.length)
        # A stated mixing length is a deliberate choice and wins; otherwise an external
        # case sets omega from a viscosity ratio, which is what free-stream turbulence
        # is actually specified by (see omega_for).
        ratio = None if opts.get("mixing_length") else viscosity_ratio(template, opts)
        writers = {
            "k": lambda: field_k(plan, flow, intensity),
            "omega": lambda: field_omega(plan, flow, intensity, mixing,
                                         nu=flow.nu, ratio=ratio),
            "epsilon": lambda: field_epsilon(plan, flow, intensity, mixing),
            "nuTilda": lambda: field_nu_tilda(plan, flow),
        }
        for field in fields:
            files[f"0/{field}"] = writers[field]()
        files["0/nut"] = field_nut(plan, model)
    return files


def write_case(target: Path, files: dict[str, str]) -> list[Path]:
    written = []
    for relative, text in sorted(files.items()):
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def turbulence_notes(name: str, plan: Plan, flow: Flow, model: str, opts) -> list[str]:
    """The two numbers a turbulent case is most often quietly wrong about.

    Both are printed because of an asymmetry found in live runs: the near-wall problem
    (F-15) is measurable from the case afterwards -- a y+ function object -- and a study
    duly found it and said the mesh was unfit. The free-stream eddy viscosity is not
    measurable from anything the product reported, so a second study inherited a
    nu_t/nu of 9,000 and never mentioned it, and the answer came back 41% high with a
    plausible-looking explanation attached. A bad default that is reported gets caught;
    a bad default that is invisible does not. So these are printed whether or not they
    look wrong, and the reading is left to whoever is holding the case.
    """
    if not turbulence_fields(model):
        return []
    intensity = free_stream_intensity(name, opts)
    mixing = float(opts.get("mixing_length") or 0.07 * plan.length)
    ratio = None if opts.get("mixing_length") else viscosity_ratio(name, opts)
    k = 1.5 * (intensity * flow.speed) ** 2
    omega = omega_for(k, flow.nu, mixing, ratio)
    nut = k / omega if omega > 0 else 0.0
    notes = [
        f"free stream I = {intensity:g}, k = {k:.4g}, omega = {omega:.4g}"
        f"   -> nu_t/nu = {nut / flow.nu:,.1f}" if flow.nu > 0 else "",
        "           (external aerodynamics wants roughly 0.1-10; far above it the "
        "boundary layer sees an ambient viscosity that is not there)",
    ]
    # The shortest edge in the mesh, which on a body wrapped in a graded O-grid is the
    # first cell off the wall -- the one the wall treatment is applied at.
    first = plan.mesh.smallest_cell
    if first and math.isfinite(first):
        y_plus = estimate_y_plus(first, flow)
        notes.append(f"near wall  first cell {first:.3g} m -> y+ ~ {y_plus:.0f}"
                     f"   ({nut_wall_function(model)})")
    return [line for line in notes if line]


def estimate_y_plus(first_cell: float, flow: Flow) -> float:
    """y+ at the centre of the first cell, from a flat-plate skin-friction correlation.

    Order of magnitude, and that is enough: the question it answers is whether the wall
    treatment is being asked for something a thousand times outside its range, not what
    the third digit is. cf = 0.0576 Re_x^-0.2 at x = L/2.
    """
    re_x = max(flow.reynolds / 2.0, 1.0)
    cf = 0.0576 * re_x ** -0.2
    u_tau = flow.speed * math.sqrt(max(cf, 1e-12) / 2.0)
    return (first_cell / 2.0) * u_tau / flow.nu if flow.nu > 0 else 0.0


def summary(name: str, plan: Plan, flow: Flow, study: str, model: str, why: str,
            opts=None) -> list[str]:
    """What was decided, in the order somebody checking it would ask."""
    x0, x1, y0, y1 = plan.mesh.bounds
    counts = plan.mesh.patch_face_counts()
    lines = [
        f"template   {name}   ({TEMPLATES[name]['what']})",
        f"study      {study}" + (f"   solver {SOLVERS[study]}" if SOLVERS.get(study) else "   mesh only, no solver"),
        f"flow       {flow.line()}",
        f"turbulence {model}   ({why})",
        f"domain     x [{x0:.4g}, {x1:.4g}] m, y [{y0:.4g}, {y1:.4g}] m, "
        f"z {plan.mesh.thickness:.4g} m, 1 cell",
        f"cells      {plan.mesh.cell_count:,}   in {len(plan.mesh.blocks)} blocks",
        "patches    " + ", ".join(f"{patch} ({counts[patch]})" for patch in patch_order(counts)),
    ]
    if opts is not None:
        lines += turbulence_notes(name, plan, flow, model, opts)
    lines += [f"           {note}" for note in plan.notes]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("template", nargs="?", help="One of the names in --list.")
    ap.add_argument("case", nargs="?", type=Path, help="Directory to write the case into.")
    ap.add_argument("--list", action="store_true", help="The templates and what they are.")
    ap.add_argument("--dry-run", action="store_true", help="Say what would be written; write nothing.")
    ap.add_argument("--force", action="store_true", help="Write into a directory that already has a case in it.")

    flow = ap.add_argument_group("the flow")
    flow.add_argument("--speed", type=float, default=1.0, help="Free-stream or inlet speed, m/s (default 1).")
    flow.add_argument("--reynolds", type=float, default=None, help="Reynolds number; nu follows (default 100).")
    flow.add_argument("--nu", type=float, default=None, help="Kinematic viscosity m2/s; Re follows.")
    flow.add_argument("--length", type=float, default=None,
                      help="Characteristic length for Re (default: the body size, or the duct width).")
    flow.add_argument("--turbulence", default="auto",
                      choices=["auto", "laminar", "kOmegaSST", "kOmegaSSTLM",
                               "kEpsilon", "SpalartAllmaras"],
                      help="auto: laminar below Re 2300, kOmegaSST above. "
                           "kOmegaSSTLM adds the Langtry-Menter transition "
                           "equations, which is what a blade or an aerofoil below "
                           "Re ~ 5e5 needs -- assume it turbulent from the leading "
                           "edge and the lift comes out low however fine the mesh.")
    flow.add_argument("--turbulent-intensity", type=float, default=None,
                      dest="turbulent_intensity",
                      help="Free-stream turbulence intensity. Default 0.001 for a body in "
                           "open flow, 0.05 inside a duct.")
    flow.add_argument("--viscosity-ratio", type=float, default=None, dest="viscosity_ratio",
                      help="Target free-stream nu_t/nu; omega follows from it (external "
                           "templates, default 10). --mixing-length overrides.")
    flow.add_argument("--mixing-length", type=float, default=None, dest="mixing_length")

    run = ap.add_argument_group("the run")
    run.add_argument("--study", default="steady", choices=["mesh", "steady", "transient"])
    run.add_argument("--iterations", type=int, default=1000, help="Steady: how many (default 1000).")
    run.add_argument("--end-time", type=float, default=1.0, dest="end_time", help="Transient: seconds.")
    run.add_argument("--delta-t", type=float, default=None, dest="delta_t",
                     help="Transient: a fixed step, which turns adjustTimeStep off.")
    run.add_argument("--courant", type=float, default=0.9, help="Transient: target max Courant (default 0.9).")
    run.add_argument("--writes", type=int, default=50, help="How many times to write (default 50).")

    shape = ap.add_argument_group("the shape")
    shape.add_argument("--size", type=float, default=0.1, help="Body size across, m (default 0.1).")
    shape.add_argument("--thickness", type=float, default=None,
                       help="The one cell's depth in z (default: a tenth of the size).")
    shape.add_argument("--profile", default="", help="profile template: a file of x,y points.")
    shape.add_argument("--aoa", type=float, default=0.0,
                       help="profile template: degrees of incidence, leading edge up "
                            "(positive gives positive lift).")
    shape.add_argument("--moving-ground", action="store_true", dest="moving_ground")
    shape.add_argument("--rotating-wheels", action="store_true", dest="rotating_wheels")

    tunnel = ap.add_argument_group("the tunnel (external templates, in body sizes)")
    # In body sizes. 8 above and below is a domain 16 sizes tall, which is 6.25%
    # blockage with free-slip top and bottom -- enough to inflate Cd by 4-7% and St by
    # 2-3% against the Re=100 cylinder benchmark, which is the difference between
    # landing inside a validation band and outside it. 12 is the field notes' own
    # recommendation and costs a modest number of cells in the coarse outer blocks.
    tunnel.add_argument("--upstream", type=float, default=10.0,
                        help="Inlet distance ahead of the body, in body sizes (default 10).")
    tunnel.add_argument("--downstream", type=float, default=20.0,
                        help="Outlet distance behind the body, in body sizes (default 20).")
    tunnel.add_argument("--above", type=float, default=12.0,
                        help="Half-height above the body, in body sizes (default 12: ~4% blockage).")
    tunnel.add_argument("--below", type=float, default=12.0,
                        help="Half-height below the body, in body sizes (default 12).")
    tunnel.add_argument("--ogrid-scale", type=float, default=2.5, dest="ogrid_scale")
    tunnel.add_argument("--ogrid-aspect", type=float, default=1.0, dest="ogrid_aspect")
    tunnel.add_argument("--cells-around", type=int, default=80, dest="cells_around",
                        help="Cells around the body per quadrant; four times this all the "
                             "way round (default 80, so 320).")
    tunnel.add_argument("--cells-radial", type=int, default=24, dest="cells_radial")
    tunnel.add_argument("--radial-grading", type=float, default=6.0, dest="radial_grading")
    tunnel.add_argument("--far-grading", type=float, default=8.0, dest="far_grading")

    duct = ap.add_argument_group("the duct and bend templates")
    duct.add_argument("--duct-width", type=float, default=0.05, dest="duct_width")
    duct.add_argument("--cells-across", type=int, default=20, dest="cells_across")
    duct.add_argument("--leg-length", type=float, default=6.0, dest="leg_length",
                      help="Inlet leg, in duct widths (default 6).")
    duct.add_argument("--branch-length", type=float, default=8.0, dest="branch_length",
                      help="Outlet legs, in duct widths (default 8).")
    duct.add_argument("--branch-angle", type=float, default=45.0, dest="branch_angle")
    duct.add_argument("--offset", type=float, default=4.0, help="Z and M templates, in duct widths.")
    duct.add_argument("--bend-radius", type=float, default=0.05, dest="bend_radius")

    args = ap.parse_args(argv)

    if args.list:
        width = max(len(name) for name in TEMPLATES)
        for family, names in (("external flow", EXTERNAL), ("ducts", DUCTS), ("bends", BENDS)):
            print(f"{family}:")
            for name in names:
                print(f"  {name:<{width}}  {TEMPLATES[name]['what']}")
            print(f"  {'':<{width}}  parameters: " + ", ".join(FAMILY_PARAMS[TEMPLATES[names[0]]['family']]))
            print()
        return 0

    if not args.template or args.case is None:
        ap.error("a template and a directory to write it into (or --list)")

    opts = vars(args)
    family = TEMPLATES.get(args.template, {}).get("family")
    if family is None:
        raise SystemExit(f"no template called {args.template}; `--list` shows them all")
    natural = args.size if family == "external" else args.duct_width
    if opts["thickness"] is None:
        opts["thickness"] = natural / 10.0

    plan = build_plan(args.template, opts)
    flow = derive_flow(opts, plan.length)
    model, why = turbulence_model(opts, flow)
    files = case_files(plan, flow, opts, args.study, model)

    for line in summary(args.template, plan, flow, args.study, model, why, vars(args)):
        print(line)
    print()

    if args.dry_run:
        print(f"would write {len(files)} files into {args.case}:")
        for relative in sorted(files):
            print(f"  {relative}   ({len(files[relative].splitlines())} lines)")
        return 0

    target = Path(args.case)
    if (target / "system" / "blockMeshDict").exists() and not args.force:
        raise SystemExit(f"{target} already holds a case; --force to write over it")
    written = write_case(target, files)
    print(f"wrote {len(written)} files into {target}")
    if SOLVERS.get(args.study):
        print(f"next: blockMesh -case {target} && checkMesh -case {target}")
    else:
        print(f"next: blockMesh -case {target}")

    # The state belongs to the STUDY, not to the case: a study with two cases in it
    # (a mesh study and the solve, or Re=100 beside Re=200) has one manifest and one
    # phase table, and `gallery.py <study>` is meant to see both. Recording against
    # the case put a `.reynolds` inside each case directory and left the study home
    # with none, so the gallery of the study came back empty.
    #
    # `study_state.find_root` walks up and prefers a `.reynolds` that already exists,
    # so passing the parent is only a starting point and an established study home
    # further up still wins. The parent is not used when it is the workspace root
    # itself -- every study under /work would then share one manifest.
    home = target.parent
    if home.name in ("work", "") or home == home.parent:
        home = target
    try:
        study_state.record("other", target, root=home, case=target.name,
                           label=f"{args.template} case, {plan.mesh.cell_count} cells",
                           template=args.template, study=args.study, reynolds=flow.reynolds,
                           nu=flow.nu, cells=plan.mesh.cell_count)
        study_state.set_phase("geometry", "done", root=home, case=target.name,
                              note=f"{args.template}, {plan.mesh.cell_count} cells")
    except OSError as exc:
        # The case is on disk either way; a manifest that could not be written is
        # worth a line and not worth losing the case over.
        print(f"(the study manifest could not be updated: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
