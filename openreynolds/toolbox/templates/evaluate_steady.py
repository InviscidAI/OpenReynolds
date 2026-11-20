#!/usr/bin/env python3
"""TEMPLATE: one round of a parametric study -- a 2D section in external flow, steady.

This file is the `build.py` of a round: copy it into a round directory under that name,
or let it write the whole round out for you, samples included:

    python3 evaluate_steady.py --write-round /work/study/rounds/000
    # -> build.py (this file), hooks.py, goal.lock.json, design_constants.py

then, per candidate, write a `design_constants.py` and let `parametric.py` do the rest:

    python3 /work/.toolbox/parametric.py /work/study/cand/007 --round /work/study/rounds/000 \\
        --lock /work/study/goal.lock.json --mode solve

Run directly in a directory that holds a `design_constants.py` (which is what
`parametric.py` does) it builds the mesh: a cambered section of `THICKNESS` at
`ALPHA_DEG` sitting `RIDE_HEIGHT_M` above a ground plane, in a box, meshed in quads
with gmsh, extruded one cell, converted with `gmshToFoam`, its patches retyped, checked
with `checkMesh` and looked at with `mesh_look.py`. The patches are `inlet`, `outlet`,
`ground`, `top`, `body` and `frontAndBack`; the roles `case_gen.py` gives them come from
the lock's `case_gen_args` and the round's `hooks.case_args()`, not from here.

what is a design constant and what is not

    `from design_constants import *` is the first thing this file does, and the section
    -- `CHORD_M`, `CAMBER_1`, `CAMBER_POS`, `THICKNESS`, `ALPHA_DEG`, `RIDE_HEIGHT_M` --
    comes entirely from there. The caller owns those numbers: an optimizer varies them,
    the trim in the lock varies `ALPHA_DEG`, and this script has no default for any of
    them, so a candidate without the file fails loudly rather than meshing the same
    shape under a new name. The mesh sizes marked below are this file's own: they are the
    fidelity, and the lock's `fidelity.cells` band is what holds them to account.

the lock, and what parametric.py and score.py read out of it

    `goal.lock.json` is harness-owned and written once. The keys the two scripts read
    are all in the sample below: `solver` (default `simpleFoam`); `case_gen_args`, the
    physics as `case_gen.py` flags; `body_patch`, where the forces are taken;
    `fidelity.ranks`, `.iters`, `.window` (the tail fraction the coefficients are averaged
    over), `.cells: {min, max}`, `.yplus: {min, max}` (either side optional -- `min` is
    left out of the sample because the stagnation point puts y+ near zero on any body
    and a lower bound would fail every candidate), `.reference` (a tag, copied through);
    `trim` (optional, the fixed-lift secant); `template_version`, copied through so a
    leaderboard row names the round layout that produced it.

Gotchas carried over from `duct2d.py`, still true here: `Mesh.Algorithm 8` plus
`Mesh.RecombinationAlgorithm 3` plus `setRecombine`, or the extrude makes prisms;
`numElements=[1], recombine=True`; `Mesh.MshFileVersion 2.2` for `gmshToFoam`; a
physical group named per curve as it is made, never from where a face sits in the
bounding box; `foamDictionary` retypes through `entry0.<patch>`; the three `system/`
stubs so `gmshToFoam` and `checkMesh` can build a `Foam::Time`. Metres, always: a chord
typed in millimetres is a wing a thousand times too big at a thousandth of the Reynolds
number, and nothing here can tell.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

try:
    from design_constants import *  # noqa: F401,F403 -- the candidate's numbers, written by the caller
except ImportError:
    if "--write-round" not in sys.argv:
        raise SystemExit(
            "no design_constants.py beside this script: the section (CHORD_M, CAMBER_1, "
            "CAMBER_POS, THICKNESS, ALPHA_DEG, RIDE_HEIGHT_M) is read from it, and there "
            "is no default -- `--write-round DIR` writes a sample"
        )

# -- edit these two, then re-run: the mesh, which is the fidelity ------------------
CELL_BODY_M = 0.002   # mesh size on the section, metres (a 1 m chord: ~1,000 cells round it)
CELL_FAR_M = 0.08     # mesh size at the box corners, metres
# ---- measured with gmsh 4.15.2 on the sample constants: 26,004 hexahedra ------------

UPSTREAM_C = 5.0      # inlet this many chords ahead of the leading edge
DOWNSTREAM_C = 10.0   # outlet this many chords behind the trailing edge
ABOVE_C = 5.0         # top this many chords above the section's highest point
POINTS_PER_SIDE = 60  # spline points on each of the upper and lower surfaces
THICK_FRACTION = 0.05  # extrude depth as a fraction of chord; 2D, so the value is arbitrary


# -- the section -------------------------------------------------------------------


def section(chord: float, camber: float, camber_pos: float, thickness: float,
            alpha_deg: float, n: int = POINTS_PER_SIDE) -> tuple[list, list]:
    """Upper and lower surface points, leading edge to trailing edge, in metres.

    NACA four-digit construction: the thickness distribution with the closed
    trailing-edge coefficient (-0.1036) so the two surfaces meet in one point, laid
    perpendicular to a parabolic camber line peaking `camber` at `camber_pos`. Then the
    rotation by `alpha_deg` about the quarter chord, nose-up positive with the flow along
    +x. Cosine-spaced in x so the leading edge, where the curvature is, gets the points.
    """
    upper, lower = [], []
    for i in range(n + 1):
        x = 0.5 * (1.0 - math.cos(math.pi * i / n))
        yt = 5.0 * thickness * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                                + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if camber > 0 and 0 < camber_pos < 1:
            if x < camber_pos:
                yc = camber / camber_pos ** 2 * (2 * camber_pos * x - x ** 2)
                dyc = 2 * camber / camber_pos ** 2 * (camber_pos - x)
            else:
                yc = camber / (1 - camber_pos) ** 2 * ((1 - 2 * camber_pos) + 2 * camber_pos * x - x ** 2)
                dyc = 2 * camber / (1 - camber_pos) ** 2 * (camber_pos - x)
        else:
            yc, dyc = 0.0, 0.0
        theta = math.atan(dyc)
        upper.append((x - yt * math.sin(theta), yc + yt * math.cos(theta)))
        lower.append((x + yt * math.sin(theta), yc - yt * math.cos(theta)))
    # The trailing edge is one point, exactly: the two constructions differ there by
    # rounding, and gmsh would otherwise see two points and an open loop.
    upper[-1] = lower[-1] = (1.0, upper[-1][1] * 0.5 + lower[-1][1] * 0.5)
    upper[0] = lower[0] = (0.0, 0.0)

    a = math.radians(alpha_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)

    def place(point):
        x, y = point[0] - 0.25, point[1]           # about the quarter chord
        return (chord * (x * cos_a + y * sin_a + 0.25), chord * (-x * sin_a + y * cos_a))

    return [place(p) for p in upper], [place(p) for p in lower]


def build_mesh(msh_path: Path) -> dict:
    """The section in its box, the quad mesh, the extrude, the msh file. Returns the
    numbers the report wants: where the section sits and how big the box is."""
    import gmsh

    upper, lower = section(CHORD_M, CAMBER_1, CAMBER_POS, THICKNESS, ALPHA_DEG)  # noqa: F405
    lowest = min(y for _, y in upper + lower)
    highest = max(y for _, y in upper + lower)
    lift = RIDE_HEIGHT_M - lowest                 # noqa: F405 -- ground is y = 0
    upper = [(x, y + lift) for x, y in upper]
    lower = [(x, y + lift) for x, y in lower]
    x0, x1 = -UPSTREAM_C * CHORD_M, CHORD_M + DOWNSTREAM_C * CHORD_M  # noqa: F405
    y0, y1 = 0.0, highest + lift + ABOVE_C * CHORD_M                    # noqa: F405
    thick = THICK_FRACTION * CHORD_M                                    # noqa: F405

    gmsh.initialize()
    gmsh.model.add("section2d")
    geo = gmsh.model.geo

    c1 = geo.addPoint(x0, y0, 0, CELL_FAR_M)
    c2 = geo.addPoint(x1, y0, 0, CELL_FAR_M)
    c3 = geo.addPoint(x1, y1, 0, CELL_FAR_M)
    c4 = geo.addPoint(x0, y1, 0, CELL_FAR_M)
    l_ground = geo.addLine(c1, c2)
    l_outlet = geo.addLine(c2, c3)
    l_top = geo.addLine(c3, c4)
    l_inlet = geo.addLine(c4, c1)
    outer = geo.addCurveLoop([l_ground, l_outlet, l_top, l_inlet])

    # One leading-edge point and one trailing-edge point, shared by both splines, so
    # the section closes exactly rather than to within a tolerance.
    le = geo.addPoint(*upper[0], 0, CELL_BODY_M)
    te = geo.addPoint(*upper[-1], 0, CELL_BODY_M)
    up = [le] + [geo.addPoint(x, y, 0, CELL_BODY_M) for x, y in upper[1:-1]] + [te]
    lo = [le] + [geo.addPoint(x, y, 0, CELL_BODY_M) for x, y in lower[1:-1]] + [te]
    s_upper = geo.addSpline(up)
    s_lower = geo.addSpline(lo)
    hole = geo.addCurveLoop([s_upper, -s_lower])

    surf = geo.addPlaneSurface([outer, hole])
    geo.synchronize()

    # quads, not triangles: Frontal-Delaunay for quads + Blossom full-quad recombination
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)

    # one layer, swept into hexahedra
    out = geo.extrude([(2, surf)], 0, 0, thick, numElements=[1], recombine=True)
    geo.synchronize()
    # AFTER the synchronize, not before it: `gmsh.model.mesh.setRecombine` is an attribute
    # on the model entity and `geo.synchronize()` rebuilds the entities, so a flag set
    # before the extrude is gone by the time the mesh is generated -- measured with gmsh
    # 4.15.2: set first, the surface meshed in quads and the volume came out as 5,616
    # prisms; set here, 3,145 hexahedra.
    gmsh.model.mesh.setRecombine(2, surf)
    # `extrude` returns the top copy of the surface, the volume, then one lateral surface
    # per boundary curve in the order the curve loops were built: the box's four, then
    # the section's two. Named from that order, never from where a face sits.
    top_surf, volume = out[0][1], out[1][1]
    side_ground, side_outlet, side_top, side_inlet, side_upper, side_lower = (
        entity[1] for entity in out[2:8])

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(2, [side_outlet], name="outlet")
    gmsh.model.addPhysicalGroup(2, [side_ground], name="ground")
    gmsh.model.addPhysicalGroup(2, [side_top], name="top")
    gmsh.model.addPhysicalGroup(2, [side_upper, side_lower], name="body")
    gmsh.model.addPhysicalGroup(3, [volume], name="internal")

    gmsh.model.mesh.generate(3)
    cells = sum(len(tags) for _, tags in
                zip(*gmsh.model.mesh.getElements(3)[:2]))

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # gmshToFoam reads only this
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return {"cells": cells, "box": [x0, y0, x1, y1], "section_y": [lowest + lift, highest + lift],
            "chord_m": CHORD_M, "alpha_deg": ALPHA_DEG, "ride_height_m": RIDE_HEIGHT_M}  # noqa: F405


# -- OpenFOAM: convert, retype, check, look -----------------------------------------


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """A command, printed, run, and raised on failure with its stderr shown."""
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode})")
    return proc


def retype(patch: str, kind: str) -> None:
    """`gmshToFoam` types every patch `patch`; `boundary` is a list, so the patch is
    reached through foamDictionary's `entry0`."""
    run(["foamDictionary", "constant/polyMesh/boundary",
         "-entry", f"entry0.{patch}.type", "-set", kind])


CONTROL_DICT_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

FV_SCHEMES_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

FV_SOLUTION_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
}
"""


def write_system_stubs(case: Path) -> None:
    """Enough `system/` for `gmshToFoam` and `checkMesh` to build a `Foam::Time`; not a
    case. `case_gen.py` writes over all three once there is something to solve."""
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(CONTROL_DICT_STUB)
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES_STUB)
    (case / "system" / "fvSolution").write_text(FV_SOLUTION_STUB)


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


# -- the rest of a round, as samples ------------------------------------------------

HOOKS_PY = '''\
"""What this round adds to `case_gen.py` beyond the lock: patch roles, nothing else.

`parametric.py` accepts only `case_args()` and imports here. A flag the lock already
sets is refused -- the lock is the physics; this is the round's own reading of its
patches -- and so is any other definition, because `score.py` is the only scorer.
"""


def case_args() -> list[str]:
    return ["--slip", "top"]
'''

GOAL_LOCK_JSON = {
    "template_version": "evaluate_steady-1",
    "solver": "simpleFoam",
    "case_gen_args": ["--speed", "10", "--reynolds", "1e6", "--length", "1.0",
                      "--belt", "ground", "--turbulence", "kOmegaSST"],
    "body_patch": "body",
    "fidelity": {
        "ranks": 4,
        "iters": 1500,
        "window": 0.2,
        "cells": {"min": 15000, "max": 60000},
        "yplus": {"max": 300},
        "reference": True,
    },
    "trim": {"variable": "ALPHA_DEG", "metric": "Cl", "target": 0.8, "tol": 0.01,
             "max_solves": 3},
}

DESIGN_CONSTANTS_PY = """\
# One line per number, `NAME = <number>`; parametric.py reads and rewrites these lines.
CHORD_M = 1.0         # chord, metres -- the Reynolds number and the coefficients are on it
CAMBER_1 = 0.04       # maximum camber as a fraction of chord
CAMBER_POS = 0.4      # where the camber peaks, fraction of chord from the leading edge
THICKNESS = 0.12      # maximum thickness as a fraction of chord
ALPHA_DEG = 2.0       # incidence, degrees, nose-up positive -- what the trim varies
RIDE_HEIGHT_M = 0.2   # gap from the section's lowest point down to the ground, metres
"""


def write_round(target: Path) -> None:
    """The whole round laid out: this file as `build.py`, and the three samples."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "build.py").write_bytes(Path(__file__).read_bytes())
    (target / "hooks.py").write_text(HOOKS_PY, encoding="utf-8")
    (target / "goal.lock.json").write_text(json.dumps(GOAL_LOCK_JSON, indent=2) + "\n",
                                           encoding="utf-8")
    (target / "design_constants.py").write_text(DESIGN_CONSTANTS_PY, encoding="utf-8")
    for name in ("build.py", "hooks.py", "goal.lock.json", "design_constants.py"):
        print(target / name)
    print("edit design_constants.py per candidate; the lock is the harness's and is written once")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--write-round":
        write_round(Path(sys.argv[2]))
        return
    case = Path.cwd()
    msh = case / "section2d.msh"
    facts = build_mesh(msh)
    print(f"section: chord {facts['chord_m']} m at {facts['alpha_deg']} deg, "
          f"y {facts['section_y'][0]:.4g}..{facts['section_y'][1]:.4g} m above the ground; "
          f"box x {facts['box'][0]:.4g}..{facts['box'][2]:.4g}, y {facts['box'][1]:.4g}.."
          f"{facts['box'][3]:.4g} m; {facts['cells']} cells")
    write_system_stubs(case)

    run(["gmshToFoam", msh.name])
    for patch in ("body", "ground"):
        retype(patch, "wall")
    retype("frontAndBack", "empty")

    run(["checkMesh", "-case", "."])
    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png", "--json", "look.json"])


if __name__ == "__main__":
    main()
