#!/usr/bin/env python3
"""TEMPLATE: an elastically mounted cylinder in cross-flow -- blockMesh O-grid, two stages.

Copy this file into an empty parent directory and run it there:

    python3 viv_cylinder.py              # writes ./fixed and ./released, 1 DOF in y
    python3 viv_cylinder.py --dof 2      # released free in x and y, isotropic spring
    python3 viv_cylinder.py --stage fixed

It writes two complete `pimpleFoam` cases that share one mesh:

* `fixed/` -- the cylinder bolted still (`staticFvMesh`, `noSlip`). Its job is the
  Strouhal number. A mesh that does not land on the published shedding frequency is not
  ready to carry a moving-body answer, and finding that out after a 400 s moving-mesh
  run is the expensive order to find it out in.
* `released/` -- the same mesh with `dynamicMotionSolverFvMesh` and
  `sixDoFRigidBodyMotion`, the body on a linear spring with zero structural damping.

Then `blockMesh`, `checkMesh` and `mesh_look.py` on the mesh, so the result is looked at
rather than only built. `viv.py` is the instrument for the finished run: amplitude,
frequency, lock-in and the energy balance.

Edit these two numbers and re-run: `DIAMETER_M` (the cylinder) and `USTAR` (the reduced
velocity U/(f_n D), which is what sweeps the response). Everything else follows --
`--re` sets the free stream from the diameter and viscosity, `--mstar` sets the body
mass from the displaced fluid mass, and the spring stiffness comes out of `USTAR`.

Physics this bakes in, each because leaving it out cost a run:

* **Zero structural damping is a free check, and it is the only one that catches loose
  coupling.** Nothing can dissipate energy, so the fluid does no net work on the body
  per cycle on a real limit cycle. The 2026-09-12 run of this configuration settled into
  a stationary, sinusoidal, spectrally clean limit cycle at A/D = 0.640 while the fluid
  did -10.1% to -10.4% of the body's energy on it every cycle, which is impossible; with
  the body moved inside the PIMPLE loop it reported +0.04% to +0.18% at A/D = 0.567
  (`benchmarks/moving_mesh/README.md`). The amplitude was 13% too wide and nothing in
  the displacement trace said so.
* So the released case ships **tightly coupled**: `moveMeshOuterCorrectors yes`,
  three outer correctors, and `accelerationRelaxation 1.0`. Relaxing the acceleration
  and moving the mesh once per time step is the cheaper scheme and it is the one that
  was 13% wrong.
* **Releasing is a second run, not a flag.** `sixDoFRigidBodyMotion` reads its
  constraints and restraints once at start-up, so the body cannot be let go mid-run;
  `released/` starts from the `fixed/` field by `mapFields` or simply from 0, which is
  why they are two directories here.
* `movingWallVelocity` on the cylinder in the released case and `noSlip` in the fixed
  one, with `0/pointDisplacement` present only in the released case. Both flip with the
  mesh type and neither is inferred.
* The morph: `innerDistance` 1 D of rigid motion around the wall, `outerDistance` inside
  the O-grid's square interface. Points between are blended, and a concave cell that is
  squeezed past flat goes negative -- which is what `checkMesh` at the largest excursion
  measures and `checkMesh` at t = 0 does not.
* Metres, always. OpenFOAM reads whatever numbers are in the mesh as metres, so a
  10 mm cylinder is 0.010 here and not 10.

The mesh is the O-grid + H-grid topology the graded 2026-09-12 run used: a radially
graded ring on the wall, a second ring out to a square, and a 3 x 3 block of rectangles
around that with the centre cell removed. 240 cells around the cylinder and 2.5%
blockage, which measured St = 0.166141 at Re = 100 against Williamson's 0.164 for
parallel shedding, +1.3%. `blockMesh` rather than gmsh on purpose: the wall-normal
grading and the exactly-matching O-grid are what that Strouhal number came from, and an
unstructured quad mesh at the same cell count does not reproduce it.

What is **not** validated here: the 2-DOF path. `--dof 2` writes a `plane` constraint
and leaves the same isotropic spring, which is the standard 2-DOF elastically mounted
cylinder, and no run of it exists in this repository to report a number from. The 1-DOF
path is the one with a graded run behind it.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
DIAMETER_M = 0.010   # cylinder diameter, metres
USTAR = 5.0          # reduced velocity U / (f_n D); 1/St is where lock-in sits
# --------------------------------------------------------------------------

# The mesh, in diameters, from the graded run. NT is per quarter, so the wall carries
# 4 x NT cells; NXM must equal NT for the H-grid to match the square.
R1_OVER_R = 4.0      # outer radius of the uniform O-ring, in cylinder radii
SQUARE_OVER_D = 5.0  # half-size of the square wrapping the O-grid, in diameters
UPSTREAM_D, DOWNSTREAM_D = 20.0, 40.0
HALF_HEIGHT_D = 20.0  # +/- 20 D, so blockage is 2.5%
NT, NR1, GR1, NR2, GR2 = 60, 50, 10.0, 25, 2.0
NXL, NXR, NYB, NYT = 45, 140, 45, 45
GXL, GXR, GYB, GYT = 1 / 6.0, 6.0, 1 / 6.0, 6.0

HEAD = """/*--------------------------------*- C++ -*----------------------------------*\\
| written by viv_cylinder.py                                                  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def write(case: Path, loc: str, obj: str, cls: str, body: str) -> None:
    (case / loc).mkdir(parents=True, exist_ok=True)
    (case / loc / obj).write_text(
        HEAD.format(cls=cls, loc=loc, obj=obj) + body
        + "\n\n// ************************************************************* //\n"
    )


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """A command, printed, run, and raised on failure with its stderr shown."""
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=None if cwd is None else str(cwd))
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode})")


# --------------------------------------------------------------------------- mesh


def block_mesh_dict(diameter: float, span: float) -> str:
    """The O-grid + H-grid blockMeshDict, built from the block corners outward.

    Four blocks per ring and eight around the square, with the arcs given explicitly so
    the wall is a circle and not a polygon: without the `arc` edges blockMesh
    interpolates straight lines between the block corners at 45 degree spacing and the
    "cylinder" is an octagon, which changes the separation point and therefore the
    answer.
    """
    r = diameter / 2.0
    r1 = R1_OVER_R * r
    a = SQUARE_OVER_D * diameter
    x0, x1 = -UPSTREAM_D * diameter, DOWNSTREAM_D * diameter
    y0, y1 = -HALF_HEIGHT_D * diameter, HALF_HEIGHT_D * diameter

    points: list[tuple[float, float]] = []
    index: dict[tuple[float, float], int] = {}

    def point(x: float, y: float) -> int:
        key = (round(x, 12), round(y, 12))
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    angles = [-45.0, 45.0, 135.0, 225.0]

    def on_circle(radius: float, degrees: float) -> int:
        t = math.radians(degrees)
        return point(radius * math.cos(t), radius * math.sin(t))

    wall = [on_circle(r, t) for t in angles]
    ring = [on_circle(r1, t) for t in angles]
    square = [point(a, -a), point(a, a), point(-a, a), point(-a, -a)]
    grid = [[point(x, y) for y in (y0, -a, a, y1)] for x in (x0, -a, a, x1)]

    # (v0, v1, v2, v3, n1, n2, g1, g2); quad counter-clockwise in xy, dir1 = v0 -> v1
    blocks = []
    for k in range(4):
        kp = (k + 1) % 4
        blocks.append((wall[k], ring[k], ring[kp], wall[kp], NR1, NT, GR1, 1.0))
        blocks.append((ring[k], square[k], square[kp], ring[kp], NR2, NT, GR2, 1.0))
    nx, gx = [NXL, NT, NXR], [GXL, 1.0, GXR]
    ny, gy = [NYB, NT, NYT], [GYB, 1.0, GYT]
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            blocks.append((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1],
                           nx[i], ny[j], gx[i], gy[j]))

    npoints = len(points)
    arcs = []
    for k in range(4):
        kp = (k + 1) % 4
        middle = angles[k] + 45.0
        for radius, nodes in ((r, wall), (r1, ring)):
            for layer in (0, 1):
                offset = layer * npoints
                arcs.append((nodes[k] + offset, nodes[kp] + offset, layer,
                             radius * math.cos(math.radians(middle)),
                             radius * math.sin(math.radians(middle))))

    def face(v0: int, v1: int) -> tuple[int, int, int, int]:
        """The z-extruded face of the 2D edge v0 -> v1, wound so the normal points out
        of the fluid: the edge direction rotated -90 degrees in xy."""
        return (v0, v1, v1 + npoints, v0 + npoints)

    faces = {"cylinder": [], "inlet": [], "outlet": [], "bottom": [], "top": []}
    for k in range(4):
        kp = (k + 1) % 4
        faces["cylinder"].append(face(wall[kp], wall[k]))  # normal into the cylinder
    for j in range(3):
        faces["inlet"].append(face(grid[0][j + 1], grid[0][j]))
        faces["outlet"].append(face(grid[3][j], grid[3][j + 1]))
    for i in range(3):
        faces["bottom"].append(face(grid[i][0], grid[i + 1][0]))
        faces["top"].append(face(grid[i + 1][3], grid[i][3]))

    def number(value: float) -> str:
        return f"{value:.9g}"

    out = ["scale   1;", "", "vertices", "("]
    for z in (0.0, span):
        out += [f"    ({number(x)} {number(y)} {number(z)})" for x, y in points]
    out += [");", "", "blocks", "("]
    for v0, v1, v2, v3, n1, n2, g1, g2 in blocks:
        hexa = [v0, v1, v2, v3, v0 + npoints, v1 + npoints, v2 + npoints, v3 + npoints]
        out += ["    hex (" + " ".join(str(v) for v in hexa) + ")",
                f"    ({n1} {n2} 1)",
                f"    simpleGrading ({number(g1)} {number(g2)} 1)"]
    out += [");", "", "edges", "("]
    for v0, v1, layer, mx, my in arcs:
        out.append(f"    arc {v0} {v1} ({number(mx)} {number(my)} {number(layer * span)})")
    out += [");", "", "boundary", "("]
    kinds = {"cylinder": "wall", "inlet": "patch", "outlet": "patch",
             "bottom": "patch", "top": "patch"}
    for name, quads in faces.items():
        out += [f"    {name}", "    {", f"        type {kinds[name]};", "        faces", "        ("]
        out += ["            (" + " ".join(str(v) for v in q) + ")" for q in quads]
        out += ["        );", "    }"]
    out += [");", ""]
    # Every face blockMesh was not told about is a z face, and in a one-cell-thick case
    # they are the two `empty` planes. Naming them through `defaultPatch` is how the
    # case becomes 2D; OpenFOAM never infers `empty` from the geometry.
    out += ["defaultPatch", "{", "    name frontAndBack;", "    type empty;", "}", "",
            "mergePatchPairs", "(", ");"]
    return "\n".join(out)


def cell_count() -> int:
    counts = [NR1 * NT, NR2 * NT] * 4
    nx, ny = [NXL, NT, NXR], [NYB, NT, NYT]
    counts += [nx[i] * ny[j] for i in range(3) for j in range(3) if not (i == 1 and j == 1)]
    return sum(counts)


# --------------------------------------------------------------------------- the case


def write_fields(case: Path, speed: float, moving: bool, perturb: float) -> None:
    wall_u = ("movingWallVelocity;\n        value           uniform (0 0 0)"
              if moving else "noSlip")
    write(case, "0", "U", "volVectorField", f"""dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({speed} {perturb} 0);

boundaryField
{{
    inlet       {{ type fixedValue; value uniform ({speed} 0 0); }}
    outlet      {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({speed} 0 0); }}
    top         {{ type slip; }}
    bottom      {{ type slip; }}
    cylinder    {{ type            {wall_u}; }}
    frontAndBack {{ type empty; }}
}}""")
    write(case, "0", "p", "volScalarField", """dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet       { type zeroGradient; }
    outlet      { type fixedValue; value uniform 0; }
    top         { type slip; }
    bottom      { type slip; }
    cylinder    { type zeroGradient; }
    frontAndBack { type empty; }
}""")
    if moving:
        # `pointDisplacement` exists only when the mesh moves, and the cylinder's entry
        # is `calculated`: the motion solver sets it, nothing else may.
        write(case, "0", "pointDisplacement", "pointVectorField", """dimensions      [0 1 0 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    cylinder    { type calculated; value uniform (0 0 0); }
    frontAndBack { type empty; }
    "(inlet|outlet|top|bottom)" { type fixedValue; value uniform (0 0 0); }
}""")


def write_constant(case: Path, nu: float, moving: bool, body: dict | None) -> None:
    write(case, "constant", "transportProperties", "dictionary",
          f"transportModel  Newtonian;\n\nnu              {nu:.9g};")
    write(case, "constant", "turbulenceProperties", "dictionary", "simulationType  laminar;")
    if not moving:
        write(case, "constant", "dynamicMeshDict", "dictionary", "dynamicFvMesh   staticFvMesh;")
        return

    assert body is not None
    zc = body["span"] / 2.0
    if body["dof"] == 1:
        constraints = f"""    yOnly
    {{
        sixDoFRigidBodyMotionConstraint line;
        centreOfRotation ({0} {0} {zc:.9g});
        direction        (0 1 0);
    }}"""
    else:
        # Free in the x-y plane. The spring below is isotropic (anchor at the rest
        # position, restLength 0), which is the usual 2-DOF elastically mounted cylinder:
        # one stiffness, the same in both directions.
        constraints = f"""    xyPlane
    {{
        sixDoFRigidBodyMotionConstraint plane;
        centreOfRotation ({0} {0} {zc:.9g});
        normal           (0 0 1);
    }}"""
    write(case, "constant", "dynamicMeshDict", "dictionary", f"""dynamicFvMesh   dynamicMotionSolverFvMesh;

motionSolverLibs (sixDoFRigidBodyMotion);

motionSolver    sixDoFRigidBodyMotion;

// Mesh morphing: points within innerDistance of the cylinder move with it as a rigid
// body, points past outerDistance do not move, cosine blend between. outerDistance
// stays inside the square that wraps the O-grid, so the blend never reaches a block
// corner -- a concave cell squeezed past flat is a negative volume, and checkMesh at
// t = 0 cannot see it.
patches         (cylinder);
innerDistance   {body['inner']:.9g};
outerDistance   {body['outer']:.9g};

mass            {body['mass']:.9g};    // m* = {body['mstar']:g} x the displaced fluid mass
centreOfMass    (0 0 {zc:.9g});
momentOfInertia (1e-6 1e-6 1e-6);      // irrelevant: the orientation is constrained
g               (0 0 0);
rho             rhoInf;
rhoInf          {body['rho']:.9g};
report          on;

// 1.0, not 0.7. Relaxing the acceleration is what let the 2026-09-12 run hold a limit
// cycle 13% too wide while its energy audit read -10%; it belongs with
// moveMeshOuterCorrectors yes in fvSolution and neither works without the other.
accelerationRelaxation 1.0;

solver
{{
    type Newmark;
}}

constraints
{{
{constraints}
    noRotation
    {{
        sixDoFRigidBodyMotionConstraint orientation;
    }}
}}

restraints
{{
    spring
    {{
        sixDoFRigidBodyMotionRestraint linearSpring;
        anchor          (0 0 {zc:.9g});
        refAttachmentPt (0 0 {zc:.9g});
        stiffness       {body['stiffness']:.9g};   // f_n = {body['fn']:.9g} Hz, U* = {body['ustar']:g}
        damping         0;                         // zero: this is the energy check
        restLength      0;
    }}
}}""")


def write_system(case: Path, flow: dict, moving: bool, end_time: float, dt: float) -> None:
    functions = f"""functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patch           cylinder;
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          {flow['rho']:.9g};
        CofR            (0 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {flow['speed']:.9g};
        lRef            {flow['diameter']:.9g};
        Aref            {flow['diameter'] * flow['span']:.9g};
    }}"""
    if moving:
        # viv.py reads this function object by the name `bodyState`. It needs a live
        # motion solver, so it is an error under a `-postProcess` pass and cannot be
        # added after the fact: it goes in before the run or the motion history is lost.
        functions += """
    bodyState
    {
        type            sixDoFRigidBodyState;
        libs            (sixDoFRigidBodyState);
        writeControl    timeStep;
        writeInterval   1;
        angleFormat     degrees;
    }"""
    functions += "\n}"

    write(case, "system", "controlDict", "dictionary", f"""application     pimpleFoam;

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.9g};

deltaT          {dt:.9g};
adjustTimeStep  no;
maxCo           1;

writeControl    runTime;
writeInterval   {end_time / 40:.9g};
purgeWrite      4;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;

{functions}""")

    write(case, "system", "fvSchemes", "dictionary", """ddtSchemes
{
    default         backward;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}""")

    # `moveMeshOuterCorrectors yes` is the tight-coupling half that lives here: the body
    # and the mesh are solved inside the PIMPLE loop, not once per time step. With
    # accelerationRelaxation 1.0 in dynamicMeshDict this pair is what moved the measured
    # energy balance from -10% to +0.1% per cycle on the graded run.
    pimple = ("""PIMPLE
{
    correctPhi          yes;
    nOuterCorrectors    3;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    momentumPredictor   yes;
    moveMeshOuterCorrectors yes;
}""" if moving else """PIMPLE
{
    nOuterCorrectors    1;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    momentumPredictor   yes;
}""")

    write(case, "system", "fvSolution", "dictionary", f"""solvers
{{
    "pcorr.*"
    {{
        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-6;
        relTol          0;
    }}

    p
    {{
        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    "(U|cellDisplacement)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0.01;
    }}

    "(U|cellDisplacement)Final"
    {{
        $U;
        relTol          0;
    }}
}}

{pimple}""")
    write(case, "system", "decomposeParDict", "dictionary",
          "numberOfSubdomains 8;\n\nmethod          scotch;")


# --------------------------------------------------------------------------- driving


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--diameter", type=float, default=DIAMETER_M, help="metres")
    parser.add_argument("--ustar", type=float, default=USTAR, help="U / (f_n D)")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number on the diameter")
    parser.add_argument("--nu", type=float, default=1.0e-6, help="m2/s; water at 20 C")
    parser.add_argument("--rho", type=float, default=1000.0, help="kg/m3")
    parser.add_argument("--mstar", type=float, default=10.0,
                        help="body mass over displaced fluid mass")
    parser.add_argument("--dof", type=int, choices=(1, 2), default=1,
                        help="1 = transverse only (graded); 2 = free in the x-y plane (untried)")
    parser.add_argument("--stage", choices=("both", "fixed", "released"), default="both")
    parser.add_argument("--cycles", type=float, default=60.0,
                        help="shedding cycles to run, which sets endTime")
    parser.add_argument("--no-mesh", action="store_true",
                        help="write the cases and skip blockMesh/checkMesh/mesh_look.py")
    args = parser.parse_args()

    d = args.diameter
    speed = args.re * args.nu / d
    span = d  # one cell thick in z, and the reference area is D * span
    displaced = args.rho * math.pi * d * d / 4.0 * span
    mass = args.mstar * displaced
    fn = speed / (args.ustar * d)
    stiffness = mass * (2 * math.pi * fn) ** 2
    shedding = 0.164 * speed / d  # Williamson's parallel-shedding value at Re = 100

    flow = {"rho": args.rho, "speed": speed, "diameter": d, "span": span}
    body = {"mass": mass, "stiffness": stiffness, "mstar": args.mstar, "fn": fn,
            "ustar": args.ustar, "rho": args.rho, "span": span, "dof": args.dof,
            "inner": d, "outer": 0.9 * SQUARE_OVER_D * d}

    print(f"Re {args.re:g} on D = {d:g} m  ->  U = {speed:.6g} m/s")
    print(f"displaced fluid mass {displaced:.6e} kg, body mass {mass:.6e} kg (m* {args.mstar:g})")
    print(f"f_n = {fn:.6g} Hz  ->  k = {stiffness:.6e} N/m, T_n = {1 / fn:.6g} s")
    print(f"f_n with the added mass = {fn * math.sqrt(args.mstar / (args.mstar + 1)):.6g} Hz"
          "   (a cylinder's added mass is its own displaced mass, Lamb art. 68)")
    print(f"expected fixed shedding f = 0.164 U/D = {shedding:.6g} Hz (T = {1 / shedding:.6g} s)")
    print(f"blockage D / H = {100 / (2 * HALF_HEIGHT_D):.2f}%, cells {cell_count()}")

    dt = 1.0 / shedding / 200.0     # 200 steps per shedding cycle
    end = args.cycles / shedding

    stages = [("fixed", False), ("released", True)]
    if args.stage != "both":
        stages = [s for s in stages if s[0] == args.stage]

    root = Path.cwd()
    for name, moving in stages:
        case = root / name
        case.mkdir(parents=True, exist_ok=True)
        # The fixed stage gets a small transverse kick in the initial field, because a
        # perfectly symmetric 2D wake at Re = 100 can take hundreds of cycles to decide
        # to shed and this is the only thing that is asymmetric about it.
        write_fields(case, speed, moving, 0.0 if moving else 0.01 * speed)
        write_constant(case, args.nu, moving, body if moving else None)
        write_system(case, flow, moving, end, dt)
        (case / "system" / "blockMeshDict").write_text(block_mesh_dict(d, span))
        print(f"wrote {case}")
        if args.no_mesh:
            continue
        run(["blockMesh", "-case", "."], cwd=case)
        run(["checkMesh", "-case", "."], cwd=case)
        run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png",
             "--json", "look.json"], cwd=case)

    print("\nnext: solve fixed/ and check the Strouhal number before solving released/,")
    print("then `python3 viv.py released --fixed fixed --out viv.png` for the amplitude,")
    print("the frequency and the energy balance. checkMesh again at the largest")
    print("excursion, not only at t = 0: the morph is what can go negative.")


if __name__ == "__main__":
    main()
