#!/usr/bin/env python3
"""TEMPLATE: two-phase VOF dam break -- gmsh tank -> gmshToFoam -> setFields -> a case
`interFoam` can actually solve.

Copy this file into an empty case directory and run it there:

    python3 free_surface.py

It builds a square tank TANK_WIDTH_M on a side, one cell thick in z (the 2D recipe:
quads extruded one layer, the two z faces typed `empty`), puts a water column
COLUMN_WIDTH_M wide and twice that tall in the bottom left corner, meshes it,
converts with `gmshToFoam`, retypes the floor and the two side walls `wall` and the
two z faces `empty`, writes the whole `interFoam` case -- `0/alpha.water`, `0/U`,
`0/p_rgh`, `constant/g`, `constant/transportProperties`,
`constant/turbulenceProperties`, `system/controlDict`, `system/fvSchemes`,
`system/fvSolution`, `system/setFieldsDict` -- runs `setFields` so the water column is
a real interface rather than a uniform field, runs `checkMesh`, and finishes with
`mesh_look.py` so the result is looked at and not just built. Edit TANK_WIDTH_M and
COLUMN_WIDTH_M and re-run; the cell size, the tank height and the column height follow
from them.

Unlike `duct2d.py` and `body_in_box.py`, which write **stub** `controlDict`/`fvSchemes`/
`fvSolution` -- just enough for `gmshToFoam` and `checkMesh` to build a `Foam::Time`
and an `fvMesh`, and expected to be overwritten by whatever writes the real case --
everything this writes is meant to be solved as it stands. Nothing here is a
placeholder. It stops short of launching the solver, because what a solve costs and
whether it is wanted is not a template's call; the command it would be is printed at
the end.

Why the dam break and not something prettier: it has published experimental data.
Martin & Moyce (1952) measured the surge front position against time for a collapsing
water column, so a run of this case has something to be wrong against. A free-surface
case with no reference is a video, not a measurement.

The physical constants below, all at 20 degrees C and 1 atm, with their sources:

* water, rho = 998.2 kg/m^3 and nu = 1.004e-06 m^2/s (mu = 1.002e-03 Pa.s divided by
  rho) -- standard water properties, e.g. White, *Fluid Mechanics*, Table A.1.
* air, rho = 1.204 kg/m^3 and nu = 1.516e-05 m^2/s (mu = 1.825e-05 Pa.s divided by
  rho) -- White, Table A.2.
* sigma = 0.0728 N/m, the surface tension of a water/air interface at 20 degrees C --
  CRC Handbook of Chemistry and Physics.
* g = 9.81 m/s^2, standard gravity (9.80665) to three figures, pointing along -y
  because that is the direction the tank's floor is at the bottom of.

They are written out as numbers with units in the dictionary comments rather than
carried as magic constants, because a VOF case is only as good as its density ratio:
`rho` for air left at 1 instead of 1.204 is a 17 percent error in the lighter phase
and a solve that runs happily to the end.

The things that make a VOF case quietly wrong, baked in below:

* **`maxAlphaCo` as well as `maxCo`.** `adjustTimeStep` with only `maxCo` limits the
  time step by the bulk Courant number, which near a free surface is set by whatever
  the fastest cell is doing anywhere in the domain. The interface has its own speed
  limit: a step that moves the surface more than a fraction of a cell smears it across
  several, and the run looks completely healthy while doing it -- no bounding warning,
  no residual spike, just a greyer and greyer band where the interface was. Both are
  set below.
* **`p_rgh`, not `p`.** `interFoam` solves for p - rho*g*h, so the hydrostatic part of
  the pressure is not in the variable being solved: a hydrostatic column has p_rgh
  flat, and the solver does not spend its iterations rediscovering gravity every step.
  `p` is derived and written by the solver, and a `0/p` field handed to it fights the
  one it computes. There is no `0/p` below on purpose.
* **An open top that lets flow back in.** `atmosphere` carries `totalPressure` on
  p_rgh, `pressureInletOutletVelocity` on U and `inletOutlet` on alpha.water. A
  `fixedValue` pressure with a `zeroGradient` velocity cannot admit the air a
  collapsing column has to pull in behind it, and the case either stalls the surge or
  goes unstable while reporting nothing wrong.
* **`alpha.water` initialised as geometry, never as a uniform field.** `setFieldsDict`
  states the water as a `boxToCell` in metres, so the interface lands where the
  parameters put it and follows them when they are edited. An `internalField uniform
  0.5` is the failure this prevents: half water everywhere is not a free surface, it is
  a fog with the mean density of neither phase, and it will solve.
* **MULES and interface compression.** `nAlphaCorr`, `nAlphaSubCycles` and `cAlpha` in
  `fvSolution` and the `interfaceCompression vanLeer` scheme family in `fvSchemes` are
  what keeps alpha bounded in [0, 1] and the interface one to three cells thick. With a
  plain `Gauss linear` on `div(phi,alpha)` the interface diffuses away over a few
  hundred steps.
* **`startFrom latestTime` and `writeControl adjustableRunTime`.** A hosted container
  has a 24 h lifetime and can be reclaimed sooner, so a long solve is resumed rather
  than restarted; `adjustableRunTime` makes `adjustTimeStep` land the writes on round
  times instead of wherever the step happened to fall, which is what makes the written
  times comparable with a published measurement at all. See "Long solves" in
  `../notes/openfoam-field-notes.md`.
* `gmshToFoam` reads only msh format 2.2 ASCII; `Mesh.MshFileVersion` is set to 2.2
  right before `gmsh.write` for exactly that reason.
* `gmshToFoam` types every patch it creates `patch`, so the floor and sides are retyped
  `wall` and the two z faces `empty` with `foamDictionary` after conversion. OpenFOAM
  never infers `empty` from the geometry, only from what the boundary file says, and a
  "2D" dam break with its z faces left `patch` is a slow 3D one with two unphysical
  open sides.
* Metres, always. OpenFOAM has no units: a tank authored in millimetres and never
  divided by 1000 is a mesh a thousand times too big, at a thousandth of the intended
  Froude number, and `checkMesh` passes it without a word.
* `setFields` overwrites `0/alpha.water`, so `0/alpha.water.orig` is written beside it
  and copied over the top before `setFields` runs. Without that, a second run of this
  script sets the field from an already-set field, which happens to be harmless here
  and is not harmless the moment the dict gains a second region.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
TANK_WIDTH_M = 0.584      # tank width and height, m -- the classic damBreak box
COLUMN_WIDTH_M = 0.1461   # initial water column width, m; it stands twice as tall
# --------------------------------------------------------------------------

CELLS_ACROSS = 60                              # cells across the tank width
CELL_SIZE_M = TANK_WIDTH_M / CELLS_ACROSS      # background mesh size, m
THICK_M = CELL_SIZE_M                          # extrude depth = one cell
COLUMN_ASPECT = 2.0                            # column height / width, Martin & Moyce n = 2


def tank_box() -> tuple[tuple[float, float], tuple[float, float]]:
    """The tank's (x0, y0) and (x1, y1) in metres -- square, cornered at the origin."""
    return (0.0, 0.0), (TANK_WIDTH_M, TANK_WIDTH_M)


def water_box() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The water column as two 3D corners in metres, which is what `setFields` wants.

    Stated as geometry rather than as a cell list so it moves with the two numbers at
    the top and survives a change of mesh resolution. The z span is padded past the
    one-cell thickness in both directions: `boxToCell` takes cells whose centre is
    inside the box, and a box whose faces sit exactly on the cell centre plane is a
    coin toss decided by floating point.
    """
    pad = THICK_M
    return ((0.0, 0.0, -pad),
            (COLUMN_WIDTH_M, COLUMN_ASPECT * COLUMN_WIDTH_M, THICK_M + pad))


def build_mesh(msh_path: Path) -> None:
    """The tank outline, the quad mesh, the one-cell extrude, the msh file."""
    import gmsh

    (x0, y0), (x1, y1) = tank_box()

    gmsh.initialize()
    gmsh.model.add("free_surface")

    p1 = gmsh.model.geo.addPoint(x0, y0, 0, CELL_SIZE_M)
    p2 = gmsh.model.geo.addPoint(x1, y0, 0, CELL_SIZE_M)
    p3 = gmsh.model.geo.addPoint(x1, y1, 0, CELL_SIZE_M)
    p4 = gmsh.model.geo.addPoint(x0, y1, 0, CELL_SIZE_M)

    l_floor = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left = gmsh.model.geo.addLine(p4, p1)

    loop = gmsh.model.geo.addCurveLoop([l_floor, l_right, l_top, l_left])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # -- quads, not triangles: Frontal-Delaunay for quads + Blossom full-quad
    # recombination. A tet or prism mesh will run, but on a free surface the
    # interface is reconstructed per cell face, and hexahedra aligned with gravity
    # keep a flat surface flat instead of giving it a sawtooth the size of a cell.
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.model.mesh.setRecombine(2, surf)

    # -- one layer, swept into hexahedra
    out = gmsh.model.geo.extrude([(2, surf)], 0, 0, THICK_M,
                                 numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    # `extrude` on a single surface returns, in order: the top copy of the surface,
    # the volume, then one lateral surface per boundary curve of the surface, in the
    # same order the curve loop was built in -- [floor, right, top, left] here.
    # Named from that order, not from where a face sits in the bounding box afterwards.
    top_surf, volume = out[0][1], out[1][1]
    side_floor, side_right, side_top, side_left = (entity[1] for entity in out[2:6])

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_floor, side_left, side_right], name="walls")
    gmsh.model.addPhysicalGroup(2, [side_top], name="atmosphere")
    gmsh.model.addPhysicalGroup(3, [volume], name="internal")

    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # gmshToFoam reads only this
    gmsh.write(str(msh_path))
    gmsh.finalize()


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
    """`gmshToFoam` types every patch `patch`; this is how it gets its real type.

    `constant/polyMesh/boundary` is a bare list, not a named dictionary, so
    `foamDictionary` addresses the whole thing as one pseudo-entry called `entry0`
    (its own error message says so: "Known entries ... : 2(FoamFile entry0)") --
    `-entry walls.type` alone fails with "not found in dictionary"; the patch name has
    to be reached through it, `entry0.<patch>.type`.
    """
    run(["foamDictionary", "constant/polyMesh/boundary",
         "-entry", f"entry0.{patch}.type", "-set", kind])


def header(cls: str, obj: str, location: str = "") -> str:
    """A `FoamFile` header. The decorative C++ banner is optional; a half-copied one
    is not harmless, so none is written -- see `../ENVIRONMENT.md`."""
    where = f"    location    \"{location}\";\n" if location else ""
    return ("FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            f"    class       {cls};\n"
            f"{where}"
            f"    object      {obj};\n"
            "}\n")


# -- the fields ----------------------------------------------------------------
#
# Three fields, and deliberately not a fourth: `p` is derived from `p_rgh` by the
# solver and writing one here would fight it.

def alpha_water_field() -> str:
    """alpha.water = 1 in water, 0 in air. The internal field is uniform 0 and
    `setFields` carves the column out of it -- the uniform value is the *background*
    phase, not the initial condition.

    `inletOutlet` on the open top, not `zeroGradient`: a cell on the boundary with
    outflow takes the internal value, and one with inflow takes `inletValue` -- air.
    `zeroGradient` on an inflow face copies whatever alpha is inside the domain
    outward and back in again, which quietly feeds water in through the ceiling once
    a splash reaches it.
    """
    return (header("volScalarField", "alpha.water", "0") + """
dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    walls
    {
        type            zeroGradient;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    frontAndBack
    {
        type            empty;
    }
}
""")


def u_field() -> str:
    """U, at rest everywhere at t = 0: a dam break is driven by gravity, not by an
    inlet.

    `pressureInletOutletVelocity` on the open top, paired with `totalPressure` on
    p_rgh: outflow is zeroGradient, inflow takes its direction from the patch normal
    and its magnitude from the flux the pressure condition asks for. A `fixedValue
    (0 0 0)` lid is a closed box with a pressure boundary condition drawn on it, and
    the column cannot collapse without compressing the air it is trapped under.
    """
    return (header("volVectorField", "U", "0") + """
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    walls
    {
        type            noSlip;
    }
    atmosphere
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }
    frontAndBack
    {
        type            empty;
    }
}
""")


def p_rgh_field() -> str:
    """p_rgh = p - rho*g*h, which is what `interFoam` solves. Zero everywhere at rest,
    whatever the depth, because the hydrostatic part is not in the variable.

    `fixedFluxPressure` on the walls, not `zeroGradient`: it sets the wall gradient
    from the flux the momentum equation needs, which is what keeps a wall from
    generating or absorbing a little flow under a buoyancy-driven field.
    `totalPressure` with p0 = 0 on the open top: a static pressure that drops by the
    dynamic head on inflow, so air can be drawn in without the boundary doing work on
    the solution. A plain `fixedValue 0` there is a boundary that holds the surface
    flat as it passes, which is exactly the measurement the case exists to make.
    """
    return (header("volScalarField", "p_rgh", "0") + """
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    walls
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    atmosphere
    {
        type            totalPressure;
        p0              uniform 0;
        value           uniform 0;
    }
    frontAndBack
    {
        type            empty;
    }
}
""")


# -- constant/ -----------------------------------------------------------------

def g_file() -> str:
    """Gravity, and the direction is half the content: -y, because the tank's floor is
    at y = 0 and its open top at y = TANK_WIDTH_M. A `g` pointing +y runs, drains the
    column upward into the atmosphere patch and reports nothing wrong.

    9.81 m/s^2 is standard gravity (9.80665 m/s^2, CGPM 1901) to three figures.
    """
    return (header("uniformDimensionedVectorField", "g", "constant") + """
dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);
""")


def transport_properties() -> str:
    """Two phases, each with its own rho and nu, and the surface tension between them.

    Every number at 20 degrees C and 1 atm, with its source, because a VOF answer is a
    density-ratio answer: water/air is about 830:1, and a lighter phase entered as
    rho = 1 (the round number the tutorials carry) is a 17 percent error in the phase
    that sets how fast air has to get out of the way of the surge.

    * water: rho = 998.2 kg/m^3, mu = 1.002e-03 Pa.s, so nu = mu/rho = 1.004e-06 m^2/s
      -- White, *Fluid Mechanics*, Table A.1.
    * air: rho = 1.204 kg/m^3, mu = 1.825e-05 Pa.s, so nu = mu/rho = 1.516e-05 m^2/s
      -- White, Table A.2.
    * sigma = 0.0728 N/m, water against air -- CRC Handbook of Chemistry and Physics.

    sigma matters least of the three at this scale and is not optional: the Bond number
    for a 0.146 m column is of order 10^4, so surface tension does nothing to the surge
    front, and the solver still wants the number before it will start.
    """
    return (header("dictionary", "transportProperties", "constant") + """
phases (water air);

water
{
    transportModel  Newtonian;
    nu              1.004e-06;  // m^2/s -- water at 20 C, mu 1.002e-03 Pa.s / rho
    rho             998.2;      // kg/m^3 -- water at 20 C, 1 atm
}

air
{
    transportModel  Newtonian;
    nu              1.516e-05;  // m^2/s -- air at 20 C, mu 1.825e-05 Pa.s / rho
    rho             1.204;      // kg/m^3 -- air at 20 C, 1 atm
}

sigma           0.0728;         // N/m -- water/air surface tension at 20 C
""")


def turbulence_properties() -> str:
    """Laminar, stated rather than defaulted.

    A dam break's surge front is an inertia-and-gravity problem and the published
    experiments are matched by a laminar solve; a turbulence model here adds an eddy
    viscosity that thickens the front. ESI OpenFOAM reads this file as
    `turbulenceProperties`; the openfoam.org line calls the same thing
    `momentumTransport`.
    """
    return (header("dictionary", "turbulenceProperties", "constant") + """
simulationType  laminar;
""")


# -- system/ -------------------------------------------------------------------

def control_dict() -> str:
    """The real thing, not a stub: `interFoam`, an interface Courant limit, an adjusted
    time step, and writes on round times.

    * `adjustTimeStep yes` with **both** `maxCo` and `maxAlphaCo`. `maxCo` alone is the
      failure this file exists to prevent: the bulk Courant limit is set by the fastest
      cell anywhere, and a step within it can still move the interface most of a cell
      per step, which smears the surface across five or ten cells over a few hundred
      steps. Nothing in the log says so.
    * `maxDeltaT` as a ceiling for the early steps, when the column is at rest and the
      Courant numbers are near zero so nothing else bounds the step.
    * `writeControl adjustableRunTime`: with an adjusted time step the solver would
      otherwise write at whatever time the step landed on, and a surge-front position
      at t = 0.0937 s cannot be put beside a published one at t = 0.1 s without an
      interpolation nobody asked for. `adjustableRunTime` shortens the step to land on
      the interval instead.
    * `startFrom latestTime`: the hosted container's lifetime is 24 h and it can be
      reclaimed sooner, so an interrupted solve resumes from its last write with one
      relaunch. See "Long solves" in `../notes/openfoam-field-notes.md`. The
      `writeInterval` below is the amount of work an interruption can destroy.
    """
    end_time = 1.0
    return (header("dictionary", "controlDict", "system") + f"""
application     interFoam;

startFrom       latestTime;     // an interrupted solve resumes, it does not restart
startTime       0;
stopAt          endTime;
endTime         {end_time};

deltaT          0.001;
adjustTimeStep  yes;
maxCo           0.5;            // bulk Courant limit
maxAlphaCo      0.5;            // INTERFACE Courant limit -- without it the surface smears
maxDeltaT       0.01;

writeControl    adjustableRunTime;  // writes land on round times, not on step ends
writeInterval   0.02;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;
""")


def fv_schemes() -> str:
    """The interface compression family, which is the difference between a surface and
    a gradient.

    `div(phi,alpha)` on `interfaceCompression vanLeer 1`: vanLeer is a bounded
    second-order limiter, and the compression term is an artificial counter-diffusion
    that pushes alpha back toward 0 and 1 at the same rate numerical diffusion spreads
    it. With `Gauss linear` here, or with vanLeer and no compression, alpha bounds hold
    and the interface still widens monotonically until the picture is a gradient from
    floor to ceiling.

    `linearUpwind grad(U)` on momentum rather than plain `linear`: a dam break's
    momentum field has a shock-like front, and central differencing on it oscillates.
    """
    return (header("dictionary", "fvSchemes", "system") + """
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
    div(U)          Gauss linear;
    "div\\(phi,alpha.*\\)"           Gauss interfaceCompression vanLeer 1;
    "div\\(phir,alpha.*,alpha.*\\)"  Gauss interfaceCompression vanLeer 1;
    "div\\(\\(\\(rho\\*nuEff\\)\\*dev2\\(T\\(grad\\(U\\)\\)\\)\\)\\)" Gauss linear;
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
""")


def fv_solution() -> str:
    """MULES on alpha, and a pressure solver that knows p_rgh is the unknown.

    * `nAlphaCorr 2`, `nAlphaSubCycles 1`, `cAlpha 1` -- the alpha corrections per
      step, the sub-cycles of the alpha equation per momentum step, and the interface
      compression coefficient. `cAlpha 0` turns compression off and the interface
      spreads; much above 1 and it stiffens into a staircase.
    * `MULESCorr yes` with `nLimiterIter 5`: the semi-implicit limiter, which is what
      allows an interface Courant number up to 0.5 instead of the 0.1 or so an explicit
      MULES tolerates.
    * `momentumPredictor no` is the usual choice for a gravity-driven free surface: the
      predictor costs a momentum solve per step and buys nothing when the pressure is
      doing the work.
    * `p_rghFinal` is written out in full rather than inherited through a `$p_rgh`
      macro, so a reader sees that only the relative tolerance changes on the last
      corrector.
    """
    return (header("dictionary", "fvSolution", "system") + """
solvers
{
    "alpha.water.*"
    {
        nAlphaCorr      2;      // alpha corrections per time step
        nAlphaSubCycles 1;      // alpha sub-cycles per momentum step
        cAlpha          1;      // interface compression; 0 lets the surface diffuse away
        MULESCorr       yes;    // semi-implicit limiter -- why maxAlphaCo 0.5 is safe
        nLimiterIter    5;
        alphaApplyPrevCorr yes;

        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0;
    }

    "pcorr.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-05;
        relTol          0;
    }

    p_rgh
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-07;
        relTol          0.05;
    }

    p_rghFinal
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-07;
        relTol          0;
    }

    "U.*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor   no;
    nOuterCorrectors    1;
    nCorrectors         3;
    nNonOrthogonalCorrectors 0;
}
""")


def set_fields_dict() -> str:
    """The water column as geometry in metres, which is the whole point of `setFields`.

    `defaultFieldValues` is the background -- air everywhere -- and the one
    `boxToCell` region overwrites it with water. Stated as two corners rather than as
    a cell list because a cell list is a statement about one mesh: edit
    COLUMN_WIDTH_M, or CELLS_ACROSS, and a box still means the same column while a
    list means a different one.
    """
    (xa, ya, za), (xb, yb, zb) = water_box()
    return (header("dictionary", "setFieldsDict", "system") + f"""
defaultFieldValues
(
    volScalarFieldValue alpha.water 0   // air everywhere, then the column below
);

regions
(
    boxToCell
    {{
        // metres: the water column, {COLUMN_WIDTH_M} m wide and {COLUMN_ASPECT * COLUMN_WIDTH_M} m tall,
        // in the corner at the origin. The z span is padded past the single cell
        // because boxToCell selects on cell centres.
        box ({xa} {ya} {za}) ({xb} {yb} {zb});
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}
);
""")


CASE_FILES = {
    "0/alpha.water": alpha_water_field,
    "0/alpha.water.orig": alpha_water_field,
    "0/U": u_field,
    "0/p_rgh": p_rgh_field,
    "constant/g": g_file,
    "constant/transportProperties": transport_properties,
    "constant/turbulenceProperties": turbulence_properties,
    "system/controlDict": control_dict,
    "system/fvSchemes": fv_schemes,
    "system/fvSolution": fv_solution,
    "system/setFieldsDict": set_fields_dict,
}
"""Every file `interFoam` reads, and nothing it does not. `0/p` is absent on purpose:
the solver derives `p` from `p_rgh` and writes it itself."""


def write_case(case: Path) -> None:
    """The whole case on disk before the mesh exists, because `gmshToFoam` builds a
    `Foam::Time` before it touches the mesh and fails "cannot find file
    .../system/controlDict" in a directory holding nothing but a `.msh`. There is no
    mesh-only exemption, and `checkMesh` goes further still -- it builds an `fvMesh`,
    so it reads `fvSchemes` too.

    Where `duct2d.py` and `body_in_box.py` satisfy that with three stubs they expect
    to be overwritten, these are the real dictionaries: the mesh tools get what they
    need as a side effect of the case being complete.
    """
    for rel, source in CASE_FILES.items():
        path = case / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source(), encoding="utf-8")


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


def main() -> None:
    case = Path.cwd()
    msh = case / "free_surface.msh"
    build_mesh(msh)
    write_case(case)

    run(["gmshToFoam", msh.name])
    retype("walls", "wall")
    # One physical group covers both z faces, so one retype makes the case 2D.
    retype("frontAndBack", "empty")

    run(["checkMesh", "-case", "."])

    # setFields overwrites 0/alpha.water, so the pristine copy goes back first and the
    # script stays re-runnable.
    shutil.copyfile(case / "0" / "alpha.water.orig", case / "0" / "alpha.water")
    run(["setFields", "-case", "."])

    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png", "--json", "look.json"])

    # Not launched from here: what a solve costs, and whether it is wanted at all, is
    # not a template's call. `startFrom latestTime` is already set, so this loop
    # resumes itself through any in-sandbox failure -- see "Long solves" in
    # ../notes/openfoam-field-notes.md.
    print("\nthe case is complete and meant to be solved. one way to launch it:\n"
          "    until interFoam >> log.interFoam 2>&1; do "
          "echo resuming >> log.interFoam; sleep 5; done\n"
          "and in parallel, with a decomposeParDict beside it:\n"
          "    decomposePar && until mpirun -np 4 interFoam -parallel >> log.interFoam 2>&1; "
          "do sleep 5; done")


if __name__ == "__main__":
    main()
