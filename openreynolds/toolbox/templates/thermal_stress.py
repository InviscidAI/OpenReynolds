#!/usr/bin/env python3
"""TEMPLATE: thermal stress in a constrained bar -- gmsh -> gmshToFoam -> a case
`solidDisplacementFoam` solves, with the temperature field and the stress it causes.

Copy this file into an empty case directory and run it there:

    python3 thermal_stress.py

It builds a 2D bar BAR_LENGTH_M long and BAR_HEIGHT_M high, one cell thick in z (the
2D recipe: quads extruded one layer, the two z faces typed `empty`), meshes it,
converts with `gmshToFoam`, retypes the two ends `symmetryPlane` and the z faces
`empty`, writes the whole `solidDisplacementFoam` case including
`constant/mechanicalProperties` and `constant/thermalProperties` with
`thermalStress yes`, runs `checkMesh`, and finishes with `mesh_look.py` so the mesh is
looked at and not just built. Edit these two numbers -- BAR_HEIGHT_M and DELTA_T_K --
and re-run; the cell size, the thermal load and every expected answer printed at the
end follow from them.

Like `free_surface.py`, `rotating.py` and `porous.py`, and unlike `duct2d.py` and
`body_in_box.py`, which write **stub** `controlDict`/`fvSchemes`/`fvSolution` just
sufficient for `gmshToFoam` and `checkMesh` to build a `Foam::Time` and an `fvMesh`,
everything written here is meant to be solved as it stands. It stops short of
launching the solver, because what a solve costs and whether it is wanted is not a
template's call; the commands it would be are printed at the end.

NOTHING HERE NEEDED INSTALLING
------------------------------
`solidDisplacementFoam` is a standard OpenFOAM solver and is in the workspace image
already, because the image is built FROM `opencfd/openfoam-default:2512`, a full ESI
distribution. It is a transient segregated finite-volume solver for linear-elastic,
small-strain deformation with optional thermal diffusion and thermal stresses, and it
lives at `applications/solvers/stressAnalysis/solidDisplacementFoam` in the OpenFOAM
sources. Its one shipped tutorial, `$FOAM_TUTORIALS/stressAnalysis/solidDisplacementFoam/
plateHole`, is a **mechanical** case: it has a `constant/thermalProperties` but with
`thermalStress no`, so the thermal half of the solver has no worked example in the
distribution. That is the gap this template fills, and it is why the numbers below are
derived here rather than copied from a tutorial.

WHAT THE SOLVER SOLVES, AND THE TRAP IN ITS TEMPERATURE FIELD
-------------------------------------------------------------
Two equations, segregated, iterated to convergence at each (pseudo-)time step:

    ddt(T)    = laplacian(DT, T)                      DT = k/(rho*C), m2/s
    d2dt2(D)  = laplacian(2*mu + lambda, D) + div(sigmaD) - grad(3*K*alpha*T)

and the stress written out is

    sigma = 2*mu*symm(grad(D)) + lambda*I*tr(grad(D)) - I*3*K*alpha*T

**The last term uses T itself, not T minus a reference temperature.** There is no
`T0`, no `TRef`, nowhere in `constant/thermalProperties` to put one: read
`readThermalProperties.H` and `calculateStress.H` in the solver's own sources and the
field is used raw. So the stress-free state of the body is T = 0 in whatever units the
T field is written in, and a case that sets `internalField uniform 300` because the
metal is at 300 K is asking for the stress of a 300 K temperature **rise** from an
unstressed state at absolute zero. On steel that is -660 MPa appearing out of nothing,
which is a plausible-looking number and a wrong one -- the plateHole tutorial ships
exactly that T field, and gets away with it only because its `thermalStress` is `no`.

Everything below therefore writes T as a temperature **rise above the stress-free
state**, and the cold face of the bar is at T = 0 rather than at an ambient.

THE CASE, AND WHY ITS ANSWER IS KNOWN EXACTLY
---------------------------------------------
A bar lying along x, held between two smooth rigid walls at x = 0 and x = L, resting
on a smooth rigid floor at y = 0, free at y = H, and conducting heat from a hot face
at y = 0 to a cold face at y = H:

    x = 0, x = L   `symmetryPlane`      D.n = 0, tangential free: a frictionless wall
    y = 0 `hot`    D fixedValue (0 0 0), T fixedValue DELTA_T_K
    y = H `cold`   D tractionDisplacement, zero traction; T fixedValue 0
    z faces        `empty`

Conduction first. Steady state between two fixed-value faces with the sides
insulated is the linear profile, exactly:

    T(y) = DELTA_T * (1 - y/H)

Elasticity second. The ends hold the bar at ex_xx = 0, the top is traction-free so
sigma_yy = 0, and the mesh is one cell thick so ex_zz = 0. Put those three into the
solver's own constitutive law above, with its plane-stress Lame coefficients
(`readMechanicalProperties.H`: lambda = nu*E/((1+nu)(1-nu)) and 3K = E/(1-nu) when
`planeStress yes`):

    sigma_yy = (2*mu + lambda)*e - 3*K*alpha*T = 0   =>  e = 3*K*alpha*T/(2*mu + lambda)
    sigma_xx = lambda*e - 3*K*alpha*T               =  -3*K*alpha*T * 2*mu/(2*mu + lambda)

and with 2*mu = E/(1+nu), 2*mu + lambda = E/((1+nu)(1-nu)), so 2*mu/(2*mu+lambda) =
(1 - nu) and 3K = E/(1-nu), the two factors of (1 - nu) cancel and what is left is the
textbook constrained-bar result with no elastic constant surviving but E:

    PLANE STRESS   sigma_xx(y) = -E*alpha*T(y)              ex_yy(y) = (1+nu)*alpha*T(y)
    PLANE STRAIN   sigma_xx(y) = -E*alpha*T(y)/(1-nu)       ex_yy(y) = (1+nu)/(1-nu)*alpha*T(y)

The plane-strain line is the same algebra with the other branch of
`readMechanicalProperties.H` (lambda = nu*E/((1+nu)(1-2nu)), 3K = E/(1-2nu)), which
gives 2*mu/(2*mu+lambda) = (1-2nu)/(1-nu); it is what `planeStress no` selects, and the
two differ by exactly 1/(1-nu). Integrating ex_yy up from the floor,

    D_y(y) = (1+nu)*alpha*DELTA_T*(y - y*y/(2*H))        D_y(H) = (1+nu)*alpha*DELTA_T*H/2

This is an exact solution and not an asymptotic one: sigma_xx varies with y and only
with y, so x-equilibrium holds; sigma_yy and sigma_xy are identically zero, so
y-equilibrium holds; D = (0, D_y(y), 0) is compatible with the strains; and D is
identically zero on the y = 0 face, which is why `fixedValue (0 0 0)` can be imposed
there without over-constraining anything and without a boundary layer near it. The
answer does not depend on the mesh, the bar's length, or how many iterations it takes.

It is also a stiffer numerical test than a uniformly heated bar would be. With T
uniform, `grad(3*K*alpha*T)` is zero everywhere in the interior and the whole thermal
load enters through `tractionDisplacement`'s boundary term alone. With T linear in y,
`grad(3*K*alpha*T)` is non-zero in every interior cell and is cancelled, to the
solver's tolerance, by the gradient of `div(sigmaD)`: the two terms are individually
large and the answer is their difference.

Known gotchas, baked in below rather than left to be rediscovered by failing:

* T is a temperature **rise**, not an absolute temperature -- see above. The single
  largest way to get a wrong thermal stress out of this solver.
* `thermalStress yes` in `constant/thermalProperties`, which is the switch that makes
  `C`, `k` and `alpha` be read at all; with it `no` the sub-dictionaries are ignored
  and the solver runs as a pure elasticity code without saying so.
* `E` is divided by `rho` on the way in (`Normalising E : E/rho` in the solver's own
  log) and the stress is multiplied back by `rho` on the way out, so the value in
  `constant/mechanicalProperties` is a real Young's modulus in Pa and `rho` matters to
  the transient and not to the converged answer -- but both must be there.
* `d2dt2Schemes default steadyState;`, so D is quasi-static at each step and the time
  loop is just the conduction relaxing. With it `Euler` the bar rings elastically
  instead, over an acoustic time thousands of times shorter than the thermal one, and
  nothing reaches a steady state in a sensible number of steps.
* `symmetryPlane` is a constraint patch type: every field on that patch must use it
  too, which is why the hot and cold faces are plain patches -- a `fixedValue` T
  cannot sit on a `symmetryPlane`, and the mechanical datum has to be found elsewhere.
* The only supported way to pin a rigid-body mode here is a boundary condition.
  `solidDisplacementFoam.C` carries two commented-out `DEqn.setComponentReference`
  calls for exactly this job, and they are commented out in the shipped source, so a
  case whose every patch is traction-free or symmetry in some direction leaves that
  direction's translation unconstrained and the linear solve singular in it.
* `MshFileVersion` 2.2 -- `gmshToFoam` reads only msh format 2.2 ASCII.
* `gmshToFoam` types every patch it creates `patch`, so the ends are retyped
  `symmetryPlane` and the z faces `empty` with `foamDictionary` after conversion;
  OpenFOAM never infers either from the geometry, only from the boundary file.
* `sigma` is a symmetric tensor and the samplers want scalars, so the run ends with
  `postProcess -func "components(sigma)"` to write `sigmaxx`, `sigmayy` and the rest,
  then `postProcess -func singleGraph` to sample them along y -- the same two commands,
  in the same order, as the plateHole tutorial's own `Allrun`.
* Metres and kelvin, always. OpenFOAM has no units: a bar authored in millimetres and
  never scaled is a mesh a thousand times too big, and `checkMesh` passes it happily.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
BAR_HEIGHT_M = 0.02     # bar height, y -- the direction the heat flows along
DELTA_T_K = 100.0       # temperature RISE of the hot face above the stress-free state
# --------------------------------------------------------------------------

BAR_LENGTH_M = 2.0 * BAR_HEIGHT_M     # x; the answer does not depend on it
CELL_SIZE_M = BAR_HEIGHT_M / 20.0     # background mesh size -- 20 cells across y
THICK_M = CELL_SIZE_M                 # extrude depth = one cell, kept roughly cubic

# Steel, and the same four numbers the plateHole tutorial uses for three of them, so a
# reader can check them against something that ships: rho 7854, nu 0.3, E 2e11,
# C 434, k 60.5, alpha 1.1e-5.
E_PA = 2.0e11           # Young's modulus
NU = 0.3                # Poisson's ratio
RHO = 7854.0            # density, kg/m3
ALPHA_PER_K = 1.1e-5    # linear thermal expansion coefficient, 1/K
C_J_PER_KG_K = 434.0    # specific heat
K_W_PER_M_K = 60.5      # thermal conductivity

# `planeStress yes` is the 2D plate state: stress-free through the thickness. `no` is
# plane strain, ex_zz = 0, which is what a long prism does. Both are exact here and
# they differ by 1/(1 - nu); the docstring derives both.
PLANE_STRESS = True

# How long to run. The slowest Fourier mode of the conduction decays as
# exp(-pi^2 * DT * t / H^2); DT = k/(rho*C) is 1.775e-5 m2/s for the steel above, so
# with H = 0.02 m the decay rate is 0.438 /s and 100 s is 44 e-foldings. The D
# equation is `steadyState` in d2dt2 and so carries no time scale of its own.
DIFFUSIVITY = K_W_PER_M_K / (RHO * C_J_PER_KG_K)
END_TIME_S = 100.0
DELTA_T_S = 1.0


def expected() -> dict[str, float]:
    """The closed-form answers for the numbers above, derived in the docstring."""
    factor = 1.0 if PLANE_STRESS else 1.0 / (1.0 - NU)
    strain = (1.0 + NU) * factor * ALPHA_PER_K * DELTA_T_K
    return {
        # sigma_xx on the hot face, where T is DELTA_T_K and the stress is largest
        "sigma_xx_hot_pa": -E_PA * ALPHA_PER_K * DELTA_T_K * factor,
        # What the solver's own `Max sigmaEq` line will say, which is NOT the number
        # above. sigma is diag(s, 0, s) here, so sqrt(1.5*magSqr(dev(sigma))) is
        # exactly |s| -- but `max` is over CELLS, and the hottest cell centre sits
        # half a cell above the hot face, so it reports (1 - dy/(2H)) of it: 2.5% low
        # on 20 cells. Quoting the hot-face value against that line would be reading
        # the mesh as an error in the physics.
        "sigma_eq_max_pa": (E_PA * ALPHA_PER_K * DELTA_T_K * factor
                            * (1.0 - CELL_SIZE_M / (2.0 * BAR_HEIGHT_M))),
        "strain_yy_hot": strain,
        "displacement_y_top_m": strain * BAR_HEIGHT_M / 2.0,
        "diffusion_e_folding_s": BAR_HEIGHT_M**2 / (3.14159265**2 * DIFFUSIVITY),
    }


def build_mesh(msh_path: Path) -> None:
    """The outline, the quad mesh, the extrude, the msh file -- everything gmsh does."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("thermal_stress")

    p1 = gmsh.model.geo.addPoint(0, 0, 0, CELL_SIZE_M)
    p2 = gmsh.model.geo.addPoint(BAR_LENGTH_M, 0, 0, CELL_SIZE_M)
    p3 = gmsh.model.geo.addPoint(BAR_LENGTH_M, BAR_HEIGHT_M, 0, CELL_SIZE_M)
    p4 = gmsh.model.geo.addPoint(0, BAR_HEIGHT_M, 0, CELL_SIZE_M)

    l_hot = gmsh.model.geo.addLine(p1, p2)        # y = 0
    l_right = gmsh.model.geo.addLine(p2, p3)      # x = L
    l_cold = gmsh.model.geo.addLine(p3, p4)       # y = H
    l_left = gmsh.model.geo.addLine(p4, p1)       # x = 0

    loop = gmsh.model.geo.addCurveLoop([l_hot, l_right, l_cold, l_left])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # -- quads, not triangles: Frontal-Delaunay for quads + Blossom full-quad recombination
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.model.mesh.setRecombine(2, surf)

    # -- one layer, swept into hexahedra
    out = gmsh.model.geo.extrude([(2, surf)], 0, 0, THICK_M,
                                 numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    # `extrude` on a single surface returns, in order: the top copy of the surface,
    # the volume, then one lateral surface per boundary curve, in the order the curve
    # loop was built in -- [hot, right, cold, left] here. Naming from this order, as
    # the line is created, and not from where a face sits in a bounding box
    # afterwards, is what keeps the hot face hot when the shape stops being a box.
    top_surf, volume = out[0][1], out[1][1]
    side_hot, side_right, side_cold, side_left = (entity[1] for entity in out[2:6])

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_hot], name="hot")
    gmsh.model.addPhysicalGroup(2, [side_cold], name="cold")
    gmsh.model.addPhysicalGroup(2, [side_left], name="end_left")
    gmsh.model.addPhysicalGroup(2, [side_right], name="end_right")
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
    `-entry hot.type` alone fails with "not found in dictionary"; the patch name has
    to be reached through it, `entry0.<patch>.type`.
    """
    run(["foamDictionary", "constant/polyMesh/boundary",
         "-entry", f"entry0.{patch}.type", "-set", kind])


HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def foam(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj) + body)


def write_case(case: Path) -> None:
    """The whole `solidDisplacementFoam` case: both fields, both property
    dictionaries, the three `system` dictionaries and the sampler.

    Written after `gmshToFoam` so nothing here has to agree with a patch list that
    does not exist yet, and written in full rather than as stubs because this case is
    meant to be solved.
    """
    # -- the displacement field. `symmetryPlane` on the two ends is the frictionless
    # rigid wall that holds ex_xx = 0 while leaving the bar free to swell across y.
    # `fixedValue (0 0 0)` on the hot face is the datum for the y translation, and it
    # is exact rather than an approximation: the analytic D is identically zero there.
    foam(case / "0" / "D", "volVectorField", "D", """
dimensions      [0 1 0 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    end_left
    {
        type            symmetryPlane;
    }
    end_right
    {
        type            symmetryPlane;
    }
    hot
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    cold
    {
        type            tractionDisplacement;
        traction        uniform (0 0 0);
        pressure        uniform 0;
        value           uniform (0 0 0);
    }
    frontAndBack
    {
        type            empty;
    }
}
""")

    # -- the temperature field, as a RISE above the stress-free state (see the
    # docstring): DELTA_T_K on the hot face, 0 on the cold one, insulated ends.
    foam(case / "0" / "T", "volScalarField", "T", f"""
dimensions      [0 0 0 1 0 0 0];

internalField   uniform 0;

boundaryField
{{
    end_left
    {{
        type            symmetryPlane;
    }}
    end_right
    {{
        type            symmetryPlane;
    }}
    hot
    {{
        type            fixedValue;
        value           uniform {DELTA_T_K:g};
    }}
    cold
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""")

    foam(case / "constant" / "mechanicalProperties", "dictionary",
         "mechanicalProperties", f"""
rho
{{
    type        uniform;
    value       {RHO:g};
}}

nu
{{
    type        uniform;
    value       {NU:g};
}}

E
{{
    type        uniform;
    value       {E_PA:g};
}}

planeStress     {"yes" if PLANE_STRESS else "no"};
""")

    # `thermalStress yes` is the switch that turns the thermal half of the solver on.
    # Without it C, k and alpha are read and never used, and the run is pure elasticity.
    foam(case / "constant" / "thermalProperties", "dictionary",
         "thermalProperties", f"""
C
{{
    type        uniform;
    value       {C_J_PER_KG_K:g};
}}

k
{{
    type        uniform;
    value       {K_W_PER_M_K:g};
}}

alpha
{{
    type        uniform;
    value       {ALPHA_PER_K:g};
}}

thermalStress   yes;
""")

    foam(case / "system" / "controlDict", "dictionary", "controlDict", f"""
application     solidDisplacementFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {END_TIME_S:g};
deltaT          {DELTA_T_S:g};

writeControl    timeStep;
writeInterval   {int(END_TIME_S / DELTA_T_S)};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")

    # `d2dt2Schemes steadyState` keeps D quasi-static: the time loop is the
    # conduction relaxing, not the bar ringing. The rest is the plateHole tutorial's,
    # which is the only shipped case that exercises these scheme names at all.
    foam(case / "system" / "fvSchemes", "dictionary", "fvSchemes", """
d2dt2Schemes
{
    default         steadyState;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         leastSquares;
    grad(D)         leastSquares;
    grad(T)         leastSquares;
}

divSchemes
{
    default         none;
    div(sigmaD)     Gauss linear;
}

laplacianSchemes
{
    default         none;
    laplacian(DD,D) Gauss linear corrected;
    laplacian(DT,T) Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         none;
}
""")

    # `stressAnalysis` is read by `readSolidDisplacementFoamControls.H`: `D` is the
    # convergence tolerance on the displacement residual, not a field name, and
    # `compactNormalStress` and `nCorrectors` are required entries with no defaults
    # between them except nCorrectors' 1. Tighter than the tutorial's 1e-6 on both
    # counts, because what is being measured here is how close to exact it gets.
    foam(case / "system" / "fvSolution", "dictionary", "fvSolution", """
solvers
{
    "(D|T)"
    {
        solver          GAMG;
        tolerance       1e-10;
        relTol          0;
        smoother        GaussSeidel;
        nCellsInCoarsestLevel 20;
    }
}

stressAnalysis
{
    compactNormalStress yes;
    nCorrectors     3;
    D               1e-09;
}
""")

    # Samples T and the stress components along a line up the middle of the bar, in
    # `raw` format, which is a header comment and then one row per point. The
    # `#includeEtc` lines are the plateHole tutorial's and resolve against $FOAM_ETC.
    # A quarter of a cell off the mid-plane: the answer is uniform in x, so where the
    # line sits does not matter, and `BAR_LENGTH_M/2` is exactly the kind of round
    # number that lands on a face plane, where which cell a sample belongs to is a
    # tie. A quarter cell is inside one cell on any mesh of this size.
    line_x = BAR_LENGTH_M / 2.0 + CELL_SIZE_M / 4.0
    foam(case / "system" / "singleGraph", "dictionary", "singleGraph", f"""
start   ({line_x:.10g} 0 {THICK_M / 2.0:.10g});
end     ({line_x:.10g} {BAR_HEIGHT_M:.10g} {THICK_M / 2.0:.10g});
fields  (T sigmaxx sigmayy sigmaxy);

#includeEtc "caseDicts/postProcessing/graphs/sampleDict.cfg"

// `sampleDict.cfg` defaults to `type uniform; nPoints 100;`, which over 20 cells
// returns each cell's value about five times, because a sampledSet interpolates
// `cell` unless told otherwise: the profile comes out a staircase about the answer,
// deviating from it by up to half a cell. `midPoint` instead puts one sample at each
// cell centre along the line, where the cell value is the answer and no
// interpolation happens at all.
setConfig
{{
    type    midPoint;
    axis    y;
}}

// Must be last entry
#includeEtc "caseDicts/postProcessing/graphs/graph.cfg"
""")


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


def main() -> None:
    case = Path.cwd()
    msh = case / "thermal_stress.msh"
    build_mesh(msh)
    # `gmshToFoam` builds a `Foam::Time` before it touches the mesh, so the case
    # dictionaries have to exist before it runs, not after.
    write_case(case)

    run(["gmshToFoam", msh.name])
    for patch in ("end_left", "end_right"):
        retype(patch, "symmetryPlane")
    for patch in ("hot", "cold"):
        retype(patch, "wall")
    # One physical group covers both z faces, so one retype makes the case 2D.
    retype("frontAndBack", "empty")

    run(["checkMesh", "-case", "."])
    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png",
         "--json", "look.json"])

    answers = expected()
    state = "plane stress" if PLANE_STRESS else "plane strain"
    print(f"\n# the case is written and the mesh is checked. {state}, "
          f"T rise {DELTA_T_K:g} K over {BAR_HEIGHT_M:g} m of steel.")
    print("#")
    print("# what it should produce, from the closed form in this file's docstring:")
    print(f"#   sigma_xx on the hot face   {answers['sigma_xx_hot_pa'] / 1e6:+.4f} MPa")
    print(f"#   the solver's Max sigmaEq   {answers['sigma_eq_max_pa'] / 1e6:.4f} MPa "
          f"(the hottest CELL, half a cell above that face)")
    print(f"#   D_y at the free top face   {answers['displacement_y_top_m'] * 1e6:.4f} um")
    print(f"#   sigma_yy, sigma_xy         0 everywhere")
    print(f"#   conduction e-folding time  {answers['diffusion_e_folding_s']:.2f} s "
          f"against an endTime of {END_TIME_S:g} s")
    print("#")
    print("# to solve it, and then sample the profile the grader reads:")
    print("#   solidDisplacementFoam")
    print('#   postProcess -func "components(sigma)" -noZero')
    print("#   postProcess -func singleGraph -noZero")
    print("#   python3 benchmarks/thermal_stress/grade.py bar .")


if __name__ == "__main__":
    main()
