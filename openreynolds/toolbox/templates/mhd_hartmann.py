#!/usr/bin/env python3
"""TEMPLATE: Hartmann flow -- a conducting liquid in a duct across a magnetic field --
blockMesh -> a case `mhdFoam` can actually solve, against a profile with a closed form.

Copy this file into an empty case directory and run it there:

    python3 mhd_hartmann.py

It builds a 2D channel 2*HALF_HEIGHT_M across and twenty times that long, one cell thick
in z (the two z faces typed `empty`), with a uniform mesh whose cell height is chosen
against the Hartmann layer rather than graded into it, writes the whole `mhdFoam` case -- `0/U`,
`0/p`, `0/B`, `0/pB`, `constant/transportProperties`, `system/controlDict`,
`system/fvSchemes`, `system/fvSolution`, `system/sample` -- runs `blockMesh`, runs
`checkMesh`, and finishes with `mesh_look.py` so the mesh is looked at and not just
built. Then it prints the velocity profile the solved case has to reproduce, which is
known in closed form and derived below.

Like `free_surface.py`, `rotating.py` and `porous.py`, and unlike `duct2d.py` and
`body_in_box.py`, which write **stub** `controlDict`/`fvSchemes`/`fvSolution` just
sufficient for the mesh tools to build a `Foam::Time` and an `fvMesh`, everything here
is meant to be solved as it stands. It stops short of launching the solver, because what
a solve costs and whether it is wanted is not a template's call; the command it would be
is printed at the end.

**edit these two** and re-run: HARTMANN_NUMBER and HALF_HEIGHT_M. The applied field, the
channel length, the Hartmann layer thickness, **the time step** and the profile printed
at the end all follow from them.


WHY HARTMANN FLOW AND NOT A PRETTIER MHD CASE
---------------------------------------------
Because its answer is a formula and not a figure in a paper. `benchmarks/README.md`
argues at length that the reference value is the thing that is wrong twice as often as
the solver, and that of the eleven reference numbers it carries, re-checking two of them
against their actual sources changed them -- one by a factor of two. A closed form
derived from the solver's own equations has nothing to misquote. This is the same choice
`benchmarks/thermal_stress/` made and for the same reason.


THE DERIVATION, FROM mhdFoam's OWN EQUATIONS
--------------------------------------------
Taken from `applications/solvers/electromagnetics/mhdFoam/mhdFoam.C` in OpenFOAM v2512,
whose two equations are, with `DBU = 1/(2*mu*rho)` and `DB = 1/(mu*sigma)` out of
`createFields.H` of the same directory:

    ddt(U) + div(phi,U) - div(phiB, 2*DBU*B) - laplacian(nu,U) + grad(DBU*magSqr(B))
        == -grad(p)
    ddt(B) + div(phi,B) - div(phiB,U) - laplacian(DB,B) == 0

`p` there is kinematic (`0/p` carries `[0 2 -2 0 0 0 0]`, m2/s2), and `2*DBU = 1/(mu*rho)`
so the two magnetic terms in the momentum equation are the Lorentz force per unit mass,

    f = ( (B.grad)B - grad(|B|^2/2) ) / (rho*mu)

written as a flux divergence minus a magnetic pressure gradient. The second equation is
the induction equation with magnetic diffusivity `DB = 1/(mu*sigma)`. Note that `mu`
here is the magnetic **permeability** in H/m, not a dynamic viscosity: the dimension set
the solver demands for it is `[1 1 -2 0 0 -2 0]`, and `sigma` is the electrical
conductivity in S/m, `[-1 -3 3 0 0 2 0]`.

Now take flow along x between walls at y = -L and y = +L, fully developed so nothing
depends on x, with an applied field B0 across the channel along y. The velocity is
`u(y)*xhat` and the field is `b(y)*xhat + B0*yhat`: the flow drags the transverse field
into an x-component, which is where every term below comes from.

**Induction.** With both fields divergence free, `div(U B) - div(B U) = (U.grad)B -
(B.grad)U`, and `(U.grad)B = 0` because nothing depends on x while `(B.grad)U =
B0*du/dy`. So the x-component of the steady induction equation is

    -B0*du/dy - DB*d2b/dy2 = 0    =>    d2b/dy2 = -(B0/DB)*du/dy

Integrate once. `DB = 1/(mu*sigma)`, so

    db/dy = -mu*sigma*B0*u + C                                               (1)

**Momentum.** `(B.grad)B = B0*db/dy*xhat`, and `grad(|B|^2/2)` has no x-component since
`b` and `B0` depend on y alone. So the x-momentum equation is

    0 = -dp/dx + nu*d2u/dy2 + B0*(db/dy)/(rho*mu)

and substituting (1) for `db/dy` turns the Lorentz term into a term proportional to `u`:

    nu*d2u/dy2 - (sigma*B0^2/rho)*u = dp/dx - B0*C/(rho*mu)  =  constant

That is the whole of it: in a transverse field the Lorentz force on this flow is a
**linear drag**, and the equation is the Poiseuille equation with a `-k*u` added. Writing

    Ha = B0*L*sqrt(sigma/(rho*nu))                       the Hartmann number

the equation is `u'' - (Ha/L)^2 * u = const`, whose solution with `u(-L) = u(L) = 0` and
even symmetry is a constant minus a cosh. Normalised on the centreline speed `u_c`:

    u(y)/u_c = ( cosh(Ha) - cosh(Ha*y/L) ) / ( cosh(Ha) - 1 )                 (2)

**Two consequences worth grading, because each fails differently.**

The *shape* (2) contains Ha and nothing else -- not the pressure gradient, not the
viscosity, not the flow rate, not even L except through y/L. That is what makes it a
good check: a solved profile divided by its own centreline value has to lie on (2) with
no fitted constant anywhere.

The *centreline speed* is an absolute number, because integrating (2) across the channel
gives the mean, and the mean is fixed by the inlet:

    u_c / u_mean = ( cosh(Ha) - 1 ) / ( cosh(Ha) - sinh(Ha)/Ha )              (3)

At Ha = 0 that is 3/2 -- Poiseuille, as it must be, since (2) degenerates to `1 - (y/L)^2`
in the limit. At Ha = 20 it is 1.05263157, which for large Ha approaches `Ha/(Ha-1)`:
the profile is a flat core with two thin layers, and the core is moving barely faster
than the average. The flattening IS the physics, and (3) measures it as one number.

The two layers have thickness `L/Ha` (the `cosh` in (2) decays over that length), which
is what the mesh below is graded for.


WHAT IS IN THE CASE, AND THE THREE THINGS THAT GO QUIETLY WRONG
---------------------------------------------------------------
* **`mu` is a permeability, not a viscosity.** `constant/transportProperties` carries
  four scalars -- `rho`, `nu`, `mu`, `sigma` -- and `mu` is the magnetic permeability.
  A dynamic viscosity written there is read without complaint, because the solver takes
  the dimension set from its own `dimensionedScalar` construction and not from the file,
  and it enters the answer through `Ha` and through `DB`. The dictionary this writes
  says so on the line above the entry.

* **Both `B` and `pB` are fields that have to exist.** `pB` is a fictitious magnetic
  flux pressure that exists only to project `phiB` divergence free, as the solver's own
  `Description` block says; it has no physical boundary condition to get right, but a
  case without `0/pB` does not start. `magneticFieldErr.H` prints
  `magnetic flux divergence error = ...` every B-PISO pass, which is
  `deltaT*weightedAverage(mag(div(phiB)))`, and that line is the one to watch: a growing
  one is not a physics result, it is the projection failing.

* **The field's wall condition is `fixedValue`, not `zeroGradient`.** The applied field
  is imposed on the walls and allowed to develop in the interior, which is what makes
  the induced `b` antisymmetric about the centreline with its own two layers. On the
  inlet and outlet `B` is `zeroGradient`, so the field is not held flat across a
  boundary the flow has to develop through.

The transport numbers are all 1 and that is deliberate rather than lazy, so it is worth
stating why: with `rho = nu = mu = sigma = 1` the magnetic Prandtl number `Pm = nu/DB =
nu*mu*sigma` is 1, so the viscous and magnetic diffusion times across the channel are
equal and the transient settles in a couple of `L^2/nu`. A real liquid metal has
`Pm` of order 1e-5 -- six orders between the two timescales -- and the same graded
answer, because (2) and (3) contain `Ha` alone and no `Pm`. Choosing a metal's numbers
would therefore buy nothing and cost a transient nobody can afford. Ha maps onto a real
duct through `Ha = B0*L*sqrt(sigma/(rho*nu))` and nothing else.

`Re = u_mean*2L/nu` is 2 here for the same reason, which is also why `laminar` is not a
modelling assumption being smuggled in: there is no turbulence model in `mhdFoam` at
all, and at Re 2 none is wanted. It is also why the channel is only twenty half-heights
long and the profile is sampled at fifteen: the development length of a Re 2 channel is
a couple of half-heights, and a transverse field shortens it further.


WHAT A HARTMANN NUMBER COSTS, WHICH IS WHY THE DEFAULT IS 5 AND NOT 20
----------------------------------------------------------------------
This is the part that was learned by breaking it, so it is written down rather than left
as a tuned constant.

The ESI tutorial runs at Ha 20 on a **uniform** 40-cell mesh, which puts **one** cell
inside its `L/Ha = 0.05` layer. One cell cannot resolve the only part of the profile
that is not a parabola, so the first version of this template graded the mesh toward the
walls -- `simpleGrading` multi-graded, reciprocal expansion ratios of 10, 80 cells, a
0.00633 m wall cell and seven cells inside the layer. That is the right instinct for a
boundary layer and `mhdFoam` diverged on it, inside eighty time steps, with a SIGFPE in
`symGaussSeidelSmoother` on the vector equation. The measurement, from the `journey` job
on that run:

    Time = 0.07
    Courant Number mean: 0.4238678571 max: 23.59931818

A Courant number of 23.6 is impossible from the flow: the advective one on this mesh is
`u*dt/dx = 1.05*0.005/0.2 = 0.026` everywhere, and 23.6 needs a transverse velocity of
about 30 m/s. 30 m/s is the **Alfven speed**, `c_A = B0/sqrt(mu*rho) = 20 m/s`, which
says what was growing. The solver's own `Description` block names the mechanism without
being asked:

    However, in this formulation discretisation error prevents the normal stresses in UB
    from cancelling with those from BU, but it is unknown whether this is a serious
    error.

Both of those stresses are `fvc::` and not `fvm::` -- `fvc::div(phiB, 2.0*DBU*B)` and
`fvc::grad(DBU*magSqr(B))` in `mhdFoam.C`. Each is of order `B0^2/(mu*rho*dy)` on its
own and they are supposed to cancel; what survives is a spurious explicit force that
grows as the cell shrinks. Grading concentrates the smallest cells exactly where the
field gradient is largest, which is the worst place to put them for this scheme. So the
answer is not to grade and then fight it, but to keep the mesh uniform and choose Ha so
that the uniform cell resolves `L/Ha`.

**The stability bracket, measured rather than derived.** With `c_A*dt/dy` as the
magnetic Courant number, the two data points are:

    dy 0.05 (ESI tutorial, uniform)     c_A*dt/dy = 2.0    runs
    dy 0.00633 (graded, this template)  c_A*dt/dy = 15.8   diverges

so the limit is somewhere between 2 and 15.8 and nothing here knows where. `delta_t_s()`
asks for 0.5, which is inside the one stable datum by a factor of four, and prints what
it got so a future divergence can be compared against this bracket instead of guessed
at.

**And the cost.** Holding the cells-per-layer fixed means `dy` falls as `1/Ha`, while
`c_A` rises as `Ha`, so the step falls as `1/Ha^2` and the cell count rises as `Ha`:
**the total work goes as Ha^3**. Ha 5 with eight cells in the layer is 800 steps on
8,000 cells. Ha 20 at the same resolution is 12,800 steps on 32,000 cells, sixty-four
times as much, which is more than the whole `journey` job's budget. That is the reason
the graded points are Ha 5 and Ha 0 and not Ha 20, and it is a property of this
solver's explicit magnetic coupling rather than of Hartmann flow.


GRADED BY
---------
    python3 benchmarks/mhd/grade.py hartmann <case>         --hartmann 5 --mean-velocity 1 --half-height 1

which reads the `postProcessing/` profile this case's `system/sample` writes and nothing
else. `benchmarks/mhd/README.md` carries the same derivation and what a real run
measured against it.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

# -- edit these two ------------------------------------------------------------
#
# HARTMANN_NUMBER is the whole physics. 0 is plane Poiseuille, worth running as the
# control that says the field is doing something at all. 5 is the shipped value, and the
# reason it is not the ESI tutorial's 20 is measured rather than preferred -- see
# WHAT A HARTMANN NUMBER COSTS in the docstring above. CELLS_ACROSS and the time step
# follow from it, and the printed summary says how many cells landed inside the layer
# and what the magnetic Courant number came out at.

HARTMANN_NUMBER = 5.0
HALF_HEIGHT_M = 1.0

# -- everything below follows ---------------------------------------------------

RHO_KG_M3 = 1.0          # density
NU_M2_S = 1.0            # kinematic viscosity
MU_H_M = 1.0             # magnetic PERMEABILITY, H/m -- not a dynamic viscosity
SIGMA_S_M = 1.0          # electrical conductivity, S/m

MEAN_VELOCITY_M_S = 1.0  # the inlet is a plug at this speed, so the mean is exactly it

LENGTH_RATIO = 20.0      # channel length in half-heights
SAMPLE_AT_RATIO = 15.0   # where the profile is sampled, in half-heights

CELLS_ALONG = 100        # cells along the channel
CELLS_ACROSS = 80        # cells across it, uniform: see the docstring for why not graded

CELLS_PER_LAYER = 4      # below this many cells inside L/Ha the summary warns

SAMPLE_POINTS = 401      # points in the sampled profile
END_TIME_S = 2.0         # two viscous times across the half-height, at nu = 1
WRITE_INTERVAL_S = 0.5   # four written times, so steadiness is checkable from the files

# The two Courant numbers the time step is taken from. Both are explicit limits on
# EXPLICIT terms, which is the whole reason this template computes its step instead of
# carrying one: mhdFoam differences the magnetic stresses explicitly.
MAGNETIC_CFL = 0.5       # c_A*dt/dy; the measured bracket is in the docstring
ADVECTIVE_CFL = 0.5      # u*dt/dx, the ordinary one
MAX_DELTA_T_S = 0.005    # the ESI tutorial's step, which is the largest ever used here

LENGTH_M = LENGTH_RATIO * HALF_HEIGHT_M
CELL_HEIGHT_M = 2.0 * HALF_HEIGHT_M / CELLS_ACROSS
CELL_LENGTH_M = LENGTH_M / CELLS_ALONG
THICK_M = CELL_HEIGHT_M        # z depth = one cell, about a cell wide


def applied_field_t() -> float:
    """B0 from the Hartmann number, which is the direction the knob turns.

    Ha = B0*L*sqrt(sigma/(rho*nu)), so B0 = Ha/L * sqrt(rho*nu/sigma). With the unit
    properties above that is simply Ha/L, which is why the ESI tutorial's `(0 20 0)` and
    its half-height of 1 are a Hartmann number of 20.
    """
    return (HARTMANN_NUMBER / HALF_HEIGHT_M) * math.sqrt(
        RHO_KG_M3 * NU_M2_S / SIGMA_S_M)


def profile(eta: float) -> float:
    """u/u_c at y/L = eta, equation (2) of the docstring. Exact.

    The Ha -> 0 limit is written out rather than left to cancel: (2) is 0/0 there, and
    `cosh(0) - cosh(0)` over `cosh(0) - 1` in floating point is a division by zero
    rather than the parabola it tends to.
    """
    if HARTMANN_NUMBER == 0.0:
        return 1.0 - eta * eta
    ha = HARTMANN_NUMBER
    return (math.cosh(ha) - math.cosh(ha * eta)) / (math.cosh(ha) - 1.0)


def centreline_over_mean() -> float:
    """u_c/u_mean, equation (3). 3/2 at Ha = 0, tending to Ha/(Ha-1) for large Ha."""
    if HARTMANN_NUMBER == 0.0:
        return 1.5
    ha = HARTMANN_NUMBER
    return (math.cosh(ha) - 1.0) / (math.cosh(ha) - math.sinh(ha) / ha)


def layer_thickness_m() -> float:
    """L/Ha, the thickness the cosh in (2) decays over. Infinite at Ha = 0."""
    if HARTMANN_NUMBER == 0.0:
        return math.inf
    return HALF_HEIGHT_M / HARTMANN_NUMBER


def alfven_speed_m_s() -> float:
    """c_A = B0/sqrt(mu*rho), the speed the explicit magnetic terms carry a wave at.

    This is the number that sets the time step, and it is worth naming rather than
    leaving inside an expression: the Lorentz terms in `mhdFoam`'s momentum equation and
    the stretching term in its induction equation are both `fvc::` and not `fvm::`, so
    the Alfven wave is advanced explicitly and has an explicit wave's stability limit.
    """
    return applied_field_t() / math.sqrt(MU_H_M * RHO_KG_M3)


def cells_in_layer() -> float:
    """How many cells land inside one Hartmann layer, on a uniform mesh.

    The one mesh fact that decides whether this case can be graded at all: the layer is
    `L/Ha` thick and the profile's whole departure from a parabola lives in it.
    """
    if HARTMANN_NUMBER == 0.0:
        return float(CELLS_ACROSS) / 2.0
    return layer_thickness_m() / CELL_HEIGHT_M


def delta_t_s() -> float:
    """The time step, from the mesh rather than from a constant.

    Three limits, and the binding one is reported by `summary()`:

    * the magnetic one, `c_A*dt/dy <= MAGNETIC_CFL`. See the docstring's measured
      bracket: this case diverged at `c_A*dt/dy = 15.8` and the ESI tutorial runs at
      2.0, so the limit is somewhere between and 0.5 is inside the stable datum by four.
    * the advective one, `u*dt/dx <= ADVECTIVE_CFL`, on the peak velocity the closed
      form predicts rather than on the mean, since the core runs faster than the mean.
    * `MAX_DELTA_T_S`, the ESI tutorial's own step, as a ceiling. At Ha 0 there is no
      Alfven wave at all and this is the only thing left holding the step down.
    """
    limits = [MAX_DELTA_T_S,
              ADVECTIVE_CFL * CELL_LENGTH_M
              / (MEAN_VELOCITY_M_S * centreline_over_mean())]
    if alfven_speed_m_s() > 0.0:
        limits.append(MAGNETIC_CFL * CELL_HEIGHT_M / alfven_speed_m_s())
    return min(limits)


def magnetic_courant() -> float:
    """`c_A*dt/dy` as actually written, which is the number to compare against the
    bracket in the docstring if a run of this ever diverges again."""
    return alfven_speed_m_s() * delta_t_s() / CELL_HEIGHT_M


# -- the case files ------------------------------------------------------------


def header(cls: str, obj: str) -> str:
    """A `FoamFile` header. The decorative C++ banner is optional; a half-copied one is
    not harmless, so none is written -- see `../ENVIRONMENT.md`."""
    return ("FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            f"    class       {cls};\n"
            f"    object      {obj};\n"
            "}\n")


def block_mesh_dict() -> str:
    """A rectangle, one cell thick, graded toward both walls.

    `blockMesh` rather than gmsh, which is what every other template here uses: the
    domain is a box, so gmsh would contribute a recombination and a non-orthogonality to
    argue about and nothing else. `checkMesh` on what this writes is exactly orthogonal.

    **Uniform, and not graded, which is a measured decision and not the obvious one.**
    Grading toward the walls is the natural way to resolve a boundary layer, and the
    first version of this template did exactly that -- `simpleGrading` multi-graded with
    reciprocal expansion ratios of 10, which put seven cells inside the Ha 20 layer and
    a 0.00633 m cell against the wall. `mhdFoam` diverged on it. The docstring's
    WHAT A HARTMANN NUMBER COSTS section has the evidence; the short version is that the
    magnetic stresses are differenced explicitly, their spurious residual grows as the
    cell shrinks, and a graded mesh concentrates the finest cells exactly where the
    field gradient is largest. A uniform mesh makes the worst cell the only cell, and
    the Hartmann number is then chosen so that `L/Ha` is resolved by it.

    The walls are typed `wall` and not `patch`. `mhdFoam` is laminar and has no wall
    functions to care, but `checkMesh` reports wall patches separately and `noSlip`
    reads as what it is.
    """
    half, length, thick = HALF_HEIGHT_M, LENGTH_M, THICK_M
    return (
        header("dictionary", "blockMeshDict") + "\n"
        "scale   1;\n\n"
        "vertices\n"
        "(\n"
        f"    (0 {-half:g} 0)\n"
        f"    ({length:g} {-half:g} 0)\n"
        f"    ({length:g} {half:g} 0)\n"
        f"    (0 {half:g} 0)\n"
        f"    (0 {-half:g} {thick:g})\n"
        f"    ({length:g} {-half:g} {thick:g})\n"
        f"    ({length:g} {half:g} {thick:g})\n"
        f"    (0 {half:g} {thick:g})\n"
        ");\n\n"
        "blocks\n"
        "(\n"
        f"    hex (0 1 2 3 4 5 6 7) ({CELLS_ALONG} {CELLS_ACROSS} 1)\n"
        "    simpleGrading (1 1 1)\n"
        ");\n\n"
        "boundary\n"
        "(\n"
        "    inlet\n"
        "    {\n"
        "        type patch;\n"
        "        faces ( (0 4 7 3) );\n"
        "    }\n"
        "    outlet\n"
        "    {\n"
        "        type patch;\n"
        "        faces ( (2 6 5 1) );\n"
        "    }\n"
        "    lowerWall\n"
        "    {\n"
        "        type wall;\n"
        "        faces ( (1 5 4 0) );\n"
        "    }\n"
        "    upperWall\n"
        "    {\n"
        "        type wall;\n"
        "        faces ( (3 7 6 2) );\n"
        "    }\n"
        "    frontAndBack\n"
        "    {\n"
        "        type empty;\n"
        "        faces ( (0 3 2 1) (4 5 6 7) );\n"
        "    }\n"
        ");\n")


def u_field() -> str:
    """A plug at the inlet, which is what makes equation (3) an absolute check.

    `fixedValue uniform (U 0 0)` fixes the volumetric flow, and the mean velocity at
    every station downstream is then exactly that number by continuity -- in a 2D case
    with both z faces `empty` there is nowhere else for it to go. So the measured
    centreline speed can be compared against `u_mean * (3)` with no constant fitted
    anywhere, which a parabolic or already-Hartmann inlet would have given away.
    """
    return (header("volVectorField", "U") + "\n"
            "dimensions      [0 1 -1 0 0 0 0];\n\n"
            f"internalField   uniform ({MEAN_VELOCITY_M_S:g} 0 0);\n\n"
            "boundaryField\n"
            "{\n"
            "    inlet\n"
            "    {\n"
            "        type            fixedValue;\n"
            f"        value           uniform ({MEAN_VELOCITY_M_S:g} 0 0);\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    lowerWall\n"
            "    {\n"
            "        type            noSlip;\n"
            "    }\n"
            "    upperWall\n"
            "    {\n"
            "        type            noSlip;\n"
            "    }\n"
            "    frontAndBack\n"
            "    {\n"
            "        type            empty;\n"
            "    }\n"
            "}\n")


def p_field() -> str:
    """Kinematic pressure, m2/s2, pinned at the outlet.

    `mhdFoam` is incompressible and solves p/rho, so the `dimensions` line is
    `[0 2 -2 0 0 0 0]` and a drop read off this field is larger than the pascal figure
    by exactly rho. With rho = 1 here the two coincide, which is a coincidence of this
    case and not a property of the solver.
    """
    return (header("volScalarField", "p") + "\n"
            "dimensions      [0 2 -2 0 0 0 0];\n\n"
            "internalField   uniform 0;\n\n"
            "boundaryField\n"
            "{\n"
            "    inlet\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type            fixedValue;\n"
            "        value           uniform 0;\n"
            "    }\n"
            "    lowerWall\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    upperWall\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    frontAndBack\n"
            "    {\n"
            "        type            empty;\n"
            "    }\n"
            "}\n")


def b_field() -> str:
    """The applied field, across the channel, held on the walls.

    `fixedValue` on the two walls and `zeroGradient` at the inlet and outlet. The
    interior is free to develop the induced x-component the flow drags out of it, which
    is the `b(y)` of the derivation and the thing that produces the linear drag. Holding
    `B` fixed everywhere would remove the induction equation from the problem and leave
    a Poiseuille flow with a body force, which has a different profile.
    """
    b0 = applied_field_t()
    return (header("volVectorField", "B") + "\n"
            "dimensions      [1 0 -2 0 0 -1 0];\n\n"
            f"internalField   uniform (0 {b0:g} 0);\n\n"
            "boundaryField\n"
            "{\n"
            "    inlet\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    lowerWall\n"
            "    {\n"
            "        type            fixedValue;\n"
            f"        value           uniform (0 {b0:g} 0);\n"
            "    }\n"
            "    upperWall\n"
            "    {\n"
            "        type            fixedValue;\n"
            f"        value           uniform (0 {b0:g} 0);\n"
            "    }\n"
            "    frontAndBack\n"
            "    {\n"
            "        type            empty;\n"
            "    }\n"
            "}\n")


def pb_field() -> str:
    """The fictitious magnetic flux pressure.

    Quoting the solver's own `Description` block: "A fictitous magnetic flux pressure pH
    is introduced in order to compensate for discretisation errors and create a magnetic
    face flux field which is divergence free as required by Maxwell's equations." It is
    not a physical field and there is no physical condition to get right; what matters
    is that the case has one at all, since `createFields.H` reads it `MUST_READ`.
    """
    return (header("volScalarField", "pB") + "\n"
            "dimensions      [1 1 -3 0 0 -1 0];\n\n"
            "internalField   uniform 0;\n\n"
            "boundaryField\n"
            "{\n"
            "    inlet\n"
            "    {\n"
            "        type            fixedValue;\n"
            "        value           uniform 0;\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type            fixedValue;\n"
            "        value           uniform 0;\n"
            "    }\n"
            "    lowerWall\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    upperWall\n"
            "    {\n"
            "        type            zeroGradient;\n"
            "    }\n"
            "    frontAndBack\n"
            "    {\n"
            "        type            empty;\n"
            "    }\n"
            "}\n")


def transport_properties() -> str:
    """Four scalars, one of which is not what its name suggests.

    `mu` is the magnetic **permeability** in H/m. `createFields.H` constructs it as
    `dimensionedScalar("mu", dimensionSet(1, 1, -2, 0, 0, -2, 0), transportProperties)`,
    which is kg.m/(s2.A2), and a dynamic viscosity written here is accepted silently
    because the dimensions come from that line and not from this file. The kinematic
    viscosity is `nu`; there is no dynamic viscosity in this dictionary at all.
    """
    return (header("dictionary", "transportProperties") + "\n"
            f"rho             {RHO_KG_M3:g};\n\n"
            f"nu              {NU_M2_S:g};\n\n"
            "// magnetic PERMEABILITY [H/m], not a dynamic viscosity:\n"
            "// dimensionSet(1, 1, -2, 0, 0, -2, 0)\n"
            f"mu              {MU_H_M:g};\n\n"
            "// electrical conductivity [S/m]\n"
            f"sigma           {SIGMA_S_M:g};\n")


def control_dict() -> str:
    """Two viscous times, written four times over so steadiness is a file comparison.

    `L^2/nu` is 1 s here, and the Hartmann damping time `rho/(sigma*B0^2)` is 1/25 s at
    Ha 5, so the magnetic part settles long before the viscous part does and the end
    time is set by the latter. Writing every 0.5 s rather than only at the end is what
    lets the last two profiles be sampled and differenced: a transient that has not
    finished looks exactly like a converged answer in a single output.

    `writeControl runTime` rather than `timeStep`, so the written times stay at 0.5
    intervals whatever `delta_t_s()` comes out at -- and it is computed rather than
    carried, because it has to follow the mesh and the field. See `delta_t_s()`.
    """
    return (header("dictionary", "controlDict") + "\n"
            "application     mhdFoam;\n\n"
            "startFrom       latestTime;\n\n"
            "startTime       0;\n\n"
            "stopAt          endTime;\n\n"
            f"endTime         {END_TIME_S:g};\n\n"
            f"deltaT          {delta_t_s():.6g};\n\n"
            "writeControl    runTime;\n\n"
            f"writeInterval   {WRITE_INTERVAL_S:g};\n\n"
            "purgeWrite      0;\n\n"
            "writeFormat     ascii;\n\n"
            "writePrecision  10;\n\n"
            "writeCompression off;\n\n"
            "timeFormat      general;\n\n"
            "timePrecision   6;\n\n"
            "runTimeModifiable true;\n")


def fv_schemes() -> str:
    """Second order everywhere, and every `div` named.

    `divSchemes` has `default none`, so each of the four divergence terms has to be
    named exactly as the solver constructs it -- including
    `div(phiB,((2*DBU)*B))`, whose name comes from `DBU.name() = "DBU"` in
    `createFields.H` and the `2.0*DBU*B` written in the momentum equation. A missing
    entry is a fatal error at the first time step rather than a silent fallback, which
    is the one good thing about `default none`.

    `Gauss linear` is central differencing and is second order and unbounded. At Re 2 it
    is also unconditionally the right choice: the cell Peclet number is tiny, there is
    nothing to stabilise, and an upwind scheme here would add a numerical viscosity that
    changes the answer being measured.
    """
    return (header("dictionary", "fvSchemes") + "\n"
            "ddtSchemes\n"
            "{\n"
            "    default         Euler;\n"
            "}\n\n"
            "gradSchemes\n"
            "{\n"
            "    default         Gauss linear;\n"
            "}\n\n"
            "divSchemes\n"
            "{\n"
            "    default         none;\n"
            "    div(phi,U)      Gauss linear;\n"
            "    div(phiB,U)     Gauss linear;\n"
            "    div(phi,B)      Gauss linear;\n"
            "    div(phiB,((2*DBU)*B)) Gauss linear;\n"
            "}\n\n"
            "laplacianSchemes\n"
            "{\n"
            "    default         Gauss linear corrected;\n"
            "}\n\n"
            "interpolationSchemes\n"
            "{\n"
            "    default         linear;\n"
            "}\n\n"
            "snGradSchemes\n"
            "{\n"
            "    default         corrected;\n"
            "}\n")


def fv_solution() -> str:
    """PISO for the flow and BPISO for the field, which is a second control dictionary.

    `mhdFoam` runs two PISO loops per time step and reads their corrector counts from
    two separate sub-dictionaries, `PISO` and `BPISO`. A case with only `PISO` in it
    fails on the missing `BPISO`, and `nCorrectors` in the wrong one is read as a
    default. There is no `pRefCell` here because the outlet pins `p`.

    Every tolerance here is the ESI tutorial's at
    `$FOAM_TUTORIALS/electromagnetics/mhdFoam/hartmann/system/fvSolution`, including
    `smoothSolver`/`symGaussSeidel` for `"(U|B)"`. An earlier version of this file
    tightened all three by two to three orders on the reasoning that tighter is safer.
    It is not: on a diverging run the only thing it bought was 165 Gauss-Seidel sweeps
    per solve instead of a few dozen, and the solver's own tutorial is the reference for
    what it expects.
    """
    return (header("dictionary", "fvSolution") + "\n"
            "solvers\n"
            "{\n"
            "    p\n"
            "    {\n"
            "        solver          PCG;\n"
            "        preconditioner  DIC;\n"
            "        tolerance       1e-06;\n"
            "        relTol          0.05;\n"
            "    }\n\n"
            "    pFinal\n"
            "    {\n"
            "        $p;\n"
            "        relTol          0;\n"
            "    }\n\n"
            "    pB\n"
            "    {\n"
            "        $p;\n"
            "        tolerance       1e-05;\n"
            "        relTol          0;\n"
            "    }\n\n"
            "    pBFinal\n"
            "    {\n"
            "        $pB;\n"
            "        relTol          0;\n"
            "    }\n\n"
            "    \"(U|B)\"\n"
            "    {\n"
            "        solver          smoothSolver;\n"
            "        smoother        symGaussSeidel;\n"
            "        tolerance       1e-05;\n"
            "        relTol          0;\n"
            "    }\n"
            "}\n\n"
            "PISO\n"
            "{\n"
            "    nCorrectors     3;\n"
            "    nNonOrthogonalCorrectors 0;\n"
            "}\n\n"
            "BPISO\n"
            "{\n"
            "    nCorrectors     3;\n"
            "}\n")


def sample_dict() -> str:
    """The profile across the channel, at fifteen half-heights, as `raw` columns.

    `axis y` puts y in the first column, which is what the grader reads the channel
    height off: a case of another size, or one sampled along another axis, is then
    graded the same way without being told its geometry.

    The line stops a thousandth of a half-height short of each wall. A `uniform` set
    silently drops sample points that land outside the mesh, and a point exactly on a
    boundary face is on that edge; losing the two end points would not break anything
    here, but a profile whose extent depends on a floating-point comparison is not worth
    the argument. `x` is offset by a thousandth of a cell for the same reason -- a line
    lying exactly on a face plane samples whichever cell wins a tie.
    """
    half = HALF_HEIGHT_M
    x = SAMPLE_AT_RATIO * HALF_HEIGHT_M + 0.001 * LENGTH_M / CELLS_ALONG
    z = 0.5 * THICK_M
    return (header("dictionary", "sample") + "\n"
            "type            sets;\n"
            "libs            (sampling);\n\n"
            "interpolationScheme cellPoint;\n\n"
            "setFormat       raw;\n\n"
            "fields          ( U );\n\n"
            "sets\n"
            "{\n"
            "    hartmannProfile\n"
            "    {\n"
            "        type    uniform;\n"
            "        axis    y;\n"
            f"        start   ({x:.6g} {-0.999 * half:.6g} {z:.6g});\n"
            f"        end     ({x:.6g} {0.999 * half:.6g} {z:.6g});\n"
            f"        nPoints {SAMPLE_POINTS};\n"
            "    }\n"
            "}\n")


CASE_FILES = {
    "0/U": u_field,
    "0/p": p_field,
    "0/B": b_field,
    "0/pB": pb_field,
    "constant/transportProperties": transport_properties,
    "system/blockMeshDict": block_mesh_dict,
    "system/controlDict": control_dict,
    "system/fvSchemes": fv_schemes,
    "system/fvSolution": fv_solution,
    "system/sample": sample_dict,
}
"""Every file `mhdFoam` reads, and nothing it does not.

There is no `constant/turbulenceProperties`: `mhdFoam` has no turbulence model to
select, so the file other incompressible solvers demand is not read here and writing one
would suggest a choice was made.
"""


def write_case(case: Path) -> None:
    """The whole case on disk before the mesh exists.

    `blockMesh` builds a `Foam::Time` before it reads its own dictionary and fails
    "cannot find file .../system/controlDict" in an otherwise empty directory;
    `checkMesh` goes further and builds an `fvMesh`, so it reads `fvSchemes` too.
    Writing the real dictionaries rather than stubs means the mesh tools get what they
    need as a side effect of the case being complete.
    """
    for rel, source in CASE_FILES.items():
        path = case / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source(), encoding="utf-8")


# -- running -------------------------------------------------------------------


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


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


def summary() -> str:
    """The closed form, the mesh against the layer it has to resolve, and the profile.

    Printed rather than only written, because the person who compares a solved
    centreline speed against a formula is not necessarily the person who read the
    docstring that derived it.
    """
    lines = [
        "Hartmann flow, and what the solved case has to reproduce:",
        f"    Ha             {HARTMANN_NUMBER:g}"
        f"   = B0*L*sqrt(sigma/(rho*nu))",
        f"    B0             {applied_field_t():.6g}      across the channel, along +y",
        f"    L              {HALF_HEIGHT_M:g} m        half-height",
        f"    u_mean         {MEAN_VELOCITY_M_S:g} m/s      fixed by the inlet plug",
        f"    Re             {MEAN_VELOCITY_M_S * 2 * HALF_HEIGHT_M / NU_M2_S:g}"
        f"        = u_mean*2L/nu, so laminar is a fact and not a model",
        f"    Pm             {NU_M2_S * MU_H_M * SIGMA_S_M:g}"
        f"        = nu*mu*sigma; equal diffusion times, which is why it is 1 here",
        "",
        "the two numbers to measure, both exact:",
        f"    u_c/u_mean     {centreline_over_mean():.8f}"
        f"   = (cosh Ha - 1)/(cosh Ha - sinh Ha/Ha)",
        f"    u_c            {centreline_over_mean() * MEAN_VELOCITY_M_S:.8f} m/s"
        f"   absolute, no constant fitted",
        "    u(y)/u_c       (cosh Ha - cosh(Ha y/L))/(cosh Ha - 1)",
        "    Poiseuille     1.50000000     u_c/u_mean at Ha = 0; the field has pulled",
        f"                                  the core "
        f"{100.0 * (1.0 - centreline_over_mean() / 1.5):.1f}% closer to the mean",
        "",
        "the mesh against the layer it has to resolve, and the step against the wave:",
        f"    L/Ha           {layer_thickness_m():.6g} m   Hartmann layer thickness",
        f"    cell height    {CELL_HEIGHT_M:.6g} m   uniform; the docstring says why",
        f"    cells in it    {cells_in_layer():.1f} of {CELLS_ACROSS // 2} across the "
        f"half-channel",
        f"    cells          {CELLS_ALONG * CELLS_ACROSS:,} in the 2D mesh",
        f"    c_A            {alfven_speed_m_s():.6g} m/s   = B0/sqrt(mu*rho), the "
        f"Alfven speed",
        f"    deltaT         {delta_t_s():.6g} s   computed, not carried",
        f"    c_A*dt/dy      {magnetic_courant():.4f}      "
        f"(this case diverged at 15.8; the ESI tutorial runs at 2.0)",
        f"    steps          {int(round(END_TIME_S / delta_t_s())):,} to "
        f"{END_TIME_S:g} s",
        "",
        "the profile, at eleven points (the case samples "
        f"{SAMPLE_POINTS}):",
        "    y/L        u/u_c",
    ]
    for index in range(11):
        eta = -1.0 + index * 0.2
        lines.append(f"    {eta:+.2f}      {profile(eta):.6f}")
    if cells_in_layer() < CELLS_PER_LAYER:
        lines += [
            "",
            f"WARNING: {cells_in_layer():.1f} cells inside the Hartmann layer, below the",
            f"{CELLS_PER_LAYER} this asks for. The layer is L/Ha thick, so raising Ha "
            "thins it and",
            "CELLS_ACROSS has to go up with it. Note what that costs: the step is capped",
            "by c_A*dt/dy, c_A rises with Ha and dy falls with it, so the step goes as",
            "1/Ha^2 and the total work as Ha^3. Ha 20 at this resolution is sixty-four",
            "times Ha 5.",
        ]
    return "\n".join(lines)


def main() -> None:
    case = Path.cwd()
    write_case(case)

    run(["blockMesh", "-case", "."])
    run(["checkMesh", "-case", "."])

    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png",
         "--json", "look.json"])

    print()
    print(summary())

    print()
    if shutil.which("mhdFoam"):
        print("mhdFoam is on PATH.")
    else:
        print("mhdFoam is NOT on PATH, so the case above cannot be solved as written.")
        print("it is a standard OpenFOAM solver, in applications/solvers/"
              "electromagnetics; to see what this image has:")
        print("    ls $FOAM_APPBIN | grep -i -e mhd -e magnetic -e electro")
        print("    ls $FOAM_TUTORIALS/electromagnetics")

    # Not launched from here: what a solve costs, and whether it is wanted at all, is
    # not a template's call. `startFrom latestTime` is already set, so this loop resumes
    # itself through any in-sandbox failure -- see "Long solves" in
    # ../notes/openfoam-field-notes.md.
    print("\nthe case is complete and meant to be solved. one way to launch it, and the"
          "\nsampling and the grading that follow:\n"
          "    until mhdFoam >> log.mhdFoam 2>&1; do echo resuming >> log.mhdFoam; "
          "sleep 5; done\n"
          "    postProcess -func sample -noZero\n"
          f"    python3 benchmarks/mhd/grade.py hartmann . "
          f"--hartmann {HARTMANN_NUMBER:g} --mean-velocity {MEAN_VELOCITY_M_S:g}\n"
          "watch the `magnetic flux divergence error` line mhdFoam prints: a growing"
          "\none is the pB projection failing and not a physics result.")


if __name__ == "__main__":
    main()
