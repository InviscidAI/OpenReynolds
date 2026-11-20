#!/usr/bin/env python3
"""TEMPLATE: a Darcy-Forchheimer porous slab across a duct -- a filter, radiator,
screen or packed bed -- gmsh -> gmshToFoam -> a case `porousSimpleFoam` can actually
solve, with `d` and `f` DERIVED from something measurable rather than demanded.

Copy this file into an empty case directory and run it there:

    python3 porous.py

It builds a 2D duct HEIGHT_M tall with a porous slab SLAB_THICKNESS_M thick across it,
one cell thick in z (the 2D recipe: quads extruded one layer, the two z faces typed
`empty`), cuts the fluid into two cell zones -- `porousZone` through the slab and
`fluid` either side of it -- meshes it, converts with `gmshToFoam`, retypes the duct
walls `wall` and the z faces `empty`, writes the whole `porousSimpleFoam` case including
`constant/porosityProperties`, runs `checkMesh`, and finishes with `mesh_look.py` so the
result is looked at and not just built. Edit SLAB_THICKNESS_M and PRESSURE_DROP_CURVE
and re-run; `d`, `f` and every derived turbulence number follow from them.

Like `free_surface.py` and `rotating.py`, and unlike `duct2d.py` and `body_in_box.py`,
which write **stub** `controlDict`/`fvSchemes`/`fvSolution` just sufficient for
`gmshToFoam` and `checkMesh` to build a `Foam::Time` and an `fvMesh`, everything written
here is meant to be solved as it stands. It stops short of launching the solver, because
what a solve costs and whether it is wanted is not a template's call; the command it
would be is printed at the end.

WHERE d AND f COME FROM, WHICH IS THE WHOLE PROBLEM
---------------------------------------------------
A porous zone in OpenFOAM is two vectors, `d` and `f`, and nobody knows them. They are
not on a datasheet, they are not a material property anybody quotes, and a template that
asks for them has handed the user back their own question. So this one does not ask: it
takes what a datasheet or a test rig *does* give -- a pressure drop at a known face
velocity, or a curve of them -- and reduces it.

OpenFOAM's Darcy-Forchheimer momentum sink, quoted from the `Description` block of
`src/finiteVolume/cfdTools/general/porosityModel/DarcyForchheimer/DarcyForchheimer.H`:

    S = - (mu*d + (rho*|U|/2)*f) * U

    d | Darcy coefficient [1/m2]
    f | Forchheimer coefficient [1/m]

The factor of a half on `f` is in the code as well as in that comment, and it is the
single easiest thing here to get wrong -- it is a factor of two in the inertial term,
which is the term that dominates everything but the slowest flows. In
`DarcyForchheimer.C`, `calcTransformModelData` builds the tensor as

    // The Forchheimer coefficient as a tensor
    // - the leading 0.5 is from 1/2*rho
    tensor forchCoeff(Zero);
    forchCoeff.xx() = 0.5*fXYZ_.value().x();

while the Darcy tensor takes no factor at all (`darcyCoeff.xx() = dXYZ_.value().x();`).
The declared dimensions are `dXYZ_("d", dimless/sqr(dimLength), coeffs_)` and
`fXYZ_("f", dimless/dimLength, coeffs_)`, which is the [1/m2] and [1/m] above.

**The reduction.** In steady flow straight through a slab of thickness L at a uniform
face velocity U, the pressure gradient balances that sink, so the drop across the slab is

    dp = (mu*d*U + 0.5*rho*f*U^2) * L          [Pa]

Divide both sides by L*U and the right-hand side becomes a straight line in U:

    dp/(L*U) = mu*d + 0.5*rho*f*U

That is the Forchheimer plot, and it is why a *curve* is worth more than a point: fit
`dp/(L*U)` against `U` and the intercept is `mu*d` and the slope is `0.5*rho*f`, so

    d = intercept / mu                 f = 2 * slope / rho

Two numbers out of a straight-line fit, with `mu = rho*nu`. `coefficients_from_curve()`
below is those four lines, using `statistics.linear_regression` from the standard
library, and `main` prints the fit, both coefficients, and then the drop the
coefficients predict back at every measured point beside the measurement itself. A
residual a reader can see is worth more than a derivation they have to trust.

**One point instead of a curve.** Two unknowns and one measurement do not determine a
split between the viscous and the inertial term, and no arithmetic can invent one. With
a single point this prints both limits -- all-Darcy (`f` = 0) and all-Forchheimer
(`d` = 0) -- so the size of the choice is visible, and writes the all-Darcy pair into
the case with a comment in `constant/porosityProperties` saying that is what happened.
The two limits differ by however much they differ; at the default slab's 1 m/s point
they are a factor of about 4 in the predicted drop at 2 m/s.

**No measurement at all: a packed bed.** For a bed of roughly spherical particles the
Ergun equation gives the drop from geometry that can be measured with a ruler and a
bucket -- particle diameter and void fraction:

    dp/L = 150*mu*(1-e)^2/(e^3*Dp^2) * U  +  1.75*rho*(1-e)/(e^3*Dp) * U^2

Term by term against the OpenFOAM form above, `mu*d` against the first coefficient and
`0.5*rho*f` against the second:

    d = 150*(1-e)^2 / (e^3*Dp^2)            f = 2*1.75*(1-e)/(e^3*Dp) = 3.5*(1-e)/(e^3*Dp)

`coefficients_from_ergun()` is those two lines. The factor of 2 on Ergun's 1.75 is the
same half as before, running the other way, and dropping it is a factor-of-two error in
the term that dominates a packed bed.

WHERE THE DEFAULT CURVE COMES FROM
----------------------------------
PRESSURE_DROP_CURVE below is **not a measurement**, and it is not invented either: it is
Ergun's prediction for a bed of 3 mm spheres at a void fraction of 0.38, in air at
20 degrees C, through the default 50 mm slab, rounded to six figures. It is there so the
shipped case carries numbers that trace to arithmetic somebody can redo, and so the two
routes in this file check each other: fitting those four points recovers Ergun's own `d`
and `f` to within a rounding error. Replace it with your own (face velocity, pressure
drop) pairs, from a datasheet curve or a rig, and the case follows.

THE QUIET FAILURES, BAKED IN BELOW
----------------------------------
* **The case's pressure is NOT in pascals.** `porousSimpleFoam` is incompressible and
  solves p/rho in m2/s2, so the drop this case reports is the measured drop divided by
  rho -- a factor of 1.204 for air, 998.2 for water. A 503 Pa measurement compared
  against a solved field reading 418 and called a 17% error is the mistake; they agree
  exactly. The printed summary gives the expected drop in both units, and `0/p` carries
  its unit in a comment where the dimensions are. `d` and `f` themselves are unaffected:
  divide `dp = (mu*d*U + 0.5*rho*f*U^2)*L` through by rho and both coefficients survive
  untouched, which is why the same pair works for either formulation.
* **`d` and `f` are vectors, and the two components you are not thinking about matter
  more than the one you are.** A filter resists flow along its thickness; a filter that
  resists flow *across* its face only as much as along it will let the flow turn and
  sneak sideways through the slab rather than passing through it, and the answer is a
  plausible, converged, wrong pressure drop. OpenFOAM's convention here is unusual and
  worth knowing: **a negative component is a multiplier of the largest component**, not
  a negative resistance. `porosityModel.C`'s `adjustNegativeResistance` applies it, and
  it is fatal only if every component is negative. The shipped
  `$FOAM_TUTORIALS/incompressible/porousSimpleFoam/straightDuctImplicit` writes
  `d (5e7 -1000 -1000)`, which means the two cross-flow directions resist a thousand
  times harder than the through-flow one. This file writes the same `-1000 -1000` and
  says so where it writes it.
* **The porous zone has to exist as a `cellZone` in the mesh.** A physical *volume* in
  the msh becomes a cellZone of the same name under `gmshToFoam`; a physical *surface*
  becomes a patch. A `porosityProperties` naming a zone the mesh does not have fails at
  the first solver step. `mesh_look.py` lists the zones a mesh actually has.
* **`nUCorrectors 2` in the SIMPLE block.** The implicit porous source puts a large
  diagonal term into the momentum matrix, and one velocity correction per outer
  iteration leaves the velocity lagging the pressure inside the zone. The shipped
  straightDuctImplicit case carries this entry; a steady porous solve without it
  converges slowly or stalls at a residual plateau.
* **`bounded Gauss upwind` on the convection terms.** A `steadyState` discretisation
  drops the `-div(phi)*field` term a transient one carries, and without it `k` and
  `epsilon` can go negative in the early iterations. The `bounded` prefix puts it back.
  The attested porous tutorial uses `upwind` on momentum as well, not a limited
  second-order scheme, and a porous case is a good place to accept that: the strong
  sink makes the momentum equation stiff, and the slab's pressure drop is set by `d` and
  `f` rather than by the convection scheme.
* **Turbulence inside the slab is not modelled, and that is a known ceiling rather than
  an oversight.** k-epsilon run through a porous zone generates eddy viscosity from the
  velocity gradients at the zone's faces, and there is no physical basis for it: the
  geometry producing the real turbulence is sub-cell and has been replaced by a source
  term. The sibling tutorial
  `$FOAM_TUTORIALS/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff` pins `k` and
  `epsilon` in the zone with a `scalarFixedValueConstraint` in `constant/fvOptions` for
  this reason. That is not written here, because it is a second dictionary and a
  judgement about values; the pressure drop across the slab is set by `d` and `f` and is
  not sensitive to it, and anything downstream of the slab is.
* **An inlet and an outlet, so no `pRefCell`.** Unlike `rotating.py`'s closed vessel,
  this domain has an outlet carrying a `fixedValue` pressure, so the level is pinned by
  a boundary condition and a reference cell would be a second, conflicting statement
  of it.
* `gmshToFoam` reads only msh format 2.2 ASCII; `Mesh.MshFileVersion` is set to 2.2
  right before `gmsh.write` for that reason, and it types every patch it creates
  `patch`, so the walls and the z faces are retyped with `foamDictionary` afterwards.
* Metres, always. OpenFOAM has no units: a duct authored in millimetres and never
  divided by 1000 is a mesh a thousand times too big, through a slab whose `d` is then
  a million times too weak per unit length, and `checkMesh` passes it without a word.

Where the OpenFOAM syntax above was read, so it can be re-checked rather than trusted:
the `porosityProperties` keys and the `d (5e7 -1000 -1000)` convention come from the
shipped `tutorials/incompressible/porousSimpleFoam/straightDuctImplicit` case, and the
momentum-sink form, the 0.5 on `f` and the declared dimensions from
`src/finiteVolume/cfdTools/general/porosityModel/DarcyForchheimer/DarcyForchheimer.{C,H}`
and `porosityModel.C`, all read off the OpenFOAM ESI repository's master branch, whose
headers currently say v2606. The workspace image is v2512. These keys are old and have
not moved between those two, and the cheap on-image confirmation is one command, printed
by this script at the end: `foamInfo DarcyForchheimer` beside
`ls $FOAM_TUTORIALS/incompressible/porousSimpleFoam`.
"""
from __future__ import annotations

import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
SLAB_THICKNESS_M = 0.05
"""Thickness of the porous slab along the flow, m. It is half of the measurement: a
pressure drop means nothing without the thickness it was measured across, because `d`
and `f` are per unit length."""

PRESSURE_DROP_CURVE = [
    # (face velocity m/s, pressure drop across the slab Pa)
    (0.5, 152.474),
    (1.0, 503.340),
    (1.5, 1052.60),
    (2.0, 1800.25),
]
"""The measurement, which is what `d` and `f` are derived from. A datasheet curve read
off at three or four points, or a rig's readings.

These defaults are NOT measurements. They are Ergun's prediction for a bed of
BED_PARTICLE_DIAMETER_M spheres at a void fraction of BED_VOID_FRACTION, in air at
20 C, across the 50 mm slab above, rounded to six figures -- so the shipped case carries
numbers that trace to arithmetic rather than to invention, and so the curve fit and the
Ergun reduction in this file check each other. One point is allowed and says what it
costs; see `coefficients_from_curve`."""
# --------------------------------------------------------------------------

HEIGHT_M = 0.1              # duct height, m
UPSTREAM_M = 0.2            # straight run before the slab, m
DOWNSTREAM_M = 0.3          # straight run after the slab, m -- longer, for the wake
LENGTH_M = UPSTREAM_M + SLAB_THICKNESS_M + DOWNSTREAM_M

CELLS_ACROSS = 40                              # cells across the duct height
CELL_SIZE_M = HEIGHT_M / CELLS_ACROSS
THICK_M = CELL_SIZE_M                          # extrude depth = one cell

GEOMETRY_TOL_M = 0.01 * CELL_SIZE_M
"""How close two coordinates have to be to count as the same place, in metres.

An ABSOLUTE tolerance, and that is the point. `occ.getBoundingBox` does not return the
exact box: OpenCASCADE pads it by about 1e-07 m in every direction, so the slab's
measured x span comes back as 0.0500002 rather than 0.05. A relative tolerance tight
enough to be meaningful -- `rel_tol=1e-06` on a 0.05 m span is 5e-08 -- is tighter than
that padding, so it rejects the very face it was written to find, and the script fails
saying the fragment did not cut the duct into three when the fragment was perfect. A
hundredth of a cell is far larger than the padding and far smaller than any distinction
this geometry makes."""

# Air at 20 degrees C and 1 atm: mu = 1.825e-05 Pa.s over rho = 1.204 kg/m^3, so
# nu = 1.516e-05 m^2/s. White, *Fluid Mechanics*, Table A.2 -- the same numbers and the
# same table `free_surface.py` uses for the air phase, rather than two templates
# quietly disagreeing about air.
RHO_KG_M3 = 1.204
NU_M2_S = 1.516e-05

# The bed the default curve was computed for, kept so the arithmetic can be redone.
BED_PARTICLE_DIAMETER_M = 0.003   # 3 mm spheres
BED_VOID_FRACTION = 0.38          # random close packing of spheres is about 0.36-0.40

CROSS_FLOW_MULTIPLIER = -1000.0
"""The two components of `d` and `f` that are not along the flow.

NEGATIVE IS NOT A NEGATIVE RESISTANCE. OpenFOAM reads a negative component as a
multiplier of the largest component -- `porosityModel.C`'s `adjustNegativeResistance` --
so -1000 means "a thousand times the through-flow resistance", which is what stops the
flow turning and sneaking across the face of the slab instead of passing through it.
The shipped straightDuctImplicit tutorial writes exactly `-1000 -1000`."""

TURBULENCE_INTENSITY = 0.05
C_MU = 0.09


def mu_pa_s() -> float:
    """Dynamic viscosity, from the kinematic one the case actually carries.

    mu = rho*nu. Spelled out because the Darcy term is `mu*d` while
    `constant/transportProperties` holds `nu`, and a `d` derived with nu where mu
    belonged is out by a factor of rho -- 1.204 for air, which is small enough to look
    like a modelling difference and large enough to matter.
    """
    return RHO_KG_M3 * NU_M2_S


def design_velocity_m_s() -> float:
    """The face velocity the case is run at: the median of the measured curve.

    A measured point rather than a round number, so the solved pressure drop has
    something to be wrong against. With an even number of points `statistics.median`
    would interpolate between two measurements and the run would have no measurement at
    its own velocity, so the lower of the two middle points is taken instead.
    """
    speeds = sorted(speed for speed, _drop in PRESSURE_DROP_CURVE)
    return speeds[(len(speeds) - 1) // 2]


def coefficients_from_curve(
    curve: list[tuple[float, float]] | None = None,
    thickness: float = SLAB_THICKNESS_M,
) -> tuple[float, float, str]:
    """(d, f, how) from measured (face velocity, pressure drop) pairs.

    The Forchheimer plot, in four lines. Dividing

        dp = (mu*d*U + 0.5*rho*f*U^2) * L

    through by L*U makes the right-hand side a straight line in U,

        dp/(L*U) = mu*d + 0.5*rho*f*U

    so a least-squares fit of `dp/(L*U)` against `U` has `mu*d` for its intercept and
    `0.5*rho*f` for its slope, and

        d = intercept/mu            f = 2*slope/rho

    `statistics.linear_regression` is the standard library's own two-parameter fit, so
    there is nothing here to get wrong except the algebra above, and numpy is not
    imported into a template that has to run as a single copied file.

    With a SINGLE point the system is underdetermined and no fit can rescue it: two
    unknowns, one equation. The all-Darcy limit (f = 0) is returned, because a `d` that
    is too large at high velocity is a conservative error for a pressure-drop
    prediction, and `how` says so in words that end up in the written dictionary. The
    all-Forchheimer alternative is printed beside it by `main` so the size of the choice
    is visible rather than buried.

    A negative intercept is possible from a real curve -- scatter at the low-velocity
    end, where `dp/(L*U)` is most sensitive -- and it would mean a negative `d`, which
    OpenFOAM would silently read as a cross-flow multiplier. It is clamped to zero and
    reported, because "this medium's loss is entirely inertial over the measured range"
    is a real answer and a negative Darcy coefficient is not.
    """
    curve = list(curve if curve is not None else PRESSURE_DROP_CURVE)
    if not curve:
        raise SystemExit("PRESSURE_DROP_CURVE is empty: there is nothing to derive d "
                         "and f from. One (face velocity, pressure drop) pair is the "
                         "minimum, and two or more determine the split.")
    for speed, drop in curve:
        if speed <= 0 or drop <= 0:
            raise SystemExit(
                f"({speed}, {drop}) is not a usable measurement: a face velocity and a "
                "pressure drop are both positive. A drop entered as a negative number "
                "because it is a loss gives a negative d, which OpenFOAM reads as a "
                "cross-flow multiplier rather than as an error."
            )

    mu = mu_pa_s()
    if len(curve) == 1:
        speed, drop = curve[0]
        darcy_only = drop / (thickness * mu * speed)
        return darcy_only, 0.0, (
            f"ONE measured point ({speed:g} m/s, {drop:g} Pa), so the viscous/inertial "
            f"split is undetermined and this is the all-Darcy limit, f = 0"
        )

    speeds = [speed for speed, _drop in curve]
    reduced = [drop / (thickness * speed) for speed, drop in curve]
    fit = statistics.linear_regression(speeds, reduced)

    darcy = fit.intercept / mu
    forchheimer = 2.0 * fit.slope / RHO_KG_M3
    how = (f"a straight-line fit of dp/(L*U) against U over {len(curve)} measured "
           f"points: intercept {fit.intercept:.6g} = mu*d, slope {fit.slope:.6g} "
           f"= 0.5*rho*f")

    if darcy < 0.0:
        how += ("; the fitted intercept was NEGATIVE, so d is clamped to 0 -- over the "
                "measured range this medium's loss is entirely inertial")
        darcy = 0.0
    if forchheimer < 0.0:
        how += ("; the fitted slope was NEGATIVE, so f is clamped to 0 -- over the "
                "measured range this medium's loss is entirely viscous")
        forchheimer = 0.0
    return darcy, forchheimer, how


def coefficients_from_ergun(
    particle_diameter: float = BED_PARTICLE_DIAMETER_M,
    void_fraction: float = BED_VOID_FRACTION,
) -> tuple[float, float]:
    """(d, f) for a packed bed, from a ruler and a bucket instead of a pressure gauge.

    Ergun's correlation for flow through a bed of roughly spherical particles:

        dp/L = 150*mu*(1-e)^2/(e^3*Dp^2) * U  +  1.75*rho*(1-e)/(e^3*Dp) * U^2

    Set that term by term against OpenFOAM's `dp/L = mu*d*U + 0.5*rho*f*U^2`. The
    viscous terms give `mu*d = 150*mu*(1-e)^2/(e^3*Dp^2)`, so mu cancels and

        d = 150*(1-e)^2 / (e^3*Dp^2)

    The inertial terms give `0.5*rho*f = 1.75*rho*(1-e)/(e^3*Dp)`, so rho cancels and

        f = 2*1.75*(1-e)/(e^3*Dp) = 3.5*(1-e)/(e^3*Dp)

    Both coefficients come out independent of the fluid, which is the useful part: a bed
    characterised once works for air or for water. The 2 on Ergun's 1.75 is the same
    half OpenFOAM puts on `f`, running the other way; dropping it halves the inertial
    term, which is the term that dominates a packed bed at anything but a crawl.

    Ergun is a correlation and not a law: it is good to roughly 20% for reasonably
    monodisperse near-spherical particles, and the void fraction is where the error
    lives, since `d` goes as e^-3 and so a 5% error in e is a 15% error in `d`.
    """
    if not 0.0 < void_fraction < 1.0:
        raise SystemExit(f"a void fraction of {void_fraction} is not between 0 and 1")
    if particle_diameter <= 0.0:
        raise SystemExit(f"a particle diameter of {particle_diameter} m is not positive")
    solid = 1.0 - void_fraction
    voids_cubed = void_fraction ** 3
    darcy = 150.0 * solid ** 2 / (voids_cubed * particle_diameter ** 2)
    forchheimer = 3.5 * solid / (voids_cubed * particle_diameter)
    return darcy, forchheimer


def predicted_drop_pa(speed: float, darcy: float, forchheimer: float,
                      thickness: float = SLAB_THICKNESS_M) -> float:
    """The pressure drop in Pa that `d` and `f` imply at a face velocity, which is the
    derivation run backwards.

    dp = (mu*d*U + 0.5*rho*f*U^2) * L, straight off OpenFOAM's own momentum sink. Used
    to print the measured drop beside the drop the fitted coefficients predict at the
    same velocity: a residual a reader can see, rather than a derivation they have to
    take on trust. It is also the closed-form answer the solved case should reproduce
    across the slab, which is the one check this case carries that needs no reference.
    """
    return (mu_pa_s() * darcy * speed
            + 0.5 * RHO_KG_M3 * forchheimer * speed * speed) * thickness


def k_seed(speed: float) -> float:
    """k = 1.5*(I*U)^2 at the inlet, I = 5%. The isotropic relation: k is 3/2 of the
    mean square of one fluctuating component."""
    return 1.5 * (TURBULENCE_INTENSITY * speed) ** 2


def epsilon_seed(speed: float) -> float:
    """epsilon = C_mu^0.75 * k^1.5 / l, with l = 0.07 * the hydraulic diameter.

    For a plane passage the hydraulic diameter is twice the gap, which is the one place
    a duct differs from a pipe and a factor of two waiting to happen.
    """
    length_scale = 0.07 * 2.0 * HEIGHT_M
    return C_MU ** 0.75 * k_seed(speed) ** 1.5 / length_scale


def build_mesh(msh_path: Path) -> None:
    """The duct in three pieces, the one-cell extrude, the msh file.

    Three rectangles made conformal with `occ.fragment` -- a shared edge, so the extrude
    produces one shared face at each slab boundary and the zone interfaces are internal
    to the mesh with no patch on them. A porous zone boundary carries no boundary
    condition; it is a surface where the source term starts.

    The pieces are identified by their bounding boxes rather than by trusting OCC's
    output ordering, which booleans do not promise, and the geometry here is
    axis-aligned so a bounding box is exact. `body_in_box.py` sorts its faces the same
    way for the same reason.
    """
    import gmsh

    gmsh.initialize()
    gmsh.model.add("porous")
    occ = gmsh.model.occ

    pieces = {
        "upstream": occ.addRectangle(0.0, 0.0, 0.0, UPSTREAM_M, HEIGHT_M),
        "slab": occ.addRectangle(UPSTREAM_M, 0.0, 0.0, SLAB_THICKNESS_M, HEIGHT_M),
        "downstream": occ.addRectangle(UPSTREAM_M + SLAB_THICKNESS_M, 0.0, 0.0,
                                       DOWNSTREAM_M, HEIGHT_M),
    }
    surfaces, _ = occ.fragment([(2, pieces["upstream"])],
                               [(2, pieces["slab"]), (2, pieces["downstream"])])
    occ.synchronize()

    # -- which surface is which, by the x span of its bounding box. The slab is the one
    # whose span is SLAB_THICKNESS_M; the x where it starts tells the other two apart,
    # and neither of those needs telling apart since they share a physical group.
    slab_surface, fluid_surfaces = None, []
    for dim, tag in surfaces:
        if dim != 2:
            continue
        xmin, _ymin, _zmin, xmax, _ymax, _zmax = occ.getBoundingBox(2, tag)
        if math.isclose(xmax - xmin, SLAB_THICKNESS_M, abs_tol=GEOMETRY_TOL_M):
            slab_surface = tag
        else:
            fluid_surfaces.append(tag)
    if slab_surface is None or len(fluid_surfaces) != 2:
        raise SystemExit(
            f"expected one slab surface and two clear-duct surfaces, got "
            f"{0 if slab_surface is None else 1} and {len(fluid_surfaces)}: the "
            f"fragment did not cut the duct into three"
        )

    gmsh.option.setNumber("Mesh.Algorithm", 8)          # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom, full-quad
    gmsh.option.setNumber("Mesh.MeshSizeMin", CELL_SIZE_M)
    gmsh.option.setNumber("Mesh.MeshSizeMax", CELL_SIZE_M)
    for tag in [slab_surface] + fluid_surfaces:
        gmsh.model.mesh.setRecombine(2, tag)

    # All three extruded in ONE call, so the edges they share become shared faces and
    # the zone interfaces are conformal. Extruded separately they would become pairs of
    # coincident boundary faces and the mesh would have two cracks across it.
    extruded = occ.extrude([(2, tag) for tag in [slab_surface] + fluid_surfaces],
                           0.0, 0.0, THICK_M, numElements=[1], recombine=True)
    occ.synchronize()

    volumes = [tag for dim, tag in extruded if dim == 3]
    if len(volumes) != 3:
        raise SystemExit(f"expected three volumes from the extrude, got {len(volumes)}")
    slab_volume, fluid_volumes = None, []
    for tag in volumes:
        xmin, _ymin, _zmin, xmax, _ymax, _zmax = occ.getBoundingBox(3, tag)
        if math.isclose(xmax - xmin, SLAB_THICKNESS_M, abs_tol=GEOMETRY_TOL_M):
            slab_volume = tag
        else:
            fluid_volumes.append(tag)
    if slab_volume is None or len(fluid_volumes) != 2:
        raise SystemExit("could not tell the slab volume from the clear-duct volumes")

    # -- the boundary faces, by bounding box. Everything on the outside of the three
    # volumes together is an inlet, an outlet, a duct wall or a z face; the two zone
    # interfaces are internal and so are not in this list at all.
    inlet, outlet, walls, z_faces = [], [], [], []
    boundary = {abs(tag) for _dim, tag in gmsh.model.getBoundary(
        [(3, tag) for tag in volumes], oriented=False, combined=True)}
    for tag in sorted(boundary):
        xmin, ymin, zmin, xmax, ymax, zmax = occ.getBoundingBox(2, tag)
        if math.isclose(zmin, zmax, abs_tol=GEOMETRY_TOL_M):
            z_faces.append(tag)                     # flat in z: the `empty` pair
        elif math.isclose(xmin, xmax, abs_tol=GEOMETRY_TOL_M):
            (inlet if math.isclose(xmin, 0.0, abs_tol=GEOMETRY_TOL_M)
             else outlet).append(tag)               # flat in x: the two duct ends
        else:
            walls.append(tag)                       # flat in y: floor and ceiling
    for name, faces in (("inlet", inlet), ("outlet", outlet),
                        ("walls", walls), ("frontAndBack", z_faces)):
        if not faces:
            raise SystemExit(
                f"no {name} face was found on the duct boundary: the bounding-box sort "
                f"above did not place every boundary face, which means the fragment or "
                f"the extrude produced a shape this does not describe"
            )

    gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
    gmsh.model.addPhysicalGroup(2, walls, name="walls")
    gmsh.model.addPhysicalGroup(2, z_faces, name="frontAndBack")
    # A physical VOLUME becomes a cellZone, which is what `porosityProperties` names.
    # Both groups are declared because gmsh writes only elements belonging to some
    # physical group once any group exists.
    gmsh.model.addPhysicalGroup(3, [slab_volume], name="porousZone")
    gmsh.model.addPhysicalGroup(3, fluid_volumes, name="fluid")

    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # gmshToFoam reads only this
    gmsh.write(str(msh_path))
    gmsh.finalize()


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """A command, printed, run, and refused by name when it fails or is not there.

    The second half is the one that was missing. A command that FAILS was already
    reported with its binary named and its stderr shown; a command that does not
    EXIST raised out of `subprocess.run` before the returncode check, so running this
    template on a machine with no OpenFOAM ended in `FileNotFoundError: [WinError 2]
    The system cannot find the file specified` (POSIX says `[Errno 2] No such file or
    directory`), naming no binary and not saying OpenFOAM was the thing missing.

    That is the laptop case, and this file is the worst place for it: a template is
    what a person copies. So the refusal names the binary, says where it comes from,
    and says what has already been written, because the mesh is usually the expensive
    part and it is not lost.
    """
    print("+", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    except FileNotFoundError:
        here = sorted(p.name for p in Path.cwd().iterdir() if p.is_file())
        raise SystemExit(
            f"refused: `{cmd[0]}` is not on PATH. It comes with OpenFOAM, which this "
            "machine does not appear to have, so the rest of this template cannot "
            "run here. What it has already written: "
            + (", ".join(here[:10]) + ("" if len(here) <= 10
                                       else f" and {len(here) - 10} more")
               if here else "nothing yet")
            + ". Copy this directory to a machine with OpenFOAM and run the template "
            "there, or use `openreynolds --local` for the Docker workspace."
        ) from None
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode})")
    return proc


def retype(patch: str, kind: str) -> None:
    """`gmshToFoam` types every patch `patch`; this is how it gets its real type.

    `constant/polyMesh/boundary` is a bare list, not a named dictionary, so
    `foamDictionary` addresses the whole thing as one pseudo-entry called `entry0` --
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
# porousSimpleFoam with kEpsilon reads five: U, p, k, epsilon and nut. kEpsilon reads
# `epsilon` and never reads `omega`; the pair is not interchangeable and a case with the
# wrong one stops before the first iteration rather than running badly.

def u_field() -> str:
    """U at the measured face velocity, straight along the duct.

    `fixedValue` at the inlet rather than a flow rate, because the velocity the
    coefficients were derived at IS the face velocity, so the case runs at the
    measurement's own condition and its pressure drop is directly comparable. The
    internal field starts at the same value, which gives the steady solve a sensible
    flux field from the first iteration instead of asking it to fill a duct from rest.

    `inletOutlet` at the outlet rather than `zeroGradient`: the slab's wake can
    recirculate, and a face with inflow on a `zeroGradient` outlet takes its value from
    inside the domain and feeds it back in, which is a feedback loop rather than a
    boundary condition.
    """
    speed = design_velocity_m_s()
    return (header("volVectorField", "U", "0") + f"""
dimensions      [0 1 -1 0 0 0 0];

// the FACE velocity the curve was measured at -- the same U that d and f were
// derived from, so the solved pressure drop is comparable with the measurement
internalField   uniform ({speed:g} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({speed:g} 0 0);
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform ({speed:g} 0 0);
    }}
    walls
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""")


def p_field() -> str:
    """Kinematic pressure, m2/s2, because incompressible OpenFOAM solves p/rho.

    This is the unit trap this template carries, and it is worth as much care as the
    coefficients: the drop this case reports across the slab is the measured drop
    DIVIDED BY rho. At the default condition that is 503.34 Pa measured against 418.06
    m2/s2 solved, and the two agree exactly. Calling it a 17% discrepancy is the
    mistake.

    `fixedValue 0` at the outlet pins the level, which is why there is no `pRefCell` in
    `fvSolution` -- unlike `rotating.py`'s closed vessel, where there is no outlet and
    a reference cell is mandatory. `zeroGradient` at the inlet, so the inlet pressure
    is a result: it is the slab's drop plus the duct's friction, and it is the number
    the case was built to produce.
    """
    return (header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];   // m2/s2 -- p/rho, NOT pascals: divide a measured
                                    // drop in Pa by rho before comparing

internalField   uniform 0;

boundaryField
{
    inlet
    {
        // a result, not an input: the slab's drop plus the duct's friction
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
""")


def k_field() -> str:
    """k, seeded from the inlet velocity and the duct height."""
    speed = design_velocity_m_s()
    k = k_seed(speed)
    return (header("volScalarField", "k", "0") + f"""
dimensions      [0 2 -2 0 0 0 0];

// k = 1.5*(I*U)^2, I = {TURBULENCE_INTENSITY}, U = {speed:g} m/s
internalField   uniform {k:.6g};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k:.6g};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {k:.6g};
        value           uniform {k:.6g};
    }}
    walls
    {{
        type            kqRWallFunction;
        value           uniform {k:.6g};
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""")


def epsilon_field() -> str:
    """epsilon, from k and a mixing length of 7% of the hydraulic diameter, which for
    a plane passage is twice the gap."""
    speed = design_velocity_m_s()
    epsilon = epsilon_seed(speed)
    return (header("volScalarField", "epsilon", "0") + f"""
dimensions      [0 2 -3 0 0 0 0];

// epsilon = C_mu^0.75 * k^1.5 / l, C_mu = {C_MU}, l = 0.07*Dh and Dh = 2*gap
//   = {0.07 * 2 * HEIGHT_M:.6g} m for a plane passage {HEIGHT_M:g} m tall
internalField   uniform {epsilon:.6g};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {epsilon:.6g};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {epsilon:.6g};
        value           uniform {epsilon:.6g};
    }}
    walls
    {{
        type            epsilonWallFunction;
        value           uniform {epsilon:.6g};
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""")


def nut_field() -> str:
    """nut, zero at t = 0 and computed by the model from then on.

    `nutUSpaldingWallFunction` rather than `nutkWallFunction`: a template cannot know
    what y+ this mesh will land at, and the Spalding blend is valid through the viscous
    sublayer, the buffer layer and the log layer alike, where `nutkWallFunction` is the
    high-Re form and is wrong below the log layer. `../case_gen.py`'s `wall_function`
    helper chooses the same one for the same argument.
    """
    return (header("volScalarField", "nut", "0") + """
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    walls
    {
        type            nutUSpaldingWallFunction;
        value           uniform 0;
    }
    frontAndBack
    {
        type            empty;
    }
}
""")


# -- constant/ -----------------------------------------------------------------

def porosity_properties() -> str:
    """The slab, with `d` and `f` derived rather than demanded, and the derivation
    written beside them.

    Keys read off the shipped
    `$FOAM_TUTORIALS/incompressible/porousSimpleFoam/straightDuctImplicit/constant/
    porosityProperties`: a named block carrying `type`, `cellZone`, `d`, `f` and a
    `coordinateSystem`. `d` and `f` are bare vectors there -- the model declares their
    dimensions in code (`dimless/sqr(dimLength)` and `dimless/dimLength`) and reads the
    coefficients out of an *optional* sub-dictionary, so both the bare form the tutorial
    uses and a nested `DarcyForchheimerCoeffs` block are accepted. The tutorial's form
    is the one written here, because it is the one attested as shipped.

    `e1` is the flow direction, so the FIRST component of each vector is the
    through-flow resistance and the other two are across the face of the slab.
    """
    darcy, forchheimer, how = coefficients_from_curve()
    speed = design_velocity_m_s()
    lines = [
        f"//   {speed_i:>6.3g} m/s measured {drop:>10.4f} Pa   "
        f"predicted {predicted_drop_pa(speed_i, darcy, forchheimer):>10.4f} Pa   "
        f"({100.0 * (predicted_drop_pa(speed_i, darcy, forchheimer) - drop) / drop:+.3g}%)"
        for speed_i, drop in PRESSURE_DROP_CURVE
    ]
    residuals = "\n".join(lines)
    return (header("dictionary", "porosityProperties", "constant") + f"""
// ============================================================================
// WHERE d AND f CAME FROM
//
// OpenFOAM's Darcy-Forchheimer momentum sink, from DarcyForchheimer.H:
//
//     S = -(mu*d + (rho*|U|/2)*f) * U        d [1/m2], f [1/m]
//
// so for steady flow straight through a slab of thickness L at face velocity U
//
//     dp = (mu*d*U + 0.5*rho*f*U^2) * L      [Pa]
//
// Dividing by L*U makes that a straight line in U, dp/(L*U) = mu*d + 0.5*rho*f*U,
// whose intercept is mu*d and whose slope is 0.5*rho*f. Hence:
//
//   derivation   {how}
//   fluid        rho = {RHO_KG_M3:g} kg/m^3, nu = {NU_M2_S:g} m^2/s, so mu = rho*nu = {mu_pa_s():.6g} Pa.s
//   thickness    L = {SLAB_THICKNESS_M:g} m
//   d            {darcy:.6g} 1/m2
//   f            {forchheimer:.6g} 1/m
//
// The derivation run backwards, against every point it was derived from -- this is
// the residual, and it is the check on the arithmetic above:
{residuals}
//
// The same coefficients predict, at the velocity this case is actually run at:
//   {speed:g} m/s  ->  {predicted_drop_pa(speed, darcy, forchheimer):.6g} Pa
//               =  {predicted_drop_pa(speed, darcy, forchheimer) / RHO_KG_M3:.6g} m2/s2 in THIS case's units,
//                  because porousSimpleFoam is incompressible and solves p/rho.
//                  That is the number to compare the solved field against; comparing
//                  it against the Pa figure is an apparent error of a factor of rho.
// ============================================================================

porosity1
{{
    type            DarcyForchheimer;

    cellZone        porousZone;   // a physical VOLUME in the msh becomes this cellZone

    // A NEGATIVE COMPONENT IS NOT A NEGATIVE RESISTANCE. OpenFOAM reads it as a
    // multiplier of the largest component (porosityModel.C, adjustNegativeResistance),
    // so {CROSS_FLOW_MULTIPLIER:g} means the two cross-flow directions resist {abs(CROSS_FLOW_MULTIPLIER):g} times harder
    // than the through-flow one. Without that, the flow turns and sneaks sideways
    // through the face of the slab instead of passing through it, and the answer is
    // a converged, plausible, wrong pressure drop. The shipped straightDuctImplicit
    // tutorial writes the same -1000 -1000.
    d   ({darcy:.6g} {CROSS_FLOW_MULTIPLIER:g} {CROSS_FLOW_MULTIPLIER:g});
    f   ({forchheimer:.6g} {CROSS_FLOW_MULTIPLIER:g} {CROSS_FLOW_MULTIPLIER:g});

    // e1 is the THROUGH-FLOW direction, so the first component of d and f above is
    // the resistance along the duct. Swap e1 and e2 and the slab resists sideways
    // a thousand times harder than it resists the flow, which solves.
    coordinateSystem
    {{
        origin  (0 0 0);
        e1      (1 0 0);
        e2      (0 1 0);
    }}
}}
""")


def transport_properties() -> str:
    """Air at 20 degrees C, with its source.

    nu = mu/rho = 1.825e-05 / 1.204 = 1.516e-05 m^2/s -- White, *Fluid Mechanics*,
    Table A.2. The same number and the same table `free_surface.py` uses for its air
    phase, rather than two templates quietly disagreeing about air. rho is not in this
    file -- an incompressible case does not carry one -- which is exactly why the
    pressure here is p/rho, and why the derivation above states the rho it used.
    """
    return (header("dictionary", "transportProperties", "constant") + f"""
transportModel  Newtonian;

nu              {NU_M2_S:g};   // m^2/s -- air at 20 C, mu 1.825e-05 Pa.s / rho {RHO_KG_M3:g}
""")


def turbulence_properties() -> str:
    """kEpsilon RAS, stated rather than defaulted.

    kEpsilon because the attested porous tutorial is kEpsilon, so the schemes, the
    relaxation factors and the wall functions below have a shipped case behind them.
    ESI OpenFOAM reads this file as `turbulenceProperties`; the openfoam.org line calls
    the same thing `momentumTransport`.
    """
    return (header("dictionary", "turbulenceProperties", "constant") + """
simulationType  RAS;

RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
""")


# -- system/ -------------------------------------------------------------------

def control_dict() -> str:
    """`porousSimpleFoam`, and the pressure drop the case exists to produce.

    `porousSimpleFoam` is a real solver in this OpenFOAM and not a historical name:
    `$FOAM_TUTORIALS/incompressible/porousSimpleFoam/straightDuctImplicit` ships with
    `application porousSimpleFoam;` and a `constant/porosityProperties` beside it. The
    same physics is also reachable as an `explicitPorositySource` entry in
    `constant/fvOptions` under an ordinary `simpleFoam`, which is what the compressible
    `angledDuctExplicitFixedCoeff` tutorial does; one route is enough, and this is the
    one whose tutorial matches this case.

    `endTime` counts SIMPLE iterations, not seconds -- this is a steady solver, and a
    Courant number for a run of this case is a number that does not exist.

    Two `surfaceFieldValue` function objects give the area-averaged pressure at the two
    ends. Their difference is the duct's total drop: the slab's, which is the closed
    form printed in `porosityProperties`, plus the two straight runs' wall friction,
    which is small here and is not zero. In m2/s2, because that is what the field is.
    """
    return (header("dictionary", "controlDict", "system") + """
application     porousSimpleFoam;

startFrom       latestTime;     // an interrupted solve resumes, it does not restart
startTime       0;
stopAt          endTime;
endTime         1000;           // SIMPLE ITERATIONS, not seconds -- this is steady

deltaT          1;
writeControl    timeStep;
writeInterval   200;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;

functions
{
    // The two ends' area-averaged pressure. Their difference is the whole duct's
    // drop -- the slab plus the friction of the two straight runs -- in m2/s2,
    // because that is the unit of the field. Multiply by rho for pascals.
    inletPressure
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        regionType      patch;
        name            inlet;
        operation       areaAverage;
        fields          (p);
        writeControl    timeStep;
        writeInterval   50;
        writeFields     no;
        log             yes;
    }

    outletPressure
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        regionType      patch;
        name            outlet;
        operation       areaAverage;
        fields          (p);
        writeControl    timeStep;
        writeInterval   50;
        writeFields     no;
        log             yes;
    }
}
""")


def fv_schemes() -> str:
    """Steady-state ddt and `bounded Gauss upwind`, which is the attested porous pair.

    `bounded Gauss` rather than plain `Gauss`: a `steadyState` discretisation drops the
    `-div(phi)*field` term a transient one carries, and without it `k` and `epsilon` can
    go negative in the first iterations, when the flux field is nowhere near
    divergence-free. The `bounded` prefix puts it back.

    `upwind` on momentum rather than a limited second-order scheme, which is what the
    shipped straightDuctImplicit case uses and is a good trade here: the strong porous
    sink makes the momentum equation stiff, and the slab's pressure drop is set by `d`
    and `f` rather than by the convection scheme. A case whose deliverable is the wake
    downstream of the slab rather than the drop across it would want
    `limitedLinearV 1` instead, and would want a finer mesh more.
    """
    return (header("dictionary", "fvSchemes", "system") + """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;

    div(phi,U)      bounded Gauss upwind;

    turbulence      bounded Gauss upwind;
    div(phi,k)      $turbulence;
    div(phi,epsilon) $turbulence;

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
""")


def fv_solution() -> str:
    """SIMPLE with `nUCorrectors 2`, which is the entry a porous case needs and an
    ordinary one does not.

    The implicit porous source puts a large diagonal term into the momentum matrix --
    at the default coefficients, `mu*d` is about 2131 and `0.5*rho*f*U` another 7936 at
    1 m/s, against a convective term of order rho*U/dx -- so one velocity correction per
    outer iteration leaves the velocity inside the zone lagging the pressure. The
    shipped straightDuctImplicit case carries this entry; without it a steady porous
    solve converges slowly or plateaus.

    No `pRefCell`: the outlet carries a `fixedValue` pressure, so the level is pinned by
    a boundary condition and a reference cell would be a second, conflicting statement
    of it. `rotating.py`'s closed vessel has the opposite problem and needs one.

    The relaxation factors are the shipped case's: p at 0.3, U at 0.7, the turbulence
    quantities at 0.9.
    """
    return (header("dictionary", "fvSolution", "system") + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-08;
        relTol          0.05;
        smoother        GaussSeidel;
        nCellsInCoarsestLevel 20;
    }

    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.1;
    }
}

SIMPLE
{
    // The porous source is a large diagonal term in the momentum matrix, so one
    // velocity correction per outer iteration leaves U lagging p inside the zone.
    // The shipped straightDuctImplicit tutorial carries this entry.
    nUCorrectors    2;

    nNonOrthogonalCorrectors 0;

    // No pRefCell: the outlet's fixedValue pressure pins the level already.

    residualControl
    {
        p               1e-04;
        U               1e-04;
        "(k|epsilon)"   1e-04;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.9;
        epsilon         0.9;
    }
}
""")


CASE_FILES = {
    "0/U": u_field,
    "0/p": p_field,
    "0/k": k_field,
    "0/epsilon": epsilon_field,
    "0/nut": nut_field,
    "constant/porosityProperties": porosity_properties,
    "constant/transportProperties": transport_properties,
    "constant/turbulenceProperties": turbulence_properties,
    "system/controlDict": control_dict,
    "system/fvSchemes": fv_schemes,
    "system/fvSolution": fv_solution,
}
"""Every file `porousSimpleFoam` with kEpsilon and a Darcy-Forchheimer zone reads, and
nothing it does not. There is no `constant/fvOptions`: the porosity is declared in
`porosityProperties`, which is the route this solver's own tutorial uses."""


def write_case(case: Path) -> None:
    """The whole case on disk before the mesh exists, because `gmshToFoam` builds a
    `Foam::Time` before it touches the mesh and fails "cannot find file
    .../system/controlDict" in a directory holding nothing but a `.msh`. `checkMesh`
    goes further still -- it builds an `fvMesh`, so it reads `fvSchemes` too.

    Where `duct2d.py` and `body_in_box.py` satisfy that with three stubs they expect to
    be overwritten, these are the real dictionaries: the mesh tools get what they need
    as a side effect of the case being complete.
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


def summary() -> str:
    """The derivation, the residual against every measured point, and the expected drop
    in both pascals and the m2/s2 the case actually reports.

    Printed as well as written into `constant/porosityProperties`, because the person
    who compares a solved pressure against a datasheet is not necessarily the person
    who opened that dictionary.
    """
    darcy, forchheimer, how = coefficients_from_curve()
    ergun_d, ergun_f = coefficients_from_ergun()
    speed = design_velocity_m_s()
    expected = predicted_drop_pa(speed, darcy, forchheimer)

    lines = [
        "d and f, derived rather than demanded:",
        f"    how            {how}",
        f"    fluid          rho = {RHO_KG_M3:g} kg/m^3, nu = {NU_M2_S:g} m^2/s, "
        f"mu = rho*nu = {mu_pa_s():.6g} Pa.s",
        f"    thickness      L = {SLAB_THICKNESS_M:g} m",
        f"    d              {darcy:.6g} 1/m2    (Darcy, the mu*d*U term)",
        f"    f              {forchheimer:.6g} 1/m     (Forchheimer, the 0.5*rho*f*U^2 term)",
        "",
        "the derivation run backwards against the points it came from:",
        "    face velocity      measured        predicted      error",
    ]
    for speed_i, drop in PRESSURE_DROP_CURVE:
        got = predicted_drop_pa(speed_i, darcy, forchheimer)
        lines.append(f"    {speed_i:>8.3g} m/s   {drop:>10.4f} Pa   {got:>10.4f} Pa   "
                     f"{100.0 * (got - drop) / drop:>+8.3g}%")
    lines += [
        "",
        "the independent route, for a packed bed with no curve at all (Ergun, from the",
        f"bed geometry: {BED_PARTICLE_DIAMETER_M:g} m particles at a void fraction of "
        f"{BED_VOID_FRACTION:g}):",
        f"    d              {ergun_d:.6g} 1/m2    "
        f"({100.0 * (darcy - ergun_d) / ergun_d:+.3g}% against the fit above)",
        f"    f              {ergun_f:.6g} 1/m     "
        f"({100.0 * (forchheimer - ergun_f) / ergun_f:+.3g}% against the fit above)",
        "    the two agree because the default curve IS Ergun's prediction, rounded to",
        "    six figures. with a real measured curve they will not, and the gap is a",
        "    statement about how spherical and how monodisperse the bed really is.",
        "",
        "what the solved case should reproduce across the slab, in BOTH units because",
        "incompressible OpenFOAM solves p/rho and the difference is a factor of rho:",
        f"    at {speed:g} m/s      {expected:.6g} Pa",
        f"                   {expected / RHO_KG_M3:.6g} m2/s2   <-- what 0/p and the function",
        "                                     objects report. comparing the Pa figure",
        f"                                     against the field is an apparent "
        f"{100.0 * (RHO_KG_M3 - 1.0):.0f}% error.",
        "    the inlet-to-outlet drop will be a little larger than this: the slab's",
        "    closed form plus the two straight runs' wall friction.",
    ]
    return "\n".join(lines)


def main() -> None:
    case = Path.cwd()
    msh = case / "porous.msh"
    build_mesh(msh)
    write_case(case)

    run(["gmshToFoam", msh.name])
    retype("walls", "wall")
    # One physical group covers both z faces, so one retype makes the case 2D.
    retype("frontAndBack", "empty")

    run(["checkMesh", "-case", "."])

    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png",
         "--json", "look.json"])

    print()
    print(summary())

    # What is on PATH is a question rather than a fact, and this solver is one people
    # assume was removed when porosity became an fvOption. It was not.
    print()
    if shutil.which("porousSimpleFoam"):
        print("porousSimpleFoam is on PATH. to confirm the dictionary keys on this image:")
    else:
        print("porousSimpleFoam is NOT on PATH, so the case above cannot be solved as")
        print("written. the same physics is reachable as an explicitPorositySource entry")
        print("in constant/fvOptions under an ordinary simpleFoam -- see the compressible")
        print("angledDuctExplicitFixedCoeff tutorial. to see what this image has:")
    print("    foamInfo DarcyForchheimer")
    print("    ls $FOAM_TUTORIALS/incompressible/porousSimpleFoam")

    # Not launched from here: what a solve costs, and whether it is wanted at all, is
    # not a template's call. `startFrom latestTime` is already set, so this loop
    # resumes itself through any in-sandbox failure -- see "Long solves" in
    # ../notes/openfoam-field-notes.md.
    print("\nthe case is complete and meant to be solved. one way to launch it:\n"
          "    until porousSimpleFoam >> log.porousSimpleFoam 2>&1; do "
          "echo resuming >> log.porousSimpleFoam; sleep 5; done\n"
          "the drop is the difference of the two surfaceFieldValue outputs in "
          "postProcessing/")


if __name__ == "__main__":
    main()
