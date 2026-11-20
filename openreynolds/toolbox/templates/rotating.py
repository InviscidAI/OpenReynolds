#!/usr/bin/env python3
"""TEMPLATE: a frozen-rotor (MRF) bladed mixer -- gmsh -> gmshToFoam -> a case
`simpleFoam` can actually solve, with the rotation as a source term rather than as a
moving mesh.

Copy this file into an empty case directory and run it there:

    python3 rotating.py

It builds a 2D vessel VESSEL_RADIUS_M in radius with a six-bladed impeller at its
centre, one cell thick in z (the 2D recipe: quads extruded one layer, the two z faces
typed `empty`), cuts the fluid into two cell zones -- `rotor` around the impeller and
`stator` out to the vessel wall -- meshes it, converts with `gmshToFoam`, retypes the
impeller and vessel walls `wall` and the z faces `empty`, writes the whole `simpleFoam`
case including `constant/MRFProperties`, runs `checkMesh`, and finishes with
`mesh_look.py` so the result is looked at and not just built. Edit VESSEL_RADIUS_M and
ROTOR_RPM and re-run; the impeller, the zone radius, the cell size and every derived
turbulence number follow from them.

Like `free_surface.py` and unlike `duct2d.py` and `body_in_box.py`, which write **stub**
`controlDict`/`fvSchemes`/`fvSolution` just sufficient for `gmshToFoam` and `checkMesh`
to build a `Foam::Time` and an `fvMesh`, everything written here is meant to be solved
as it stands. It stops short of launching the solver, because what a solve costs and
whether it is wanted is not a template's call; the command it would be is printed at the
end.

MRF, AND WHAT MRF CANNOT TELL YOU
---------------------------------
There are two ways to turn a rotor in OpenFOAM and they answer different questions.

* **MRF -- a frozen rotor.** The mesh never moves. Inside a named `cellZone` the
  momentum equation gains the rotating-frame source terms and the flux is made relative
  to a frame turning at `omega`, so the blade is held at one azimuthal position and the
  flow around it is solved steady. One steady run, `simpleFoam`, minutes to hours. This
  is what this template writes.
* **Sliding mesh / `cyclicAMI`.** The rotor's cells really rotate past the stator's
  across a sliding interface, transient, `pimpleFoam`. It computes blade passing
  properly and costs one to three orders of magnitude more.

This template delivers MRF, and the reason is measured rather than preferred: on this
repository's own `cyclicAMI` benchmark (`benchmarks/moving_mesh/couette_ami.txt`, and
the numbers in `../notes/openfoam-field-notes.md`) the sliding interface converges at
observed order 0.79 -- first order -- and its error scales with `omega*dt`, the angle the
rotor slides in one step, not with the cell count: four times the cells at a fixed `dt`
moved the inner-wall torque error from +0.2174% to +0.2205%. A template cannot choose a
user's `omega*dt`, so a sliding-mesh template would be shipping a case whose dominant
error is a number it left blank. The frozen rotor's limitations, below, are at least
nameable.

What this case therefore does **not** compute, stated the same way in
`constant/MRFProperties` where a reader editing `omega` will meet it:

* **No blade passing.** The impeller is frozen. The blade-passing tone at
  N_BLADES * ROTOR_RPM / 60 Hz, and every quantity that is a function of blade position
  -- unsteady casing pressure, tonal noise, fatigue cycles -- is absent from the answer
  rather than small in it. The printed summary gives that frequency in Hz so the
  question "can this case see my X Hz vibration" has a number beside it.
* **No unsteady wake.** The wake behind each blade is steady in the rotating frame, so
  vortex shedding off the blade, rotating stall and any wake/baffle interaction do not
  appear. A steady run cannot be converged into an unsteady answer.
* **One azimuthal position out of many.** Where the vessel is azimuthally asymmetric (a
  baffle, a volute, a side outlet) the frozen rotor gives the answer for the blade angle
  it was meshed at. The honest error bar is the spread of the quantity over several blade
  positions, which is several runs.
* **An answer that depends on where the zone boundary was put.** The zone face is a
  surface where nothing physical happens, so moving it cannot change a correct answer.
  `ladder.py`'s frozen-rotor rung states the check: the torque should move by no more
  than a few per cent when the zone radius changes by about a quarter, and the absolute
  velocity should have no step across the zone face that is visible on a slice. A torque
  that tracks ZONE_RADIUS_FRACTION is a number set by a modelling choice, and quoting it
  without saying so is the trap.

rad/s VERSUS rpm, WHICH IS A FACTOR OF 9.5493
---------------------------------------------
`omega` in `constant/MRFProperties` is in **radians per second**. OpenFOAM's own
`MRFZone.H` says so where the member is declared -- `//- Angular velocity (rad/sec)` --
and there is no `rpm` keyword in that class to catch the mistake. A number meant as rpm
and written into `omega` runs happily at

    1 rpm = 2*pi/60 rad/s = 0.104719755 rad/s
    1 rad/s = 60/(2*pi) rpm = 9.549296586 rpm

so a rotor entered as "1000" meaning rpm turns at 1000 rad/s, which is 9549 rpm, and a
rotor entered as "104.72" meaning rad/s when rpm was meant turns 9.55 times too slowly.
Nothing in the log mentions it; the torque is simply wrong by roughly the square of
9.5493.

The shipped tutorial is the evidence that `omega` is rad/s rather than a convention
somebody remembered: `$FOAM_TUTORIALS/incompressible/simpleFoam/mixerVessel2D/constant/
MRFProperties` carries `omega 104.72;`, and 104.72 rad/s is 1000.0 rpm to five figures,
a round nameplate number. Read as rpm it would be 104.72 rpm, which is nobody's round
number.

So the editable constant here is named ROTOR_RPM, in the unit a nameplate is written
in, the conversion happens in one function with the factor spelled out, and `omega` is
written with both numbers in a comment beside it. The printed summary states rad/s, rpm,
the tip speed in m/s and the rpm converted back, because a round trip that does not
return where it started is the cheapest possible check on a unit conversion.

THE REST OF THE QUIET FAILURES, BAKED IN BELOW
----------------------------------------------
* **The rotor's walls are `noSlip`, and that is not a mistake.** MRF declares
  `void correctBoundaryVelocity(volVectorField& U) const;` -- "Correct the boundary
  velocity for the rotation of the MRF region" -- so the patches belonging to the zone
  get their velocity set from `Omega x r` by the MRF machinery each iteration, on a mesh
  that never moves. The shipped mixerVessel2D puts plain `noSlip` on its `rotor` patch
  for exactly this reason. Writing `rotatingWallVelocity` here instead would impose the
  rotation twice.
* **`nonRotatingPatches` is the other half of that, and it is the classic MRF defect.**
  Any patch whose faces lie inside the rotating `cellZone` but which is physically
  stationary -- a baffle, a shaft seal, a stationary shroud -- has to be named there or
  it gets spun with the zone and silently stirs the fluid. The class calls the member
  `excludedPatchNames_`, commented "Non-rotating patches". This vessel has no such patch,
  so the entry is written as an empty list **explicitly**, with the comment saying what
  would go in it, rather than omitted.
* **A closed vessel has no pressure level, so `pRefCell`/`pRefValue` are mandatory.**
  There is no inlet and no outlet here: every patch is a wall or `empty`. An
  incompressible steady solve of a closed domain has a singular pressure equation
  without a reference cell, and the symptom is a pressure field that drifts without
  bound while the velocity field is perfectly healthy.
* **`bounded Gauss` on the convection terms.** `steadyState` ddt with an unbounded
  convection scheme lets `k` and `epsilon` go negative in the first iterations before
  the field is anywhere near the solution. The `bounded` prefix adds the
  `-div(phi)*field` term that a steady-state discretisation drops, and it is what the
  shipped mixerVessel2D uses.
* **kEpsilon, not kOmegaSST, and the reason is the reference.** The attested MRF
  tutorial is kEpsilon, so the schemes, relaxation and wall functions below have a
  shipped case behind them rather than being reasoned out. kEpsilon reads `epsilon` and
  never reads `omega`; the two are not interchangeable and a case with the wrong pair
  stops before the first iteration.
* **`nutUSpaldingWallFunction` rather than `nutkWallFunction`.** A template cannot know
  what y+ the mesh will land at. `nutkWallFunction` is the high-Re Launder-Spalding form
  and is only valid above the buffer layer; the Spalding blend covers the viscous
  sublayer, the buffer layer and the log layer, so it is the one form that is not wrong
  at whichever y+ this mesh happens to produce. `../case_gen.py` picks it for the same
  reason, with the argument at its `wall_function` helper.
* **The zone is a `cellZone`, and `gmshToFoam` is what makes one.** A physical *volume*
  in the msh becomes a cellZone of the same name; a physical *surface* becomes a patch.
  A `MRFProperties` naming a zone the mesh does not have fails at the first solver step.
  Two physical volumes are declared below, `rotor` and `stator`, and `mesh_look.py`
  lists the zones a mesh actually has.
* **A rotating case with no torque output has delivered nothing**, so a `forces`
  function object on the impeller is written into `controlDict`. Its z-moment is the
  shaft torque. `rhoInf` is stated because incompressible OpenFOAM solves p/rho and a
  moment taken from that field without the density is out by a factor of 998.
* `gmshToFoam` reads only msh format 2.2 ASCII; `Mesh.MshFileVersion` is set to 2.2
  right before `gmsh.write` for that reason, and it types every patch it creates
  `patch`, so the walls and the z faces are retyped with `foamDictionary` afterwards.
* Metres, always. OpenFOAM has no units: a vessel authored in millimetres and never
  divided by 1000 is a mesh a thousand times too big at a thousandth of the intended
  Reynolds number, and `checkMesh` passes it without a word.

Where the OpenFOAM syntax above was read, so it can be re-checked rather than trusted:
`MRFProperties` keys and the `omega 104.72` value come from the shipped
`tutorials/incompressible/simpleFoam/mixerVessel2D` case and the `omega_` declaration
from `src/finiteVolume/cfdTools/general/MRF/MRFZone.H`, both read off the OpenFOAM ESI
repository's master branch, whose headers currently say v2606. The workspace image is
v2512. These dictionary keys are old and have not moved between those two, and the
cheap on-image confirmation is one command, printed by this script at the end:
`foamInfo MRFZone` beside `ls $FOAM_TUTORIALS/incompressible/simpleFoam/mixerVessel2D`.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
VESSEL_RADIUS_M = 0.1      # inner radius of the vessel, m
ROTOR_RPM = 1000.0         # impeller speed in REVOLUTIONS PER MINUTE; omega below is rad/s
# --------------------------------------------------------------------------

N_BLADES = 6                                   # blades on the impeller
HUB_FRACTION = 0.20                            # hub radius / vessel radius
BLADE_FRACTION = 0.55                          # blade tip radius / vessel radius
ZONE_FRACTION = 0.70                           # MRF zone radius / vessel radius
BLADE_WIDTH_FRACTION = 0.10                    # blade thickness / vessel radius

CELLS_ACROSS = 80                              # cells across the vessel diameter
CELL_SIZE_M = 2.0 * VESSEL_RADIUS_M / CELLS_ACROSS
THICK_M = CELL_SIZE_M                          # extrude depth = one cell

HUB_RADIUS_M = HUB_FRACTION * VESSEL_RADIUS_M
BLADE_RADIUS_M = BLADE_FRACTION * VESSEL_RADIUS_M
ZONE_RADIUS_M = ZONE_FRACTION * VESSEL_RADIUS_M
BLADE_WIDTH_M = BLADE_WIDTH_FRACTION * VESSEL_RADIUS_M

# Water at 20 degrees C and 1 atm: mu = 1.002e-03 Pa.s over rho = 998.2 kg/m^3.
# White, *Fluid Mechanics*, Table A.1. rho is here because a moment computed from
# incompressible OpenFOAM's p/rho field is out by this factor without it.
NU_M2_S = 1.004e-06
RHO_KG_M3 = 998.2

TURBULENCE_INTENSITY = 0.05   # 5% of tip speed, the usual guess for a stirred vessel
C_MU = 0.09                   # kEpsilon's constant, used for the epsilon seed

RAD_S_PER_RPM = 2.0 * math.pi / 60.0
"""0.10471975511965977. The whole of the rad/s-versus-rpm problem is this number and
its reciprocal, 9.549296585513721."""


def omega_rad_per_s(rpm: float = ROTOR_RPM) -> float:
    """rpm -> rad/s, which is the unit `constant/MRFProperties` wants.

    One revolution is 2*pi radians and one minute is 60 seconds, so the factor is
    2*pi/60 = 0.104719755 and nothing else. Spelled out as a named constant rather
    than folded into a literal because the inverse, 9.5493, is the size of the error
    when it is left out, and that error is invisible in every log the run produces.
    """
    return rpm * RAD_S_PER_RPM


def rpm_from_rad_per_s(omega: float) -> float:
    """rad/s -> rpm, the round trip. Used by the printed summary: a conversion that
    does not come back to where it started is the cheapest check there is."""
    return omega / RAD_S_PER_RPM


def tip_speed_m_s() -> float:
    """The blade tip's speed, omega * r, in m/s -- the number that says whether a
    stated rotor speed is physically sane at all, and the velocity scale the
    turbulence seeds below are built on."""
    return omega_rad_per_s() * BLADE_RADIUS_M


def blade_passing_hz() -> float:
    """The tone this case cannot produce: blades * revolutions per second.

    N_BLADES * ROTOR_RPM / 60. Printed rather than merely mentioned, so that "can
    this case see my 100 Hz vibration" has a number beside it instead of a
    paragraph.
    """
    return N_BLADES * ROTOR_RPM / 60.0


def k_seed() -> float:
    """k = 1.5*(I*U)^2 with U the tip speed and I the turbulence intensity.

    The isotropic relation: k is 3/2 of the mean square of one velocity fluctuation
    component, and the fluctuation is taken as I times the velocity scale. A closed
    vessel has no inlet to take a free-stream value from, so the tip speed is the
    only velocity scale the geometry offers.
    """
    return 1.5 * (TURBULENCE_INTENSITY * tip_speed_m_s()) ** 2


def epsilon_seed() -> float:
    """epsilon = C_mu^0.75 * k^1.5 / l, with l = 0.07 * vessel diameter.

    The standard equilibrium estimate. The 0.07 of a width is the usual mixing-length
    fraction for a bounded flow; both numbers are a starting field for a steady solve
    rather than a result, and a steady run converges away from them.
    """
    length_scale = 0.07 * 2.0 * VESSEL_RADIUS_M
    return C_MU ** 0.75 * k_seed() ** 1.5 / length_scale


def blade_angles() -> list[float]:
    """The azimuth of each blade, in radians, evenly spaced from zero.

    Returned rather than recomputed at each use because the frozen rotor's answer is
    a function of these angles -- that is the modelling limitation stated in the
    docstring -- so the positions the case was meshed at are worth being able to read
    off one place.
    """
    return [index * 2.0 * math.pi / N_BLADES for index in range(N_BLADES)]


def build_mesh(msh_path: Path) -> None:
    """The vessel, the impeller, the two fluid zones, the one-cell extrude, the msh.

    The fluid is built as two surfaces that share an edge at ZONE_RADIUS_M rather than
    as one surface cut afterwards, and they are made conformal with `occ.fragment` --
    a shared edge, so the extrude produces one shared face and the zone interface is
    internal to the mesh with no patch on it. An MRF zone boundary carries no boundary
    condition; it is a surface where nothing happens.

    Everything is identified after the fact by **measurement** rather than by trusting
    OCC's output ordering, which booleans do not promise: the two fluid surfaces are
    told apart by their area against the ring's exact `pi*(R_vessel^2 - R_zone^2)`, and
    the boundary faces by the radius and z of their centre of mass. `body_in_box.py`
    does the same thing with `getEntitiesInBoundingBox` for the same reason.
    """
    import gmsh

    gmsh.initialize()
    gmsh.model.add("rotating")
    occ = gmsh.model.occ

    # -- the impeller solid: a hub with N_BLADES rectangular blades fused onto it.
    hub = occ.addDisk(0.0, 0.0, 0.0, HUB_RADIUS_M, HUB_RADIUS_M)
    blades = []
    for angle in blade_angles():
        # Each blade starts inside the hub so the fuse leaves no sliver, and reaches
        # BLADE_RADIUS_M. Built on the x axis and rotated, so the blades are identical
        # by construction instead of by four lines of trigonometry each.
        start = 0.5 * HUB_RADIUS_M
        tag = occ.addRectangle(start, -0.5 * BLADE_WIDTH_M, 0.0,
                               BLADE_RADIUS_M - start, BLADE_WIDTH_M)
        occ.rotate([(2, tag)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, angle)
        blades.append((2, tag))
    impeller, _ = occ.fuse([(2, hub)], blades)

    # -- the two fluid regions. The zone disk is copied before it is consumed: `cut`
    # removes its tool by default, and the ring needs the same circle the rotor zone
    # was bounded by, not a second one authored separately.
    zone_disk = occ.addDisk(0.0, 0.0, 0.0, ZONE_RADIUS_M, ZONE_RADIUS_M)
    zone_copy = occ.copy([(2, zone_disk)])
    vessel_disk = occ.addDisk(0.0, 0.0, 0.0, VESSEL_RADIUS_M, VESSEL_RADIUS_M)

    rotor_fluid, _ = occ.cut([(2, zone_disk)], impeller)
    ring_fluid, _ = occ.cut([(2, vessel_disk)], zone_copy)
    surfaces, _ = occ.fragment(rotor_fluid, ring_fluid)
    occ.synchronize()

    # -- which surface is which, by area. The ring's area is exact.
    ring_area = math.pi * (VESSEL_RADIUS_M ** 2 - ZONE_RADIUS_M ** 2)
    rotor_surfaces, stator_surfaces = [], []
    for dim, tag in surfaces:
        if dim != 2:
            continue
        if math.isclose(occ.getMass(2, tag), ring_area, rel_tol=1e-06):
            stator_surfaces.append(tag)
        else:
            rotor_surfaces.append(tag)
    if len(stator_surfaces) != 1 or len(rotor_surfaces) != 1:
        raise SystemExit(
            f"expected one rotor and one stator surface, got {len(rotor_surfaces)} and "
            f"{len(stator_surfaces)}: the boolean did not produce two fluid regions"
        )

    # -- quads, then one layer swept into hexahedra. Frontal-Delaunay for quads plus
    # Blossom full-quad recombination, the same pair `duct2d.py` uses.
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.MeshSizeMin", CELL_SIZE_M)
    gmsh.option.setNumber("Mesh.MeshSizeMax", CELL_SIZE_M)
    for tag in rotor_surfaces + stator_surfaces:
        gmsh.model.mesh.setRecombine(2, tag)

    # Both surfaces extruded in ONE call, so the edge they share becomes one shared
    # face: the zone interface is internal and conformal. Extruded separately it would
    # become two coincident boundary faces and the mesh would have a crack in it.
    extruded = occ.extrude([(2, tag) for tag in rotor_surfaces + stator_surfaces],
                           0.0, 0.0, THICK_M, numElements=[1], recombine=True)
    occ.synchronize()

    volumes = [tag for dim, tag in extruded if dim == 3]
    if len(volumes) != 2:
        raise SystemExit(f"expected two volumes from the extrude, got {len(volumes)}")
    # The volume whose own mass is the ring's area times the thickness is the stator.
    rotor_volume = stator_volume = None
    for tag in volumes:
        if math.isclose(occ.getMass(3, tag), ring_area * THICK_M, rel_tol=1e-06):
            stator_volume = tag
        else:
            rotor_volume = tag
    if rotor_volume is None or stator_volume is None:
        raise SystemExit("could not tell the rotor volume from the stator volume by mass")

    # -- the boundary faces, by measurement. Every surface in the model after the
    # extrude is either a z face (centre of mass at z = 0 or z = THICK_M) or a lateral
    # face at some radius. The zone interface is shared by the two volumes and is not
    # on the boundary of either, so it is excluded by asking for the boundary.
    z_faces, wall_faces = [], []
    boundary = {abs(tag) for _dim, tag in gmsh.model.getBoundary(
        [(3, rotor_volume), (3, stator_volume)], oriented=False, combined=True)}
    for tag in sorted(boundary):
        xmin, ymin, zmin, xmax, ymax, zmax = occ.getBoundingBox(2, tag)
        if math.isclose(zmin, zmax, abs_tol=0.01 * THICK_M):
            # Flat in z: one of the two faces the `empty` type belongs on.
            z_faces.append(tag)
        else:
            # How far out the face reaches, from its bounding box rather than from its
            # centre of mass: the vessel wall is a complete cylinder, and a complete
            # cylinder's centre of mass is ON THE AXIS, so a centroid radius calls it
            # r = 0 and files it with the hub. The bounding box cannot be fooled that
            # way. `body_in_box.py` sorts its faces by bounding box for the same reason.
            wall_faces.append((max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)), tag))

    # A lateral face belongs to the vessel if it reaches out to the vessel radius;
    # everything else lateral is the impeller. The midpoint between the blade tip and
    # the vessel wall is the separator, and it is unambiguous because the zone
    # interface is internal and so is not in this list at all.
    cutoff = 0.5 * (BLADE_RADIUS_M + VESSEL_RADIUS_M)
    stator_wall = [tag for reach, tag in wall_faces if reach >= cutoff]
    rotor_wall = [tag for reach, tag in wall_faces if reach < cutoff]
    if not stator_wall or not rotor_wall:
        raise SystemExit(
            f"expected both a vessel wall and an impeller wall, got {len(stator_wall)} "
            f"and {len(rotor_wall)} lateral faces either side of r = {cutoff} m"
        )

    # -- physical groups. A physical SURFACE becomes a patch; a physical VOLUME becomes
    # a cellZone, which is what `MRFProperties` names. Both volumes are named because
    # gmsh writes only elements belonging to some physical group once any group exists.
    gmsh.model.addPhysicalGroup(2, rotor_wall, name="rotor")
    gmsh.model.addPhysicalGroup(2, stator_wall, name="stator")
    gmsh.model.addPhysicalGroup(2, z_faces, name="frontAndBack")
    gmsh.model.addPhysicalGroup(3, [rotor_volume], name="rotorZone")
    gmsh.model.addPhysicalGroup(3, [stator_volume], name="statorZone")

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
    `foamDictionary` addresses the whole thing as one pseudo-entry called `entry0` --
    `-entry rotor.type` alone fails with "not found in dictionary"; the patch name has
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
# simpleFoam with kEpsilon reads five: U, p, k, epsilon and nut. kEpsilon reads
# `epsilon` and never reads `omega`; the pair is not interchangeable and a case with
# the wrong one stops before the first iteration rather than running badly.

def u_field() -> str:
    """U, at rest everywhere at t = 0, and `noSlip` on both walls including the rotor.

    `noSlip` on the impeller is the point worth reading twice. The mesh never moves
    here, and the blade is still turning: MRF declares
    `void correctBoundaryVelocity(volVectorField& U) const;` -- "Correct the boundary
    velocity for the rotation of the MRF region" -- and sets the zone's patch velocities
    from `Omega x r` itself, every iteration. The shipped mixerVessel2D tutorial puts
    plain `noSlip` on its `rotor` patch for this reason. A `rotatingWallVelocity` here
    would impose the rotation a second time on top of the one MRF already applied.
    """
    return (header("volVectorField", "U", "0") + """
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    rotor
    {
        // MRF corrects this to Omega x r inside the zone -- the blade turns, the
        // mesh does not, and this entry stays noSlip. See constant/MRFProperties.
        type            noSlip;
    }
    stator
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}
""")


def p_field() -> str:
    """Kinematic pressure, m2/s2, because incompressible OpenFOAM solves p/rho.

    Naming the unit here is not decoration: a torque computed from this field as if it
    were pascals is out by a factor of rho -- 998 for water -- and looks entirely
    plausible. The `forces` function object in `controlDict` carries `rhoInf` for
    exactly that reason.

    `zeroGradient` on both walls, which is what the shipped mixerVessel2D uses: there
    is no inlet or outlet to set a level against, so the level comes from
    `pRefCell`/`pRefValue` in `fvSolution` instead.
    """
    return (header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];   // m2/s2 -- p/rho, NOT pascals

internalField   uniform 0;

boundaryField
{
    rotor
    {
        type            zeroGradient;
    }
    stator
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
    """k, seeded from the tip speed, with `kqRWallFunction` on the walls.

    A closed vessel has no inlet to take a free-stream value from, so the velocity
    scale is the blade tip's own speed: k = 1.5*(I*U_tip)^2 at I = 5%. It is a starting
    field, not a result -- a steady solve converges away from it -- and it is written
    as a number with its arithmetic in a comment rather than as a round guess, so a
    reader can see which way it was wrong if the solve struggles.
    """
    k = k_seed()
    return (header("volScalarField", "k", "0") + f"""
dimensions      [0 2 -2 0 0 0 0];

// k = 1.5*(I*U_tip)^2, I = {TURBULENCE_INTENSITY}, U_tip = omega*r_blade
//   = {tip_speed_m_s():.6g} m/s at {ROTOR_RPM:g} rpm and r_blade = {BLADE_RADIUS_M:.6g} m
internalField   uniform {k:.6g};

boundaryField
{{
    rotor
    {{
        type            kqRWallFunction;
        value           uniform {k:.6g};
    }}
    stator
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
    """epsilon, from k and a mixing length, with `epsilonWallFunction` on the walls."""
    epsilon = epsilon_seed()
    return (header("volScalarField", "epsilon", "0") + f"""
dimensions      [0 2 -3 0 0 0 0];

// epsilon = C_mu^0.75 * k^1.5 / l, C_mu = {C_MU}, l = 0.07*D = {0.07 * 2 * VESSEL_RADIUS_M:.6g} m
internalField   uniform {epsilon:.6g};

boundaryField
{{
    rotor
    {{
        type            epsilonWallFunction;
        value           uniform {epsilon:.6g};
    }}
    stator
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
    what y+ this mesh will land at, and the Spalding blend is the one form that is
    valid through the viscous sublayer, the buffer layer and the log layer alike.
    `nutkWallFunction` is the high-Re Launder-Spalding form and is wrong below the log
    layer. `../case_gen.py`'s `wall_function` helper chooses the same one, for the same
    argument.
    """
    return (header("volScalarField", "nut", "0") + """
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    rotor
    {
        type            nutUSpaldingWallFunction;
        value           uniform 0;
    }
    stator
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

def mrf_properties() -> str:
    """The frozen rotor itself, and the file that says what this case cannot compute.

    Keys read off the shipped
    `$FOAM_TUTORIALS/incompressible/simpleFoam/mixerVessel2D/constant/MRFProperties`:
    a named block carrying `cellZone`, `active`, `nonRotatingPatches`, `origin`, `axis`
    and `omega`. `omega` is a bare number in radians per second -- OpenFOAM's
    `MRFZone.H` declares it `//- Angular velocity (rad/sec)` -- and the tutorial's own
    value, 104.72, is 1000 rpm, which is the evidence rather than the recollection.

    `nonRotatingPatches` is written as an explicit empty list with its explanation,
    because the defect it prevents is silent: a stationary baffle whose faces lie inside
    the rotating cellZone gets spun with the zone unless it is named here, and the run
    reports nothing.
    """
    omega = omega_rad_per_s()
    return (header("dictionary", "MRFProperties", "constant") + f"""
// ============================================================================
// WHAT THIS CASE COMPUTES, AND WHAT IT CANNOT
//
// This is a FROZEN ROTOR (MRF): the mesh never moves. Inside the cellZone named
// below the momentum equation carries the rotating-frame source terms and the
// flux is relative to a frame turning at omega, so the impeller is held at one
// azimuthal position and the flow around it is solved steady.
//
// It cannot tell you:
//   * anything about BLADE PASSING. The blade-passing tone at
//     {N_BLADES} blades * {ROTOR_RPM:g} rpm / 60 = {blade_passing_hz():.6g} Hz is absent from this
//     answer, not small in it, and so is every quantity that is a function of
//     blade position -- unsteady casing pressure, tonal noise, fatigue cycles.
//   * anything about the UNSTEADY WAKE. The wake behind each blade is steady in
//     the rotating frame. Vortex shedding off the blade, rotating stall and
//     wake/baffle interaction do not appear. A steady run cannot be converged
//     into an unsteady answer.
//   * more than ONE BLADE POSITION. Where the vessel is azimuthally asymmetric
//     this is the answer for the angle it was meshed at; the honest error bar is
//     the spread over several positions, which is several runs.
// For any of those, the case has to be rebuilt as a sliding mesh: a cyclicAMI
// pair through the zone interface and pimpleFoam, transient, one to three orders
// of magnitude more expensive. Its own error scales with omega*dt rather than
// with the cell count.
//
// The check that this frozen rotor is being read honestly: move the zone radius
// by about a quarter and re-run. The zone face is a surface where nothing
// physical happens, so a correct torque moves by no more than a few per cent. A
// torque that tracks the zone radius is a number set by a modelling choice.
// ============================================================================

// omega is in RADIANS PER SECOND. 1 rpm = 2*pi/60 = 0.104719755 rad/s and
// 1 rad/s = 9.549296586 rpm, so a value meant as rpm and written here runs 9.55
// times too fast and nothing in the log says so.
//
//   {ROTOR_RPM:g} rpm * 2*pi/60 = {omega:.9g} rad/s
//   back again: {omega:.9g} * 60/(2*pi) = {rpm_from_rad_per_s(omega):.9g} rpm
//   blade tip speed: omega * {BLADE_RADIUS_M:.6g} m = {tip_speed_m_s():.6g} m/s

MRF1
{{
    cellZone    rotorZone;      // a physical VOLUME in the msh becomes this cellZone
    active      yes;

    // Patches whose faces lie inside the cellZone but which do NOT turn with it --
    // a stationary baffle, a shaft seal, a fixed shroud. MRF corrects the velocity
    // of every patch in the zone to Omega x r unless it is named here, so a
    // stationary patch left out of this list is silently stirred. This vessel has
    // none: the only wall inside the zone is the impeller, which does turn.
    nonRotatingPatches ();

    origin    (0 0 0);
    axis      (0 0 1);          // the z axis, which is the one-cell direction here
    omega     {omega:.9g};      // rad/s == {ROTOR_RPM:g} rpm
}}
""")


def transport_properties() -> str:
    """Water at 20 degrees C, with its source, because a Reynolds number is only as
    good as its viscosity.

    nu = mu/rho = 1.002e-03 / 998.2 = 1.004e-06 m^2/s -- White, *Fluid Mechanics*,
    Table A.1. The same number `free_surface.py` uses for the same fluid, from the
    same table, rather than two templates quietly disagreeing about water.
    """
    return (header("dictionary", "transportProperties", "constant") + f"""
transportModel  Newtonian;

nu              {NU_M2_S:g};   // m^2/s -- water at 20 C, mu 1.002e-03 Pa.s / rho
""")


def turbulence_properties() -> str:
    """kEpsilon RAS, stated rather than defaulted.

    kEpsilon rather than kOmegaSST because the attested MRF tutorial is kEpsilon, so
    the schemes, the relaxation factors and the wall functions below have a shipped
    case behind them instead of being reasoned out from scratch. ESI OpenFOAM reads
    this file as `turbulenceProperties`; the openfoam.org line calls the same thing
    `momentumTransport`.
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
    """`simpleFoam`, and the torque the case exists to produce.

    `endTime` counts SIMPLE iterations, not seconds: `simpleFoam` is a steady solver
    and its `deltaT` is an iteration counter. Reporting a Courant number for a run of
    this case would be reporting a number that does not exist.

    The `forces` function object is here because a rotating-machinery case with no
    torque output has delivered nothing. Its z-moment about the axis is the shaft
    torque. `rho rhoInf` with `rhoInf` stated is not optional: incompressible OpenFOAM
    solves p/rho in m2/s2, so a moment taken off that field without the density is out
    by a factor of 998 and reads as a plausible small number.

    `startFrom latestTime` because the hosted container's lifetime is 24 h and it can
    be reclaimed sooner, so an interrupted solve resumes from its last write with one
    relaunch -- see "Long solves" in `../notes/openfoam-field-notes.md`.
    """
    return (header("dictionary", "controlDict", "system") + f"""
application     simpleFoam;

startFrom       latestTime;     // an interrupted solve resumes, it does not restart
startTime       0;
stopAt          endTime;
endTime         2000;           // SIMPLE ITERATIONS, not seconds -- this is steady

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
{{
    impellerTorque
    {{
        type            forces;
        libs            ("libforces.so");
        patches         (rotor);

        // p is kinematic here (m2/s2), so the density has to be supplied or every
        // force and moment comes out a factor of rho too small.
        rho             rhoInf;
        rhoInf          {RHO_KG_M3:g};       // kg/m^3 -- water at 20 C

        CofR            (0 0 0);        // the rotation axis passes through the origin
        writeControl    timeStep;
        writeInterval   50;
    }}
}}
""")


def fv_schemes() -> str:
    """Steady-state ddt and `bounded` convection, which is the pair that matters.

    `bounded Gauss` rather than plain `Gauss`: a `steadyState` discretisation drops the
    `-div(phi)*field` term that a transient one carries, and without it `k` and
    `epsilon` can go negative during the first iterations, when the flux field is
    nowhere near divergence-free. The `bounded` prefix puts that term back. The shipped
    mixerVessel2D uses the same prefix on the same three terms.

    `limitedLinearV 1` on momentum: the V form limits on the vector field as a whole
    rather than component by component, which is what keeps a swirling flow from being
    limited differently in x and y and acquiring a preferred direction from the
    discretisation. `upwind` on k and epsilon, because a bounded first-order scheme on
    a quantity that must stay positive is a better trade than a limited second-order
    one.
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

    div(phi,U)      bounded Gauss limitedLinearV 1;

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
    """SIMPLE, with the pressure reference a closed vessel cannot do without.

    `pRefCell` and `pRefValue` are the entries this file exists to not forget. There is
    no inlet and no outlet in this case -- every patch is a wall or `empty` -- so the
    incompressible pressure equation is singular up to a constant. Without a reference
    the level drifts without bound while the velocity field stays perfectly healthy,
    and the only symptom is a pressure plot nobody can read.

    The relaxation factors are the shipped mixerVessel2D's: p at 0.3 and the equations
    at 0.5. A frozen rotor's source terms are stiff and a steady solve of one diverges
    at the relaxation an external-aerodynamics case runs happily at.
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
    nNonOrthogonalCorrectors 0;

    // A CLOSED domain: no inlet, no outlet, so the pressure equation is singular
    // up to a constant and the level has to be pinned somewhere. Without these two
    // the pressure drifts without bound and the velocity field looks fine.
    pRefCell        0;
    pRefValue       0;

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
        U               0.5;
        k               0.5;
        epsilon         0.5;
    }
}
""")


CASE_FILES = {
    "0/U": u_field,
    "0/p": p_field,
    "0/k": k_field,
    "0/epsilon": epsilon_field,
    "0/nut": nut_field,
    "constant/MRFProperties": mrf_properties,
    "constant/transportProperties": transport_properties,
    "constant/turbulenceProperties": turbulence_properties,
    "system/controlDict": control_dict,
    "system/fvSchemes": fv_schemes,
    "system/fvSolution": fv_solution,
}
"""Every file `simpleFoam` with kEpsilon and an MRF zone reads, and nothing it does
not. There is no `constant/dynamicMeshDict` and no `0/pointDisplacement`: the mesh does
not move, which is the whole of what MRF is."""


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
    """The rotation stated in both units, with the round trip, the tip speed and the
    tone this case cannot produce.

    Printed rather than merely written into the dictionary because the rpm/rad/s
    mistake is made by a reader who never opens `constant/MRFProperties`, and because
    a conversion that does not come back to where it started is the cheapest check on
    a unit factor there is.
    """
    omega = omega_rad_per_s()
    back = rpm_from_rad_per_s(omega)
    return "\n".join([
        "rotation, in both units, because rad/s against rpm is a factor of 9.5493:",
        f"    ROTOR_RPM          {ROTOR_RPM:.6g} rpm      (what you edited, nameplate units)",
        f"    omega              {omega:.9g} rad/s    (what constant/MRFProperties carries)",
        f"    round trip         {back:.9g} rpm      (rad/s * 60/(2*pi); must match the first line)",
        f"    blade tip speed    {tip_speed_m_s():.6g} m/s      (omega * {BLADE_RADIUS_M:.6g} m)",
        "",
        "what this frozen rotor cannot tell you:",
        f"    no blade passing   the {blade_passing_hz():.6g} Hz tone "
        f"({N_BLADES} blades * {ROTOR_RPM:.6g} rpm / 60) is absent, not small",
        "    no unsteady wake   shedding, rotating stall and wake/baffle interaction",
        "    one blade angle    the spread over several positions is the error bar",
        "    for any of those, a cyclicAMI sliding mesh and pimpleFoam, transient",
        "",
        "the honest check on the result, from ladder.py's frozen-rotor rung:",
        f"    move ZONE_FRACTION from {ZONE_FRACTION} by about a quarter and re-run. the zone",
        "    face is a surface where nothing physical happens, so the torque should move",
        "    by no more than a few per cent. one that tracks the zone radius is a number",
        "    set by a modelling choice rather than by the flow.",
    ])


def main() -> None:
    case = Path.cwd()
    msh = case / "rotating.msh"
    build_mesh(msh)
    write_case(case)

    run(["gmshToFoam", msh.name])
    retype("rotor", "wall")
    retype("stator", "wall")
    # One physical group covers both z faces, so one retype makes the case 2D.
    retype("frontAndBack", "empty")

    run(["checkMesh", "-case", "."])

    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png",
         "--json", "look.json"])

    print()
    print(summary())

    # What is on PATH is a question rather than a fact: MRF in this OpenFOAM is a
    # feature of the ordinary solvers and not a binary of its own, and on this image
    # `which MRFSimpleFoam simpleFoam` answered with simpleFoam alone.
    print()
    if shutil.which("simpleFoam"):
        print("simpleFoam is on PATH. there is no MRFSimpleFoam in this OpenFOAM and")
        print("there does not need to be: MRF is an MRFProperties feature of the")
        print("ordinary solvers. to confirm the dictionary keys on this image:")
    else:
        print("simpleFoam is NOT on PATH, so the case above cannot be solved here.")
        print("to confirm what this image does have:")
    print("    foamInfo MRFZone")
    print("    ls $FOAM_TUTORIALS/incompressible/simpleFoam/mixerVessel2D/constant")

    # Not launched from here: what a solve costs, and whether it is wanted at all, is
    # not a template's call. `startFrom latestTime` is already set, so this loop
    # resumes itself through any in-sandbox failure -- see "Long solves" in
    # ../notes/openfoam-field-notes.md.
    print("\nthe case is complete and meant to be solved. one way to launch it:\n"
          "    until simpleFoam >> log.simpleFoam 2>&1; do "
          "echo resuming >> log.simpleFoam; sleep 5; done\n"
          "the torque is the z moment in postProcessing/impellerTorque/*/moment.dat")


if __name__ == "__main__":
    main()
