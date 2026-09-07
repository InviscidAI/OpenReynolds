#!/usr/bin/env python3
"""TEMPLATE: closed planar outline -> gmsh quads -> extrude -> gmshToFoam -> checkMesh.

Copy this file into an empty case directory and run it there:

    python3 duct2d.py

It builds a straight rectangular duct LENGTH_M x HEIGHT_M x THICK_M, meshes it in
quads, extrudes one cell into hexahedra, converts to a polyMesh with `gmshToFoam`,
retypes the walls and the two z faces, runs `checkMesh`, and finishes by calling
`mesh_look.py` so the result is looked at, not just built. Edit LENGTH_M and
HEIGHT_M and re-run; CELL_SIZE_M and THICK_M follow from HEIGHT_M so the cells stay
roughly cubic. Everything from `gmsh.model.geo.synchronize()` on does not care what
the outline was, so for a shape that is not a plain box, replace the four
`addPoint`/`addLine` calls below with your own outline (and its own patch names) and
leave the rest alone.

This is the "2D" recipe from `../notes/openfoam-field-notes.md` (see "Keeping a
study 2D"): OpenFOAM has no 2D mesh, only a 3D mesh one cell thick with the two faces
normal to the thickness typed `empty`. The common way a "2D" case is actually a slow,
thin 3D one is a mesh built several cells deep, or a side patch left `patch`/`wall`
where `empty` was meant.

Known gotchas, baked in below rather than left to be rediscovered by failing:

* `Mesh.Algorithm 8` (Frontal-Delaunay for quads) plus `Mesh.RecombinationAlgorithm 3`
  (Blossom, full-quad) and `setRecombine` on the surface -- without all three the 2D
  mesh comes out triangulated and the extrude below produces prisms, not hexahedra.
* `extrude(..., numElements=[1], recombine=True)` -- one layer, swept into hexahedra
  rather than left as a solid gmsh would otherwise mesh with tets.
* `gmshToFoam` reads only msh format 2.2 ASCII; `Mesh.MshFileVersion` is set to 2.2
  right before `gmsh.write` for exactly that reason.
* A physical group is named per edge **as the line is created** (`inlet`, `outlet`,
  `wall_bottom`, `wall_top` below), never worked out afterwards from where a face
  happens to sit in the bounding box -- that scheme silently mislabels every L-bend,
  U-bend or elbow, and says nothing when it does.
* `gmshToFoam` types every patch it creates `patch`, so the walls are retyped `wall`
  and the two z faces `empty` with `foamDictionary` after conversion -- OpenFOAM never
  infers `empty` from the geometry, only from what the boundary file says.
* Metres, always. OpenFOAM has no units: it reads whatever numbers are in the mesh as
  metres, so a duct authored in millimetres and never scaled becomes a mesh a
  thousand times too big, at a thousandth of the intended Reynolds number, and
  `checkMesh` is perfectly happy to pass it. Divide a millimetre drawing by 1000
  before the numbers go in below.
* `gmshToFoam` (like every OpenFOAM utility) builds a `Foam::Time` before it touches
  the mesh, so it fails with "cannot find file .../system/controlDict" in a directory
  that has nothing but a `.msh` in it -- there is no mesh-only exemption. Measured on
  the instance: `gmshToFoam` needs only `controlDict`, but `checkMesh` goes one step
  further and builds an `fvMesh` (its non-orthogonality and skewness metrics are
  computed with the interpolation weights `fvSchemes` sets), so it also wants
  `system/fvSchemes` -- "cannot find file .../system/fvSchemes" otherwise -- and
  `fvSolution` follows for free since nothing here reads it either. Minimal stubs of
  all three are written below before `gmshToFoam` runs; they are not a real case,
  only enough for a mesh to be converted and checked, and are overwritten by whatever
  `case_gen.py` (or you) writes once there is something to solve.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
LENGTH_M = 0.10   # duct length, x
HEIGHT_M = 0.02   # duct height, y
# --------------------------------------------------------------------------

CELL_SIZE_M = HEIGHT_M / 10.0   # background mesh size -- ~10 cells across the height
THICK_M = CELL_SIZE_M           # extrude depth = one cell, kept roughly cubic


def build_mesh(msh_path: Path) -> None:
    """The outline, the quad mesh, the extrude, the msh file -- everything gmsh does."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("duct2d")

    p1 = gmsh.model.geo.addPoint(0, 0, 0, CELL_SIZE_M)
    p2 = gmsh.model.geo.addPoint(LENGTH_M, 0, 0, CELL_SIZE_M)
    p3 = gmsh.model.geo.addPoint(LENGTH_M, HEIGHT_M, 0, CELL_SIZE_M)
    p4 = gmsh.model.geo.addPoint(0, HEIGHT_M, 0, CELL_SIZE_M)

    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_outlet = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_inlet = gmsh.model.geo.addLine(p4, p1)

    loop = gmsh.model.geo.addCurveLoop([l_bottom, l_outlet, l_top, l_inlet])
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
    # the volume, then one lateral surface per boundary curve of the surface, in the
    # same order the curve loop was built in -- [bottom, outlet, top, inlet] here.
    # Naming from this order (not from where a face sits afterwards) is the point.
    top_surf, volume = out[0][1], out[1][1]
    side_bottom, side_outlet, side_top, side_inlet = (entity[1] for entity in out[2:6])

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(2, [side_outlet], name="outlet")
    gmsh.model.addPhysicalGroup(2, [side_bottom], name="wall_bottom")
    gmsh.model.addPhysicalGroup(2, [side_top], name="wall_top")
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
    `-entry wall_bottom.type` alone fails with "not found in dictionary"; the patch
    name has to be reached through it, `entry0.<patch>.type`.
    """
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

# checkMesh builds an fvMesh, not just a polyMesh, so it wants these two as well --
# measured on the instance: `gmshToFoam` is satisfied by controlDict alone, but
# `checkMesh` fails on a missing fvSchemes right after it. Nothing here is read by a
# mesh conversion or a mesh check; it only has to parse.
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
    """`gmshToFoam` and `checkMesh` both build a `Foam::Time` before they read the
    mesh, so a directory with nothing but a `.msh` in it fails "cannot find file
    .../system/controlDict" -- there is no mesh-only exemption, and `checkMesh` goes
    on to want `fvSchemes` too (it builds an `fvMesh`) with `fvSolution` following for
    free. These are minimal stubs, not a real case: enough for a mesh to be converted
    and checked, and overwritten by whatever `case_gen.py` (or you) writes once there
    is something to solve.
    """
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


def main() -> None:
    case = Path.cwd()
    msh = case / "duct2d.msh"
    build_mesh(msh)
    write_system_stubs(case)

    run(["gmshToFoam", msh.name])
    for patch in ("wall_bottom", "wall_top"):
        retype(patch, "wall")
    # One physical group covers both z faces, so one retype makes the case 2D.
    retype("frontAndBack", "empty")

    run(["checkMesh", "-case", "."])
    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png", "--json", "look.json"])


if __name__ == "__main__":
    main()
