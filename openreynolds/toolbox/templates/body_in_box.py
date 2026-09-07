#!/usr/bin/env python3
"""TEMPLATE: gmsh-OCC primitives + booleans -> body-fitted volume mesh -> gmshToFoam
-> checkMesh.

Copy this file into an empty case directory and run it there:

    python3 body_in_box.py

It builds a sphere of radius BODY_RADIUS_M inside a rectangular flow box sized off
that radius, cuts the sphere out of the box (an OpenCASCADE boolean), meshes the
remaining volume body-fitted with tetrahedra -- no STL, no snappyHexMesh -- converts
with `gmshToFoam`, retypes the body wall, runs `checkMesh`, and finishes with
`mesh_look.py`. Edit BODY_RADIUS_M and CELL_SIZE_M and re-run. Swap `addSphere` for
`addBox`, `addCylinder`, or a `gmsh.model.occ.cut`/`fuse` of several primitives to
change the shape without touching anything from the boolean on.

How the flow box is sized and its faces named, and the other gotchas baked in below:

* The box runs UPSTREAM_R body-radii ahead of the body, DOWNSTREAM_R behind it (a
  wake needs more room than an approach), and SIDE_R to each side, above and below --
  all in multiples of the body radius, which is the rule of thumb in
  `../notes/openfoam-field-notes.md` for how far a farfield boundary has to sit
  before it starts shaping the answer it was meant to measure.
* Faces are named **right after the boolean, from the numbers the box was built
  with** (`getEntitiesInBoundingBox` against the box's own known corners) -- never by
  looking at the finished mesh and guessing which face is which. The same rule as the
  2D template's physical groups, applied to a box instead of an outline. Whatever
  boundary faces are left over once the six box faces are accounted for are the body,
  named as a group rather than assumed to be exactly one face.
* `gmsh.model.occ.cut` (not `fuse` or `fragment`) so the sphere becomes a hole: the
  mesh fills the box minus the body, which is the fluid region.
* Body-fitted tetrahedra straight from the OCC kernel -- no STL export, no
  snappyHexMesh octree refinement rediscovering edges the CAD already has exactly.
* `gmshToFoam` reads only msh format 2.2 ASCII; `Mesh.MshFileVersion` is set to 2.2
  right before `gmsh.write` for exactly that reason.
* `gmshToFoam` types every patch it creates `patch`; only the body needs retyping to
  `wall` so a wall function reads it -- the box faces can stay `patch` and take
  whatever boundary condition the case actually needs (`inletOutlet`,
  `freestreamPressure`, a fixed velocity, ...).
* Metres, always. A body radius written in millimetres and never divided by 1000
  gives `checkMesh` a mesh a thousand times too big, and it will not complain.
* `gmshToFoam` (like every OpenFOAM utility) builds a `Foam::Time` before it touches
  the mesh, so it fails with "cannot find file .../system/controlDict" in a directory
  that has nothing but a `.msh` in it. Measured on the instance: `gmshToFoam` needs
  only `controlDict`, but `checkMesh` goes one step further and builds an `fvMesh`
  (its quality metrics are computed with the interpolation weights `fvSchemes`
  sets), so it also wants `system/fvSchemes` -- and `fvSolution` follows for free
  since nothing here reads it either. Minimal stubs of all three are written below
  before `gmshToFoam` runs; they are not a real case, only enough for a mesh to be
  converted and checked.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# -- edit these two, then re-run -----------------------------------------------
BODY_RADIUS_M = 0.02   # sphere radius, m
CELL_SIZE_M = 0.01     # background mesh size, m -- refine by lowering this
# --------------------------------------------------------------------------

# The flow box, in multiples of the body radius -- see the module docstring.
UPSTREAM_R, DOWNSTREAM_R, SIDE_R = 5.0, 10.0, 5.0


def build_mesh(msh_path: Path) -> None:
    """The body, the box, the boolean, the named faces, the tet mesh -- all of it."""
    import gmsh

    r = BODY_RADIUS_M
    x0, x1 = -UPSTREAM_R * r, DOWNSTREAM_R * r
    y0, y1 = -SIDE_R * r, SIDE_R * r
    z0, z1 = -SIDE_R * r, SIDE_R * r

    gmsh.initialize()
    gmsh.model.add("body_in_box")

    body = gmsh.model.occ.addSphere(0, 0, 0, r)
    box = gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
    gmsh.model.occ.synchronize()

    cut, _ = gmsh.model.occ.cut([(3, box)], [(3, body)])
    gmsh.model.occ.synchronize()
    volume = cut[0][1]
    faces = gmsh.model.getBoundary([(3, volume)], oriented=False)

    eps = 1e-6 + 0.001 * max(x1 - x0, y1 - y0, z1 - z0)

    def box_face(lo: tuple, hi: tuple) -> list:
        found = gmsh.model.getEntitiesInBoundingBox(
            lo[0] - eps, lo[1] - eps, lo[2] - eps,
            hi[0] + eps, hi[1] + eps, hi[2] + eps, dim=2)
        return [tag for _, tag in found if (2, tag) in faces]

    named = {
        "inlet": box_face((x0, y0, z0), (x0, y1, z1)),
        "outlet": box_face((x1, y0, z0), (x1, y1, z1)),
        "side_ymin": box_face((x0, y0, z0), (x1, y0, z1)),
        "side_ymax": box_face((x0, y1, z0), (x1, y1, z1)),
        "side_zmin": box_face((x0, y0, z0), (x1, y1, z0)),
        "side_zmax": box_face((x0, y0, z1), (x1, y1, z1)),
    }
    on_box = {tag for tags in named.values() for tag in tags}
    body_faces = [tag for _, tag in faces if tag not in on_box]

    for name, tags in named.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(2, body_faces, name="body")
    gmsh.model.addPhysicalGroup(3, [volume], name="internal")

    gmsh.option.setNumber("Mesh.MeshSizeMin", CELL_SIZE_M * 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", CELL_SIZE_M)
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


def retype(patch: str, kind: str) -> None:
    """`gmshToFoam` types every patch `patch`; this is how it gets its real type.

    `constant/polyMesh/boundary` is a bare list, not a named dictionary, so
    `foamDictionary` addresses the whole thing as one pseudo-entry called `entry0`
    (its own error message says so: "Known entries ... : 2(FoamFile entry0)") --
    `-entry body.type` alone fails with "not found in dictionary"; the patch name has
    to be reached through it, `entry0.<patch>.type`.
    """
    run(["foamDictionary", "constant/polyMesh/boundary",
         "-entry", f"entry0.{patch}.type", "-set", kind])


def find_mesh_look() -> Path:
    """The deployed toolbox first, then this file's own siblings for local testing."""
    deployed = Path("/work/.toolbox/mesh_look.py")
    if deployed.exists():
        return deployed
    return Path(__file__).resolve().parent.parent / "mesh_look.py"


def main() -> None:
    case = Path.cwd()
    msh = case / "body_in_box.msh"
    build_mesh(msh)
    write_system_stubs(case)

    run(["gmshToFoam", msh.name])
    retype("body", "wall")

    run(["checkMesh", "-case", "."])
    run([sys.executable, str(find_mesh_look()), ".", "--out", "look.png", "--json", "look.json"])


if __name__ == "__main__":
    main()
