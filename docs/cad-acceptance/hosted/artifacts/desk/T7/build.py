"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess
print(subprocess.run(["python3", "/work/.toolbox/cad_convert.py", "--help"], capture_output=True, text=True).stdout)
print(subprocess.run(["python3", "/work/.toolbox/cad_audit.py", "--help"], capture_output=True, text=True).stdout)
print(subprocess.run(["python3", "/work/.toolbox/mesh_look.py", "--help"], capture_output=True, text=True).stdout)
print(subprocess.run(["python3", "/work/.toolbox/domain_probe.py", "--help"], capture_output=True, text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
# Given the geometry is a plain rectangular box with two internal cell zones (not separate solids), th
from build123d import *
import os

CASE_DIR = "/work/accept-t7"
os.chdir(CASE_DIR)

# Cavity dimensions, metres (spec given in mm -> convert here, once)
LX_M, LY_M, LZ_M = 0.100, 0.060, 0.040

# Box built with one corner at the origin: x in [0,LX], y in [0,LY], z in [0,LZ]
# z is the vertical axis: z=0 floor, z=LZ ceiling.
cavity = Pos(LX_M / 2, LY_M / 2, LZ_M / 2) * Box(LX_M, LY_M, LZ_M)

bb = cavity.bounding_box()
print("bounds:", bb.min, bb.max)
print("volume (m^3):", cavity.volume, "expected:", LX_M * LY_M * LZ_M)
print("num faces:", len(cavity.faces()))

# -- cell 3 -------------------------------------------------------------------------
import subprocess
# Inspect cad_convert.py to see how export_patches is invoked (module import vs CLI)
print(subprocess.run(["sed", "-n", "1,60p", "/work/.toolbox/cad_convert.py"], capture_output=True, text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
import subprocess
print(subprocess.run(["grep", "-n", "def export_patches", "-A", "40", "/work/.toolbox/cad_convert.py"], capture_output=True, text=True).stdout)

# -- cell 5 -------------------------------------------------------------------------
import subprocess
print(subprocess.run(["cat", "/work/.toolbox/templates/prep/README.md"], capture_output=True, text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
import sys
sys.path.insert(0, "/work/.toolbox")
import cad_convert

# All six faces of the sealed cavity are the same BC type (adiabatic wall) --
# one patch group "walls" covering every face, named where the surface is made.
export_step(cavity, "cavity.step", unit=Unit.M)

faces = cavity.faces()
key = lambda fs: [tuple(f.center(CenterOf.MASS)) for f in fs]

# clmax chosen to follow the blockMesh cell size we will use (5 mm cubes):
# ~0.5 * dx_surface = 0.0025 m
CLMAX_M = 0.0025

report = cad_convert.export_patches(
    "cavity.step",
    {"walls": {"role": "wall", "faces": key(faces)}},
    "constant/triSurface",
    clmax=CLMAX_M,
    location_in_mesh=[0.05, 0.03, 0.02],  # geometric centre of the cavity, all fluid
)
print(cad_convert.render(report))

# -- cell 7 -------------------------------------------------------------------------
import subprocess
print(subprocess.run(["python3", "/work/.toolbox/cad_audit.py", "constant/triSurface"], capture_output=True, text=True).stdout)

# -- cell 8 -------------------------------------------------------------------------
# Surface export is closed, manifold, watertight, and disjoint — good record of the geometry. Now let'
import subprocess
r = subprocess.run(["python3", "/work/.toolbox/domain_probe.py", "constant/triSurface",
                     "--point", "0.05", "0.03", "0.02"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)

# -- cell 9 -------------------------------------------------------------------------
import os

os.makedirs("system", exist_ok=True)
os.makedirs("constant", exist_ok=True)

LX_M, LY_M, LZ_M = 0.100, 0.060, 0.040
DX_M = 0.005  # 5 mm cells, coarse first pass
NX, NY, NZ = round(LX_M / DX_M), round(LY_M / DX_M), round(LZ_M / DX_M)
print("cells:", NX, NY, NZ, "total", NX * NY * NZ)

blockMeshDict = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1.0;

vertices
(
    (0        0        0       )   // 0
    ({LX_M}   0        0       )   // 1
    ({LX_M}   {LY_M}   0       )   // 2
    (0        {LY_M}   0       )   // 3
    (0        0        {LZ_M} )   // 4
    ({LX_M}   0        {LZ_M} )   // 5
    ({LX_M}   {LY_M}   {LZ_M} )   // 6
    (0        {LY_M}   {LZ_M} )   // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1)
);

edges ();

// Sealed cavity: every boundary is the same adiabatic-wall type, so one patch
// group "walls" carries all six faces of the box.
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 3 2 1)   // z = 0    floor
            (4 5 6 7)   // z = LZ   ceiling
            (0 1 5 4)   // y = 0
            (3 7 6 2)   // y = LY
            (0 4 7 3)   // x = 0
            (1 2 6 5)   // x = LX
        );
    }}
);

mergePatchPairs ();
"""

with open("system/blockMeshDict", "w") as f:
    f.write(blockMeshDict)

print(blockMeshDict)
