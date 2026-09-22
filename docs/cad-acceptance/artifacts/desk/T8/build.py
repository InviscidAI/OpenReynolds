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
print(subprocess.run(["ls", "-la"], cwd="/work/accept-t8", capture_output=True, text=True).stdout)
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
print(subprocess.run(["ls", toolbox], capture_output=True, text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
import subprocess
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
print(subprocess.run(["find", toolbox+"templates", "-maxdepth", "3"], capture_output=True, text=True).stdout)

# -- cell 3 -------------------------------------------------------------------------
print(open(toolbox+"templates/snappy/README.md").read())

# -- cell 4 -------------------------------------------------------------------------
import subprocess
print(open(toolbox+"templates/prep/README.md").read())

# -- cell 5 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-rl", "-i", "cellzone\|splitMeshRegions\|multiregion\|multi-region\|chtMultiRegion", toolbox],
                    capture_output=True, text=True)
print(r.stdout)
print(r.stderr)

# -- cell 6 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-n", "-i", "cellzone\\|splitMeshRegions\\|multiregion\\|multi-region\\|chtMultiRegion\\|region",
                     toolbox+"templates/snappy/snappyHexMeshDict"], capture_output=True, text=True)
print(r.stdout)

# -- cell 7 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-n", "-i", "cellzone\\|splitMeshRegions\\|chtMultiRegion\\|conjugate\\|region",
                     toolbox+"notes/openfoam-agent-architecture.md"], capture_output=True, text=True)
print(r.stdout)
r2 = subprocess.run(["grep", "-n", "-i", "cellzone\\|splitMeshRegions\\|chtMultiRegion\\|conjugate\\|region",
                     toolbox+"notes/openfoam-field-notes.md"], capture_output=True, text=True)
print(r2.stdout)

# -- cell 8 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["python3", toolbox+"cad_convert.py", "--help"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)

# -- cell 9 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["python3", "-c", "import sys; sys.path.insert(0,'.'); import inspect; sys.path.insert(0,'/tmp/x')"], capture_output=True, text=True)
import sys
sys.path.insert(0, toolbox)
import cad_convert, inspect
print(inspect.signature(cad_convert.export_patches))
print(cad_convert.export_patches.__doc__)

# -- cell 10 ------------------------------------------------------------------------
print(cad_convert.PATCH_ROLES if hasattr(cad_convert, "PATCH_ROLES") else "no attr")
import subprocess
r = subprocess.run(["grep", "-n", "ROLE\|role", toolbox+"cad_convert.py"], capture_output=True, text=True)
print(r.stdout[:3000])

# -- cell 11 ------------------------------------------------------------------------
print(open(toolbox+"templates/snappy/snappyHexMeshDict").read())

# -- cell 12 ------------------------------------------------------------------------
from build123d import *
import numpy as np

# All dimensions in METRES from the start (30 mm square duct etc. -> 0.030 m).
DUCT_L = 0.200   # duct length, x
DUCT_W = 0.030   # duct width, y
DUCT_H = 0.030   # duct height, z (fluid occupies z in [0, DUCT_H])

BLOCK_L = 0.040  # copper block length, x
BLOCK_W = 0.030  # copper block width, y (== duct width: flush wall to wall)
BLOCK_T = 0.010  # copper block thickness, z (sits below the floor, z in [-BLOCK_T, 0])

# block centred halfway along the duct
BLOCK_X0 = DUCT_L/2 - BLOCK_L/2
BLOCK_X1 = DUCT_L/2 + BLOCK_L/2

fluid_box = Pos(DUCT_L/2, DUCT_W/2, DUCT_H/2) * Box(DUCT_L, DUCT_W, DUCT_H)
solid_box = Pos((BLOCK_X0+BLOCK_X1)/2, DUCT_W/2, -BLOCK_T/2) * Box(BLOCK_L, BLOCK_W, BLOCK_T)

print("fluid bbox:", fluid_box.bounding_box())
print("solid bbox:", solid_box.bounding_box())
print("fluid volume (m^3):", fluid_box.volume, " expected:", DUCT_L*DUCT_W*DUCT_H)
print("solid volume (m^3):", solid_box.volume, " expected:", BLOCK_L*BLOCK_W*BLOCK_T)

assembly = fluid_box + solid_box
print("assembly solids:", len(assembly.solids()))
print("assembly bbox:", assembly.bounding_box())

# -- cell 13 ------------------------------------------------------------------------
import os, numpy as np

CASE = "/work/accept-t8"
os.makedirs(f"{CASE}/system", exist_ok=True)
os.makedirs(f"{CASE}/constant", exist_ok=True)

# geometry parameters, metres (same numbers as the build123d model above)
DUCT_L, DUCT_W, DUCT_H = 0.200, 0.030, 0.030
BLOCK_L, BLOCK_W, BLOCK_T = 0.040, 0.030, 0.010
BLOCK_X0 = DUCT_L/2 - BLOCK_L/2
BLOCK_X1 = DUCT_L/2 + BLOCK_L/2

X0, X1, X2, X3 = 0.0, BLOCK_X0, BLOCK_X1, DUCT_L
Y0, Y1 = 0.0, DUCT_W
ZS0, ZS1, ZF1 = -BLOCK_T, 0.0, DUCT_H

def gen_blockMeshDict(dx, path):
    """Rectilinear, block-structured mesh: 3 fluid blocks (up/mid/down duct)
    plus 1 solid block (copper) sharing the mid-block floor exactly -> a
    literal, point-for-point conformal interface, no snapping involved."""
    verts = {}
    vlist = []
    def vid(p):
        key = tuple(round(c, 9) for c in p)
        if key not in verts:
            verts[key] = len(vlist)
            vlist.append(key)
        return verts[key]

    def hex_block(xlo, xhi, ylo, yhi, zlo, zhi, n, zone):
        pts = [(xlo,ylo,zlo),(xhi,ylo,zlo),(xhi,yhi,zlo),(xlo,yhi,zlo),
               (xlo,ylo,zhi),(xhi,ylo,zhi),(xhi,yhi,zhi),(xlo,yhi,zhi)]
        ids = [vid(p) for p in pts]
        return {"ids": ids, "pts": pts, "n": n, "zone": zone}

    def n_for(length):
        n = round(length/dx)
        assert abs(n*dx - length) < 1e-9, (length, dx, n)
        return max(int(n), 1)

    nxA, nxB, nxC = n_for(X1-X0), n_for(X2-X1), n_for(X3-X2)
    ny = n_for(Y1-Y0)
    nzF = n_for(ZF1-ZS1)
    nzS = n_for(ZS1-ZS0)

    blockA = hex_block(X0, X1, Y0, Y1, ZS1, ZF1, (nxA, ny, nzF), "fluid")   # upstream
    blockB = hex_block(X1, X2, Y0, Y1, ZS1, ZF1, (nxB, ny, nzF), "fluid")   # middle, over block
    blockC = hex_block(X2, X3, Y0, Y1, ZS1, ZF1, (nxC, ny, nzF), "fluid")   # downstream
    blockD = hex_block(X1, X2, Y0, Y1, ZS0, ZS1, (nxB, ny, nzS), "solid")   # copper block
    blocks = [blockA, blockB, blockC, blockD]

    def face_ids(b, axis, side):
        # axis in 'x','y','z'; side in '-','+' ; returns 4 vertex ids, unordered
        p = b["pts"]; ids = b["ids"]
        idxmap = {
            ('x','-'): [0,3,7,4], ('x','+'): [1,2,6,5],
            ('y','-'): [0,1,5,4], ('y','+'): [3,2,6,7],
            ('z','-'): [0,1,2,3], ('z','+'): [4,5,6,7],
        }
        sel = idxmap[(axis, side)]
        return [ids[i] for i in sel], [p[i] for i in sel]

    def oriented(ids, pts, outward):
        v0,v1,v2 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
        n = np.cross(v1-v0, v2-v0)
        if np.dot(n, outward) < 0:
            return [ids[0], ids[3], ids[2], ids[1]]
        return ids

    def patch_face(b, axis, side):
        ids, pts = face_ids(b, axis, side)
        outward = {'x':(1,0,0),'y':(0,1,0),'z':(0,0,1)}[axis]
        outward = tuple(o*(1 if side=='+' else -1) for o in outward)
        return oriented(ids, pts, outward)

    patches = {
        "inlet":          [patch_face(blockA,'x','-')],
        "outlet":         [patch_face(blockC,'x','+')],
        "fluidTop":       [patch_face(b,'z','+') for b in (blockA,blockB,blockC)],
        "fluidSideY0":    [patch_face(b,'y','-') for b in (blockA,blockB,blockC)],
        "fluidSideY1":    [patch_face(b,'y','+') for b in (blockA,blockB,blockC)],
        "fluidFloor":     [patch_face(blockA,'z','-'), patch_face(blockC,'z','-')],
        "solidBottom":    [patch_face(blockD,'z','-')],
        "solidSideY0":    [patch_face(blockD,'y','-')],
        "solidSideY1":    [patch_face(blockD,'y','+')],
        "solidStepXNeg":  [patch_face(blockD,'x','-')],
        "solidStepXPos":  [patch_face(blockD,'x','+')],
    }
    # blockB z- <-> blockD z+ share identical vertices -> merged internal face,
    # left out of `boundary` on purpose (that IS the conformal interface).

    lines = []
    lines.append('FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n')
    lines.append('scale 1;\n')
    lines.append('vertices\n(\n')
    for p in vlist:
        lines.append(f'    ({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})\n')
    lines.append(');\n\n')
    lines.append('blocks\n(\n')
    for b in blocks:
        ids = " ".join(str(i) for i in b["ids"])
        n = b["n"]
        lines.append(f'    hex ({ids}) {b["zone"]} ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n')
    lines.append(');\n\n')
    lines.append('edges\n(\n);\n\n')
    lines.append('boundary\n(\n')
    for name, faces in patches.items():
        lines.append(f'    {name}\n    {{\n        type wall;\n        faces\n        (\n')
        for f in faces:
            lines.append(f'            ({f[0]} {f[1]} {f[2]} {f[3]})\n')
        lines.append('        );\n    }\n')
    lines.append(');\n\n')
    lines.append('mergePatchPairs\n(\n);\n')

    with open(path, "w") as fh:
        fh.writelines(lines)
    return dict(nxA=nxA, nxB=nxB, nxC=nxC, ny=ny, nzF=nzF, nzS=nzS,
                n_fluid_cells=(nxA+nxB+nxC-0)*ny*nzF, n_solid_cells=nxB*ny*nzS)

DX_COARSE = 0.010
info = gen_blockMeshDict(DX_COARSE, f"{CASE}/system/blockMeshDict")
print(info)

# -- cell 14 ------------------------------------------------------------------------
import subprocess, os

CASE = "/work/accept-t8"

controlDict = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application     foamRun;
startFrom        startTime;
startTime        0;
stopAt           endTime;
endTime          1;
deltaT           1;
writeControl     timeStep;
writeInterval    1;
purgeWrite       0;
writeFormat      ascii;
writePrecision   6;
writeCompression off;
timeFormat       general;
timePrecision    6;
runTimeModifiable true;
"""

fvSchemes = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes {}
gradSchemes {}
divSchemes {}
laplacianSchemes {}
interpolationSchemes {}
snGradSchemes {}
"""

fvSolution = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
"""

for name, content in [("controlDict", controlDict), ("fvSchemes", fvSchemes), ("fvSolution", fvSolution)]:
    with open(f"{CASE}/system/{name}", "w") as fh:
        fh.write(content)

r = subprocess.run(["blockMesh", "-case", CASE], capture_output=True, text=True)
print(r.stdout[-3000:])
print("STDERR", r.stderr[-2000:])
print("returncode", r.returncode)

# -- cell 15 ------------------------------------------------------------------------
import subprocess

CASE = "/work/accept-t8"
r = subprocess.run(["splitMeshRegions", "-cellZones", "-overwrite", "-case", CASE],
                    capture_output=True, text=True)
print(r.stdout[-5000:])
print("STDERR", r.stderr[-3000:])
print("returncode", r.returncode)

# -- cell 16 ------------------------------------------------------------------------
import subprocess

CASE = "/work/accept-t8"
r = subprocess.run(["checkMesh", "-case", CASE, "-allRegions"], capture_output=True, text=True)
print(r.stdout[-6000:])
print("STDERR", r.stderr[-2000:])
print("rc", r.returncode)

# -- cell 17 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["bash", "-lc", "find /work/accept-t8/constant -maxdepth 2 | sort"], capture_output=True, text=True)
print(r.stdout)

# -- cell 18 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["python3", toolbox+"mesh_look.py", "--help"], capture_output=True, text=True).stdout)

# -- cell 19 ------------------------------------------------------------------------
# Good — `mesh_look.py` supports `--region`, confirming the region-per-mesh layout is the expected sha
import subprocess, shutil, os

CASE = "/work/accept-t8"

regionProperties = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object regionProperties;
}
regions
(
    fluid       (fluid)
    solid       (solid)
);
"""
with open(f"{CASE}/constant/regionProperties", "w") as fh:
    fh.write(regionProperties)

# the unsplit base mesh is no longer the mesh of record now that fluid/ and solid/
# each carry their own polyMesh; drop it so nothing downstream reads it by mistake.
if os.path.isdir(f"{CASE}/constant/polyMesh"):
    shutil.rmtree(f"{CASE}/constant/polyMesh")
if os.path.exists(f"{CASE}/constant/cellToRegion"):
    os.remove(f"{CASE}/constant/cellToRegion")
if os.path.isdir(f"{CASE}/0"):
    shutil.rmtree(f"{CASE}/0")   # cellToRegion volScalarField, not a real field of either region

print(subprocess.run(["bash", "-lc", "find /work/accept-t8/constant -maxdepth 2 | sort"],
                      capture_output=True, text=True).stdout)

r = subprocess.run(["checkMesh", "-case", CASE, "-allRegions"], capture_output=True, text=True)
print(r.stdout[-3000:])
print("rc", r.returncode)

# -- cell 20 ------------------------------------------------------------------------
print(r.stdout[:4500])

# -- cell 21 ------------------------------------------------------------------------
print(open("/work/accept-t8/constant/fluid/polyMesh/boundary").read())

# -- cell 22 ------------------------------------------------------------------------
import subprocess, os, shutil

CASE = "/work/accept-t8"

# inlet/outlet are flow boundaries, not walls: give them "patch" type so a later
# solver setup can put an inletOutlet/fixedValue BC on them; interface faces are
# left out of `boundary` entirely (they merge as internal, then splitMeshRegions
# turns them into the mappedWall pair).
PATCH_TYPES = {
    "inlet": "patch", "outlet": "patch",
    "fluidTop": "wall", "fluidSideY0": "wall", "fluidSideY1": "wall", "fluidFloor": "wall",
    "solidBottom": "wall", "solidSideY0": "wall", "solidSideY1": "wall",
    "solidStepXNeg": "wall", "solidStepXPos": "wall",
}

def gen_blockMeshDict(dx, path, patch_types):
    verts = {}
    vlist = []
    def vid(p):
        key = tuple(round(c, 9) for c in p)
        if key not in verts:
            verts[key] = len(vlist)
            vlist.append(key)
        return verts[key]

    def hex_block(xlo, xhi, ylo, yhi, zlo, zhi, n, zone):
        pts = [(xlo,ylo,zlo),(xhi,ylo,zlo),(xhi,yhi,zlo),(xlo,yhi,zlo),
               (xlo,ylo,zhi),(xhi,ylo,zhi),(xhi,yhi,zhi),(xlo,yhi,zhi)]
        ids = [vid(p) for p in pts]
        return {"ids": ids, "pts": pts, "n": n, "zone": zone}

    def n_for(length):
        n = round(length/dx)
        assert abs(n*dx - length) < 1e-9, (length, dx, n)
        return max(int(n), 1)

    nxA, nxB, nxC = n_for(X1-X0), n_for(X2-X1), n_for(X3-X2)
    ny = n_for(Y1-Y0)
    nzF = n_for(ZF1-ZS1)
    nzS = n_for(ZS1-ZS0)

    blockA = hex_block(X0, X1, Y0, Y1, ZS1, ZF1, (nxA, ny, nzF), "fluid")
    blockB = hex_block(X1, X2, Y0, Y1, ZS1, ZF1, (nxB, ny, nzF), "fluid")
    blockC = hex_block(X2, X3, Y0, Y1, ZS1, ZF1, (nxC, ny, nzF), "fluid")
    blockD = hex_block(X1, X2, Y0, Y1, ZS0, ZS1, (nxB, ny, nzS), "solid")

    def face_ids(b, axis, side):
        p = b["pts"]; ids = b["ids"]
        idxmap = {
            ('x','-'): [0,3,7,4], ('x','+'): [1,2,6,5],
            ('y','-'): [0,1,5,4], ('y','+'): [3,2,6,7],
            ('z','-'): [0,1,2,3], ('z','+'): [4,5,6,7],
        }
        sel = idxmap[(axis, side)]
        return [ids[i] for i in sel], [p[i] for i in sel]

    def oriented(ids, pts, outward):
        v0,v1,v2 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
        n = np.cross(v1-v0, v2-v0)
        if np.dot(n, outward) < 0:
            return [ids[0], ids[3], ids[2], ids[1]]
        return ids

    def patch_face(b, axis, side):
        ids, pts = face_ids(b, axis, side)
        outward = {'x':(1,0,0),'y':(0,1,0),'z':(0,0,1)}[axis]
        outward = tuple(o*(1 if side=='+' else -1) for o in outward)
        return oriented(ids, pts, outward)

    patches = {
        "inlet":          [patch_face(blockA,'x','-')],
        "outlet":         [patch_face(blockC,'x','+')],
        "fluidTop":       [patch_face(b,'z','+') for b in (blockA,blockB,blockC)],
        "fluidSideY0":    [patch_face(b,'y','-') for b in (blockA,blockB,blockC)],
        "fluidSideY1":    [patch_face(b,'y','+') for b in (blockA,blockB,blockC)],
        "fluidFloor":     [patch_face(blockA,'z','-'), patch_face(blockC,'z','-')],
        "solidBottom":    [patch_face(blockD,'z','-')],
        "solidSideY0":    [patch_face(blockD,'y','-')],
        "solidSideY1":    [patch_face(blockD,'y','+')],
        "solidStepXNeg":  [patch_face(blockD,'x','-')],
        "solidStepXPos":  [patch_face(blockD,'x','+')],
    }

    lines = []
    lines.append('FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n')
    lines.append('scale 1;\n')
    lines.append('vertices\n(\n')
    for p in vlist:
        lines.append(f'    ({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})\n')
    lines.append(');\n\n')
    lines.append('blocks\n(\n')
    for b in (blockA, blockB, blockC, blockD):
        ids = " ".join(str(i) for i in b["ids"])
        n = b["n"]
        lines.append(f'    hex ({ids}) {b["zone"]} ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n')
    lines.append(');\n\n')
    lines.append('edges\n(\n);\n\n')
    lines.append('boundary\n(\n')
    for name, faces in patches.items():
        ptype = patch_types[name]
        lines.append(f'    {name}\n    {{\n        type {ptype};\n        faces\n        (\n')
        for f in faces:
            lines.append(f'            ({f[0]} {f[1]} {f[2]} {f[3]})\n')
        lines.append('        );\n    }\n')
    lines.append(');\n\n')
    lines.append('mergePatchPairs\n(\n);\n')

    with open(path, "w") as fh:
        fh.writelines(lines)
    return dict(nxA=nxA, nxB=nxB, nxC=nxC, ny=ny, nzF=nzF, nzS=nzS,
                n_fluid_cells=(nxA+nxB+nxC)*ny*nzF, n_solid_cells=nxB*ny*nzS)

def rebuild_mesh(dx, case=CASE):
    """(Re)generate blockMeshDict at cell size dx and rebuild both regions from
    scratch: blockMesh -> splitMeshRegions -cellZones -> regionProperties.
    Safe to re-run: each step overwrites rather than accumulates."""
    for d in ("fluid", "solid", "polyMesh"):
        p = f"{case}/constant/{d}"
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in ("cellToRegion",):
        p = f"{case}/constant/{f}"
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(f"{case}/0"):
        shutil.rmtree(f"{case}/0")

    info = gen_blockMeshDict(dx, f"{case}/system/blockMeshDict", PATCH_TYPES)

    r1 = subprocess.run(["blockMesh", "-case", case], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stdout[-3000:] + r1.stderr

    r2 = subprocess.run(["splitMeshRegions", "-cellZones", "-overwrite", "-case", case],
                         capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr

    with open(f"{case}/constant/regionProperties", "w") as fh:
        fh.write('FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n'
                  '    location "constant";\n    object regionProperties;\n}\n'
                  'regions\n(\n    fluid       (fluid)\n    solid       (solid)\n);\n')

    for f in ("cellToRegion",):
        p = f"{case}/constant/{f}"
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(f"{case}/0"):
        shutil.rmtree(f"{case}/0")
    if os.path.isdir(f"{case}/constant/polyMesh"):
        shutil.rmtree(f"{case}/constant/polyMesh")

    return info

DX_COARSE = 0.010
info = rebuild_mesh(DX_COARSE)
print(info)

r = subprocess.run(["checkMesh", "-case", CASE, "-allRegions"], capture_output=True, text=True)
print(r.stdout[-1500:])
print("rc", r.returncode)

# -- cell 23 ------------------------------------------------------------------------
# Both regions pass at coarse resolution and the interface is correctly typed. Now let's look at it wi
import subprocess
r1 = subprocess.run(["python3", toolbox+"mesh_look.py", "/work/accept-t8", "--region", "fluid",
                      "--out", "look_fluid.png"], capture_output=True, text=True, cwd="/work/accept-t8")
print(r1.stdout[-3000:]); print(r1.stderr[-2000:])

# -- cell 24 ------------------------------------------------------------------------
from PIL import Image
im = Image.open("/work/accept-t8/look_fluid.png")
print(im.size)
im

# -- cell 25 ------------------------------------------------------------------------
import os
print(os.path.getsize("/work/accept-t8/look_fluid.png"))
from PIL import Image
im = Image.open("/work/accept-t8/look_fluid.png")
print(im.size, im.mode)

# -- cell 26 ------------------------------------------------------------------------
print(open(toolbox+"ENVIRONMENT.md").read())

# -- cell 27 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-n", "savefig\\|screenshot\\|except\\|fallback\\|172\\|26\\b", toolbox+"mesh_look.py"],
                    capture_output=True, text=True)
print(r.stdout)

# -- cell 28 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["sed", "-n", "380,525p", toolbox+"mesh_look.py"], capture_output=True, text=True)
print(r.stdout)
