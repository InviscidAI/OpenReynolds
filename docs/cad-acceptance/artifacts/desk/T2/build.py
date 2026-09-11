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
print(subprocess.run(["ls", "-la"], capture_output=True, text=True).stdout)
print(subprocess.run(["cat", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/README.md"], capture_output=True, text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["cat", f"{tb}/templates/prep/README.md"], capture_output=True, text=True).stdout)
print(subprocess.run(["ls", f"{tb}/templates/snappy"], capture_output=True, text=True).stdout)
print(subprocess.run(["cat", f"{tb}/templates/snappy/blockMeshDict"], capture_output=True, text=True).stdout)

# -- cell 3 -------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["grep", "-n", "-i", "sphere", f"{tb}/b123d_api.md"], capture_output=True, text=True).stdout[:3000])
print(subprocess.run(["python3", f"{tb}/cad_convert.py", "--help"], capture_output=True, text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["grep", "-n", "def export_patches", "-A", "40", f"{tb}/cad_convert.py"], capture_output=True, text=True).stdout)

# -- cell 5 -------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["grep", "-n", "PATCH_ROLES", f"{tb}/cad_convert.py"], capture_output=True, text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
import os
os.makedirs("/work/accept-t2", exist_ok=True)
os.chdir("/work/accept-t2")

from build123d import *

# Sphere: 40 mm diameter -> 20 mm radius, built directly in metres.
SPHERE_D_M = 0.040
SPHERE_R_M = SPHERE_D_M / 2.0

sphere = Sphere(radius=SPHERE_R_M)

print("solids:", len(sphere.solids()))
print("faces:", len(sphere.faces()))
print("volume (m^3):", sphere.volume, " expected:", 4/3*3.141592653589793*SPHERE_R_M**3)
bb = sphere.bounding_box()
print("bbox min/max (m):", bb.min, bb.max)

# -- cell 7 -------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["grep", "-n", "-i", "export_step", f"{tb}/b123d_api.md"], capture_output=True, text=True).stdout)

# -- cell 8 -------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["grep", "-n", "-i", "class Unit", "-A", "10", f"{tb}/b123d_api.md"], capture_output=True, text=True).stdout)

# -- cell 9 -------------------------------------------------------------------------
from build123d import Unit
print(list(Unit))

# -- cell 10 ------------------------------------------------------------------------
# Good — `Unit.M` exists. Export the sphere STEP declaring metres, since it was built with metre-value
from build123d import export_step, Unit

SPHERE_STEP = "sphere_m.step"
export_step(sphere, SPHERE_STEP, unit=Unit.M)
print("wrote", SPHERE_STEP)

# -- cell 11 ------------------------------------------------------------------------
import sys
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
sys.path.insert(0, tb)
import cad_convert
import importlib
importlib.reload(cad_convert)

from pathlib import Path

groups = {
    "sphere": {"role": "wall", "faces": None},
}

report = cad_convert.export_patches(
    path=Path("sphere_m.step"),
    groups=groups,
    out_dir=Path("constant/triSurface"),
    clmax=0.003,
    unit="m",
    curvature=20,
)
import json
print(json.dumps(report, indent=2, default=str)[:3000])

# -- cell 12 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["grep", "-a", "-i", "SI_UNIT\|FILE_SCHEMA\|CARTESIAN_POINT", "sphere_m.step"], capture_output=True, text=True).stdout[:3000])

# -- cell 13 ------------------------------------------------------------------------
import os
os.makedirs("system", exist_ok=True)
os.makedirs("constant", exist_ok=True)

# Domain: sphere D=0.04 m centered at origin.
# 5D upstream, 15D downstream, 5D each side and above/below.
D_M = 0.040
XMIN = -5 * D_M   # -0.20
XMAX = 15 * D_M   #  0.60
YMIN = -5 * D_M   # -0.20
YMAX =  5 * D_M   #  0.20
ZMIN = -5 * D_M   # -0.20
ZMAX =  5 * D_M   #  0.20

DX_BG = 0.020  # coarse background cell, metres -- first look only

nx = round((XMAX - XMIN) / DX_BG)
ny = round((YMAX - YMIN) / DX_BG)
nz = round((ZMAX - ZMIN) / DX_BG)
print("domain (m):", (XMIN, XMAX), (YMIN, YMAX), (ZMIN, ZMAX))
print("background cells:", nx, ny, nz, "total", nx*ny*nz)

blockMeshDict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;   // geometry built and STLs exported in metres

vertices
(
    ({XMIN} {YMIN} {ZMIN})
    ({XMAX} {YMIN} {ZMIN})
    ({XMAX} {YMAX} {ZMIN})
    ({XMIN} {YMAX} {ZMIN})
    ({XMIN} {YMIN} {ZMAX})
    ({XMAX} {YMIN} {ZMAX})
    ({XMAX} {YMAX} {ZMAX})
    ({XMIN} {YMAX} {ZMAX})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    farfield
    {{
        type patch;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs ();

// ************************************************************************* //
"""

with open("system/blockMeshDict", "w") as f:
    f.write(blockMeshDict)
print("wrote system/blockMeshDict")

# -- cell 14 ------------------------------------------------------------------------
import subprocess
res = subprocess.run(["blockMesh"], capture_output=True, text=True, cwd=".")
print(res.stdout[-3000:])
print(res.stderr[-3000:])

# -- cell 15 ------------------------------------------------------------------------
controlDict = """/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleFoam;
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

// ************************************************************************* //
"""
with open("system/controlDict", "w") as f:
    f.write(controlDict)

fvSchemes = """/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
// ************************************************************************* //
"""
with open("system/fvSchemes", "w") as f:
    f.write(fvSchemes)

fvSolution = """/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
// ************************************************************************* //
"""
with open("system/fvSolution", "w") as f:
    f.write(fvSolution)

import subprocess
res = subprocess.run(["blockMesh"], capture_output=True, text=True, cwd=".")
print(res.stdout[-2000:])
print(res.stderr[-1000:])

# -- cell 16 ------------------------------------------------------------------------
# Background mesh built: 16000 cells, bounds match the requested domain (-0.2..0.6, -0.2..0.2, -0.2..0
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["cat", f"{tb}/templates/snappy/snappyHexMeshDict"], capture_output=True, text=True).stdout)

# -- cell 17 ------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
d = subprocess.run(["cat", f"{tb}/templates/snappy/snappyHexMeshDict"], capture_output=True, text=True).stdout
print(d[1400:3200])

# -- cell 18 ------------------------------------------------------------------------
# Now let's check `patch_entries.py --help` and `domain_probe.py --suggest` to fill in locationInMesh 
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["python3", f"{tb}/patch_entries.py", "--help"], capture_output=True, text=True).stdout)
print(subprocess.run(["python3", f"{tb}/domain_probe.py", "--help"], capture_output=True, text=True).stdout)

# -- cell 19 ------------------------------------------------------------------------
res.poll() if hasattr(res, "poll") else None
print("waiting...")

# -- cell 20 ------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
res = subprocess.run(["python3", f"{tb}/domain_probe.py", "constant/triSurface", "--point", "0.3", "0.0", "0.0"],
                      capture_output=True, text=True)
print(res.stdout)
print(res.stderr[-2000:])

# -- cell 21 ------------------------------------------------------------------------
print("still checking previous cell status")

# -- cell 22 ------------------------------------------------------------------------
print(res.stdout)
print(res.stderr[-2000:])

# -- cell 23 ------------------------------------------------------------------------
import subprocess
tb = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"

# Copy the template dictionaries into place.
for name in ["blockMeshDict"]:  # blockMeshDict already hand-written for the named 6 far-field patches
    pass

subprocess.run(["cp", f"{tb}/templates/snappy/snappyHexMeshDict", "system/snappyHexMeshDict"], check=True)
subprocess.run(["cp", f"{tb}/templates/snappy/surfaceFeatureExtractDict", "system/surfaceFeatureExtractDict"], check=True)

res = subprocess.run(
    ["python3", f"{tb}/patch_entries.py", "accept-t2" if False else ".",
     "--insert", "system/snappyHexMeshDict",
     "--insert", "system/surfaceFeatureExtractDict",
     "--level", "2 3"],   # sphere surface refined to level 2-3 relative to background dx=0.02
    capture_output=True, text=True,
)
print(res.stdout)
print(res.stderr[-3000:])

# -- cell 24 ------------------------------------------------------------------------
# Now set `locationInMesh` manually (outside the sphere, inside the box, off any background cell face)
import re

LOCATION_IN_MESH = (0.25, 0.03, 0.03)  # well inside the domain, outside the sphere, off any dx=0.02 cell face

with open("system/snappyHexMeshDict") as f:
    content = f.read()

content = content.replace(
    "locationInMesh (CHANGE_ME CHANGE_ME CHANGE_ME);",
    f"locationInMesh ({LOCATION_IN_MESH[0]} {LOCATION_IN_MESH[1]} {LOCATION_IN_MESH[2]});",
)

with open("system/snappyHexMeshDict", "w") as f:
    f.write(content)

print("CHANGE_ME" in content)
import subprocess
print(subprocess.run(["grep", "-n", "locationInMesh", "system/snappyHexMeshDict"], capture_output=True, text=True).stdout)

# -- cell 25 ------------------------------------------------------------------------
# Good, no `CHANGE_ME` left. Now enable prism layers on the sphere patch — the request asks specifical
with open("system/snappyHexMeshDict") as f:
    content = f.read()

# turn on the layer stage and request layers on the sphere patch
content = content.replace("addLayers       false;", "addLayers       true;")

old_layers_block = """    layers
    {
        // One entry per patch that is to carry a layer, by name or by regex. A patch
        // that is not named here gets no layer and no warning. Coverage is what was
        // achieved, not what was asked for:
        //   python3 layer_report.py <case> --patch <wall> --ref <the same case, no layers>
        //
        // "walls"
        // {
        //     nSurfaceLayers 3;
        // }
    }"""

new_layers_block = """    layers
    {
        // One entry per patch that is to carry a layer, by name or by regex. A patch
        // that is not named here gets no layer and no warning. Coverage is what was
        // achieved, not what was asked for:
        //   python3 layer_report.py <case> --patch <wall> --ref <the same case, no layers>
        "sphere"
        {
            nSurfaceLayers 3;   // coarse first pass; raise once castellate+snap is clean
        }
    }"""

assert old_layers_block in content
content = content.replace(old_layers_block, new_layers_block)

with open("system/snappyHexMeshDict", "w") as f:
    f.write(content)

print("addLayers true" in content)
import subprocess
print(subprocess.run(["grep", "-n", "-A3", "\"sphere\"", "system/snappyHexMeshDict"], capture_output=True, text=True).stdout)

# -- cell 26 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["grep", "-n", "addLayers", "system/snappyHexMeshDict"], capture_output=True, text=True).stdout)

# -- cell 27 ------------------------------------------------------------------------
# Good — it did apply; the earlier check string just wasn't exact due to the trailing comment. Now let
import subprocess
print(subprocess.run(["python3",
    "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/patch_entries.py",
    ".", "--print"], capture_output=True, text=True).stdout)

# -- cell 28 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["grep", "-n", "-A5", "refinementSurfaces", "system/snappyHexMeshDict"], capture_output=True, text=True).stdout)
