"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The shape is a plain rectangular box (0.100 × 0.060 × 0.040 m) with one boundary patch (`walls`, sin
# Positions nobody gave me (assumptions, to be recorded): `heater` sits on the floor near the −x end, 
import build123d as bd

# Dimensions in metres (OpenFOAM reads raw numbers as metres)
L, W, H = 0.100, 0.060, 0.040          # cavity: x, y, z
CUBE = 0.020                            # heater/cooler cube side

# Cavity solid, corner at origin, base on z=0
cavity = bd.Box(L, W, H, align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN))

# Cell-zone regions (assumed placement):
#  heater: 20 mm cube on the floor (z=0), near x- end, centred in y
#  cooler: 20 mm cube on the ceiling (z=H), near x+ end, centred in y
HEATER_BOX = (0.010, 0.010 + CUBE)            # x range
HEATER_Y   = (W/2 - CUBE/2, W/2 + CUBE/2)     # y: 0.020..0.040
HEATER_Z   = (0.0, CUBE)                      # z: floor-touching
COOLER_BOX = (L - 0.030, L - 0.030 + CUBE)    # x: 0.070..0.090
COOLER_Y   = HEATER_Y
COOLER_Z   = (H - CUBE, H)                    # z: ceiling-touching

bb = cavity.bounding_box()
print("cavity bbox:", tuple(round(v,4) for v in bb.min), "->", tuple(round(v,4) for v in bb.max))
print("cavity volume (m^3):", cavity.volume, " expected:", L*W*H)

import os
os.makedirs("constant/triSurface", exist_ok=True)
bd.export_stl(cavity, "constant/triSurface/walls.stl")
print("STL written")

# -- cell 2 -------------------------------------------------------------------------
# Geometry checks out: 0.100 × 0.060 × 0.040 m, volume exact. Now the `blockMeshDict` — one hex block,
import os
os.makedirs("system", exist_ok=True)

block_mesh_dict = """/*--------------------------------*- C++ -*----------------------------------*\\
| Sealed rectangular cavity 0.1 x 0.06 x 0.04 m, all walls adiabatic           |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1;

vertices
(
    (0.000 0.000 0.000)   // 0
    (0.100 0.000 0.000)   // 1
    (0.100 0.060 0.000)   // 2
    (0.000 0.060 0.000)   // 3
    (0.000 0.000 0.040)   // 4
    (0.100 0.000 0.040)   // 5
    (0.100 0.060 0.040)   // 6
    (0.000 0.060 0.040)   // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (25 15 10) simpleGrading (1 1 1)
);

boundary
(
    walls
    {
        type wall;
        faces
        (
            (0 3 2 1)   // z = 0 floor
            (4 5 6 7)   // z = H ceiling
            (0 1 5 4)   // y = 0
            (3 7 6 2)   // y = W
            (0 4 7 3)   // x = 0
            (1 2 6 5)   // x = L
        );
    }
);
"""

with open("system/blockMeshDict", "w") as f:
    f.write(block_mesh_dict)

import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:])
print("STDERR:", r.stderr[-500:])
print("returncode:", r.returncode)

# -- cell 3 -------------------------------------------------------------------------
# `blockMesh` needs a `controlDict` to exist. Adding a minimal one, then rerunning.
control_dict = """/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     none;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
"""
with open("system/controlDict", "w") as f:
    f.write(control_dict)

import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-800:])
print("STDERR:", r.stderr[-300:])
print("returncode:", r.returncode)

# -- cell 4 -------------------------------------------------------------------------
# Mesh built: 3750 cells, single `walls` patch. Now mark the two cell zones with `topoSet`.
topo_set_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
    {{
        name    heaterSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({HEATER_BOX[0]} {HEATER_Y[0]} {HEATER_Z[0]})
                ({HEATER_BOX[1]} {HEATER_Y[1]} {HEATER_Z[1]});
    }}
    {{
        name    heater;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     heaterSet;
    }}
    {{
        name    coolerSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({COOLER_BOX[0]} {COOLER_Y[0]} {COOLER_Z[0]})
                ({COOLER_BOX[1]} {COOLER_Y[1]} {COOLER_Z[1]});
    }}
    {{
        name    cooler;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     coolerSet;
    }}
);
"""
with open("system/topoSetDict", "w") as f:
    f.write(topo_set_dict)

import subprocess
r = subprocess.run(["topoSet"], capture_output=True, text=True)
import re
for line in r.stdout.splitlines():
    if re.search(r"(cellZone|cells|Set|heater|cooler)", line, re.I):
        print(line)
print("returncode:", r.returncode, "| stderr tail:", r.stderr[-200:])

# -- cell 5 -------------------------------------------------------------------------
# Re-place zones on cell-boundary planes (grid spacing 0.004 m everywhere):
HEATER_BOX = (0.008, 0.028)   # x: cell faces, 8 mm off the x- end wall
COOLER_BOX = (0.072, 0.092)   # x: cell faces, 8 mm off the x+ end wall

topo_set_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
    {{
        name    heaterSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({HEATER_BOX[0]} {HEATER_Y[0]} {HEATER_Z[0]})
                ({HEATER_BOX[1]} {HEATER_Y[1]} {HEATER_Z[1]});
    }}
    {{
        name    heater;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     heaterSet;
    }}
    {{
        name    coolerSet;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({COOLER_BOX[0]} {COOLER_Y[0]} {COOLER_Z[0]})
                ({COOLER_BOX[1]} {COOLER_Y[1]} {COOLER_Z[1]});
    }}
    {{
        name    cooler;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     coolerSet;
    }}
);
"""
with open("system/topoSetDict", "w") as f:
    f.write(topo_set_dict)

import subprocess
r = subprocess.run(["topoSet"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "now size" in line or "Created" in line:
        print(line)
print("returncode:", r.returncode)

# -- cell 6 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
# print the interesting sections
keep = False
for line in out.splitlines():
    s = line.strip()
    if s.startswith(("Mesh stats", "Checking", "Overall", "  ", "cells", "points", "faces")) or "cellZones" in s or "OK" in s or "FAILED" in s.lower():
        print(line)
print("---- tail ----")
print("\n".join(out.splitlines()[-25:]))
print("returncode:", r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The output got truncated before the actual checks — printing the full log.
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout)
print("==== STDERR ====")
print(r.stderr)
print("returncode:", r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# `checkMesh` wants `fvSchemes`/`fvSolution` to exist. Adding minimal placeholders.
fv_schemes = """/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
ddtSchemes  { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes  { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
fv_solution = """/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
solvers {}
relaxationFactors {}
"""
with open("system/fvSchemes", "w") as f: f.write(fv_schemes)
with open("system/fvSolution", "w") as f: f.write(fv_solution)

r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout)
print("returncode:", r.returncode)

# -- cell 9 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["foamToVTK"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-200:])

import glob
print(glob.glob("VTK/**/*", recursive=True)[:20])

# -- cell 10 ------------------------------------------------------------------------
import numpy as np
import pyvista as pv
pv.set_jupyter_backend("static")

mesh = pv.read("VTK/t7_0/internal.vtu")
centers = mesh.cell_centers().points

def in_box(c, lo, hi):
    return np.all((c >= np.array(lo)) & (c <= np.array(hi)), axis=1)

zone = np.zeros(mesh.n_cells, dtype=int)
zone[in_box(centers, (HEATER_BOX[0], HEATER_Y[0], HEATER_Z[0]), (HEATER_BOX[1], HEATER_Y[1], HEATER_Z[1]))] = 1
zone[in_box(centers, (COOLER_BOX[0], COOLER_Y[0], COOLER_Z[0]), (COOLER_BOX[1], COOLER_Y[1], COOLER_Z[1]))] = 2
mesh.cell_data["zone"] = zone

heater_m = mesh.threshold([0.5, 1.5], scalars="zone")
cooler_m = mesh.threshold([1.5, 2.5], scalars="zone")
print("heater cells:", heater_m.n_cells, " centre:", np.round(heater_m.center, 4))
print("cooler cells:", cooler_m.n_cells, " centre:", np.round(cooler_m.center, 4))

sl = mesh.slice(normal=(0, 1, 0), origin=(0.05, 0.03, 0.02))
pl = pv.Plotter(off_screen=True, window_size=(900, 600))
pl.add_mesh(sl, color="lightgrey", opacity=0.3)
pl.add_mesh(heater_m, color="red", show_edges=True, label="heater (floor, x- end)")
pl.add_mesh(cooler_m, color="blue", show_edges=True, label="cooler (ceiling, x+ end)")
pl.add_legend()
pl.add_axes()
pl.view_vector((-1.6, -1.2, 1.0))   # look from -x,-y so x- end (heater) is near-left
pl.show()

# -- cell 11 ------------------------------------------------------------------------
with open("system/blockMeshDict") as f:
    bmd = f.read()
bmd = bmd.replace("(25 15 10)", "(50 30 20)")
with open("system/blockMeshDict", "w") as f:
    f.write(bmd)

import subprocess
for cmd in (["blockMesh"], ["topoSet"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    for line in r.stdout.splitlines():
        if "now size" in line or "nCells" in line:
            print(cmd[0], "|", line.strip())

# -- cell 12 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
lines = r.stdout.splitlines()
for i, line in enumerate(lines):
    s = line.strip()
    if (s.startswith(("Mesh stats", "points", "faces", "cells", "cell zones", "CellZone", "heater", "cooler",
                      "hexahedra", "Overall domain", "Max aspect", "Min volume", "Mesh non-orthogonality",
                      "Max skewness", "Mesh OK", "Boundary openness", "Patch"))
        or "FAILED" in s):
        print(line)
print("returncode:", r.returncode)

# -- cell 13 ------------------------------------------------------------------------
# Mesh OK at the final resolution. Removing the stale VTK export from the coarse mesh, then a final in
import shutil, glob, os
shutil.rmtree("VTK", ignore_errors=True)
for f in sorted(glob.glob("**/*", recursive=True)):
    if os.path.isfile(f):
        print(f, os.path.getsize(f), "bytes")
