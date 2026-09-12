"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: the cavity is a plain rectangular box, so a single graded-free `blockMesh` block is the right 
from build123d import *
import numpy as np, os

# ---- parameters (metres) ----
LX, LY, LZ = 0.100, 0.060, 0.040      # cavity 100 x 60 x 40 mm
CUBE = 0.020                          # 20 mm marker cubes
GAP_Z = 0.005                         # gap from floor / ceiling
GAP_X = 0.010                         # gap from the end walls

cavity = Box(LX, LY, LZ).moved(Location((LX/2, LY/2, LZ/2)))   # corner at origin

heater = Box(CUBE, CUBE, CUBE).moved(Location((GAP_X + CUBE/2, LY/2, GAP_Z + CUBE/2)))
cooler = Box(CUBE, CUBE, CUBE).moved(Location((LX - GAP_X - CUBE/2, LY/2, LZ - GAP_Z - CUBE/2)))

for nm, s in [("cavity", cavity), ("heater", heater), ("cooler", cooler)]:
    bb = s.bounding_box()
    print(f"{nm:7s} bbox min={np.round(bb.min.to_tuple(),4)} max={np.round(bb.max.to_tuple(),4)} "
          f"size={np.round((bb.max - bb.min).to_tuple(),4)} vol={s.volume:.3e} m^3")
print("cubes inside cavity:", (cavity & heater).volume/heater.volume, (cavity & cooler).volume/cooler.volume)

# -- cell 2 -------------------------------------------------------------------------
# Dimensions are exactly as asked. Now look at the arrangement — cavity outline with the two cubes ins
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
pl = pv.Plotter(off_screen=True, window_size=(900, 650))
pl.add_mesh(pv.Box(bounds=(0, LX, 0, LY, 0, LZ)), style="wireframe", color="k", line_width=2)
pl.add_mesh(pv.Box(bounds=(0.01, 0.03, 0.02, 0.04, 0.005, 0.025)), color="red", opacity=0.8)
pl.add_mesh(pv.Box(bounds=(0.07, 0.09, 0.02, 0.04, 0.015, 0.035)), color="blue", opacity=0.8)
pl.add_axes()
pl.camera_position = [(0.25, -0.22, 0.16), (0.05, 0.03, 0.02), (0, 0, 1)]
pl.screenshot("layout.png")
print("red=heater (low, x~0.02)  blue=cooler (high, x~0.08)")

# -- cell 3 -------------------------------------------------------------------------
# The screenshot didn't come back. Check whether the file was actually written and display it explicit
import os
from IPython.display import Image, display
print(os.path.exists("layout.png"), os.path.getsize("layout.png") if os.path.exists("layout.png") else None)
if os.path.exists("layout.png"):
    display(Image("layout.png"))

# -- cell 4 -------------------------------------------------------------------------
# Layout is right: heater low near x=0, cooler high near x=0.1. Now the coarse mesh — 5 mm cells (20×1
import subprocess, os, textwrap
os.makedirs("system", exist_ok=True)

CELL = 0.005   # coarse cell size, m
nx, ny, nz = round(LX/CELL), round(LY/CELL), round(LZ/CELL)

open("system/controlDict","w").write(textwrap.dedent(f"""\
    FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
    application     buoyantFoam;
    startFrom       startTime;  startTime 0;
    stopAt          endTime;    endTime   1;
    deltaT          1;          writeControl timeStep; writeInterval 1;
    """))

open("system/blockMeshDict","w").write(textwrap.dedent(f"""\
    FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
    scale 1;
    vertices
    (
        (0 0 0) ({LX} 0 0) ({LX} {LY} 0) (0 {LY} 0)
        (0 0 {LZ}) ({LX} 0 {LZ}) ({LX} {LY} {LZ}) (0 {LY} {LZ})
    );
    blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
    edges ();
    boundary
    (
        walls
        {{
            type wall;
            faces
            (
                (0 3 2 1)   // floor  z=0
                (4 5 6 7)   // ceiling z=LZ
                (0 1 5 4)   // y=0
                (3 7 6 2)   // y=LY
                (0 4 7 3)   // x=0
                (1 2 6 5)   // x=LX
            );
        }}
    );
    mergePatchPairs ();
    """))

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-900:])

# -- cell 5 -------------------------------------------------------------------------
# Now the two cell zones via `topoSet` box sources (boxes coincide with cell faces, so cell centres fa
import subprocess, textwrap

def zone_entries(name, bmin, bmax):
    return textwrap.dedent(f"""\
    {{ name {name}Set; type cellSet; action new; source boxToCell;
      box ({bmin[0]} {bmin[1]} {bmin[2]}) ({bmax[0]} {bmax[1]} {bmax[2]}); }}
    {{ name {name}; type cellZoneSet; action new; source setToCellZone; set {name}Set; }}
    """)

H = ((0.010, 0.020, 0.005), (0.030, 0.040, 0.025))
C = ((0.070, 0.020, 0.015), (0.090, 0.040, 0.035))

open("system/topoSetDict","w").write(
    "FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }\n"
    "actions (\n" + zone_entries("heater", *H) + zone_entries("cooler", *C) + ");\n")

r = subprocess.run(["topoSet"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1200:], r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
# Zones are 64 cells each = 8.0e-6 m³, matching the 20 mm cubes exactly. Shape confirmed at the coarse
import subprocess, textwrap

CELL_FINE = 0.0025
nx, ny, nz = round(LX/CELL_FINE), round(LY/CELL_FINE), round(LZ/CELL_FINE)
d = open("system/blockMeshDict").read()
d = d.replace(f"({20} {12} {8})", f"({nx} {ny} {nz})")
open("system/blockMeshDict","w").write(d)

print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-500:])
print(subprocess.run(["topoSet"], capture_output=True, text=True).stdout[-450:])

# -- cell 7 -------------------------------------------------------------------------
# checkMesh, then dump each cellZone to VTK so I can measure its actual bounds and see it in place.
import subprocess
cm = subprocess.run(["checkMesh"], capture_output=True, text=True).stdout
print(cm[cm.find("Checking geometry"):][:1500])
for z in ("heater","cooler"):
    r = subprocess.run(["foamToVTK","-cellZone",z,"-time","0","-overwrite"], capture_output=True, text=True)
    print(z, r.returncode, r.stdout[-120:])

# -- cell 8 -------------------------------------------------------------------------
# checkMesh printed nothing useful from my slice, and foamToVTK exited 1. Look at both in full.
import subprocess
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-2000:])
r2 = subprocess.run(["foamToVTK","-cellZone","heater","-overwrite"], capture_output=True, text=True)
print("foamToVTK:", r2.returncode, r2.stdout[-400:], r2.stderr[-400:])

# -- cell 9 -------------------------------------------------------------------------
# checkMesh and foamToVTK both need `fvSchemes`/`fvSolution`. Write minimal ones (part of any case dir
import subprocess, textwrap
open("system/fvSchemes","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
    ddtSchemes { default Euler; }
    gradSchemes { default Gauss linear; }
    divSchemes { default none; }
    laplacianSchemes { default Gauss linear corrected; }
    interpolationSchemes { default linear; }
    snGradSchemes { default corrected; }
    """))
open("system/fvSolution","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
    solvers {}
    """))
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[r.stdout.find("Mesh stats"):][:2200])

# -- cell 10 ------------------------------------------------------------------------
# Mesh OK, and checkMesh itself reports both zone bounding boxes exactly at the requested cubes. Final
import subprocess, glob, pyvista as pv
from IPython.display import Image, display

for z in ("heater","cooler"):
    subprocess.run(["foamToVTK","-cellZone",z,"-overwrite","-name",f"VTK_{z}"], capture_output=True, text=True)
files = {z: sorted(glob.glob(f"VTK_{z}/**/*.vtk", recursive=True) + glob.glob(f"VTK_{z}/**/*.vtu", recursive=True)) for z in ("heater","cooler")}
print(files)

pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(pv.Box(bounds=(0,LX,0,LY,0,LZ)), style="wireframe", color="k")
for z, col in (("heater","red"), ("cooler","blue")):
    m = pv.read(files[z][0]); print(z, m.n_cells, m.bounds)
    pl.add_mesh(m, color=col, show_edges=True)
pl.add_axes(); pl.camera_position = [(0.25,-0.22,0.16),(0.05,0.03,0.02),(0,0,1)]
pl.screenshot("zones.png"); display(Image("zones.png"))
