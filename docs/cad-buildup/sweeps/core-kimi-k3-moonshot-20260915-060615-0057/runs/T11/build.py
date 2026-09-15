"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll plan this build: a square duct (80×80 mm) running from z=-40 mm (inlet) to z=+100 mm (outlet), 
# First cell: parameters and the hex lattice, with a quick pattern plot to check coverage of the 80×80
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon as MplHex

# ---- parameters (metres) ----
ACROSS_FLATS = 0.008          # hex cell width across flats
APOTHEM = ACROSS_FLATS / 2    # 4 mm
WALL_T  = 0.001               # wall between cells
PITCH   = 2*APOTHEM + WALL_T  # nearest-neighbour distance = 9 mm
R_HEX   = APOTHEM / math.cos(math.radians(30))  # circumradius
HALF    = 0.040               # duct half width (80 mm square)
Z_CORE0, Z_CORE1 = 0.0, 0.020 # honeycomb extent
Z_IN,  Z_OUT     = -0.040, 0.100   # inlet / outlet planes

# hex lattice: a1=(PITCH,0), a2=(PITCH/2, PITCH*sqrt(3)/2), flats normal to a1
a1 = np.array([PITCH, 0.0])
a2 = np.array([PITCH/2, PITCH*math.sqrt(3)/2])
centers = []
n = int(HALF/PITCH) + 3
for i in range(-n, n+1):
    for j in range(-n, n+1):
        c = i*a1 + j*a2
        # keep any hexagon that touches the duct square
        if abs(c[0]) < HALF + R_HEX and abs(c[1]) < HALF + R_HEX:
            centers.append(c)
centers = np.array(centers)
print("hexes:", len(centers), " pitch(mm):", PITCH*1000, " R(mm):", R_HEX*1000)

fig, ax = plt.subplots(figsize=(6,6))
for c in centers:
    ax.add_patch(MplHex(c, 6, radius=R_HEX, orientation=math.radians(30),
                        fill=False, edgecolor="b", lw=0.7))
ax.add_patch(plt.Rectangle((-HALF,-HALF), 2*HALF, 2*HALF, fill=False, edgecolor="r", lw=1.5))
ax.set_aspect("equal"); ax.set_xlim(-0.05,0.05); ax.set_ylim(-0.05,0.05)
ax.set_title("cell pattern vs duct section")
plt.savefig("pattern.png", dpi=110); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The lattice gives 105 hexagons. Let me look at the pattern figure.
from IPython.display import Image
Image("pattern.png")

# -- cell 3 -------------------------------------------------------------------------
import build123d as bd
from functools import reduce

duct = bd.Pos(0, 0, (Z_IN+Z_OUT)/2) * bd.Box(2*HALF, 2*HALF, Z_OUT-Z_IN)
core = bd.Pos(0, 0, (Z_CORE0+Z_CORE1)/2) * bd.Box(2*HALF, 2*HALF, Z_CORE1-Z_CORE0)

# pointy-top hexagon: flats normal to x (rotation 30 deg), extruded through the core
def cell_prism(cx, cy):
    hexsk = bd.RegularPolygon(radius=R_HEX, side_count=6, rotation=30)
    prism = bd.extrude(hexsk, amount=Z_CORE1-Z_CORE0)
    return bd.Pos(cx, cy, Z_CORE0) * prism

cells = []
for c in centers:
    p = cell_prism(c[0], c[1]) & core      # clip partial edge cells to the plate
    if p.volume > 1e-12:
        cells.append(p)
print("cells kept:", len(cells))

cells_union = reduce(lambda a, b: a + b, cells)
honeycomb_solid = core - cells_union
air = duct - honeycomb_solid

exp_mm3 = (2*HALF*1000)**2 * (Z_OUT-Z_IN)*1000 - honeycomb_solid.volume*1e9
print(f"air volume      : {air.volume*1e9:,.1f} mm^3 (expected {exp_mm3:,.1f})")
print(f"honeycomb walls : {honeycomb_solid.volume*1e9:,.1f} mm^3")
print(f"open fraction at core: {cells_union.volume*1e9 / (80*80*20):.4f}")
print("air solids:", len(air.solids()), " faces:", len(air.faces()))
bd.export_step(air, "air.step")

# -- cell 4 -------------------------------------------------------------------------
# The tessellation returns build123d Vectors — converting to a plain float array:
import pyvista as pv
pv.set_jupyter_backend("static")

verts, tris = air.tessellate(2e-4)
pts = np.array([[v.X, v.Y, v.Z] for v in verts], dtype=float)
faces_pv = np.hstack([[3, *t] for t in tris]).astype(np.int64)
mesh = pv.PolyData(pts, faces_pv)
print("triangles:", mesh.n_cells)

pl = pv.Plotter(off_screen=True, window_size=(900,700))
pl.add_mesh(mesh.clip(normal="-z", origin=(0,0,0.03)), color="tan")
pl.view_isometric(); pl.add_text("air volume, clipped at z=30 mm", font_size=10)
pl.show()

# -- cell 5 -------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(900,700))
core_part = mesh.clip(normal="z", origin=(0,0,0.015))   # keep z <= 15 mm
pl.add_mesh(core_part, color="tan")
pl.view_xy()
pl.add_text("core region from outlet side: honeycomb voids in the plate", font_size=10)
pl.show()

# -- cell 6 -------------------------------------------------------------------------
import os
os.makedirs("constant/triSurface", exist_ok=True)

EPS = 1e-6
groups = {"inlet": [], "outlet": [], "ductWalls": [], "honeycombWalls": []}
for f in air.faces():
    c = f.center()
    if abs(c.Z - Z_IN) < EPS:
        groups["inlet"].append(f)
    elif abs(c.Z - Z_OUT) < EPS:
        groups["outlet"].append(f)
    elif abs(abs(c.X) - HALF) < EPS or abs(abs(c.Y) - HALF) < EPS:
        groups["ductWalls"].append(f)
    else:
        groups["honeycombWalls"].append(f)

for name, fl in groups.items():
    comp = bd.Compound(children=fl)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-5, angular_tolerance=0.2)
    print(f"{name:14s}: {len(fl):4d} faces, area {comp.area*1e6:12.1f} mm^2")

# sanity: expected areas
open_area = cells_union.volume / (Z_CORE1-Z_CORE0)
print(f"\nexpected inlet area : {(2*HALF)**2*1e6:.1f} mm^2 (full square)")
print(f"duct walls expected : {4*2*HALF*(Z_OUT-Z_IN)*1e6:.1f} mm^2")
print(f"cell open area      : {open_area*1e6:.1f} mm^2 of { (2*HALF)**2*1e6:.0f} mm^2")

# -- cell 7 -------------------------------------------------------------------------
import subprocess, shutil, textwrap, os

os.makedirs("system", exist_ok=True); os.makedirs("0", exist_ok=True)
print("surfaceFeatures:", shutil.which("surfaceFeatures"),
      "| surfaceFeatureExtract:", shutil.which("surfaceFeatureExtract"))

open("system/controlDict","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
    application snappyHexMesh;
    startFrom latestTime; startTime 0; stopAt endTime; endTime 1;
    deltaT 1; writeControl timeStep; writeInterval 1;
    writeFormat ascii; writePrecision 8; runTimeModifiable false;
"""))

# background mesh: 4 mm cells over the whole duct
open("system/blockMeshDict","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
    scale 1;
    vertices
    (
        (-0.04 -0.04 -0.04) ( 0.04 -0.04 -0.04) ( 0.04  0.04 -0.04) (-0.04  0.04 -0.04)
        (-0.04 -0.04  0.10) ( 0.04 -0.04  0.10) ( 0.04  0.04  0.10) (-0.04  0.04  0.10)
    );
    blocks ( hex (0 1 2 3 4 5 6 7) (20 20 35) simpleGrading (1 1 1) );
    edges ();
    boundary
    (
        walls { type wall; faces ( (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }
    );
"""))

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:] if r.returncode==0 else r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
# Background mesh is 14k cells. Now the surfaceFeatureExtract dict, snappyHexMeshDict (coarse: level 2
open("system/surfaceFeatureExtractDict","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
    honeycombWalls.stl { extractionMethod extractFromSurface; includedAngle 150; }
    ductWalls.stl      { extractionMethod extractFromSurface; includedAngle 150; }
    inlet.stl          { extractionMethod extractFromSurface; includedAngle 150; }
    outlet.stl         { extractionMethod extractFromSurface; includedAngle 150; }
    writeObj yes;
"""))

open("system/snappyHexMeshDict","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
    castellatedMesh true;
    snap            true;
    addLayers       false;

    geometry
    {
        inlet         { type triSurfaceMesh; file "inlet.stl";         patchInfo { type patch; } }
        outlet        { type triSurfaceMesh; file "outlet.stl";        patchInfo { type patch; } }
        ductWalls     { type triSurfaceMesh; file "ductWalls.stl";     patchInfo { type wall;  } }
        honeycombWalls{ type triSurfaceMesh; file "honeycombWalls.stl";patchInfo { type wall;  } }
        coreBox { type searchableBox; min (-0.0405 -0.0405 -0.003); max (0.0405 0.0405 0.023); }
    }

    castellatedMeshControls
    {
        maxLocalCells 10000000; maxGlobalCells 20000000; minRefinementCells 10;
        maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
        resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
        features ( { file "honeycombWalls.eMesh"; level 2; } );
        refinementSurfaces
        {
            honeycombWalls { level (2 2); }
            ductWalls      { level (1 1); }
        }
        refinementRegions { coreBox { mode inside; levels ((1e15 2)); } }
        locationInMesh (0.0 0.0 -0.02);
    }

    snapControls
    {
        nSmoothPatch 3; tolerance 2.0; nSolveIter 100; nRelaxIter 5;
        nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
        multiRegionFeatureSnap false;
    }

    addLayersControls
    {
        relativeSizes true;
        layers {}
        expansionRatio 1.0; finalLayerThickness 0.5; minThickness 0.1;
        nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1;
        nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
        maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
        nLayerIter 50; nRelaxedIter 20;
    }

    meshQualityControls
    {
        maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
        maxConcave 80; minVol 1e-13; minTetQuality -1e30; minArea -1;
        minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05;
        minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
        errorReduction 0.75;
    }
    mergeTolerance 1e-6;
"""))

r1 = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print("sfe:", r1.returncode, (r1.stdout or r1.stderr)[-200:])
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("shm:", r2.returncode)
print(r2.stdout[-1200:] if r2.returncode==0 else r2.stderr[-1500:])

# -- cell 9 -------------------------------------------------------------------------
# This ESI version wants `fvSchemes`/`fvSolution` present even for meshing. Adding minimal placeholder
open("system/fvSchemes","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
    ddtSchemes { default steadyState; }
    gradSchemes { default Gauss linear; }
    divSchemes { default Gauss upwind; }
    laplacianSchemes { default Gauss linear corrected; }
    interpolationSchemes { default linear; }
    snGradSchemes { default corrected; }
"""))
open("system/fvSolution","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
    solvers { p { solver GAMG; tolerance 1e-6; relTol 0.1; smoother GaussSeidel; } }
    SIMPLE { nNonOrthogonalCorrectors 0; }
"""))

r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("shm:", r2.returncode)
print(r2.stdout[-1000:] if r2.returncode==0 else (r2.stdout[-600:]+r2.stderr[-900:]))

# -- cell 10 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 11 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())
open("case.foam","w").write("")
import pyvista as pv
reader = pv.POpenFOAMReader("case.foam")
reader.cell_to_point_creation = False
reader.enable_all_patch_arrays()
mb = reader.read()
print(list(mb.keys()))

# -- cell 12 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace("""        refinementSurfaces
        {
            honeycombWalls { level (2 2); }
            ductWalls      { level (1 1); }
        }
        refinementRegions { coreBox { mode inside; levels ((1e15 2)); } }""",
"""        refinementSurfaces
        {
            honeycombWalls { level (3 3); }
            ductWalls      { level (1 1); }
            inlet          { level (0 0); }
            outlet         { level (0 0); }
        }
        refinementRegions { coreBox { mode inside; levels ((1e15 2)); } }""")
s = s.replace('{ file "honeycombWalls.eMesh"; level 2; }', '{ file "honeycombWalls.eMesh"; level 3; }')
s = s.replace("nCellsBetweenLevels 2;", "nCellsBetweenLevels 3;")
open("system/snappyHexMeshDict","w").write(s)

r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("shm:", r2.returncode)
print(r2.stdout[-900:] if r2.returncode==0 else (r2.stdout[-500:]+r2.stderr[-900:]))

# -- cell 13 ------------------------------------------------------------------------
# It crashed inside `featureEdgeRefine`. Let me see the actual error text above the stack trace:
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc:", r2.returncode)
out = r2.stdout
i = out.find("error")
print(out[max(0,i-1500):i+1500] if i>=0 else out[-3000:])
print("---STDERR---"); print(r2.stderr[:1500])

# -- cell 14 ------------------------------------------------------------------------
# My mistake — the second snappy run started from the already-refined mesh in `constant/polyMesh` inst
import glob
for f in glob.glob("cell_*.obj"): os.remove(f)
subprocess.run(["blockMesh"], capture_output=True, text=True)
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("shm:", r2.returncode)
print(r2.stdout[-800:] if r2.returncode==0 else r2.stderr[-800:])

# -- cell 15 ------------------------------------------------------------------------
# 667k cells, snappy's internal checks pass. Now `checkMesh` and the patch list:
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
i = out.find("Checking patch topology")
print(out[out.find("Mesh stats"):out.find("Breakdown")])
print(out[i:i+700])
j = out.find("Checking geometry")
print(out[j:])
subprocess.run(["awk", "/^[a-zA-Z]/ {print}", "constant/polyMesh/boundary"], capture_output=False)
