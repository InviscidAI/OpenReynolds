"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# `Pos` here is itself a Location. Quick API check, then rebuild:
import build123d as bd
print(type(bd.Pos(1,2,3)), bd.Pos.__mro__[:3])
print(hasattr(bd.Pos(1,2,3), "position"))
# check algebra-mode relocate returns a new shape
b = bd.Box(1,1,1)
b2 = bd.Pos(5,0,0) * b
print("moved bbox min x:", b2.bounding_box().min.X, " original:", b.bounding_box().min.X)

# -- cell 2 -------------------------------------------------------------------------
# BoundingBox uses uppercase `.X/.Y/.Z`. Fixing:
import numpy as np
import build123d as bd

# ---- Request dimensions (metres) ----
W       = 0.080          # duct / plate square section
T       = 0.020          # honeycomb thickness in flow direction
Z_UP    = -0.040         # inlet plane (40 mm upstream of plate)
Z_DN    =  0.100         # outlet plane (80 mm downstream of plate)
F_CELL  = 0.008          # hex cell opening, across the flats
T_WALL  = 0.001          # honeycomb wall thickness

# ---- Hexagonal lattice: openings F_CELL on flat-to-flat pitch F_CELL+T_WALL ----
F_LAT = F_CELL + T_WALL
S_LAT = F_LAT / np.sqrt(3.0)
PX    = 1.5 * S_LAT              # column pitch (x)
PY    = F_LAT                    # row pitch within a column (y)
DY    = F_LAT / 2.0              # alternate-column offset
R_CH  = F_CELL / np.sqrt(3.0)    # channel opening circumradius

hex_face = bd.Plane.XY * bd.RegularPolygon(radius=R_CH, side_count=6)
ch0 = bd.extrude(hex_face, amount=T)

centres = []
nx = int(np.ceil((W/2 + R_CH) / PX)) + 1
ny = int(np.ceil((W/2 + R_CH + DY) / PY)) + 1
for i in range(-nx, nx + 1):
    x = i * PX
    off = DY if (i % 2) else 0.0
    for j in range(-ny, ny + 1):
        y = j * PY + off
        if abs(x) <= W/2 + R_CH and abs(y) <= W/2 + R_CH:
            centres.append((x, y))
prisms = [bd.Pos(x, y, 0.0) * ch0 for x, y in centres]

duct  = bd.Pos(0, 0, (Z_UP + Z_DN)/2) * bd.Box(W, W, Z_DN - Z_UP)
plate = bd.Pos(0, 0, T/2) * bd.Box(W, W, T)

walls = plate - bd.Compound(children=prisms)
air   = duct - walls

# ---- verification numbers ----
bb_ch = ch0.bounding_box()
print(f"channel opening across flats (y size): {bb_ch.size.Y*1e3:.4f} mm   (asked 8)")
print(f"channel opening across vertices (x size): {bb_ch.size.X*1e3:.4f} mm (expect {2*R_CH*1e3:.4f})")
print(f"number of channel prisms in lattice: {len(prisms)}")

dmin = min(prisms[0].distance(p) for k, p in enumerate(prisms) if k != 0)
print(f"min gap from channel 0 to any other: {dmin*1e3:.4f} mm (asked 1)")

open_area  = np.sqrt(3)/2 * F_CELL**2
pitch_area = PX * PY
n_equiv    = W*W / pitch_area
est_air    = (W*W*(Z_DN-Z_UP) - W*W*T) + n_equiv*open_area*T
print(f"air volume: {air.volume:.6e} m^3   estimate: {est_air:.6e} m^3")
bb = air.bounding_box()
print(f"air bbox: x [{bb.min.X:.4f},{bb.max.X:.4f}] y [{bb.min.Y:.4f},{bb.max.Y:.4f}] z [{bb.min.Z:.4f},{bb.max.Z:.4f}]")
print(f"plate porosity (open fraction): {open_area/pitch_area*100:.1f} %")
print(f"air faces: {len(air.faces())},  solids: {len(air.solids())}")

# -- cell 3 -------------------------------------------------------------------------
import pyvista as pv
import numpy as np

# tessellate the air solid and the walls solid for display
def to_pv(shape, tol=2e-4):
    verts, tris = shape.tessellate(tol)
    v = np.array([(p.X, p.Y, p.Z) for p in verts])
    f = np.hstack([np.full((len(tris),1),3), np.array(tris)]).astype(np.int64)
    return pv.PolyData(v, f)

pv_air   = to_pv(air)
pv_walls = to_pv(walls)

pv.global_theme.jupyter_backend = 'static'
pl = pv.Plotter(shape=(1,2), window_size=(1400,650))
pl.subplot(0,0)
pl.add_text("air volume (honeycomb walls = holes)", font_size=10)
pl.add_mesh(pv_air, color="lightblue", show_edges=False)
pl.add_mesh(pv_walls, color="orange", opacity=0.35)
pl.view_isometric(); 
pl.subplot(0,1)
pl.add_text("cross-section at z = 10 mm (mid-plate)", font_size=10)
sl_a = pv_air.slice(normal=(0,0,1), origin=(0,0,0.01))
sl_w = pv_walls.slice(normal=(0,0,1), origin=(0,0,0.01))
# clip both solids at the plane for a filled section view
sec_a = pv_air.clip(normal=(0,0,-1), origin=(0,0,0.0101))
sec_w = pv_walls.clip(normal=(0,0,-1), origin=(0,0,0.0101))
pl.add_mesh(sec_a, color="lightblue", show_edges=False)
pl.add_mesh(sec_w, color="orange", show_edges=False)
pl.view_xy(); pl.camera.zoom(1.4)
pl.show()
print("air triangles:", pv_air.n_cells, " walls triangles:", pv_walls.n_cells)

# -- cell 4 -------------------------------------------------------------------------
pl = pv.Plotter(shape=(1,2), window_size=(1400,650))
pl.subplot(0,0)
pl.add_text("honeycomb walls alone (plate with hex holes)", font_size=10)
pl.add_mesh(pv_walls, color="orange", show_edges=False)
pl.view_isometric()
pl.subplot(0,1)
pl.add_text("air volume, quadrant x>0,y>0 clipped away", font_size=10)
clip_air = pv_air.clip_box(pv.Box(bounds=(0, 0.05, 0, 0.05, -0.045, 0.105)), invert=True)
pl.add_mesh(clip_air, color="lightblue", show_edges=False)
pl.add_mesh(pv_walls.clip_box(pv.Box(bounds=(0, 0.05, 0, 0.05, -0.045, 0.105)), invert=True),
            color="orange", opacity=0.5)
pl.view_isometric()
pl.show()

# -- cell 5 -------------------------------------------------------------------------
import os

# classify faces of the air solid by the exact construction planes of this
# straight duct: z=Z_UP (inlet), z=Z_DN (outlet), x/y=+-W/2 (duct walls),
# everything else belongs to the honeycomb lattice (channel walls + web faces
# at z=0 and z=T)
tol = 1e-7
groups = {"inlet": [], "outlet": [], "duct_walls": [], "honeycomb_walls": []}
for f in air.faces():
    c = f.center()
    n = f.normal_at()
    if abs(n.Z) > 0.999 and abs(c.Z - Z_UP) < 1e-6:
        groups["inlet"].append(f)
    elif abs(n.Z) > 0.999 and abs(c.Z - Z_DN) < 1e-6:
        groups["outlet"].append(f)
    elif abs(abs(c.X) - W/2) < 1e-6 or abs(abs(c.Y) - W/2) < 1e-6:
        groups["duct_walls"].append(f)
    else:
        groups["honeycomb_walls"].append(f)

for name, fl in groups.items():
    area = sum(f.area for f in fl)
    print(f"{name:16s} faces={len(fl):4d}  area={area:.6e} m^2")

# sanity: total air surface area
print("total area:", sum(f.area for f in air.faces()))

# export one STL per patch
os.makedirs("constant/triSurface", exist_ok=True)
patch_shapes = {}
for name, fl in groups.items():
    comp = bd.Compound(children=fl)
    patch_shapes[name] = comp
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-5)
    print("wrote", name, os.path.getsize(f"constant/triSurface/{name}.stl"), "bytes")

# -- cell 6 -------------------------------------------------------------------------
import subprocess, os, textwrap

os.makedirs("system", exist_ok=True)

def w(path, txt):
    with open(path, "w") as f: f.write(textwrap.dedent(txt))
    print("wrote", path)

HDR = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
"""

w("system/controlDict", HDR + """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     snappyHexMesh;
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
""")

# background block: 4 mm cells, extends 1 cell beyond the duct on every side
w("system/blockMeshDict", HDR + """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale   1;

vertices
(
    (-0.042 -0.042 -0.042)
    ( 0.042 -0.042 -0.042)
    ( 0.042  0.042 -0.042)
    (-0.042  0.042 -0.042)
    (-0.042 -0.042  0.102)
    ( 0.042 -0.042  0.102)
    ( 0.042  0.042  0.102)
    (-0.042  0.042  0.102)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (21 21 36) simpleGrading (1 1 1)
);

boundary
(
    outer
    {
        type patch;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
""")

w("system/snappyHexMeshDict", HDR + """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    inlet           { type triSurfaceMesh; file "inlet.stl"; }
    outlet          { type triSurfaceMesh; file "outlet.stl"; }
    duct_walls      { type triSurfaceMesh; file "duct_walls.stl"; }
    honeycomb_walls { type triSurfaceMesh; file "honeycomb_walls.stl"; }
}

castellatedMeshControls
{
    maxLocalCells       5000000;
    maxGlobalCells      10000000;
    minRefinementCells  10;
    nCellsBetweenLevels 2;

    features    ();

    refinementSurfaces
    {
        honeycomb_walls { level (1 1); }
        duct_walls      { level (0 0); }
        inlet           { level (0 0); }
        outlet          { level (0 0); }
    }

    resolveFeatureAngle 30;

    refinementRegions   {}

    locationInMesh      (0 0 -0.02);

    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      30;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes   true;
    layers          {}
    expansionRatio  1.0;
    finalLayerThickness 0.3;
    minThickness    0.1;
    nGrow           0;
    featureAngle    60;
    nRelaxIter      3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals  3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter      50;
}

meshQualityControls
{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}

mergeTolerance 1e-6;
""")

for cmd in (["blockMesh"], ["snappyHexMesh", "-overwrite"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(" ".join(cmd), "-> rc", r.returncode)
    print("\n".join((r.stdout + r.stderr).splitlines()[-12:]))

# -- cell 7 -------------------------------------------------------------------------
# ESI's snappyHexMesh wants `fvSchemes`/`fvSolution` present even though we never solve. Adding minima
w("system/fvSchemes", HDR + """
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
""")

w("system/fvSolution", HDR + """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
PIMPLE {}
""")

r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", r.returncode)
print("\n".join((r.stdout + r.stderr).splitlines()[-25:]))

# -- cell 8 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-45:]))
print("---- boundary file ----")
print(open("constant/polyMesh/boundary").read())

# -- cell 9 -------------------------------------------------------------------------
print("\n".join(r.stdout.splitlines()[-8:]))
import glob
print(glob.glob("VTK/**/*", recursive=True)[:20])

# -- cell 10 ------------------------------------------------------------------------
import pyvista as pv
grid = pv.read("VTK/t11_0/internal.vtu")
print(grid.n_cells, "cells")
hc = pv.read("VTK/t11_0/boundary/honeycomb_walls.vtp")

pl = pv.Plotter(shape=(1,2), window_size=(1400,650))
pl.subplot(0,0); pl.add_text("coarse mesh section z=10 mm + honeycomb_walls patch", font_size=10)
sl = grid.slice(normal=(0,0,1), origin=(0,0,0.01))
pl.add_mesh(sl, show_edges=True, color="lightblue")
pl.add_mesh(hc, color="orange", opacity=0.9)
pl.view_xy(); pl.camera.zoom(1.3)
pl.subplot(0,1); pl.add_text("honeycomb_walls patch (coarse)", font_size=10)
pl.add_mesh(hc, color="orange", show_edges=False)
pl.view_isometric()
pl.show()

# -- cell 11 ------------------------------------------------------------------------
# finer background block: 2 mm cells
w("system/blockMeshDict", HDR + """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale   1;

vertices
(
    (-0.042 -0.042 -0.042)
    ( 0.042 -0.042 -0.042)
    ( 0.042  0.042 -0.042)
    (-0.042  0.042 -0.042)
    (-0.042 -0.042  0.102)
    ( 0.042 -0.042  0.102)
    ( 0.042  0.042  0.102)
    (-0.042  0.042  0.102)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (42 42 72) simpleGrading (1 1 1)
);

boundary
(
    outer
    {
        type patch;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
""")

# final snappy settings: honeycomb surface cells 0.5 mm (level 2 on 2 mm base),
# plate volume held at 1 mm via a refinement box, duct walls at 1 mm
w("system/snappyHexMeshDict", HDR + """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    inlet           { type triSurfaceMesh; file "inlet.stl"; }
    outlet          { type triSurfaceMesh; file "outlet.stl"; }
    duct_walls      { type triSurfaceMesh; file "duct_walls.stl"; }
    honeycomb_walls { type triSurfaceMesh; file "honeycomb_walls.stl"; }
    plateZone
    {
        type        searchableBox;
        min         (-0.041 -0.041 -0.004);
        max         ( 0.041  0.041  0.024);
    }
}

castellatedMeshControls
{
    maxLocalCells       10000000;
    maxGlobalCells      20000000;
    minRefinementCells  10;
    nCellsBetweenLevels 2;

    features    ();

    refinementSurfaces
    {
        honeycomb_walls { level (2 2); }
        duct_walls      { level (1 1); }
        inlet           { level (1 1); }
        outlet          { level (1 1); }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
        plateZone
        {
            mode    inside;
            levels  ((1E15 1));
        }
    }

    locationInMesh      (0 0 -0.02);

    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes   true;
    layers          {}
    expansionRatio  1.0;
    finalLayerThickness 0.3;
    minThickness    0.1;
    nGrow           0;
    featureAngle    60;
    nRelaxIter      3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals  3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter      50;
}

meshQualityControls
{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}

mergeTolerance 1e-6;
""")

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc", r.returncode, r.stdout.splitlines()[-8:])

# launch snappyHexMesh in background, poll from later cells
proc = subprocess.Popen(["snappyHexMesh", "-overwrite"],
                        stdout=open("log.snappyHexMesh", "w"),
                        stderr=subprocess.STDOUT)
print("snappyHexMesh started, pid", proc.pid)

# -- cell 12 ------------------------------------------------------------------------
import time, subprocess
time.sleep(90)
rc = proc.poll()
print("running" if rc is None else f"done rc={rc}")
print("\n".join(open("log.snappyHexMesh").read().splitlines()[-15:]))

# -- cell 13 ------------------------------------------------------------------------
# In the snap phase already. Polling again:
time.sleep(120)
rc = proc.poll()
print("running" if rc is None else f"done rc={rc}")
out = open("log.snappyHexMesh").read()
print("\n".join(out.splitlines()[-20:]))

# -- cell 14 ------------------------------------------------------------------------
time.sleep(150)
rc = proc.poll()
print("running" if rc is None else f"done rc={rc}")
out = open("log.snappyHexMesh").read()
print("\n".join(out.splitlines()[-25:]))

# -- cell 15 ------------------------------------------------------------------------
# Snapped with 735k cells and zero error faces. Waiting for the write to finish, then checking:
time.sleep(30)
rc = proc.poll()
print("running" if rc is None else f"done rc={rc}")
out = open("log.snappyHexMesh").read()
print("\n".join(out.splitlines()[-14:]))

# -- cell 16 ------------------------------------------------------------------------
import re
bp = "constant/polyMesh/boundary"
txt = open(bp).read()
# inlet/outlet are openings for the flow -> type patch (not wall)
for name in ("inlet", "outlet"):
    pat = re.compile(r"(\n\s*" + name + r"\s*\{\s*\n\s*)type\s+wall;\s*\n\s*inGroups\s+1\(wall\);\s*\n")
    txt2 = pat.sub(r"\1type            patch;\n", txt)
    assert txt2 != txt, name
    txt = txt2
open(bp, "w").write(txt)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-42:]))

# -- cell 17 ------------------------------------------------------------------------
import shutil
shutil.rmtree("VTK", ignore_errors=True)
r = subprocess.run(["foamToVTK", "-ascii", "-latestTime"], capture_output=True, text=True)
print("foamToVTK rc", r.returncode)
grid = pv.read("VTK/t11_0/internal.vtu")
hc   = pv.read("VTK/t11_0/boundary/honeycomb_walls.vtp")
dw   = pv.read("VTK/t11_0/boundary/duct_walls.vtp")
print(f"cells {grid.n_cells}, honeycomb patch faces {hc.n_cells}, duct_walls faces {dw.n_cells}")

pl = pv.Plotter(shape=(1,2), window_size=(1400,650))
pl.subplot(0,0); pl.add_text("final mesh, section z=10 mm (mid honeycomb)", font_size=10)
sl = grid.slice(normal=(0,0,1), origin=(0,0,0.01))
pl.add_mesh(sl, show_edges=False, color="lightblue")
pl.add_mesh(hc, color="orange")
pl.view_xy(); pl.camera.zoom(1.25)
pl.subplot(0,1); pl.add_text("final mesh, section y=0 (side view, flow up)", font_size=10)
sl2 = grid.slice(normal=(0,1,0), origin=(0,0.0005,0))
pl.add_mesh(sl2, show_edges=False, color="lightgreen")
pl.add_mesh(hc, color="orange", opacity=0.6)
pl.view_xz(); pl.camera.zoom(0.9)
pl.show()

# -- cell 18 ------------------------------------------------------------------------
import shutil, glob, os
shutil.rmtree("VTK", ignore_errors=True)
shutil.rmtree("constant/polyMesh/sets", ignore_errors=True)   # stale skewFaces set from coarse check

import pyvista as pv
parts = {n: pv.read(f"constant/triSurface/{n}.stl")
         for n in ("inlet", "outlet", "duct_walls", "honeycomb_walls")}
union = pv.merge([p for p in parts.values()])
print("union points (raw):", union.n_points)
free = union.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                   feature_edges=False, manifold_edges=False)
print("free (boundary) edges in welded union:", free.n_cells)
nm = union.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
                                 feature_edges=False, manifold_edges=False)
print("non-manifold edges:", nm.n_cells)

# final directory listing
for root, dirs, files in os.walk("."):
    if "polyMesh" in root and root.count(os.sep) > 2: continue
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"{os.path.getsize(p):>12d}  {p}")

# -- cell 19 ------------------------------------------------------------------------
# 40 free edges in the welded union needs investigating before I declare. Locating them:
import numpy as np
fe_pts = free.points
print("free-edge vertices (mm):")
print(np.round(fe_pts*1e3, 3))
# histogram of z and x/y locations
zs = np.round(fe_pts[:,2]*1e3, 2)
xs = np.round(fe_pts[:,0]*1e3, 2)
ys = np.round(fe_pts[:,1]*1e3, 2)
print("z values:", sorted(set(zs)))
print("x values:", sorted(set(xs)))
print("y values:", sorted(set(ys)))

# -- cell 20 ------------------------------------------------------------------------
# The free edges sit in the web planes (z=0 and z=20) at hexagon vertices of the central row. Let me l
# inspect individual free-edge segments
lines = free.lines.reshape(-1, 3)[:, 1:]
for a, b in lines[:12]:
    pa, pb = free.points[a]*1e3, free.points[b]*1e3
    print(np.round(pa,3), "->", np.round(pb,3))
print("total segments:", len(lines))

# -- cell 21 ------------------------------------------------------------------------
hc_stl = parts["honeycomb_walls"]
target = np.array([-35.796e-3, 0.0, 0.0])
d = np.linalg.norm(hc_stl.points - target, axis=1)
near = np.argsort(d)[:12]
for i in near:
    print(i, np.round(hc_stl.points[i]*1e3, 6), " dist mm:", round(d[i]*1e3, 6))

# -- cell 22 ------------------------------------------------------------------------
import inspect
print(inspect.signature(bd.Shape.tessellate))
help(bd.Shape.tessellate)
