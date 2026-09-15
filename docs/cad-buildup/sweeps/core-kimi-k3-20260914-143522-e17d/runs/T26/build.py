"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The air volume is the shaft cylinder minus the column minus 20 swept treads. I'll build it in build1
import build123d as bd
import numpy as np

# Parameters (metres)
R_shaft, H_shaft = 1.2, 3.0
R_col = 0.1
N_tread = 20
tread_L, tread_D, tread_T = 1.1, 0.25, 0.04   # radial length, tangential depth, thickness
rise, ang = 0.15, 18.0

shaft = bd.Cylinder(R_shaft, H_shaft)                      # axis Z, base at z=0
column = bd.Cylinder(R_col, H_shaft)

def tread(k):
    b = bd.Box(tread_L, tread_D, tread_T, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN))
    b = bd.Pos(bd.Align.MIN if False else R_col, 0, rise*k) * b  # x from R_col outward
    return bd.Rot(0, 0, ang*k) * b

treads = [tread(k) for k in range(N_tread)]
air = shaft - column - sum(treads[1:], treads[0])
print("air volume m^3:", air.volume)
print("faces:", len(air.faces()))

# -- cell 2 -------------------------------------------------------------------------
import collections
groups = collections.defaultdict(list)
for f in air.faces():
    c = f.center().to_tuple()
    gt = str(f.geom_type)
    if gt == "GeomType.CYLINDER" or "CYLINDER" in gt:
        r = np.hypot(c[0], c[1])
        groups["shaftWall" if r > 0.5 else "column"].append(f)
    else:
        n = f.normal_at().to_tuple()
        if gt == "GeomType.PLANE" and abs(abs(n[2]) - 1) < 1e-6 and (abs(c[2]) < 1e-9 or abs(c[2]-H_shaft) < 1e-9):
            groups["floor" if c[2] < 1e-9 else "topOpening"].append(f)
        else:
            groups["treads"].append(f)
print({k: len(v) for k, v in groups.items()})

# -- cell 3 -------------------------------------------------------------------------
for f in air.faces():
    gt = str(f.geom_type)
    if "CYLINDER" not in gt:
        n = f.normal_at().to_tuple()
        c = f.center().to_tuple()
        if abs(abs(n[2])-1) < 1e-6 and (c[2] < 1e-9 or abs(c[2]-H_shaft) < 1e-9 or c[2]==0):
            print(gt, n[2], c[2], f.area)
        elif abs(abs(n[2])-1) < 1e-6 and abs(c[2]) > 2.9:
            print("cand:", gt, n[2], c[2], f.area)

# -- cell 4 -------------------------------------------------------------------------
# The Cylinder primitive is centered at the origin, so my shaft ran from −1.5 to +1.5 and clipped half
shaft  = bd.Pos(0, 0, H_shaft/2) * bd.Cylinder(R_shaft, H_shaft)
column = bd.Pos(0, 0, H_shaft/2) * bd.Cylinder(R_col,   H_shaft)
air = shaft - column - sum(treads[1:], treads[0])
air = bd.ShapeList([air])[0]
print("air volume m^3:", air.volume)

groups = collections.defaultdict(list)
for f in air.faces():
    c = tuple(f.center())
    gt = str(f.geom_type)
    if "CYLINDER" in gt:
        groups["shaftWall" if np.hypot(c[0], c[1]) > 0.5 else "column"].append(f)
    elif "PLANE" in gt:
        n = tuple(f.normal_at())
        if abs(abs(n[2]) - 1) < 1e-6 and abs(c[2]) < 1e-9: groups["floor"].append(f)
        elif abs(abs(n[2]) - 1) < 1e-6 and abs(c[2] - H_shaft) < 1e-9: groups["topOpening"].append(f)
        else: groups["treads"].append(f)
    else:
        groups["treads"].append(f)
print({k: len(v) for k, v in groups.items()})

# -- cell 5 -------------------------------------------------------------------------
import inspect
print(inspect.signature(bd.export_stl))
import build123d
print([m for m in dir(bd.Compound) if 'comp' in m.lower()])

# -- cell 6 -------------------------------------------------------------------------
import os
os.makedirs("constant/triSurface", exist_ok=True)
for name, faces in groups.items():
    comp = bd.Compound(children=faces)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", angular_tolerance=0.15, tolerance=0.008)
print(sorted(os.listdir("constant/triSurface")))

import pyvista as pv
p = pv.Plotter(off_screen=True)
colors = {"shaftWall":"tan","column":"brown","treads":"gray","floor":"green","topOpening":"cyan"}
for name in groups:
    p.add_mesh(pv.read(f"constant/triSurface/{name}.stl"), color=colors[name], show_edges=False, opacity=0.8)
p.camera_position = [(6,-9,6),(0,0,1.5),(0,0,1)]
p.show(screenshot="view.png")

# -- cell 7 -------------------------------------------------------------------------
# The geometry reads exactly like the spec: 20 treads spiraling a full turn (20×18°) between floor and
import textwrap, os
os.makedirs("system", exist_ok=True)

open("system/blockMeshDict","w").write(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices
(
    (-1.5 -1.5 -0.2)
    ( 1.5 -1.5 -0.2)
    ( 1.5  1.5 -0.2)
    (-1.5  1.5 -0.2)
    (-1.5 -1.5  3.2)
    ( 1.5 -1.5  3.2)
    ( 1.5  1.5  3.2)
    (-1.5  1.5  3.2)
);
blocks ( hex (0 1 2 3 4 5 6 7) (15 15 17) simpleGrading (1 1 1) );
boundary ();
"""))

open("system/meshQualityDict","w").write(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object meshQualityDict; }
maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
maxConcave 80; minVol 1e-13; minTetQuality 1e-9; minArea -1;
minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05;
minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
errorReduction 0.75;
"""))

open("system/snappyHexMeshDict","w").write(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
    shaftWall.stl  { type triSurfaceMesh; name shaftWall; }
    column.stl     { type triSurfaceMesh; name column; }
    treads.stl     { type triSurfaceMesh; name treads; }
    floor.stl      { type triSurfaceMesh; name floor; }
    topOpening.stl { type triSurfaceMesh; name topOpening; }
}
castellatedMeshControls
{
    maxLocalCells 100000; maxGlobalCells 2000000; minRefinementCells 10; maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        shaftWall  { level (1 1); }
        column     { level (2 2); }
        treads     { level (2 2); }
        floor      { level (1 1); }
        topOpening { level (1 1); }
    }
    resolveFeatureAngle 30;
    refinementRegions {}
    locationInMesh (0.3 0.3 1.5);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls { };
meshQualityControls #include "meshQualityDict";
mergeTolerance 1e-6;
"""))

open("system/controlDict","w").write(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
print("case files written")

# -- cell 8 -------------------------------------------------------------------------
import subprocess
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r1.stdout[-600:], r1.stderr[-300:])
r2 = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r2.stdout[-800:])

# -- cell 9 -------------------------------------------------------------------------
r2_stdout = r2.stdout
print(r2_stdout[r2_stdout.rfind("Successfully added"):][:400] if "Successfully" in r2_stdout else r2_stdout[-1200:])
print("rc:", r2.returncode, os.path.exists("1/polyMesh/boundary"))

# -- cell 10 ------------------------------------------------------------------------
print(r2.stderr[-1500:])

# -- cell 11 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n")
open("system/fvSolution","w").write("FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n")
# no feature snapping without eMesh files
s = open("system/snappyHexMeshDict").read()
s = s.replace("explicitFeatureSnap true", "explicitFeatureSnap false").replace("implicitFeatureSnap false", "implicitFeatureSnap true")
open("system/snappyHexMeshDict","w").write(s)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode, "err:", r.stderr[-400:], os.path.exists("1/polyMesh/boundary"))

# -- cell 12 ------------------------------------------------------------------------
open("system/fvSchemes","w").write(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default none; laplacian((1|A(U)),p) Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""))
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-1500:], r.stderr[-300:])

# -- cell 13 ------------------------------------------------------------------------
print(r.stderr[-1200:])

# -- cell 14 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace('meshQualityControls #include "meshQualityDict";',
              'meshQualityControls { #include "meshQualityDict"; }')
open("system/snappyHexMeshDict","w").write(s)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-900:], r.stderr[-300:])

# -- cell 15 ------------------------------------------------------------------------
# Mesh built with 24,527 cells and snappy's own checks pass. Verify patches and run checkMesh, then lo
import re
print(open("constant/polyMesh/boundary").read()[:600])
rc = subprocess.run(["checkMesh","-time","constant"], capture_output=True, text=True, check=False)
print(rc.stdout[-900:])

# -- cell 16 ------------------------------------------------------------------------
b = open("constant/polyMesh/boundary").read()
names = re.findall(r"\b(shaftWall|column|treads|floor|topOpening|defaultFaces)\b\s*\{[^}]*?nFaces\s+(\d+)", b, re.S)
print(names)
# visual slice
mesh = pv.read("foamdummy.foam") if os.path.exists("foamdummy.foam") else None
print("done")

# -- cell 17 ------------------------------------------------------------------------
mb = mesh.combine()
s = mb.slice(normal=[0,0,1], origin=[0,0,1.2])
p = pv.Plotter(off_screen=True)
p.add_mesh(s, show_edges=True)
p.view_xy()
p.show(screenshot="slice.png")

# -- cell 18 ------------------------------------------------------------------------
# The winding conflicts come from coincidence: tread 0's bottom lies exactly on the floor plane and ev
def tread2(k):
    b = bd.Box(tread_L-0.002, tread_D, tread_T, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN))
    b = bd.Pos(R_col, 0, 0.01 + rise*k) * b
    return bd.Rot(0, 0, ang*k) * b

treads = [tread2(k) for k in range(N_tread)]
air = shaft - column - sum(treads[1:], treads[0])
air = bd.ShapeList([air])[0]
print("air volume m^3:", air.volume)

groups = collections.defaultdict(list)
for f in air.faces():
    c = tuple(f.center()); gt = str(f.geom_type)
    if "CYLINDER" in gt:
        groups["shaftWall" if np.hypot(c[0], c[1]) > 0.5 else "column"].append(f)
    elif "PLANE" in gt:
        n = tuple(f.normal_at())
        if abs(abs(n[2])-1) < 1e-6 and abs(c[2]) < 1e-9: groups["floor"].append(f)
        elif abs(abs(n[2])-1) < 1e-6 and abs(c[2]-H_shaft) < 1e-9: groups["topOpening"].append(f)
        else: groups["treads"].append(f)
    else: groups["treads"].append(f)
print({k: len(v) for k, v in groups.items()})
for name, faces in groups.items():
    bd.export_stl(bd.Compound(children=faces), f"constant/triSurface/{name}.stl", angular_tolerance=0.15, tolerance=0.008)
print("re-exported")

# -- cell 19 ------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode, r.stdout[-300:])
rc = subprocess.run(["checkMesh","-time","constant"], capture_output=True, text=True)
print(rc.stdout[-400:])

# -- cell 20 ------------------------------------------------------------------------
print(r.stderr[-2000:]); print(r.stdout[-600:])

# -- cell 21 ------------------------------------------------------------------------
# sHM started from the old refined mesh — need to reset the background first:
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
subprocess.run(["blockMesh"], capture_output=True, check=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode, r.stdout[-400:])

# -- cell 22 ------------------------------------------------------------------------
rc = subprocess.run(["checkMesh","-time","constant"], capture_output=True, text=True)
print(rc.stdout[-600:])
b = open("constant/polyMesh/boundary").read()
print(re.findall(r"\b(shaftWall|column|treads|floor|topOpening|defaultFaces)\b\s*\{[^}]*?nFaces\s+(\d+)", b, re.S))

# -- cell 23 ------------------------------------------------------------------------
mq = open("system/meshQualityDict").read()
mq = mq.replace("maxInternalSkewness 4;", "maxInternalSkewness 8;")
mq = mq.replace("maxBoundarySkewness 20;", "maxBoundarySkewness 20;")
open("system/meshQualityDict","w").write(mq)
shutil.rmtree("constant/polyMesh", ignore_errors=True)
subprocess.run(["blockMesh"], capture_output=True, check=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
rc = subprocess.run(["checkMesh","-time","constant"], capture_output=True, text=True)
print(rc.stdout[-350:])

# -- cell 24 ------------------------------------------------------------------------
# Skip topoSet; better remedy: turn on feature-edge snapping with extracted eMesh files, which targets
open("system/surfaceFeatureExtractDict","w").write(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
column.stl  { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 120; } writeObj yes; }
treads.stl  { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 120; } writeObj yes; }
"""))
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print("rc:", r.returncode, r.stdout[-300:], r.stderr[-200:])
print([f for f in os.listdir("constant/triSurface") if f.endswith("eMesh")])
