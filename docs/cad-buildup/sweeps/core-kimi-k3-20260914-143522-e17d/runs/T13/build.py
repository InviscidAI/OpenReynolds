"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# No shapely — I'll use build123d, which has polygon offset natively, and classify faces geometrically
import inspect, build123d as bd
print([n for n in dir(bd) if n.lower() in ("sweep","helix","offset","face","polyline","solid")])
print(inspect.signature(bd.sweep))
print(inspect.signature(bd.offset))
print(inspect.signature(bd.Helix.__init__))

# -- cell 2 -------------------------------------------------------------------------
# build123d has everything. Building the ridge, the outer envelope swept 0.2 mm larger, the gap as the
import numpy as np, build123d as bd, math
R_MAJOR,R_MINOR,PITCH,HEIGHT,GAP = 20.0,19.0,2.0,12.0,0.2
Rc=(R_MAJOR+R_MINOR)/2
prof=[(R_MINOR,-0.7),(R_MINOR,0.7),(R_MAJOR,0.45),(R_MAJOR,-0.45)]  # (x=r, y=z) on Plane.XZ
face_in = bd.Face(bd.Polyline(*[bd.Vector(x,0,y) for x,y in prof], close=True))
face_out = bd.offset(face_in, GAP).face()
hx = bd.Helix(pitch=PITCH, height=HEIGHT, radius=Rc)
ridge = bd.sweep(face_in, path=hx)
big   = bd.sweep(face_out, path=hx)
gap   = big - ridge
gap = gap.clean()
print("gap volume mm^3:", gap.volume, "faces:", len(gap.faces()))
# classify
neck, lid, inner, outer = [],[],[],[]
for f in gap.faces():
    c = f.center(); n = f.normal_at()
    if abs(c.Z) < 0.35 and len(f.vertices())<=6: inner.append(f); continue
    if abs(c.Z-HEIGHT) < 0.35 and len(f.vertices())<=6: outer.append(f); continue
    probe = bd.Vector(*c.to_tuple()) - n*1e-3
    inside = ridge.is_inside(probe)
    (neck if inside else lid).append(f)
print("neck",len(neck),"lid",len(lid),"inner",len(inner),"outer",len(outer))

# -- cell 3 -------------------------------------------------------------------------
# Caps weren't caught — let me inspect face centers:
for i,f in enumerate(gap.faces()):
    c=f.center()
    print(i, "nv",len(f.vertices()), "c=[%.2f %.2f %.2f]"%(c.X,c.Y,c.Z), "R=%.2f"%(math.hypot(c.X,c.Y)), "area=%.2f"%f.area)

# -- cell 4 -------------------------------------------------------------------------
# Caps found (faces 2 and 4, small ~1 mm² annular sections — area matches expectation). Neck-probe nee
neck,lid,inner,outer=[],[],[],[]
V=lambda c: bd.Vector(c.X,c.Y,c.Z)
for i,f in enumerate(gap.faces()):
    c=f.center(); n=f.normal_at()
    if i in (2,4):
        (inner if c.Z<6 else outer).append(f); continue
    ins = ridge.is_inside(V(c)+n*1e-3) or ridge.is_inside(V(c)-n*1e-3)
    (neck if ins else lid).append(f)
print("neck",len(neck),"lid",len(lid),"inner",len(inner),"outer",len(outer))
import os
os.makedirs("constant/triSurface",exist_ok=True)
names={"neckFlank":neck,"lidFlank":lid,"innerEnd":inner,"outerEnd":outer}
for nm,fl in names.items():
    bd.export_stl(bd.Compound(fl),f"constant/triSurface/{nm}.stl")
    print(nm, sum(f.area for f in fl))

# -- cell 5 -------------------------------------------------------------------------
# Counts and areas match expectations (neck ≈ inner trapezoid perimeter × helix length ≈ 4.4 × 750 mm 
import pyvista as pv
pv.set_jupyter_backend("static")
pl=pv.Plotter(off_screen=True)
cols={"neckFlank":"red","lidFlank":"green","innerEnd":"blue","outerEnd":"orange"}
for nm in cols:
    m=pv.read(f"constant/triSurface/{nm}.stl")
    pl.add_mesh(m,color=cols[nm],opacity=1.0)
pl.add_axes(); pl.view_xy()
img=pl.screenshot(return_img=True)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8)); plt.imshow(img); plt.axis("off"); plt.show()

# -- cell 6 -------------------------------------------------------------------------
# Ring looks right from top. Side view to check the helix pitch and end caps:
pl=pv.Plotter(off_screen=True)
for nm,c in cols.items():
    pl.add_mesh(pv.read(f"constant/triSurface/{nm}.stl"),color=c)
pl.view_xz()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8)); plt.imshow(pl.screenshot(return_img=True)); plt.axis("off"); plt.show()

# -- cell 7 -------------------------------------------------------------------------
# The helix stack and tiny end caps look right. Scaling STLs to metres and setting up a snappyHexMesh 
import subprocess, os, glob
os.makedirs("system",exist_ok=True)
for f in glob.glob("constant/triSurface/*.stl"):
    subprocess.run(["surfaceTransformPoints","-scale","0.001",f,f],check=True)
# blockMesh: box around geometry at 2e-4 m cells → 250x250x70
bm="""convertToMeters 1; vertices ( (-0.025 -0.025 -0.001) (0.025 -0.025 -0.001) (0.025 0.025 -0.001) (-0.025 0.025 -0.001) (-0.025 -0.025 0.013) (0.025 -0.025 0.013) (0.025 0.025 0.013) (-0.025 0.025 0.013) );
blocks ( hex (0 1 2 3 4 5 6 7) (250 250 70) simpleGrading (1 1 1) );
edges (); boundary ();"""
open("system/blockMeshDict","w").write(bm)
open("system/controlDict","w").write('application snappyHexMesh;\n')
snp="""castellatedMesh true; snap true; addLayers false;
geometry { neckFlank.stl { type triSurfaceMesh; name neckFlank; }
  lidFlank.stl { type triSurfaceMesh; name lidFlank; }
  innerEnd.stl { type triSurfaceMesh; name innerEnd; }
  outerEnd.stl { type triSurfaceMesh; name outerEnd; } }
castellatedMeshControls { maxLocalCells 100000; maxGlobalCells 4000000; minRefinementCells 5; maxLoadBalanceTolerance 0.1;
  nCellsBetweenLevels 2; features (); refinementSurfaces ((neckFlank lidFlank innerEnd outerEnd) { level (2 2); });
  resolveFeatureAngle 30; refinementRegions (); locationInMesh (0.01944 0 1e-4); allowFreeStandingZoneFaces false; }
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10;
  implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers (); expansionRatio 1; finalLayerThickness 0.5;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; nMedialAxisIter 90;
  nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minFlatness -1; minVol 1e-21; minTetQuality 1e-9; minArea -1; minTwist 0.05; minDeterminant 0.001;
  minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
  errorReduction 0.75; relaxed { maxNonOrtho 75; } }
mergeTolerance 1e-6; debug 0; writeFlags ();"""
open("system/snappyHexMeshDict","w").write(snp)
print("ready")

# -- cell 8 -------------------------------------------------------------------------
# STLs are scaled (replay-safe: export overwrites with mm each run, then scale runs after). Now blockM
import subprocess
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-300:])

# -- cell 9 -------------------------------------------------------------------------
# controlDict needs a FoamFile header. Fixing all dict files:
HDR="""FoamFile { version 2.0; format ascii; class dictionary; object %s; }
"""
ctrl=HDR%"controlDict"+ "application snappyHexMesh;\nstartFrom latestTime;\n"
open("system/controlDict","w").write(ctrl)
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-200:] or r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
# blockMesh still failing — capturing full diagnostics, then launching snappy as a background subproce
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print("rc",r.returncode)
print(r.stdout[-800:]); print("ERR",r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
# controlDict needs full time entries. Then blockMesh and launch snappy in background to `log.snappy`:
ctrl="""FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable true;
"""
open("system/controlDict","w").write(ctrl)
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print("blockMesh rc",r.returncode)
import subprocess as sp
sn=sp.Popen(["snappyHexMesh","-overwrite"],stdout=open("log.snappy","w"),stderr=sp.STDOUT)
print("snappy pid",sn.pid)

# -- cell 12 ------------------------------------------------------------------------
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.stdout[-500:]); print(r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
# blockMeshDict has a syntax error — rewriting it cleanly and rerunning:
bm="""FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    (-0.025 -0.025 -0.001)
    ( 0.025 -0.025 -0.001)
    ( 0.025  0.025 -0.001)
    (-0.025  0.025 -0.001)
    (-0.025 -0.025  0.013)
    ( 0.025 -0.025  0.013)
    ( 0.025  0.025  0.013)
    (-0.025  0.025  0.013)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (250 250 70) simpleGrading (1 1 1)
);
edges ();
boundary
(
    def { type patch; faces (); }
);
"""
open("system/blockMeshDict","w").write(bm)
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-150:])

# -- cell 14 ------------------------------------------------------------------------
# blockMesh OK (13M base cells). Relaunching snappy:
import subprocess as sp
sn=sp.Popen(["snappyHexMesh","-overwrite"],stdout=open("log.snappy","w"),stderr=sp.STDOUT)
print("pid",sn.pid)

# -- cell 15 ------------------------------------------------------------------------
import time
print(sn.poll())  # None = still running
print(open("log.snappy").read()[-400:])
