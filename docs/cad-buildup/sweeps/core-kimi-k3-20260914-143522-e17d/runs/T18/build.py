"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Re-emitting as one self-contained cell that builds all pieces and exports all patch STLs:
import os
import build123d as bd

# ---- parameters (metres) ----
BAR, H   = 0.045, 0.120    # barrel radius, height
FIN, FT, GAP, NF = 0.070, 0.003, 0.006, 12
DUCT = 0.300
BELOW, ABOVE = 0.150, 0.300

stack = NF*FT + (NF-1)*GAP     # 0.102
z0 = (H - stack)/2             # 0.009, fin stack centred on barrel
z_top = H + BAR                # dome apex 0.165
z_min, z_max = -BELOW, z_top + ABOVE
Lcyl = z_max - z_min

# cylinder pieces (separate overlapping solids)
barrel = bd.Cylinder(BAR, H).moved(bd.Location((0,0,H/2)))
sph = bd.Sphere(BAR).moved(bd.Location((0,0,H)))
cap_box = bd.Pos(0,0,H+BAR/2) * bd.Box(2*BAR,2*BAR,BAR)
head_s = sph & cap_box
fin_list = [bd.Cylinder(FIN,FT).moved(bd.Location((0,0,z0 + i*(FT+GAP) + FT/2)))
            for i in range(NF)]
fins_s = fin_list[0]
for f in fin_list[1:]:
    fins_s = fins_s + f

duct = bd.Pos(0,0,(z_min+z_max)/2) * bd.Box(DUCT, DUCT, Lcyl)

def face_group(shape, target):
    return bd.Compound([f for f in shape.faces()
                        if tuple(round(v,6) for v in f.normal_at())==target])
inlet  = face_group(duct,(0,0,-1))
outlet = face_group(duct,(0,0, 1))
walls  = bd.Compound([f for f in duct.faces() if abs(f.normal_at().Z)<1e-6])

os.makedirs("constant/triSurface", exist_ok=True)
for name, shp in [("barrel",barrel),("fins",fins_s),("head",head_s),
                  ("inlet",inlet),("outlet",outlet),("walls",walls)]:
    bd.export_stl(shp, "constant/triSurface/"+name+".stl")
    print(name, "faces:", len(shp.faces()), " vol/area:",
          round(shp.volume if hasattr(shp,'volume') and shp.volume else shp.area, 6))
print("duct z:", z_min, "->", z_max, "  fin zs:", z0, "...", round(z0+11*(FT+GAP)+FT,4))

# -- cell 2 -------------------------------------------------------------------------
# All six patch STLs are exported. Rendering them to eyeball the geometry:
import pyvista as pv, glob
pl = pv.Plotter(off_screen=True, window_size=(900,900))
colors = {"barrel":"tan","fins":"tomato","head":"lightblue","inlet":"lime",
          "outlet":"yellow","walls":"grey"}
for f in glob.glob("constant/triSurface/*.stl"):
    name = f.split("/")[-1][:-4]
    m = pv.read(f)
    print(name, m.n_points, m.n_cells)
    pl.add_mesh(m, color=colors[name], opacity=0.6 if name in ("walls","inlet","outlet") else 1.0)
pl.camera_position = 'iso'
p = pl.screenshot("/tmp/view.png")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
plt.figure(figsize=(8,8)); plt.imshow(mpimg.imread("/tmp/view.png")); plt.axis("off"); plt.show()

# -- cell 3 -------------------------------------------------------------------------
# Geometry looks correct — finned cylinder with dome in a square duct. Now write the blockMeshDict (co
import subprocess, textwrap, os
os.makedirs("system", exist_ok=True)

x0,x1,y0,y1,z0b,z1b = -DUCT/2, DUCT/2, -DUCT/2, DUCT/2, z_min, z_max
nx, ny, nz = 40, 40, 82
verts = [(x0,y0,z0b),(x1,y0,z0b),(x1,y1,z0b),(x0,y1,z0b),
         (x0,y0,z1b),(x1,y0,z1b),(x1,y1,z1b),(x0,y1,z1b)]
bmd = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices (%s);
blocks ( hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1) );
edges ();
boundary ();
""" % ("\n".join("    (%g %g %g)"%v for v in verts), nx, ny, nz)
open("system/blockMeshDict","w").write(bmd)

# locationInMesh: in the air gap between fin OD and duct wall, e.g. x=0.12,z=0.01
shmd = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       false;
geometry
{
    barrel  { type triSurfaceMesh; file "barrel.stl"; }
    fins    { type triSurfaceMesh; file "fins.stl"; }
    head    { type triSurfaceMesh; file "head.stl"; }
    inlet   { type triSurfaceMesh; file "inlet.stl"; }
    outlet  { type triSurfaceMesh; file "outlet.stl"; }
    walls   { type triSurfaceMesh; file "walls.stl"; }
}
castellatedMeshControls
{
    maxLocalCells 100000; maxGlobalCells 2000000;
    minRefinementCells 10; maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    {
        barrel { level (2 2); }
        fins   { level (3 3); }
        head   { level (2 2); }
        inlet  { level (1 1); }
        outlet { level (1 1); }
        walls  { level (1 1); }
    }
    resolveFeatureAngle 30;
    refinementRegions { }
    locationInMesh (0.12 0.0 0.006);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3; tolerance 2.0;
    nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    layers { }
    expansionRatio 1.0; finalLayerThickness 0.3;
    minThickness 0.1; nGrow 0;
    featureAngle 60; slipFeatureAngle 30;
    nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}
meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality -1;
    minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1;
    nSmoothScale 4; errorReduction 0.75;
}
"""
open("system/snappyHexMeshDict","w").write(shmd)
print("files written")
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-500:])

# -- cell 4 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=230)
print(r.stdout[-2000:])
print("rc", r.returncode)

# -- cell 5 -------------------------------------------------------------------------
# It exited with rc=1 quickly — need the actual error:
import subprocess
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
log = r.stdout + r.stderr
import re
# print last 40 lines
print("\n".join(log.splitlines()[-40:]))
print("rc", r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# Just a missing controlDict:
cd = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
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
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
open("system/controlDict","w").write(cd)
import subprocess
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=230)
print("rc", r.returncode)
print("\n".join(r.stdout.splitlines()[-25:]))

# -- cell 7 -------------------------------------------------------------------------
print("\n".join(r.stderr.splitlines()[-25:]))
print("---- stdout full ----")
print("\n".join(r.stdout.splitlines()[35:80]))

# -- cell 8 -------------------------------------------------------------------------
import subprocess, os
r2 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("rc", r2.returncode)
print("\n".join(r2.stderr.splitlines()[-10:]))
print("\n".join(r2.stdout[-800:].splitlines()))
print(os.path.exists("constant/polyMesh/points"))

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=235)
print("rc", r.returncode)
out = r.stdout
print("\n".join(out.splitlines()[-30:]))

# -- cell 10 ------------------------------------------------------------------------
print("\n".join(r.stderr.splitlines()[-15:]))

# -- cell 11 ------------------------------------------------------------------------
# Just needs empty fvSchemes/fvSolution stubs:
open("system/fvSchemes","w").write(
"FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n")
open("system/fvSolution","w").write(
"FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=235)
print("rc", r.returncode)
print("\n".join((r.stdout+r.stderr).splitlines()[-20:]))

# -- cell 12 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default none; }
gradSchemes { default none; }
divSchemes { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default none; }
snGradSchemes { default none; }
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=235)
print("rc", r.returncode)
print("\n".join((r.stdout+r.stderr).splitlines()[-20:]))

# -- cell 13 ------------------------------------------------------------------------
# It's meshing but slow (fin refinement level 3 everywhere). Launching snappy in the background with a
import subprocess, os
p = subprocess.Popen(["snappyHexMesh","-overwrite"], stdout=open("log.snappy","w"),
                     stderr=subprocess.STDOUT)
print("pid", p.pid)

# -- cell 14 ------------------------------------------------------------------------
import time, subprocess, os
time.sleep(90)
if os.path.exists("log.snappy"):
    t = open("log.snappy").read()
    print("\n".join(t.splitlines()[-20:]))
print("pid alive:", subprocess.run(["ps","-p","280920"],capture_output=True,text=True).stdout.count("snappy"))

# -- cell 15 ------------------------------------------------------------------------
# It crashed — inspecting the error part of the log:
t = open("log.snappy").read()
lines = t.splitlines()
for i,l in enumerate(lines):
    if "FATAL" in l or "error" in l.lower() or "Error" in l:
        print("\n".join(lines[max(0,i-6):i+8])); print("=====")
        break
