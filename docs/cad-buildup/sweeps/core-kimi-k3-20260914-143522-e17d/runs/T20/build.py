"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The fluid domain is a simple cylinder: R = 30 mm, L = 42 mm (20 flange + 2 gasket + 20 flange). I'll
import os, build123d as bd
R, L, t = 0.03, 0.042, 0.02
g = 0.002  # gasket band
os.makedirs("constant/triSurface", exist_ok=True)
wall1 = bd.Cylinder(radius=R, height=t, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
gasket = bd.Cylinder(radius=R, height=g, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)).moved(bd.Location((0,0,t)))
wall2 = bd.Cylinder(radius=R, height=t, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)).moved(bd.Location((0,0,t+g)))
inlet  = bd.Cylinder(radius=R, height=0.0005, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
outlet = bd.Cylinder(radius=R, height=0.0005, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)).moved(bd.Location((0,0,L-0.0005)))
for name, s in [("flange1", wall1), ("gasket", gasket), ("flange2", wall2), ("inlet", inlet), ("outlet", outlet)]:
    bd.export_stl(s, f"constant/triSurface/{name}.stl")
print("volumes:", wall1.volume, inlet.volume, "L =", L)

# -- cell 2 -------------------------------------------------------------------------
import json, subprocess
bm = f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices ( (-0.035 -0.035 -0.01) (0.035 -0.035 -0.01) (0.035 0.035 -0.01) (-0.035 0.035 -0.01)
           (-0.035 -0.035  0.052) (0.035 -0.035  0.052) (0.035 0.035  0.052) (-0.035 0.035  0.052) );
blocks ( hex (0 1 2 3 4 5 6 7) (16 16 16) simpleGrading (1 1 1) );
edges (); boundary ( defaultFaces {{ type wall; faces ((0 1 5 4)); }} ); mergePairs ();
"""
os.makedirs("system", exist_ok=True)
open("system/blockMeshDict","w").write(bm)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:])

# -- cell 3 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode); print(r.stderr[-800:])

# -- cell 4 -------------------------------------------------------------------------
# blockMesh needs a controlDict; I'll write one plus the snappyHexMeshDict in the same step.
open("system/controlDict","w").write('''
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom latestTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6; timePrecision 6; runTimeModifiable true;
''')
smd = '''
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
 inlet.stl  { type triSurfaceMesh; name inlet; }
 outlet.stl { type triSurfaceMesh; name outlet; }
 flange1.stl { type triSurfaceMesh; name flange1; }
 flange2.stl { type triSurfaceMesh; name flange2; }
 gasket.stl  { type triSurfaceMesh; name gasket; }
}
castellatedMeshControls
{
 maxLocalCells 100000; maxGlobalCells 2000000; minRefinementCells 10; maxLoadUnbalance 0.1;
 nCellsBetweenLevels 1; resolveFeatureAngle 30;
 features ();
 refinementSurfaces { inlet { level (1 1); } outlet { level (1 1); }
   flange1 { level (1 1); } flange2 { level (1 1); } gasket { level (1 1); } }
 refinementRegions {};
 locationInMesh (0 0 0.021);
 allowFreeStandingZoneFaces false;
}
snapControls { nSmoothPatch 3; tolerance 1.5; nSolveIter 20; nRelaxIter 5; nFeatureSnapIter 10; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
  maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02;
  minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01; minTriangleRotation 180;
  nSmoothScale 4; errorReduction 0.75; }
'''
open("system/snappyHexMeshDict","w").write(smd)
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=200)
print(r.returncode); print(r.stdout[-600:])

# -- cell 5 -------------------------------------------------------------------------
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=200)
print(p.returncode); print(p.stderr[-1500:])

# -- cell 6 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1000:]); print(r.stderr[-1500:])

# -- cell 7 -------------------------------------------------------------------------
# The boundary dict must cover all six faces explicitly. Fixing:
bm = open("system/blockMeshDict").read()
bm = bm.replace("boundary ( defaultFaces { type wall; faces ((0 1 5 4)); } );",
 "boundary ( defaultFaces { type wall; faces ((0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7) (4 5 6 7) (0 3 2 1)); } );")
open("system/blockMeshDict","w").write(bm)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).returncode, "blockMesh ok")
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=200)
print(p.returncode); print(p.stdout[-500:])

# -- cell 8 -------------------------------------------------------------------------
import subprocess
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.returncode); print(p.stderr[-2000:])

# -- cell 9 -------------------------------------------------------------------------
# It needs fvSchemes/fvSolution too. Adding minimal ones:
open("system/fvSchemes","w").write('''
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {} interpolationSchemes {} snGradSchemes {} wallDist { meshWave; }
''')
open("system/fvSolution","w").write('''
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
relaxationFactors {} solvers {}
''')
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.returncode, p.stderr[-300:])
print(p.stdout[p.stdout.rfind("Finished"):][:800] if "Finished" in p.stdout else p.stdout[-1500:])

# -- cell 10 ------------------------------------------------------------------------
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.stderr[-3000:])

# -- cell 11 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace("snapControls {",
 "addLayersControls { relativeSizes true; layers (); thicknessRatio 1.3; thickness 0.5; minThickness 0.1; nGrow 1; maxFaceASqr 3; fAngleScale 1; "
 "featureAngle -180; slipFeatureAngle 30; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; "
 "maxLayeringIterations 3; meshGamma 0.6; nPerCell 50; nIterator 5; moveMedian false; "
 "minMedialAxisAngle 90; maxThicknessToMedialRatio 0.3; nSmoothMesh 10; nSmoothPost 10; nSmoothTopology 10; layersOn { } } snapControls {")
open("system/snappyHexMeshDict","w").write(s)
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.returncode); print(p.stderr[-500:])

# -- cell 12 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace("meshQualityControls", "mergeTolerance 1e-6; meshQualityControls")
open("system/snappyHexMeshDict","w").write(s)
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.returncode); print(p.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
import re
m = p.stderr.find("Entry")
print(p.stderr[max(0,m-100):m+300])

# -- cell 14 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace("minTriangleRotation 180;", "minTriangleRotation 0.5; minTriangleTwist 0;")
open("system/snappyHexMeshDict","w").write(s)
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.returncode)
import re
m = p.stderr.find("Entry"); print(p.stderr[max(0,m-100):m+200] if m>=0 else p.stderr[-300:])

# -- cell 15 ------------------------------------------------------------------------
# snappyHexMesh succeeded. Now checkMesh:
p = subprocess.run(["checkMesh"], capture_output=True, text=True, timeout=120)
print(p.stdout[-2000:])

# -- cell 16 ------------------------------------------------------------------------
# Let me visualize properly with foamToVTK:
import subprocess, glob, pyvista as pv, numpy as np
subprocess.run(["foamToVTK", "-allPatches" if False else ""], capture_output=True, text=True)
files = sorted(glob.glob("VTK/*/boundary_*.vtu"))
print(files)
c = sorted(glob.glob("VTK/*/internal.vtu"))
m = pv.read(c[-1]) if c else None
if m is not None:
    import matplotlib
    pv.set_plot_theme("document")
    pl = pv.Plotter(off_screen=True)
    clip = m.slice_orthogonal()
    pl.add_mesh(m.slice(normal=[1,0,0]), opacity=0.7)
    pl.add_mesh(m.extract_surface(), opacity=0.3)
    pl.show_bounds()
    pl.screenshot("mesh.png")
    pv.read("mesh.png").plot(console_enabled=False)
    print("bbox", m.bounds)

# -- cell 17 ------------------------------------------------------------------------
import subprocess, glob
p = subprocess.run(["foamToVTK"], capture_output=True, text=True)
print(p.returncode, p.stderr[-300:]); print(p.stdout[-300:])
print(sorted(glob.glob("VTK/**/*", recursive=True)))

# -- cell 18 ------------------------------------------------------------------------
import pyvista as pv, matplotlib.pyplot as plt, glob
for f in sorted(glob.glob("VTK/t20_?/internal.vtu")):
    mm = pv.read(f); print(f, mm.n_cells, mm.bounds)
m = pv.read("VTK/t20_3/internal.vtu")
pts = m.slice(normal=[1,0,0]).points
plt.figure(figsize=(8,3)); plt.scatter(pts[:,1], pts[:,2], s=1)
plt.gca().set_aspect("equal"); plt.title("final internal slice (y-z)"); plt.show()

# -- cell 19 ------------------------------------------------------------------------
# Re-emitting the full surface export self-contained:
import build123d as bd, os
from build123d import Cylinder, Face, Align, Location
R, L, t, g = 0.03, 0.042, 0.02, 0.002
os.makedirs("constant/triSurface", exist_ok=True)
def side(z0, h):
    c = Cylinder(radius=R, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0,0,z0)))
    return [f for f in c.faces() if f.geom_type == bd.GeomType.CYLINDER][0]
flange1_f, gasket_f, flange2_f = side(0,t), side(t,g), side(t+g,t)
thin = Cylinder(radius=R, height=0.001, align=(Align.CENTER, Align.CENTER, Align.MIN))
caps = sorted([f for f in thin.faces() if f.geom_type == bd.GeomType.PLANE], key=lambda f: f.center().Z)
inlet_f, outlet_f = caps[0], caps[-1].moved(Location((0,0,L-0.001)))
for name, f in [("flange1",flange1_f),("gasket",gasket_f),("flange2",flange2_f),("inlet",inlet_f),("outlet",outlet_f)]:
    bd.export_stl(f, f"constant/triSurface/{name}.stl")
    print(name, round(f.area,7), tuple(f.center()))

# -- cell 20 ------------------------------------------------------------------------
# Surfaces are correct (areas match 2πR·h and πR²). Re-running the whole meshing chain:
import subprocess
subprocess.run(["blockMesh"], capture_output=True)
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True, timeout=220)
print(p.returncode, p.stderr[-200:])
q = subprocess.run(["checkMesh"], capture_output=True, text=True)
import re
print(q.stdout.split("Checking geometry")[1][:600])
print("patches:", re.findall(r"\n\n?(\w+)\n", q.stdout)) if False else None

# -- cell 21 ------------------------------------------------------------------------
p = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=220)
print(p.returncode)
print(p.stdout[-3500:])

# -- cell 22 ------------------------------------------------------------------------
txt = p.stdout
i = txt.find("Removing disconnected")
print(txt[max(0,i-500):i+800] if i>=0 else "not found")

# -- cell 23 ------------------------------------------------------------------------
for k in ["region", "Zone", "zone", "inside", "walk"]:
    idx = txt.find(k)
    print("==", k, idx)
    if idx>=0: print(txt[max(0,idx-300):idx+600], "\n")

# -- cell 24 ------------------------------------------------------------------------
q = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(q.stdout[-1500:])
