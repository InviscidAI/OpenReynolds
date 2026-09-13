"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("constant/triSurface", exist_ok=True)

def naca2412(n=80):
    m, p, t = 0.02, 0.40, 0.12
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    px = np.concatenate([xu[::-1], xl[1:]])
    py = np.concatenate([yu[::-1], yl[1:]])
    return px, py

px, py = naca2412()
print(len(px), px[0], py[0], px[-1], py[-1], "thickness span:", (py.max()-py.min()))
plt.figure(figsize=(8,2)); plt.plot(px, py, '-o', ms=2); plt.axis('equal'); plt.grid(True); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# Profile is clean and closed at the TE. Now the solid: loft root (c=0.12, z=0) to tip (c=0.08, z=0.2)
from build123d import *

CROOT, CTIP, SPAN = 0.120, 0.080, 0.200
def section(chord, z):
    pts = [(chord*a, chord*b, z) for a, b in zip(px, py)]
    return Face(Wire(Polyline(*pts, close=True)))

root_f = section(CROOT, 0.0)
tip_f  = section(CTIP, SPAN)
wing = loft([root_f, tip_f])
stub = extrude(section(CROOT, 0.0), amount=-0.02)   # below root wall, trimmed away later
wing = wing + stub
print("wing volume m^3:", wing.volume, " bbox:", wing.bounding_box())
print("root chord:", root_f.bounding_box().size.X, " tip chord:", tip_f.bounding_box().size.X)

# -- cell 3 -------------------------------------------------------------------------
# Volume check: sectional area coefficient implied is 0.0818·c², matching NACA 2412 (~0.082). Now the 
XMIN, XMAX = -4*CROOT, CROOT + 8*CROOT
YMIN, YMAX = -3*CROOT, 3*CROOT
ZMIN, ZMAX = 0.0, SPAN + 3*CROOT
box = Pos((XMIN+XMAX)/2, 0, (ZMIN+ZMAX)/2) * Box(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
fluid = box - wing
print("domain:", (XMIN,XMAX),(YMIN,YMAX),(ZMIN,ZMAX))
print("fluid vol:", fluid.volume, "box vol:", box.volume, "diff:", box.volume-fluid.volume)
print("n faces:", len(fluid.faces()))

# -- cell 4 -------------------------------------------------------------------------
# Volume removed (1.657e-4) equals the wing above z=0 exactly. Now group the fluid's faces into the fi
import numpy as np
tol = 1e-9
groups = {"inlet":[], "outlet":[], "root":[], "farfield":[], "wing":[]}
for f in fluid.faces():
    c = f.center(); n = f.normal_at(c); bb = f.bounding_box()
    planar = f.geom_type == GeomType.PLANE
    if   planar and abs(bb.min.X-XMIN)<tol and abs(bb.max.X-XMIN)<tol: groups["inlet"].append(f)
    elif planar and abs(bb.min.X-XMAX)<tol and abs(bb.max.X-XMAX)<tol: groups["outlet"].append(f)
    elif planar and abs(bb.min.Z-ZMIN)<tol and abs(bb.max.Z-ZMIN)<tol: groups["root"].append(f)
    elif planar and (abs(bb.min.Y-YMIN)<tol and abs(bb.max.Y-YMIN)<tol or
                     abs(bb.min.Y-YMAX)<tol and abs(bb.max.Y-YMAX)<tol or
                     abs(bb.min.Z-ZMAX)<tol and abs(bb.max.Z-ZMAX)<tol): groups["farfield"].append(f)
    else: groups["wing"].append(f)
for k,v in groups.items(): print(k, len(v), "area:", sum(f.area for f in v))
print("expected inlet/outlet 0.72*0.56 =", 0.72*0.56, " farfield =", 2*1.56*0.56+1.56*0.72)
print("root face area (box face minus root section):", 1.56*0.72 - root_f.area)

# -- cell 5 -------------------------------------------------------------------------
# All five patch areas match analytically. Export one STL per patch (metres, no scaling needed).
from build123d import Compound, Mesher
import subprocess
for name, faces in groups.items():
    shp = Compound(children=[f for f in faces]) if len(faces) > 1 else faces[0]
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=2e-5, angular_tolerance=0.1, ascii_format=True)
    # rename solid to patch name so snappy/surfaceToPatch sees it
    p = f"constant/triSurface/{name}.stl"
    t = open(p).read().replace("solid ", f"solid {name} ", 1)
    import re
    lines = t.splitlines()
    lines = [f"solid {name}" if l.strip().startswith("solid") else (f"endsolid {name}" if l.strip().startswith("endsolid") else l) for l in lines]
    open(p,"w").write("\n".join(lines)+"\n")
subprocess.run("ls -la constant/triSurface", shell=True)
print(subprocess.run(["surfaceCheck","constant/triSurface/wing.stl"],capture_output=True,text=True).stdout[-900:])

# -- cell 6 -------------------------------------------------------------------------
# Wing STL is open only along its root curve — closed by root.stl in the union. Now a coarse backgroun
import os, textwrap, subprocess
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(head%"controlDict"+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head%"fvSchemes"+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head%"fvSolution"+"solvers{} \n")
NX, NY, NZ = 39, 18, 14
bm = head%"blockMeshDict" + f"""
scale 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet    {{ type patch; faces ((0 4 7 3)); }}
 outlet   {{ type patch; faces ((1 2 6 5)); }}
 root     {{ type wall;  faces ((0 3 2 1)); }}
 farfield {{ type patch; faces ((0 1 5 4) (3 7 6 2) (4 5 6 7)); }}
);
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(bm)
r = subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-600:], r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# Now snappy: feature extraction on the wing, refinement levels (3–4) plus a near-field box, castellat
open("system/surfaceFeatureExtractDict","w").write(head%"surfaceFeatureExtractDict"+
 "wing.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }\n")
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.stdout[-400:], r.returncode)
snap = head%"snappyHexMeshDict" + """
castellatedMesh true; snap true; addLayers false;
geometry
{
    wing.stl { type triSurfaceMesh; name wing; }
    nearBox { type searchableBox; min (-0.06 -0.09 -0.01); max (0.20 0.09 0.24); }
}
castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ( { file "wing.eMesh"; level 4; } );
    refinementSurfaces { wing { level (3 4); patchInfo { type wall; } } }
    refinementRegions { nearBox { mode inside; levels ((1e15 2)); } }
    locationInMesh (-0.30 0.10 0.30);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers{} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
    nGrow 0; featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snap)
print("ok")

# -- cell 8 -------------------------------------------------------------------------
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-2500:])

# -- cell 9 -------------------------------------------------------------------------
# Meshed to 41.6k cells with no quality errors. Check patch areas and look at the wing surface in the 
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-2300:])

# -- cell 10 ------------------------------------------------------------------------
# Volume matches the CAD fluid to 1e-6. Now look at it: render the wing patch and root plane.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
subprocess.run(["foamToVTK","-constant","-latestTime"],capture_output=True,text=True)
import glob
print(glob.glob("VTK/*"))

# -- cell 11 ------------------------------------------------------------------------
m = pv.read("VTK/t14_0.vtm")
names = [b for b in m.keys()] if hasattr(m,'keys') else None
print(m)

# -- cell 12 ------------------------------------------------------------------------
bnd = m[1]
print(bnd.keys() if hasattr(bnd,'keys') else bnd)
wing_s = bnd["wing"]; root_s = bnd["root"]
print("wing patch area:", wing_s.compute_cell_sizes(length=False,volume=False)["Area"].sum(), "CAD:", sum(f.area for f in groups["wing"]))
pl = pv.Plotter(off_screen=True, window_size=(1100,700))
pl.add_mesh(wing_s, color="orange", show_edges=True, line_width=0.4)
pl.add_mesh(root_s, color="lightsteelblue", show_edges=True, opacity=0.5, line_width=0.3)
pl.camera_position = [(0.6,-0.7,0.6),(0.1,0,0.12),(0,0,1)]
pl.screenshot("view_wing.png"); print("shot")

# -- cell 13 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view_wing.png"))

# -- cell 14 ------------------------------------------------------------------------
# Shape is right. Refine one level on the wing (surface 4–5, features 5) and re-check.
s = open("system/snappyHexMeshDict").read()
s = s.replace('file "wing.eMesh"; level 4;', 'file "wing.eMesh"; level 5;').replace("wing { level (3 4);", "wing { level (4 5);")
open("system/snappyHexMeshDict","w").write(s)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-700:])

# -- cell 15 ------------------------------------------------------------------------
# Right — the previous `-overwrite` replaced the background mesh; snappy must start from blockMesh aga
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-900:]); print(r.stderr[-500:])

# -- cell 16 ------------------------------------------------------------------------
# 69.6k cells, no errors. Now measure the snapped wing geometry itself: chord at root/mid/tip and max 
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print([l for l in r.stdout.splitlines() if any(k in l for k in ("Max skew","non-orthogonality","Total volume","Mesh OK","***","aspect"))])
subprocess.run(["foamToVTK","-constant","-latestTime"],capture_output=True,text=True)
m = pv.read("VTK/t14_0.vtm"); w = m[1]["wing"]
for z in (0.0005, 0.1, 0.1995):
    sl = w.slice(normal=(0,0,1), origin=(0,0,z))
    if sl.n_points:
        b=sl.bounds; print(f"z={z}: chord={b[1]-b[0]:.5f} expect {0.12-0.2*z:.5f}  thick={b[3]-b[2]:.5f} t/c={(b[3]-b[2])/(b[1]-b[0]):.4f}")
print("wing patch area:", w.compute_cell_sizes(length=False,volume=False)["Area"].sum())
print("span (z extent of wing):", w.bounds[4], w.bounds[5])

# -- cell 17 ------------------------------------------------------------------------
# All dimensions match. Final visual check plus a weld test of the five STLs (they should form one clo
import glob
subprocess.run("cat constant/triSurface/{inlet,outlet,root,farfield,wing}.stl > /tmp/all.stl", shell=True)
o=subprocess.run(["surfaceCheck","/tmp/all.stl"],capture_output=True,text=True).stdout
print([l for l in o.splitlines() if "closed" in l or "unconnected parts" in l or "one face" in l])
pl = pv.Plotter(off_screen=True, window_size=(1100,600))
pl.add_mesh(m[1]["wing"], color="orange", show_edges=True, line_width=0.3)
pl.add_mesh(m[0].slice(normal=(0,0,1), origin=(0,0,0.1)), color="w", show_edges=True, line_width=0.3, opacity=0.6)
pl.camera_position = [(0.2,-0.05,0.9),(0.2,-0.05,0.1),(0,1,0)]
pl.screenshot("view_slice.png"); display(Image("view_slice.png"))

# -- cell 18 ------------------------------------------------------------------------
# Slice looks right (cambered section, graded refinement). Redo the weld test without brace expansion.
names=["inlet","outlet","root","farfield","wing"]
with open("/tmp/all.stl","w") as out:
    for n in names: out.write(open(f"constant/triSurface/{n}.stl").read())
o=subprocess.run(["surfaceCheck","/tmp/all.stl"],capture_output=True,text=True).stdout
print([l.strip() for l in o.splitlines() if "closed" in l or "unconnected parts" in l or "one face" in l])
