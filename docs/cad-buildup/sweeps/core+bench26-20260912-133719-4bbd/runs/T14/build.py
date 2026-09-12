"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Re-sending as a tool call.
import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("constant/triSurface", exist_ok=True)

def naca2412(n=80):
    m, p, t = 0.02, 0.4, 0.12
    beta = np.linspace(0, np.pi, n)
    x = 0.5*(1-np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
    dyc = np.where(x < p, 2*m/p**2*(p-x), 2*m/(1-p)**2*(p-x))
    th = np.arctan(dyc)
    xu, yu = x-yt*np.sin(th), yc+yt*np.cos(th)
    xl, yl = x+yt*np.sin(th), yc-yt*np.cos(th)
    return (xu, yu), (xl, yl)

(xu, yu), (xl, yl) = naca2412()
print("max thickness/c =", (np.interp(xl, xu, yu)-yl).max())
print("TE gap =", yu[-1]-yl[-1], " LE point:", xu[0], yu[0])
plt.plot(xu, yu, '-', xl, yl, '-'); plt.axis('equal'); plt.grid(True); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The previous cell errored only on the `is_valid` call; re-emitting the wing construction with the fi
from build123d import *

C_ROOT, C_TIP, SPAN, STUB = 0.120, 0.080, 0.200, 0.010

def section(chord, y):
    pts = list(zip(xu, yu))[::-1] + list(zip(xl, yl))[1:]
    p3 = [(px*chord, y, pz*chord) for px, pz in pts]
    return make_face(Polyline(*p3, close=True))

wing = loft([section(C_ROOT, 0.0), section(C_TIP, SPAN)]) \
       + extrude(section(C_ROOT, 0.0), amount=STUB, dir=(0,-1,0))
print("solid:", wing.is_valid, "vol =", wing.volume)
print("bbox:", wing.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
# Now export the wing STL and measure the actual chords/thickness from the tessellation.
import numpy as np
from build123d import export_stl
export_stl(wing, "constant/triSurface/wing.stl", tolerance=1e-5, angular_tolerance=0.1)
import pyvista as pv
m = pv.read("constant/triSurface/wing.stl")
p = m.points
for yq, cexp in [(0.0, 0.120), (0.100, 0.100), (0.200, 0.080)]:
    sl = m.slice(normal=(0,1,0), origin=(0.06, yq if yq > 0 else 1e-4, 0))
    q = sl.points
    print(f"y={yq}: chord={q[:,0].max()-q[:,0].min():.5f} (expect {cexp}), "
          f"t/c={(q[:,2].max()-q[:,2].min())/(q[:,0].max()-q[:,0].min()):.4f}, LE x={q[:,0].min():.5f}")
print("span root->tip:", p[:,1].max()-0.0, " stub to y=", p[:,1].min())

# -- cell 4 -------------------------------------------------------------------------
# Chords 0.120/0.100/0.080 with straight unswept LE — taper and no-sweep confirmed. Now the domain box
XMIN, XMAX = -4*C_ROOT, 0.12 + 8*C_ROOT          # 4c upstream of LE, 8c downstream of TE
YMIN, YMAX = 0.0, SPAN + 3*C_ROOT                 # root wall at y=0, 3c beyond tip
ZMIN, ZMAX = -3*C_ROOT, 3*C_ROOT
print("domain x", XMIN, XMAX, "y", YMIN, YMAX, "z", ZMIN, ZMAX)

def rect(p0, p1):   # axis-aligned rectangle face from two opposite corners
    a, b = np.array(p0, float), np.array(p1, float)
    k = int(np.argmin(np.abs(b-a)))
    idx = [i for i in range(3) if i != k]
    c1, c2 = a.copy(), a.copy()
    c1[idx[0]] = b[idx[0]]; c2[idx[1]] = b[idx[1]]
    return make_face(Polyline(tuple(a), tuple(c1), tuple(b), tuple(c2), close=True))

patch_faces = {
  "inlet":    [rect((XMIN,YMIN,ZMIN), (XMIN,YMAX,ZMAX))],
  "outlet":   [rect((XMAX,YMIN,ZMIN), (XMAX,YMAX,ZMAX))],
  "rootWall": [rect((XMIN,YMIN,ZMIN), (XMAX,YMIN,ZMAX))],
  "farField": [rect((XMIN,YMAX,ZMIN), (XMAX,YMAX,ZMAX)),
               rect((XMIN,YMIN,ZMIN), (XMAX,YMAX,ZMIN)),
               rect((XMIN,YMIN,ZMAX), (XMAX,YMAX,ZMAX))],
}
for name, fs in patch_faces.items():
    export_stl(Compound(children=fs) if len(fs) > 1 else fs[0], f"constant/triSurface/{name}.stl")
    print(name, "area =", sum(f.area for f in fs))

# -- cell 5 -------------------------------------------------------------------------
# Background hex mesh: uniform 0.04 m cells (39×14×18 ≈ 10k) with the four patch names set on the bloc
import os, textwrap, subprocess
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(head % "controlDict" + "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head % "fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head % "fvSolution" + "solvers{} \n")

BASE = 0.04
nx, ny, nz = [max(1, int(round(l/BASE))) for l in (XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)]
bm = head % "blockMeshDict" + f"""
scale 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary
(
 inlet    {{ type patch; faces ((0 4 7 3)); }}
 outlet   {{ type patch; faces ((1 2 6 5)); }}
 rootWall {{ type wall; faces ((0 1 5 4)); }}
 farField {{ type patch; faces ((3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(bm)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(nx, ny, nz, r.returncode); print(r.stdout[-700:])

# -- cell 6 -------------------------------------------------------------------------
sfe = head % "surfaceFeatureExtractDict" + """
wing.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
"""
open("system/surfaceFeatureExtractDict","w").write(sfe)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-400:])

# -- cell 7 -------------------------------------------------------------------------
# Coarse snappy pass: surface levels (3 4), trailing-edge features, no layers yet.
def write_snappy(lev_min, lev_max, layers=0):
    d = head % "snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers {"true" if layers else "false"};
geometry {{ wing.stl {{ type triSurfaceMesh; name wing; }} }};
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 3; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ( {{ file "wing.eMesh"; level {lev_max}; }} );
  refinementSurfaces {{ wing {{ level ({lev_min} {lev_max}); patchInfo {{ type wall; }} }} }}
  refinementRegions {{ }}
  locationInMesh (-0.30 0.30 0.20);
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{ relativeSizes true; layers {{ wing {{ nSurfaceLayers {layers}; }} }}
  expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1; nGrow 0;
  featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50; }}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
"""
    open("system/snappyHexMeshDict","w").write(d)

write_snappy(3, 4)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 8 -------------------------------------------------------------------------
# 43k cells, no quality errors. Look at it before refining.
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
mesh = rd.read()
bnd = mesh["boundary"]
print(list(bnd.keys()))
pl = pv.Plotter(off_screen=True, window_size=(1100,500), shape=(1,2))
pl.subplot(0,0); pl.add_mesh(bnd["wing"], color="orange", show_edges=True)
pl.add_mesh(bnd["rootWall"].extract_feature_edges(), color="k")
pl.camera_position = "iso"
pl.subplot(0,1)
sl = mesh["internalMesh"].slice(normal=(0,1,0), origin=(0.06,0.10,0))
pl.add_mesh(sl, show_edges=True, color="w"); pl.camera_position="xz"; pl.camera.zoom(3.0)
pl.screenshot("look1.png"); print("ok")

# -- cell 9 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look1.png"))

# -- cell 10 ------------------------------------------------------------------------
# Shape is right. Now the production pass: surface levels (4 5) (≈1.25 mm on the wing), a refinement b
dict_txt = open("system/snappyHexMeshDict").read()
write_snappy(4, 5, layers=3)
d = open("system/snappyHexMeshDict").read()
d = d.replace('geometry { wing.stl { type triSurfaceMesh; name wing; } };',
  'geometry { wing.stl { type triSurfaceMesh; name wing; } '
  'nearBox { type searchableBox; min (-0.06 -0.01 -0.09); max (0.55 0.27 0.09); } };')
d = d.replace('refinementRegions {  }', 'refinementRegions { nearBox { mode inside; levels ((1e15 2)); } }')
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1200:])

# -- cell 11 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 12 ------------------------------------------------------------------------
# Mesh OK. Final geometric verification from the meshed wing patch itself.
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
mesh = rd.read(); bnd = mesh["boundary"]; w = bnd["wing"]
print("wing patch bbox:", np.round(w.bounds, 5))
for yq in (0.002, 0.100, 0.198):
    q = w.slice(normal=(0,1,0), origin=(0.06, yq, 0)).points
    print(f" y={yq}: chord={q[:,0].max()-q[:,0].min():.5f}  LE x={q[:,0].min():.5f}")
box = (XMAX-XMIN)*(YMAX-YMIN)*(ZMAX-ZMIN)
print("box vol", round(box,6), " mesh vol 0.628826  -> displaced", round(box-0.628826,7),
      " expected wing(y>0)", round(wing.volume - 0.010*wing.volume/0.21*0 - 1.176e-5, 7))
print("rootWall patch area:", bnd["rootWall"].compute_cell_sizes()["Area"].sum(), "vs solid-wall 1.1232 minus root section")
