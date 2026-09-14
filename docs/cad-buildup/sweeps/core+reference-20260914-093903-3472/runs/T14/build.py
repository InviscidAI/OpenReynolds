"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, build123d as bd

def naca2412(chord, n=120, m=0.02, p=0.4, t=0.12):
    beta = np.linspace(0, np.pi, n)
    xc = (1 - np.cos(beta)) / 2
    yt = 5*t*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    yc = np.where(xc < p, m/p**2*(2*p*xc - xc**2), m/(1-p)**2*((1-2*p) + 2*p*xc - xc**2))
    dyc = np.where(xc < p, 2*m/p**2*(p - xc), 2*m/(1-p)**2*(p - xc))
    th = np.arctan(dyc)
    xu, yu = xc - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = xc + yt*np.sin(th), yc - yt*np.cos(th)
    pts = list(zip(xu, yu))[::-1] + list(zip(xl, yl))[1:]
    return [(chord*a, chord*b) for a, b in pts]

r = naca2412(1.0)
xs = np.array([p[0] for p in r]); ys = np.array([p[1] for p in r])
print("pts", len(r), "x range", xs.min(), xs.max())
xu = np.linspace(0.001, 0.999, 2000)
m, p, t = 0.02, 0.4, 0.12
yt = 5*t*(0.2969*np.sqrt(xu)-0.1260*xu-0.3516*xu**2+0.2843*xu**3-0.1015*xu**4)
yc = np.where(xu < p, m/p**2*(2*p*xu-xu**2), m/(1-p)**2*((1-2*p)+2*p*xu-xu**2))
print("t/c =", 2*yt.max(), "at x/c =", xu[np.argmax(yt)])
print("max camber =", yc.max(), "at x/c =", xu[np.argmax(yc)])

# -- cell 2 -------------------------------------------------------------------------
# The previous cell errored, so re-emit the whole build (measurements inside), using `is_valid` as a p
from build123d import *

C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200   # m, from the request
ROOT_EXT = 0.010                            # my choice: stub below the wall for a clean snappy cut

def section(chord, z):
    pts = [(x, y, z) for x, y in naca2412(chord)]
    return make_face(Polyline(*pts, close=True))

f_root = section(C_ROOT, 0.0)
f_tip  = section(C_TIP, SPAN)
wing_solid = loft([f_root, f_tip]) + extrude(f_root, amount=-ROOT_EXT)

print("volume", wing_solid.volume, "valid", wing_solid.is_valid, "solids", len(wing_solid.solids()))
bb = wing_solid.bounding_box()
print("bbox min", bb.min, "max", bb.max)
for z in [0.0, 0.1, SPAN - 1e-6]:
    b = (wing_solid & Plane(origin=(0, 0, z), z_dir=(0, 0, 1))).bounding_box()
    print(f"z={z:.3f} chord={b.max.X-b.min.X:.5f} y-extent={b.max.Y-b.min.Y:.5f} LE x={b.min.X:.5f}")

# -- cell 3 -------------------------------------------------------------------------
# Chords 120.0 / 100.0 / 80.0 mm, leading edge at x≈0 at every z (no sweep), sections unrotated (no tw
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

XUP, XDN, SIDE = 4*C_ROOT, 8*C_ROOT, 3*C_ROOT      # 0.48, 0.96, 0.36 m from the request
X0, X1 = -XUP, XDN
Y0, Y1 = -SIDE, SIDE
Z0, Z1 = 0.0, SPAN + SIDE                           # root wall at z=0, 3 chords above the tip
print("domain", (X0, X1), (Y0, Y1), (Z0, Z1))

box = Box(X1-X0, Y1-Y0, Z1-Z0).moved(Location(((X0+X1)/2, (Y0+Y1)/2, (Z0+Z1)/2)))
fs = box.faces()
inlet    = fs.filter_by(lambda f: abs(f.center().X - X0) < 1e-9)
outlet   = fs.filter_by(lambda f: abs(f.center().X - X1) < 1e-9)
root     = fs.filter_by(lambda f: abs(f.center().Z - Z0) < 1e-9)
farfield = fs.filter_by(lambda f: abs(f.center().Y - Y0) < 1e-9 or abs(f.center().Y - Y1) < 1e-9
                                  or abs(f.center().Z - Z1) < 1e-9)
print([len(x) for x in (inlet, outlet, root, farfield)])

for name, shp in [("wing", wing_solid), ("inlet", Compound(inlet)), ("outlet", Compound(outlet)),
                  ("root", Compound(root)), ("farfield", Compound(farfield))]:
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.2)
print(subprocess.run(["ls", "-l", "constant/triSurface"], capture_output=True, text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# Case files: blockMesh box with the four named external patches, then snappyHexMesh to cut the wing. 
import os, textwrap
for d in ["system", "constant"]:
    os.makedirs(d, exist_ok=True)

head = lambda cls, obj: f"""FoamFile
{{ version 2.0; format ascii; class {cls}; object {obj}; }}
"""
open("system/controlDict", "w").write(head("dictionary", "controlDict") + """
application     simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
""")
open("system/fvSchemes", "w").write(head("dictionary", "fvSchemes") + """
ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; }
divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; }
""")
open("system/fvSolution", "w").write(head("dictionary", "fvSolution") + """
solvers {} 
""")

NB = 0.04  # base cell size, m
nx, ny, nz = [int(round(l/NB)) for l in (X1-X0, Y1-Y0, Z1-Z0)]
print("blockMesh cells", nx, ny, nz, nx*ny*nz)
bm = head("dictionary", "blockMeshDict") + f"""
scale 1;
vertices
(
 ({X0} {Y0} {Z0}) ({X1} {Y0} {Z0}) ({X1} {Y1} {Z0}) ({X0} {Y1} {Z0})
 ({X0} {Y0} {Z1}) ({X1} {Y0} {Z1}) ({X1} {Y1} {Z1}) ({X0} {Y1} {Z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
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
open("system/blockMeshDict", "w").write(bm)
import subprocess
r = subprocess.run(["blockMesh", "-dict", "system/blockMeshDict"], capture_output=True, text=True)
print(r.stdout[-700:], r.stderr[-500:])

# -- cell 5 -------------------------------------------------------------------------
snappy = head("dictionary", "snappyHexMeshDict") + """
castellatedMesh true; snap true; addLayers false;

geometry
{
    wing.stl { type triSurfaceMesh; name wing; }
}

castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10;
    nCellsBetweenLevels 2; maxLoadUnbalance 0.1; resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    locationInMesh (-0.30 0.20 0.40);
    features ();
    refinementSurfaces { wing { level (3 3); patchInfo { type wall; } } }
    refinementRegions { wing { mode distance; levels ((0.03 2) (0.08 1)); } }
}

snapControls
{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }

addLayersControls
{ relativeSizes true; layers {}; expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }

meshQualityControls
{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }

mergeTolerance 1e-6;
writeFlags ( );
"""
open("system/snappyHexMeshDict", "w").write(snappy)
r = subprocess.run(["snappyHexMesh", "-dict", "system/snappyHexMeshDict", "-overwrite"], capture_output=True, text=True)
print(r.stdout[-1500:]); print("ERR", r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
# Coarse snap done, 16.7k cells. Look at the wing patch in the mesh.
import pyvista as pv
pv.start_xvfb() if False else None
pv.OFF_SCREEN = True
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
rd.enable_all_patch_arrays(); mesh = rd.read()
print(mesh.keys())
bd_ = mesh["boundary"]
print(bd_.keys())
w = bd_["wing"]
print("wing patch bounds", w.bounds, "area", w.extract_surface().area)
pl = pv.Plotter(off_screen=True, window_size=(1000,700))
pl.add_mesh(w, color="tan", show_edges=True)
pl.add_mesh(bd_["root"], color="lightblue", opacity=0.35, show_edges=False)
pl.camera_position = [(0.5,-0.6,0.5),(0.06,0,0.1),(0,0,1)]
pl.screenshot("wing_coarse.png"); print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("wing_coarse.png"))

# -- cell 8 -------------------------------------------------------------------------
# Topology is right; the trailing edge is ragged at level 3. Add feature-edge extraction and refine th
open("system/surfaceFeatureExtractDict","w").write(head("dictionary","surfaceFeatureExtractDict") + """
wing.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    subsetFeatures { nonManifoldEdges no; openEdges yes; }
    writeObj no;
}
""")
r = subprocess.run(["surfaceFeatureExtract","-dict","system/surfaceFeatureExtractDict"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-300:])
print(subprocess.run(["ls","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 9 -------------------------------------------------------------------------
snappy2 = snappy.replace("features ();", 'features ( { file "wing.eMesh"; level 5; } );') \
                .replace("level (3 3);", "level (4 5);") \
                .replace("levels ((0.03 2) (0.08 1));", "levels ((0.02 4) (0.06 3) (0.15 2));")
open("system/snappyHexMeshDict","w").write(snappy2)
subprocess.run(["blockMesh","-dict","system/blockMeshDict"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-dict","system/snappyHexMeshDict","-overwrite"], capture_output=True, text=True)
open("log.snappy","w").write(r.stdout)
print(r.stdout[-1200:]); print("ERR", r.stderr[-300:])

# -- cell 10 ------------------------------------------------------------------------
rd2 = pv.OpenFOAMReader("case.foam"); rd2.enable_all_patch_arrays(); m2 = rd2.read()
b2 = m2["boundary"]; w2 = b2["wing"]
print("wing bounds", w2.bounds)
print("wing area", w2.extract_surface(algorithm="dataset_surface").area)
print("patches", list(b2.keys()), [b2[k].n_cells for k in b2.keys()])
pl = pv.Plotter(off_screen=True, window_size=(1100,750), shape=(1,2))
pl.subplot(0,0); pl.add_mesh(w2, color="tan", show_edges=True); pl.add_mesh(b2["root"], color="lightblue", opacity=0.3)
pl.camera_position=[(0.45,-0.45,0.4),(0.06,0,0.1),(0,0,1)]
pl.subplot(0,1)
sl = m2["internalMesh"].slice(normal="z", origin=(0,0,0.1))
pl.add_mesh(sl, color="white", show_edges=True); pl.camera_position="xy"; pl.camera.zoom(1.0)
pl.screenshot("wing_ref.png"); display(Image("wing_ref.png"))

# -- cell 11 ------------------------------------------------------------------------
# Trailing edge now snapped (x_max 0.11999 m). Measure chords from the meshed wing patch itself, then 
ws = w2.extract_surface(algorithm="dataset_surface")
for z in [0.002, 0.100, 0.198]:
    c = ws.slice(normal="z", origin=(0,0,z))
    print(f"z={z:.3f}  chord={c.bounds[1]-c.bounds[0]:.5f}  expected={0.120-(0.120-0.080)*z/0.200:.5f}  LE x={c.bounds[0]:+.5f}")
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[-1600:])
