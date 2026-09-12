"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: sphere D = 0.04 m centred at origin; background box x ∈ [−0.2, 0.6] (5D up, 15D down), y,z ∈ [
import os, subprocess
from build123d import *

D = 0.040          # sphere diameter [m]
R = D / 2.0

sphere = Sphere(radius=R)

print("volume  :", sphere.volume, " expected", 4/3*3.141592653589793*R**3)
print("area    :", sphere.area,   " expected", 4*3.141592653589793*R**2)
print("bbox    :", sphere.bounding_box())
os.makedirs("constant/triSurface", exist_ok=True)
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=1e-5, angular_tolerance=0.1)
print(os.path.getsize("constant/triSurface/sphere.stl"), "bytes")

# -- cell 2 -------------------------------------------------------------------------
# STL came out at ~1M triangles — far too fine. Re-tessellate coarser and check the triangle count and
import struct
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=1e-4, angular_tolerance=0.3)
sz = os.path.getsize("constant/triSurface/sphere.stl")
with open("constant/triSurface/sphere.stl","rb") as f:
    head = f.read(84)
ntri = struct.unpack("<I", head[80:84])[0]
print(sz, "bytes,", ntri, "triangles, header:", head[:40])

# -- cell 3 -------------------------------------------------------------------------
# Deflection isn't biting as expected; sweep it.
for tol, ang in [(5e-4,0.5),(2e-3,0.5),(1e-2,1.0)]:
    export_stl(sphere, "/tmp/s.stl", tolerance=tol, angular_tolerance=ang)
    with open("/tmp/s.stl","rb") as f: h=f.read(84)
    print(tol, ang, struct.unpack("<I", h[80:84])[0])

# -- cell 4 -------------------------------------------------------------------------
# Pick the 5e-4 tessellation (~21k triangles) and verify the actual surface deviation from the true sp
import numpy as np
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=5e-4, angular_tolerance=0.5)
raw = open("constant/triSurface/sphere.stl","rb").read()
n = struct.unpack("<I", raw[80:84])[0]
arr = np.frombuffer(raw[84:84+n*50].reshape if False else raw[84:84+n*50], dtype=np.uint8).reshape(n,50)
verts = np.frombuffer(arr[:,12:48].tobytes(), dtype="<f4").reshape(-1,3).astype(float)
r = np.linalg.norm(verts, axis=1)
print("tris", n, "vertex radius min/max:", r.min(), r.max(), "target", R)
# facet centroid radius -> sag
c = verts.reshape(n,3,3).mean(axis=1)
print("max sag:", R - np.linalg.norm(c,axis=1).min())

# -- cell 5 -------------------------------------------------------------------------
# Sag 17 µm (0.09 % of D) — good. Now the background box and blockMesh: domain x ∈ [−0.2, 0.6], y,z ∈ 
XMIN, XMAX = -5*D, 15*D
YMIN, YMAX = -5*D, 5*D
ZMIN, ZMAX = -5*D, 5*D
CELL = 0.02
NX = int(round((XMAX-XMIN)/CELL)); NY = int(round((YMAX-YMIN)/CELL)); NZ = int(round((ZMAX-ZMIN)/CELL))
print("domain", XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, "cells", NX, NY, NZ, NX*NY*NZ)

os.makedirs("system", exist_ok=True)
HEAD = """FoamFile{{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"""

blockMeshDict = f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile{{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}

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
    inlet    {{ type patch; faces ( (0 4 7 3) ); }}
    outlet   {{ type patch; faces ( (1 2 6 5) ); }}
    farfield {{ type patch; faces ( (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }}
);

mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(blockMeshDict)

controlDict = """FoamFile{ version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""
open("system/controlDict","w").write(controlDict)
open("system/fvSchemes","w").write("""FoamFile{ version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}
laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}
snGradSchemes{default corrected;}
""")
open("system/fvSolution","w").write("""FoamFile{ version 2.0; format ascii; class dictionary; object fvSolution; }
solvers{} SIMPLE{nNonOrthogonalCorrectors 0;}
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
# Now snappyHexMesh, castellate + snap only for the first look. Sphere refinement level 3 (2.5 mm ≈ 16
snappy = """FoamFile{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    sphere.stl { type triSurfaceMesh; name sphere; }
    wakeBox { type searchableBox; min (-0.06 -0.06 -0.06); max (0.30 0.06 0.06); }
}

castellatedMeshControls
{
    maxLocalCells 2000000;
    maxGlobalCells 5000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    maxLoadUnbalance 0.1;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces
    {
        sphere { level (3 3); patchInfo { type wall; } }
    }
    refinementRegions
    {
        wakeBox { mode inside; levels ((1e15 2)); }
    }
    locationInMesh (0.3 0.1 0.1);
}

snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers { sphere { nSurfaceLayers 5; } }
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
    nGrow 0;
    featureAngle 120;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}

meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01;
    minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}

writeFlags ( );
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:]); print(r.stderr[-500:])

# -- cell 7 -------------------------------------------------------------------------
# Re-emitting with the import inside the cell so it enters the script, and displaying the PNG.
import pyvista as pv
from IPython.display import Image, display
rd = pv.OpenFOAMReader("case.foam")
rd.enable_all_patch_arrays()
mesh = rd.read()
internal, bnd = mesh["internalMesh"], mesh["boundary"]
print(internal.n_cells, list(bnd.keys()))
sl = internal.slice(normal="z", origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(1400,700))
p.add_mesh(sl, show_edges=True, color="lightgray", line_width=0.5)
p.add_mesh(bnd["sphere"], color="red")
p.view_xy(); p.camera.zoom(1.4)
p.screenshot("look_slice.png")
display(Image("look_slice.png"))

# -- cell 8 -------------------------------------------------------------------------
# All measured numbers match the request (D within 0.02 %, 5/15/5/5 D extents). Now switch layers on a
snappy_l = snappy.replace("addLayers       false;", "addLayers       true;")
open("system/snappyHexMeshDict","w").write(snappy_l)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode)
import re
out = r.stdout
i = out.find("Layer mesh :")
print(out[out.find("Extruding")-200 if out.find("Extruding")>0 else 0:][:0])
print("\n".join([l for l in out.splitlines() if "overall" in l or "Layer mesh" in l or "thickness" in l.lower()][-15:]))
print(out[-700:])

# -- cell 9 -------------------------------------------------------------------------
# I must stop adding illustrative output. One block only:
lines = r.stdout.splitlines()
k = [i for i,l in enumerate(lines) if "overall thickness" in l]
for i in k:
    print("\n".join(lines[i-2:i+6]))
    print("---")

# -- cell 10 ------------------------------------------------------------------------
# Layers grew 5/5 to full thickness. Now the production pass: sphere at level 4 (1.25 mm ≈ 32 cells ac
snappy_fine = (snappy
    .replace("addLayers       false;", "addLayers       true;")
    .replace("sphere { level (3 3); patchInfo { type wall; } }",
             "sphere { level (4 4); patchInfo { type wall; } }")
    .replace("nSurfaceLayers 5;", "nSurfaceLayers 6;")
    .replace("finalLayerThickness 0.5;", "finalLayerThickness 0.4;"))
open("system/snappyHexMeshDict","w").write(snappy_fine)
subprocess.run(["blockMesh"], capture_output=True, text=True)
rf = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(rf.returncode)
L = rf.stdout.splitlines()
for i,l in enumerate(L):
    if "overall thickness" in l: print("\n".join(L[i-2:i+6]))
print(rf.stdout[-400:])

# -- cell 11 ------------------------------------------------------------------------
# Run checkMesh and look at the near-sphere slice with the layers visible.
cm = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
tail = cm.stdout
print("\n".join([l for l in tail.splitlines() if any(s in l for s in
      ["Mesh OK","***","Max ","cells:","non-orthogonality","skewness","aspect ratio","faces","points:"])][-25:]))

# -- cell 12 ------------------------------------------------------------------------
# checkMesh: Mesh OK; volume 0.127967 vs box−sphere = 0.1279665 (0.0004 % off). Final look at the laye
import pyvista as pv
from IPython.display import Image, display
rd2 = pv.OpenFOAMReader("case.foam"); rd2.enable_all_patch_arrays()
m2 = rd2.read(); im, bd = m2["internalMesh"], m2["boundary"]
sl2 = im.slice(normal="z", origin=(0,0,0)).clip_box([-0.05,0.09,-0.05,0.05,-1,1], invert=False)
p = pv.Plotter(off_screen=True, window_size=(1200,900))
p.add_mesh(sl2, show_edges=True, color="white", line_width=0.8)
p.view_xy(); p.camera.zoom(1.3)
p.screenshot("look_layers.png")
display(Image("look_layers.png"))

# -- cell 13 ------------------------------------------------------------------------
# Measure the first prism layer thickness directly from the mesh (cell volume / wall face area for cel
import numpy as np
sphp = bd["sphere"]
areas = sphp.compute_cell_sizes(length=False, volume=False)["Area"]
cent = sphp.cell_centers().points
# find owner cell of each wall face by nearest cell centre
cc = im.cell_centers().points
vol = im.compute_cell_sizes(length=False, area=False)["Volume"]
tree = None
from scipy.spatial import cKDTree
tree = cKDTree(cc)
d, idx = tree.query(cent)
t = vol[idx] / areas
print("first layer thickness: mean %.4e  min %.4e  max %.4e m" % (t.mean(), t.min(), t.max()))
print("expected 0.4*1.25mm/1.2^5 = %.4e" % (0.4*0.00125/1.2**5))
print("total layer thickness 0.002 m = %.1f%% of D" % (100*0.002/D))
print("sphere wall faces:", sphp.n_cells, " total cells:", im.n_cells)
