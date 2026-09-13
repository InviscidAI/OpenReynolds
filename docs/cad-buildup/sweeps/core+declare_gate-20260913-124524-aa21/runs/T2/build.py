"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess, numpy as np
from build123d import *
D = 0.040           # sphere diameter [m]
R = D/2
sphere = Solid.make_sphere(R)
bb = sphere.bounding_box()
print("case dir:", os.getcwd())
print("bbox size (m):", bb.size, " target D:", D)
print("volume:", sphere.volume, " analytic:", 4/3*np.pi*R**3)
print("area:", sphere.area, " analytic:", 4*np.pi*R**2)

# -- cell 2 -------------------------------------------------------------------------
# Binary STL; re-export as ASCII with a named solid.
import time
t0=time.time()
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=5e-5, angular_tolerance=0.1, ascii_format=True)
lines = open("constant/triSurface/sphere.stl").read().splitlines()
lines[0] = "solid sphere"; lines[-1] = "endsolid sphere"
open("constant/triSurface/sphere.stl","w").write("\n".join(lines)+"\n")
print(time.time()-t0, "s; facets:", sum(l.strip().startswith("facet") for l in lines))

# -- cell 3 -------------------------------------------------------------------------
# 201k facets is overkill for a 40 mm sphere; coarsen chordal tolerance to 20 µm-ish equivalent and ch
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=2e-5, angular_tolerance=0.3, ascii_format=True)
lines = open("constant/triSurface/sphere.stl").read().splitlines()
lines[0] = "solid sphere"; lines[-1] = "endsolid sphere"
open("constant/triSurface/sphere.stl","w").write("\n".join(lines)+"\n")
pts = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.strip().startswith("vertex")])
r = np.linalg.norm(pts,axis=1)
print("facets:", sum(l.strip().startswith("facet") for l in lines))
print("vertex radius min/max:", r.min(), r.max(), " target", R)
print("bbox:", pts.min(0), pts.max(0))

# -- cell 4 -------------------------------------------------------------------------
# Wrong direction — larger tolerance is coarser. Aim for ~20k facets with ≤10 µm chord error.
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=1e-5*R, angular_tolerance=0.2, ascii_format=True)
lines = open("constant/triSurface/sphere.stl").read().splitlines()
lines[0] = "solid sphere"; lines[-1] = "endsolid sphere"
open("constant/triSurface/sphere.stl","w").write("\n".join(lines)+"\n")
pts = np.array([[float(x) for x in l.split()[1:]] for l in lines if l.strip().startswith("vertex")])
r = np.linalg.norm(pts,axis=1)
print("facets:", sum(l.strip().startswith("facet") for l in lines), "radius min/max:", r.min(), r.max())

# -- cell 5 -------------------------------------------------------------------------
# `export_stl` deflection isn't behaving monotonically here. Build the tessellation directly as an ico
def icosphere(radius, subdiv):
    t = (1+5**0.5)/2
    v = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
                  [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], float)
    f = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                  [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
    v = v/np.linalg.norm(v,axis=1)[:,None]
    for _ in range(subdiv):
        tris = v[f]
        mid = np.concatenate([(tris[:,0]+tris[:,1])/2,(tris[:,1]+tris[:,2])/2,(tris[:,2]+tris[:,0])/2])
        mid = mid/np.linalg.norm(mid,axis=1)[:,None]
        n = len(v); m = len(f)
        v = np.vstack([v, mid])
        a,b,c = f[:,0],f[:,1],f[:,2]
        ab = n+np.arange(m); bc = n+m+np.arange(m); ca = n+2*m+np.arange(m)
        f = np.vstack([np.c_[a,ab,ca], np.c_[b,bc,ab], np.c_[c,ca,bc], np.c_[ab,bc,ca]])
    return v*radius, f

V, F = icosphere(R, 4)
tris = V[F]
cen = tris.mean(1)
chord_err = R - np.linalg.norm(cen,axis=1)
print("facets:", len(F), "chord error max (m):", chord_err.max(), "=", chord_err.max()/R*100, "% of R")
print("vertex radius min/max:", np.linalg.norm(V,axis=1).min(), np.linalg.norm(V,axis=1).max())

# -- cell 6 -------------------------------------------------------------------------
# Use subdiv 5 (≈5.7 µm chord error), enforce outward normals, and write the ASCII STL named `sphere`.
V, F = icosphere(R, 5)
tris = V[F]
nrm = np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0])
nrm /= np.linalg.norm(nrm,axis=1)[:,None]
flip = (nrm*tris.mean(1)).sum(1) < 0
F[flip] = F[flip][:,[0,2,1]]; nrm[flip] *= -1
tris = V[F]
out = ["solid sphere"]
for n,t in zip(nrm,tris):
    out.append(" facet normal %.9e %.9e %.9e\n  outer loop" % tuple(n))
    for p in t: out.append("   vertex %.9e %.9e %.9e" % tuple(p))
    out.append("  endloop\n endfacet")
out.append("endsolid sphere")
open("constant/triSurface/sphere.stl","w").write("\n".join(out)+"\n")
area = 0.5*np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1).sum()
print("facets:", len(F), "outward:", flip.sum(), "flipped")
print("STL area:", area, "vs exact", 4*np.pi*R**2, " chord err max:", (R-np.linalg.norm(tris.mean(1),axis=1)).max())

# -- cell 7 -------------------------------------------------------------------------
# Background box: x ∈ [−0.2, 0.6] (5D up, 15D down), y,z ∈ [±0.2] (5D each way), base cell 20 mm → 40×
for d in ["system","constant/polyMesh"]: os.makedirs(d, exist_ok=True)
XMIN,XMAX,YZ = -5*D, 15*D, 5*D
NX,NY,NZ = 40,20,20
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/blockMeshDict","w").write(head%"blockMeshDict" + f"""
scale 1;
vertices
(
 ({XMIN} {-YZ} {-YZ}) ({XMAX} {-YZ} {-YZ}) ({XMAX} {YZ} {-YZ}) ({XMIN} {YZ} {-YZ})
 ({XMIN} {-YZ} {YZ}) ({XMAX} {-YZ} {YZ}) ({XMAX} {YZ} {YZ}) ({XMIN} {YZ} {YZ})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet    {{ type patch; faces ((0 4 7 3)); }}
 outlet   {{ type patch; faces ((1 2 6 5)); }}
 farfield {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
open("system/controlDict","w").write(head%"controlDict" + "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head%"fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head%"fvSolution" + "solvers{} \n")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-700:])

# -- cell 8 -------------------------------------------------------------------------
# Coarse castellate+snap first (refinement level 2–3, no layers yet), then look at it.
snappy = head%"snappyHexMeshDict" + """
castellatedMesh true; snap true; addLayers false;
geometry { sphere.stl { type triSurfaceMesh; name sphere; } }
castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 3;
    maxLoadUnbalance 0.10; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces { sphere { level (2 3); patchInfo { type wall; } } }
    refinementRegions {}
    locationInMesh (-0.15 0.15 0.15);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
   nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
   nGrow 0; featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
   maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
   nLayerIter 50; nRelaxedIter 20; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
   minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
   minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:])

# -- cell 9 -------------------------------------------------------------------------
# Snapped. Look at a z=0 slice to confirm the sphere is there and correctly placed.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
subprocess.run(["foamToVTK","-latestTime","-overwrite"],capture_output=True,text=True)
import glob; print(glob.glob("VTK/*"))

# -- cell 10 ------------------------------------------------------------------------
pv.global_theme.allow_empty_mesh = True
m = pv.read("VTK/t2_0.vtm").combine()
sl = m.slice(normal='z', origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(1200,600))
p.add_mesh(sl, show_edges=True, color='lightgray')
p.view_xy(); p.camera.zoom(1.5)
p.screenshot("slice_coarse.png")
print(m.bounds, m.n_cells)

# -- cell 11 ------------------------------------------------------------------------
print(os.path.exists("slice_coarse.png"), os.path.getsize("slice_coarse.png") if os.path.exists("slice_coarse.png") else 0)
from IPython.display import Image, display
display(Image("slice_coarse.png"))

# -- cell 12 ------------------------------------------------------------------------
# Shape and placement are right. Now refine the sphere to level 4 (1.25 mm ≈ 32 cells/diameter), add d
snappy = head%"snappyHexMeshDict" + """
castellatedMesh true; snap true; addLayers true;
geometry
{
    sphere.stl { type triSurfaceMesh; name sphere; }
    wakeBox { type searchableBox; min (-0.04 -0.06 -0.06); max (0.30 0.06 0.06); }
}
castellatedMeshControls
{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; nCellsBetweenLevels 3;
    maxLoadUnbalance 0.10; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces { sphere { level (4 4); patchInfo { type wall; } } }
    refinementRegions
    {
        sphere  { mode distance; levels ((0.02 3) (0.06 2)); }
        wakeBox { mode inside; levels ((1e15 2)); }
    }
    locationInMesh (-0.15 0.15 0.15);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 100; nRelaxIter 5;
   nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls
{
    relativeSizes true; layers { sphere { nSurfaceLayers 5; } }
    expansionRatio 1.2; finalLayerThickness 0.4; minThickness 0.05;
    nGrow 0; featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50; nRelaxedIter 20;
}
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
   minVol 1e-16; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
   minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
   relaxed { maxNonOrtho 75; } }
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
open("log.snappy","w").write(r.stdout)
print(r.stdout[-1200:])

# -- cell 13 ------------------------------------------------------------------------
print(r.returncode); print(r.stderr[-1500:])

# -- cell 14 ------------------------------------------------------------------------
# Crash came from re-refining an already-snapped mesh; regenerate the background mesh first, then run 
subprocess.run(["blockMesh"],capture_output=True,text=True,check=True)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
open("log.snappy","w").write(r.stdout)
print("rc",r.returncode)
print("\n".join(l for l in r.stdout.splitlines() if "Layer mesh" in l or "cells:" in l or "Overall" in l or "thickness" in l)[-1500:])

# -- cell 15 ------------------------------------------------------------------------
log = open("log.snappy").read()
i = log.rfind("patch  faces        layers        overall thickness")
print(log[i:i+600])

# -- cell 16 ------------------------------------------------------------------------
# 5 of 5 layers everywhere, 1.79 mm total. Now checkMesh and measure the meshed sphere patch (area, bb
cm = subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print(cm[cm.find("Checking geometry"):][:2600])

# -- cell 17 ------------------------------------------------------------------------
# Volume check: 0.128 − 3.351e−5 = 0.1279665 vs mesh 0.127967 ✓. Now measure the sphere patch surface 
subprocess.run(["foamToVTK","-latestTime","-overwrite"],capture_output=True,text=True)
mb = pv.read("VTK/t2_0.vtm")
names=[]
def walk(b,path=""):
    for k in range(b.n_blocks):
        blk=b[k]; nm=b.get_block_name(k)
        if isinstance(blk,pv.MultiBlock): walk(blk,path+"/"+str(nm))
        elif blk is not None: names.append((path+"/"+str(nm),blk))
walk(mb)
for n,b in names:
    print(n, b.n_cells, np.round(b.bounds,4), "area=%.6g"%b.area if b.n_cells and b.area else "")

# -- cell 18 ------------------------------------------------------------------------
# Patch measurements: sphere bbox 40.0×40.0×40.0 mm, area 0.005019 m² (−0.15% vs exact sphere, facetin
m = pv.read("VTK/t2_0.vtm").combine()
sl = m.slice(normal='z', origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(1000,800))
p.add_mesh(sl, show_edges=True, color='white', line_width=1)
p.view_xy(); p.camera.focal_point=(0,0,0); p.camera.parallel_projection=True; p.camera.parallel_scale=0.05
p.screenshot("slice_layers.png")
display(Image("slice_layers.png"))
