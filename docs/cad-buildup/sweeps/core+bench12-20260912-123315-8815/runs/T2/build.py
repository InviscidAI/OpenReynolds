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

D = 0.040          # sphere diameter [m]
R = D/2
XU, XD, YZ = 5*D, 15*D, 5*D   # upstream, downstream, lateral half-extent

sphere = Sphere(radius=R)
print("volume", sphere.volume, "expected", 4/3*np.pi*R**3)
print("bbox", sphere.bounding_box())
os.makedirs("constant/triSurface", exist_ok=True)
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=1e-5, angular_tolerance=0.1)
print(os.path.getsize("constant/triSurface/sphere.stl"))

# -- cell 2 -------------------------------------------------------------------------
# Export was far too fine (50 MB). Re-export at a sane tolerance and name the solid `sphere` so snappy
export_stl(sphere, "constant/triSurface/sphere.stl", tolerance=2e-4, angular_tolerance=0.3, ascii_format=True)
txt = open("constant/triSurface/sphere.stl").read()
n = txt.count("facet normal")
txt = txt.replace("solid ", "solid sphere ", 1) if txt.startswith("solid ") else txt
open("constant/triSurface/sphere.stl","w").write(txt)
print(n, os.path.getsize("constant/triSurface/sphere.stl"), txt.splitlines()[0])

# -- cell 3 -------------------------------------------------------------------------
# Background block: 0.8 × 0.4 × 0.4 m with 0.02 m cells (40×20×20 = 16k). Patches named at the face wh
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
CELL = 0.02
nx, ny, nz = int(round((XU+XD)/CELL)), int(round(2*YZ/CELL)), int(round(2*YZ/CELL))
bm = f"""FoamFile{{version 2.0; format ascii; class dictionary; object blockMeshDict;}}
scale 1;
vertices
(
 ({-XU} {-YZ} {-YZ}) ({XD} {-YZ} {-YZ}) ({XD} {YZ} {-YZ}) ({-XU} {YZ} {-YZ})
 ({-XU} {-YZ} {YZ})  ({XD} {-YZ} {YZ})  ({XD} {YZ} {YZ})  ({-XU} {YZ} {YZ})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet    {{ type patch; faces ((0 4 7 3)); }}
 outlet   {{ type patch; faces ((1 2 6 5)); }}
 farfield {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
"""
open("system/blockMeshDict","w").write(bm)
ctrl = """FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;
runTimeModifiable true;
"""
open("system/controlDict","w").write(ctrl)
open("system/fvSchemes","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSchemes;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSolution;}\nsolvers{}\n")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 4 -------------------------------------------------------------------------
# Coarse snappy run: castellate + snap only (layers after I've seen the shape). Sphere refinement leve
snappy_tmpl = """FoamFile{{version 2.0; format ascii; class dictionary; object snappyHexMeshDict;}}
castellatedMesh true; snap true; addLayers {layers};
geometry
{{
    sphere.stl {{ type triSurfaceMesh; name sphere; }}
    refBox {{ type searchableBox; min (-0.06 -0.06 -0.06); max (0.30 0.06 0.06); }}
}}
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 2;
    maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces {{ sphere {{ level ({lmin} {lmax}); patchInfo {{ type wall; }} }} }}
    refinementRegions {{ refBox {{ mode inside; levels ((1E15 {lbox})); }} }}
    locationInMesh (-0.15 0.15 0.15);
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
   implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{
    relativeSizes true;
    layers {{ sphere {{ nSurfaceLayers {nlayers}; }} }}
    expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
    nGrow 0; featureAngle 130; slipFeatureAngle 30; nRelaxIter 5; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50; nRelaxedIter 20;
}}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
   minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
   minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
   relaxed {{ maxNonOrtho 75; }} }}
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy_tmpl.format(layers="false", lmin=2, lmax=3, lbox=1, nlayers=3))
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 5 -------------------------------------------------------------------------
# Snapped fine. Let me look at it: a slice through z=0 plus the sphere patch surface.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,"start_xvfb") else None
subprocess.run(["foamToVTK","-latestTime","-constant"], capture_output=True, text=True)
import glob
print(sorted(glob.glob("VTK/*")))

# -- cell 6 -------------------------------------------------------------------------
pv.global_theme.allow_empty_mesh = True
blk = pv.read("VTK/t2_0.vtm")
mesh = blk[0] if blk[0].n_cells else blk
print(blk.keys() if hasattr(blk,'keys') else "", mesh.n_cells, mesh.bounds)
sl = mesh.slice(normal="z", origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(1100,600))
p.add_mesh(sl, show_edges=True, color="lightblue")
p.view_xy(); p.camera.zoom(1.4)
p.screenshot("slice_coarse.png")
print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("slice_coarse.png"))

# -- cell 8 -------------------------------------------------------------------------
# Shape is right. Now the production run: sphere refinement level 3–4 (≈1.25 mm on the surface), wake 
open("system/snappyHexMeshDict","w").write(snappy_tmpl.format(layers="true", lmin=3, lmax=4, lbox=2, nlayers=3))
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
open("log.snappy","w").write(r.stdout)
print(r.returncode, r.stdout[-1200:])

# -- cell 9 -------------------------------------------------------------------------
# Level 4 never triggered (curvature-based), so surface sits at 2.5 mm with only 744 faces. Set a unif
open("system/snappyHexMeshDict","w").write(snappy_tmpl.format(layers="true", lmin=4, lmax=4, lbox=2, nlayers=3))
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
open("log.snappy","w").write(r.stdout)
print(r.returncode, r.stdout[-1000:])

# -- cell 10 ------------------------------------------------------------------------
# 3 layers everywhere on 2792 sphere faces. Now checkMesh plus geometric verification: sphere patch ar
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
open("log.checkMesh","w").write(r.stdout)
print(r.stdout[-2200:])

# -- cell 11 ------------------------------------------------------------------------
# checkMesh passes. Now measure the sphere patch from the mesh itself against the requested 40 mm diam
subprocess.run(["foamToVTK","-latestTime","-constant","-overwrite"], capture_output=True, text=True)
b = pv.read("VTK/t2_0.vtm")
bnd = b["boundary"]
names = list(bnd.keys())
print(names)
sp = bnd["sphere"]
rad = np.linalg.norm(sp.points, axis=1)
print("sphere patch area %.6e m2, ideal pi*D^2 = %.6e" % (sp.area, np.pi*D**2))
print("point radius min/mean/max [mm]", rad.min()*1e3, rad.mean()*1e3, rad.max()*1e3, "-> D =", 2*rad.mean()*1e3)
for nm in ["inlet","outlet","farfield"]:
    pm = bnd[nm]; print(nm, pm.n_cells, "bounds", np.round(pm.bounds,4), "area %.4f"%pm.area)

# -- cell 12 ------------------------------------------------------------------------
# Geometry checks out. Last look: zoomed slice at z=0 to confirm the prism layers wrap the sphere.
mesh = b["internal"]
sl = mesh.slice(normal="z", origin=(0,0,1e-6))
p = pv.Plotter(off_screen=True, window_size=(900,900))
p.add_mesh(sl, show_edges=True, color="white", line_width=1)
p.view_xy(); p.camera.SetParallelProjection(True)
p.camera.focal_point=(0,0,0); p.camera.position=(0,0,1); p.camera.parallel_scale=0.045
p.screenshot("slice_zoom.png"); display(Image("slice_zoom.png"))
