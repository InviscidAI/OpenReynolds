"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run that cell now.
import subprocess, os
from build123d import *

mm = 1e-3
PL, PW, PH = 60.0, 40.0, 8.0      # plate 60 x 40 x 8
CW, CD     = 4.0, 5.0             # channel width, depth
RIB        = 3.0                  # metal between passes
EDGE       = 2.0                  # metal between U-turn outer wall and plate edge
ZC0, ZC1   = PH - CD, PH          # channel milled from top: z 3 -> 8

pitch = CW + RIB
yc = [PW/2 - pitch, PW/2, PW/2 + pitch]
xt_r = PL - EDGE - CW/2
xt_l = EDGE + CW/2

def box(x0, x1, y0, y1):
    return Pos((x0+x1)/2, (y0+y1)/2, (ZC0+ZC1)/2) * Box(x1-x0, y1-y0, CD)

leg1  = box(0.0,        xt_r+CW/2, yc[0]-CW/2, yc[0]+CW/2)
turn1 = box(xt_r-CW/2,  xt_r+CW/2, yc[0]-CW/2, yc[1]+CW/2)
leg2  = box(xt_l-CW/2,  xt_r+CW/2, yc[1]-CW/2, yc[1]+CW/2)
turn2 = box(xt_l-CW/2,  xt_l+CW/2, yc[1]-CW/2, yc[2]+CW/2)
leg3  = box(xt_l-CW/2,  PL,        yc[2]-CW/2, yc[2]+CW/2)

fluid_mm = leg1 + turn1 + leg2 + turn2 + leg3
print("bbox mm:", fluid_mm.bounding_box())
print("volume mm^3:", fluid_mm.volume)
print("pass centres y:", yc, " rib gap:", (yc[1]-CW/2)-(yc[0]+CW/2))

# -- cell 2 -------------------------------------------------------------------------
import numpy as np
fluid = fluid_mm.scale(mm)
print("bbox m:", fluid.bounding_box(), "vol m3:", fluid.volume)
def pick(xval):
    return [f for f in fluid.faces()
            if abs(f.center().X - xval) < 1e-9 and abs(abs(f.normal_at(f.center()).X)-1) < 1e-6]
inlet_fs, outlet_fs = pick(0.0), pick(PL*mm)
for nm, fs in [("inlet", inlet_fs), ("outlet", outlet_fs)]:
    print(nm, len(fs), [round(f.area/mm**2,3) for f in fs],
          [tuple(round(c/mm,2) for c in f.center()) for f in fs])
wall_fs = [f for f in fluid.faces() if not (f in inlet_fs or f in outlet_fs)]
print("wall faces", len(wall_fs), "of", len(fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
# Mesher rejects open face sheets; I'll tessellate faces myself and write ASCII STL per patch.
import numpy as np, os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

def write_stl(faces, name):
    tris = []
    for f in faces:
        vs, ts = f.tessellate(1e-4)
        P = np.array([[v.X, v.Y, v.Z] for v in vs])
        for t in ts:
            tris.append(P[list(t)])
    with open(f"constant/triSurface/{name}.stl", "w") as fh:
        fh.write(f"solid {name}\n")
        for tr in tris:
            n = np.cross(tr[1]-tr[0], tr[2]-tr[0]); ln = np.linalg.norm(n)
            n = n/ln if ln > 0 else n
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in tr: fh.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
    return len(tris)

for fs, nm in [(inlet_fs,"inlet"), (outlet_fs,"outlet"), (wall_fs,"walls")]:
    print(nm, write_stl(fs, nm), "tris")
print(subprocess.run(["surfaceCheck","constant/triSurface/walls.stl"],capture_output=True,text=True).stdout[-600:])

# -- cell 4 -------------------------------------------------------------------------
# Walls open only at the two ends (8 free edges), as expected. Render to look at it.
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
p = pv.Plotter(off_screen=True, window_size=(1000,700))
p.add_mesh(pv.read("constant/triSurface/walls.stl"), color="lightsteelblue", show_edges=True, opacity=0.9)
p.add_mesh(pv.read("constant/triSurface/inlet.stl"), color="green")
p.add_mesh(pv.read("constant/triSurface/outlet.stl"), color="red")
p.add_mesh(pv.Box(bounds=(0,0.06,0,0.04,0,0.008)), style="wireframe", color="black")
p.add_axes(); p.camera_position = "iso"; p.screenshot("geom.png")
print("done")

# -- cell 5 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 6 -------------------------------------------------------------------------
# Geometry looks right. Now a coarse snappyHexMesh case: 1 mm background grid, no refinement.
import os, textwrap
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(head%"controlDict"+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head%"fvSchemes"+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head%"fvSolution"+"solvers{} \n")

# background: 1 mm cells, 0.5 mm offset from channel planes to avoid exact coincidence
x0,x1,y0,y1,z0,z1 = -2.5, 62.5, 8.5, 31.5, 1.5, 9.5   # mm
nx,ny,nz = int(x1-x0), int(y1-y0), int(z1-z0)
verts = [(a,b,c) for c in (z0,z1) for b in (y0,y1) for a in (x0,x1)]
order = [0,1,3,2,4,5,7,6]
vtxt = "\n".join("(%g %g %g)"%tuple(v*1e-3 for v in verts[i]) for i in order)
open("system/blockMeshDict","w").write(head%"blockMeshDict"+f"""
scale 1;
vertices ( {vtxt} );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges (); boundary ( bg {{ type patch; faces ( (0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3) ); }} ); mergePatchPairs ();
""")
print(nx,ny,nz, nx*ny*nz)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 7 -------------------------------------------------------------------------
snappy = head%"snappyHexMeshDict" + """
castellatedMesh true; snap true; addLayers false;
geometry {
  walls.stl  { type triSurfaceMesh; name walls; }
  inlet.stl  { type triSurfaceMesh; name inlet; }
  outlet.stl { type triSurfaceMesh; name outlet; }
}
castellatedMeshControls {
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 0; nCellsBetweenLevels 1;
  maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces {
    walls  { level (0 0); patchInfo { type wall; } }
    inlet  { level (0 0); patchInfo { type patch; } }
    outlet { level (0 0); patchInfo { type patch; } }
  }
  refinementRegions {}
  locationInMesh (0.030 0.013 0.0055);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers{} expansionRatio 1.2; finalLayerThickness 0.4;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:], r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print(r[r.find("Mesh stats"):r.find("Mesh stats")+900])
print(r[-1200:])

# -- cell 9 -------------------------------------------------------------------------
# Coarse mesh is good (volume 3.56002e-6 m³ vs exact 3.56e-6). Now refine to 0.5 mm cells (8×10 across
H = 0.5  # mm cell size
x0,x1,y0,y1,z0,z1 = -2.25, 62.25, 8.75, 31.75, 1.75, 9.75
nx,ny,nz = [int(round((b-a)/H)) for a,b in [(x0,x1),(y0,y1),(z0,z1)]]
verts = [(a,b,c) for c in (z0,z1) for b in (y0,y1) for a in (x0,x1)]
vtxt = "\n".join("(%g %g %g)"%tuple(v*1e-3 for v in verts[i]) for i in [0,1,3,2,4,5,7,6])
open("system/blockMeshDict","w").write(head%"blockMeshDict"+f"""
scale 1;
vertices ( {vtxt} );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges (); boundary ( bg {{ type patch; faces ( (0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3) ); }} ); mergePatchPairs ();
""")
print(nx,ny,nz,nx*ny*nz)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).returncode)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-300:])

# -- cell 10 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
import re
print(r[r.find("Mesh stats"):r.find("Checking topology")])
print("\n".join(l for l in r.splitlines() if any(k in l for k in
  ["Total volume","non-orthogonality","Max skewness","aspect ratio","Mesh OK","***","bounding box","walls","inlet","outlet"])))

# -- cell 11 ------------------------------------------------------------------------
import pyvista as pv
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays(); m = rd.read()
bl = m["boundary"]
for nm in bl.keys():
    s = bl[nm].extract_surface()
    print(nm, "area mm2 =", round(s.compute_cell_sizes(length=False,volume=False)["Area"].sum()/1e-6,4),
          "bbox mm", [round(v*1e3,2) for v in s.bounds])

# -- cell 12 ------------------------------------------------------------------------
p = pv.Plotter(off_screen=True, window_size=(1000,700))
p.add_mesh(bl["walls"].extract_surface(), color="lightsteelblue", show_edges=True, line_width=0.4)
p.add_mesh(bl["inlet"].extract_surface(), color="green"); p.add_mesh(bl["outlet"].extract_surface(), color="red")
p.add_mesh(pv.Box(bounds=(0,0.06,0,0.04,0,0.008)), style="wireframe", color="black")
p.camera_position="iso"; p.add_axes(); p.screenshot("mesh.png")
from IPython.display import Image, display; display(Image("mesh.png"))
