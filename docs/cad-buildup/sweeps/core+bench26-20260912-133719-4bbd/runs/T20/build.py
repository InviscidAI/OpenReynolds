"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess
from build123d import *

R      = 0.030   # bore radius, m (60 mm dia)
T_FL   = 0.020   # flange thickness
T_GK   = 0.002   # gasket thickness
L      = 2*T_FL + T_GK

seg1 = Pos(0, 0, T_FL/2)              * Cylinder(R, T_FL)
seg2 = Pos(0, 0, T_FL + T_GK/2)       * Cylinder(R, T_GK)
seg3 = Pos(0, 0, T_FL + T_GK + T_FL/2)* Cylinder(R, T_FL)
fluid = seg1 + seg2 + seg3

bb = fluid.bounding_box()
print("bbox", bb.min, bb.max)
print("length z =", bb.max.Z - bb.min.Z, "want", L)
print("diameter =", bb.max.X - bb.min.X, "want", 2*R)
print("volume   =", fluid.volume, "want", 3.141592653589793*R*R*L)
os.makedirs("constant/triSurface", exist_ok=True)

# -- cell 2 -------------------------------------------------------------------------
def lateral(seg):
    return [f for f in seg.faces() if f.geom_type == GeomType.CYLINDER]
def planar_at(seg, z):
    return [f for f in seg.faces() if f.geom_type == GeomType.PLANE and abs(f.center().Z - z) < 1e-9]

patches = {
    "inlet":       planar_at(seg1, 0.0),
    "outlet":      planar_at(seg3, L),
    "flangeBore1": lateral(seg1),
    "gasketInner": lateral(seg2),
    "flangeBore2": lateral(seg3),
}
for k, v in patches.items():
    print(k, len(v), [round(f.area, 8) for f in v], "zc", [round(f.center().Z, 4) for f in v])

# -- cell 3 -------------------------------------------------------------------------
for name, fs in patches.items():
    shp = fs[0] if len(fs) == 1 else Compound(children=fs)
    ok = export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.1)
    print(name, ok, os.path.getsize(f"constant/triSurface/{name}.stl"))
print(subprocess.run(["ls","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
import textwrap, os
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
head = """FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"""

open("system/controlDict","w").write(head%"controlDict" + textwrap.dedent("""
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(head%"fvSchemes" + "\nddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\ninterpolationSchemes{default linear;}\nsnGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head%"fvSolution" + "\nsolvers{}\n")

# background box: snug around the bore, 3 mm cells
bm = head%"blockMeshDict" + textwrap.dedent("""
scale 1;
vertices ((-0.0345 -0.0345 -0.006)(0.0345 -0.0345 -0.006)(0.0345 0.0345 -0.006)(-0.0345 0.0345 -0.006)
          (-0.0345 -0.0345 0.048)(0.0345 -0.0345 0.048)(0.0345 0.0345 0.048)(-0.0345 0.0345 0.048));
blocks (hex (0 1 2 3 4 5 6 7) (23 23 18) simpleGrading (1 1 1));
edges ();
boundary ( background { type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); } );
""")
open("system/blockMeshDict","w").write(bm)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-500:])

# -- cell 5 -------------------------------------------------------------------------
names = ["inlet","outlet","flangeBore1","gasketInner","flangeBore2"]
sfe = head%"surfaceFeatureExtractDict" + "\n" + "\n".join(
  f'{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}' for n in names)
open("system/surfaceFeatureExtractDict","w").write(sfe)
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True)
print(r.stdout[-1500:], r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
geo = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs = "\n".join(f'    {n} {{ level (0 0); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in names)
feats = "\n".join(f'      {{ file "{n}.eMesh"; level 1; }}' for n in names)
snap = head%"snappyHexMeshDict" + textwrap.dedent(f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geo}
}};
castellatedMeshControls {{
  maxLocalCells 1000000; maxGlobalCells 5000000; minRefinementCells 0; nCellsBetweenLevels 2;
  maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features (
{feats}
  );
  refinementSurfaces {{
{refs}
  }}
  refinementRegions {{}}
  locationInMesh (0.001 0.002 0.021);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.5;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
open("system/snappyHexMeshDict","w").write(snap)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-2000:], r.stderr[-800:])

# -- cell 7 -------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2500:])

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv, numpy as np
open("case.foam","w").write("")
pv.global_theme.window_size=[900,700]
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read()
bnd = m["boundary"]
p = pv.Plotter(off_screen=True)
cols = {"inlet":"blue","outlet":"red","flangeBore1":"lightgray","gasketInner":"green","flangeBore2":"tan"}
for k in bnd.keys():
    b = bnd[k]
    z = b.points[:,2]
    print(f"{k:12s} faces={b.n_cells:4d} z=[{z.min():.4f},{z.max():.4f}] rmax={np.hypot(b.points[:,0],b.points[:,1]).max():.5f}")
    p.add_mesh(b, color=cols[k], show_edges=True, line_width=1)
p.camera_position=[(0.12,-0.13,0.10),(0,0,0.021),(0,0,1)]
p.screenshot("view.png")

# -- cell 9 -------------------------------------------------------------------------
p = pv.Plotter(off_screen=True)
for k in bnd.keys():
    p.add_mesh(bnd[k], color=cols[k], show_edges=True, line_width=1)
p.view_isometric(); p.camera.zoom(1.2)
img = p.screenshot("view.png", return_img=True)
print(img.shape, img.min(), img.max())
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 10 ------------------------------------------------------------------------
# Shape is right. Now one refinement step (2 mm background cells) and a clearance check on the bolt ci
PCD, D_BOLT = 0.140, 0.014
print("bolt hole inner radius =", PCD/2 - D_BOLT/2, "> bore radius", R, "->", PCD/2-D_BOLT/2 > R)
bm2 = bm.replace("(23 23 18)", "(35 35 27)")
open("system/blockMeshDict","w").write(bm2)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-300:])
c=subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print("\n".join(l for l in c.splitlines() if any(k in l for k in ["cells:","Total volume","non-orthogonality","skewness","Mesh OK","***","aspect"])))

# -- cell 11 ------------------------------------------------------------------------
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0); bnd = rd.read()["boundary"]
for k in bnd.keys():
    b = bnd[k]; z=b.points[:,2]; rr=np.hypot(b.points[:,0],b.points[:,1])
    print(f"{k:12s} faces={b.n_cells:5d} area={b.area:.6f} z=[{z.min():.4f},{z.max():.4f}] r=[{rr.min():.4f},{rr.max():.4f}]")
print("exact areas: inlet/outlet", np.pi*R*R, " bore(20mm)", 2*np.pi*R*0.020, " gasket", 2*np.pi*R*0.002)
