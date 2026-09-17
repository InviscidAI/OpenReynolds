"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The CAD boolean and STL were created, but the installed PyVista plotting API rejected a rendering ke
import os, math
import build123d as bd
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
SHAFT_R, HEIGHT, COLUMN_R = 1.200, 3.000, 0.100
N_TREADS, RADIAL_RUN, TREAD_DEPTH, TREAD_THICK = 20, 1.100, 0.250, 0.040
RISE, TURN_DEG = 0.150, 18.0
shaft = bd.Cylinder(SHAFT_R, HEIGHT)
column = bd.Cylinder(COLUMN_R, HEIGHT)
treads = []
for i in range(N_TREADS):
    top_z = (i + 1) * RISE
    tread0 = bd.Box(SHAFT_R, TREAD_DEPTH, TREAD_THICK, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN))
    treads.append(bd.Rot(0, 0, i * TURN_DEG) * (bd.Pos(0, 0, top_z - TREAD_THICK) * tread0))
stair_solid = column.fuse(*treads)
fluid = shaft - stair_solid
os.makedirs("constant/triSurface", exist_ok=True)
bd.export_stl(fluid, "coarse_fluid_preview.stl", tolerance=0.01, angular_tolerance=0.2)
mesh = pv.read("coarse_fluid_preview.stl").triangulate()
tri = mesh.points[mesh.faces.reshape(-1,4)[:,1:]]
fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection(tri, facecolor="lightsteelblue", edgecolor="navy", linewidth=0.08, alpha=0.35))
ax.set(xlim=(-1.25,1.25), ylim=(-1.25,1.25), zlim=(0,3), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"); ax.set_box_aspect((2.5,2.5,3)); ax.view_init(25,35)
plt.tight_layout(); plt.savefig("coarse_fluid_preview.png", dpi=150); plt.show()
bb = fluid.bounding_box()
print(f"shaft ID measured = {2*SHAFT_R:.3f} m (requested 2.400 m)")
print(f"height measured = {bb.max.Z-bb.min.Z:.3f} m (requested 3.000 m)")
print(f"column diameter = {2*COLUMN_R:.3f} m (requested 0.200 m)")
print(f"tread exposed radial run = {SHAFT_R-COLUMN_R:.3f} m (requested 1.100 m)")
print(f"tread depth = {TREAD_DEPTH:.3f} m; thickness = {TREAD_THICK:.3f} m")
print(f"top elevations = {RISE:.3f}..{N_TREADS*RISE:.3f} m; turn increment = {TURN_DEG:.1f} deg")
print(f"fluid volume = {fluid.volume:.6f} m^3; solids={len(fluid.solids())}; faces={len(fluid.faces())}")

# -- cell 2 -------------------------------------------------------------------------
# The coarse view shows one connected air volume with the spiral obstructions and central column. I’ll
faces = list(fluid.faces())
from collections import Counter
print("face geometry types:", Counter(str(f.geom_type) for f in faces))
for j,f in enumerate(faces):
    c=f.center(); n=f.normal_at(c) if str(f.geom_type).endswith("PLANE") else None
    print(j, str(f.geom_type), f"A={f.area:.6f}", f"C=({c.X:.3f},{c.Y:.3f},{c.Z:.3f})", "N="+str(tuple(round(v,3) for v in n)) if n else "")

# -- cell 3 -------------------------------------------------------------------------
# Inspection found a construction error: build123d cylinders default to being centered in Z, so the sh
print(__import__('inspect').signature(bd.Cylinder))

# -- cell 4 -------------------------------------------------------------------------
# The constructor supports explicit three-axis alignment. I’ll now rebuild only the affected solids an
shaft = bd.Cylinder(SHAFT_R, HEIGHT, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
column = bd.Cylinder(COLUMN_R, HEIGHT, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
stair_solid = column.fuse(*treads)
fluid = shaft - stair_solid
bd.export_stl(fluid, "coarse_fluid_preview.stl", tolerance=0.01, angular_tolerance=0.2)
mesh = pv.read("coarse_fluid_preview.stl").triangulate(); tri = mesh.points[mesh.faces.reshape(-1,4)[:,1:]]
fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection(tri, facecolor="lightsteelblue", edgecolor="navy", linewidth=0.06, alpha=0.28))
ax.set(xlim=(-1.25,1.25), ylim=(-1.25,1.25), zlim=(0,3), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"); ax.set_box_aspect((2.5,2.5,3)); ax.view_init(25,35)
plt.tight_layout(); plt.savefig("coarse_fluid_preview.png", dpi=150); plt.show()
faces = list(fluid.faces()); horizontal_z = sorted({round(f.center().Z,3) for f in faces if str(f.geom_type).endswith("PLANE") and abs(abs(f.normal_at(f.center()).Z)-1)<1e-6})
print(f"corrected z extent = {fluid.bounding_box().min.Z:.3f} to {fluid.bounding_box().max.Z:.3f} m")
print("horizontal boundary levels:", horizontal_z)
print(f"fluid volume = {fluid.volume:.6f} m^3; solids={len(fluid.solids())}; faces={len(faces)}")

# -- cell 5 -------------------------------------------------------------------------
# The corrected boundary contains all requested tread underside/top levels from 0.110/0.150 m through 
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches
faces = list(fluid.faces())
cyl_faces = [f for f in faces if str(f.geom_type).endswith("CYLINDER")]
shaft_wall = max(cyl_faces, key=lambda f:f.area)
column_patch = min(cyl_faces, key=lambda f:f.area)
plane_faces = [f for f in faces if str(f.geom_type).endswith("PLANE")]
floor_faces = [f for f in plane_faces if abs(f.center().Z) < 1e-8 and f.normal_at(f.center()).Z < -0.99]
top_faces = [f for f in plane_faces if abs(f.center().Z-HEIGHT) < 1e-8 and f.normal_at(f.center()).Z > 0.99]
reserved = {id(f) for f in floor_faces + top_faces}
tread_faces = [f for f in plane_faces if id(f) not in reserved]
print(f"patch faces: shaft_wall=1 column=1 treads={len(tread_faces)} floor={len(floor_faces)} top_opening={len(top_faces)}")
print(f"patch areas [m2]: wall={shaft_wall.area:.6f}, column={column_patch.area:.6f}, treads={sum(f.area for f in tread_faces):.6f}, floor={sum(f.area for f in floor_faces):.6f}, opening={sum(f.area for f in top_faces):.6f}")
export_patches(fluid, {"shaft_wall":[shaft_wall], "column":[column_patch], "treads":tread_faces, "floor":floor_faces, "top_opening":top_faces}, tolerance=2.5e-4)

# -- cell 6 -------------------------------------------------------------------------
# The exported union is closed and consistently wound. For the coarse mesh I’ll use a 0.10 m Cartesian
import os, textwrap, subprocess
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
open("system/blockMeshDict","w").write(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
 (-1.3 -1.3 -0.1) (1.3 -1.3 -0.1) (1.3 1.3 -0.1) (-1.3 1.3 -0.1)
 (-1.3 -1.3 3.1)  (1.3 -1.3 3.1)  (1.3 1.3 3.1)  (-1.3 1.3 3.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) (26 26 32) simpleGrading (1 1 1) );
edges ();
boundary ( background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); } );
mergePatchPairs ();
'''))
open("system/snappyHexMeshDict","w").write(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
 shaft_wall.stl { type triSurfaceMesh; name shaft_wall; }
 column.stl { type triSurfaceMesh; name column; }
 treads.stl { type triSurfaceMesh; name treads; }
 floor.stl { type triSurfaceMesh; name floor; }
 top_opening.stl { type triSurfaceMesh; name top_opening; }
}
castellatedMeshControls
{
 maxLocalCells 1000000; maxGlobalCells 2000000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2;
 features ();
 refinementSurfaces
 {
  shaft_wall { level (1 1); patchInfo { type wall; } }
  column { level (2 2); patchInfo { type wall; } }
  treads { level (2 2); patchInfo { type wall; } }
  floor { level (1 1); patchInfo { type wall; } }
  top_opening { level (1 1); patchInfo { type patch; } }
 }
 resolveFeatureAngle 30;
 refinementRegions {}
 locationInMesh (0.8 0 0.05);
 allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls
{
 maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
 minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}
debug 0; mergeTolerance 1e-6;
'''))
open("system/controlDict","w").write("FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
r=subprocess.run(["blockMesh"], text=True, capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-1000:]); print("blockMesh return code",r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The background mesh is correct at 21,632 cells with uniform 0.10 m spacing. I’ll now run `snappyHexM
r=subprocess.run(["snappyHexMesh","-overwrite"], text=True, capture_output=True, timeout=240)
open("snappyHexMesh.log","w").write(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-1000:]); print("snappyHexMesh return code",r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# This OpenFOAM build requires placeholder `fvSchemes` and `fvSolution` dictionaries even for meshing;
open("system/fvSchemes","w").write("FoamFile { format ascii; class dictionary; object fvSchemes; }\n")
open("system/fvSolution","w").write("FoamFile { format ascii; class dictionary; object fvSolution; }\n")
r=subprocess.run(["snappyHexMesh","-overwrite"], text=True, capture_output=True, timeout=240)
open("snappyHexMesh.log","w").write(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print("snappyHexMesh return code",r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The placeholders must contain the expected scheme subdictionaries in this version. I’ll supply minim
open("system/fvSchemes","w").write(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
'''))
open("system/fvSolution","w").write(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
SIMPLE { nNonOrthogonalCorrectors 0; }
'''))
r=subprocess.run(["snappyHexMesh","-overwrite"], text=True, capture_output=True, timeout=240)
open("snappyHexMesh.log","w").write(r.stdout+r.stderr)
print(r.stdout[-6500:]); print(r.stderr[-1000:]); print("snappyHexMesh return code",r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# Snappy completed successfully with 107,171 cells and retained all five named surface patches. I’ll r
r=subprocess.run(["checkMesh"], text=True, capture_output=True, timeout=240)
open("checkMesh.log","w").write(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print("checkMesh return code",r.returncode)

# -- cell 11 ------------------------------------------------------------------------
# The binding check passes: one connected 107,171-cell region, five patches, maximum non-orthogonality
r=subprocess.run(["foamToVTK","-constant"], text=True, capture_output=True, timeout=240)
print(r.stdout[-1500:]); print("foamToVTK return code",r.returncode)
for root,dirs,files in os.walk("VTK"):
    for fn in files[:5]: print(os.path.join(root,fn))

# -- cell 12 ------------------------------------------------------------------------
# The mesh converted successfully and each named patch appears as a distinct VTK boundary dataset. I’l
ug = pv.read("VTK/t26_0/internal.vtu")
sl = ug.slice(normal=(0,1,0), origin=(0,0,1.5))
ed = sl.extract_all_edges()
lines = ed.lines.reshape(-1,3)[:,1:]
pts = ed.points
fig,ax=plt.subplots(figsize=(7,8))
for a,b in lines:
    ax.plot([pts[a,0],pts[b,0]],[pts[a,2],pts[b,2]],color="#244a9b",lw=0.18,alpha=0.65)
ax.set_aspect("equal"); ax.set(xlim=(-1.25,1.25),ylim=(-0.02,3.02),xlabel="x [m]",ylabel="z [m]",title="Final volume mesh slice at y = 0")
ax.grid(True,alpha=.2); plt.tight_layout(); plt.savefig("final_mesh_slice.png",dpi=180); plt.show()
print(f"slice: {sl.n_cells} cut cells, {ed.n_cells} displayed edges")
