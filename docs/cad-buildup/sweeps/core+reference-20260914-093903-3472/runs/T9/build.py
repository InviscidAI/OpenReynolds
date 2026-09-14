"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os, math
# --- parameters (all from the request), METRES ---
R_ring_o, R_ring_i, H_ring = 0.030, 0.023, 0.010      # outer ring
R_sp_o,  R_sp_i,  H_sp     = 0.022, 0.015, 0.008      # inner spinner
# oil region = everything inside the ring bore (r<=R_ring_i, |z|<=H_ring/2) minus the spinner solid
spinner = bd.Cylinder(R_sp_o, H_sp) - bd.Cylinder(R_sp_i, H_sp)
oil     = bd.Cylinder(R_ring_i, H_ring) - spinner
print("radial clearance mm :", (R_ring_i-R_sp_o)*1000)
print("axial gap each end mm:", (H_ring-H_sp)/2*1000)
print("oil volume mm^3 :", oil.volume*1e9)
print("analytic mm^3   :", (math.pi*23**2*10 - math.pi*(22**2-15**2)*8))
print("oil bbox m:", oil.bounding_box())
print("n solids:", len(oil.solids()))

# -- cell 2 -------------------------------------------------------------------------
# Oil volume matches the analytic value and is a single connected solid. Now the four patch surfaces, 
# patch sheets, each authored directly so the name is decided where the surface is made
ringWall = bd.Cylinder(R_ring_i, H_ring).faces().filter_by(bd.GeomType.CYLINDER)[0]   # r=23 bore of the ring
endTop   = bd.Pos(0,0, H_ring/2) * bd.Circle(R_ring_i).face()   # oil cap in the plane of the ring's top end
endBot   = bd.Pos(0,0,-H_ring/2) * bd.Circle(R_ring_i).face()
spinWall = bd.Shell(spinner.faces())                                                   # whole spinner surface
patches = {"ringWall":bd.Shell([ringWall]), "spinnerWall":spinWall,
           "ringEndTop":bd.Shell([endTop]), "ringEndBottom":bd.Shell([endBot])}
for n,s in patches.items(): print(n, "area mm^2 %.1f"%(s.area*1e6), "nfaces", len(s.faces()))
print("sum patch area mm^2 %.1f"%(sum(s.area for s in patches.values())*1e6), " oil solid area mm^2 %.1f"%(oil.area*1e6))

# -- cell 3 -------------------------------------------------------------------------
# Patch areas sum exactly to the oil boundary area, so the four sheets tile it with no overlap. Export
import subprocess, inspect
os.makedirs("constant/triSurface", exist_ok=True)
print(inspect.signature(bd.export_stl))
for n,s in patches.items():
    bd.export_stl(s, f"constant/triSurface/{n}.stl", tolerance=5e-5, angular_tolerance=0.1)
print(subprocess.run(["ls","-l","constant/triSurface"],capture_output=True,text=True).stdout)
print(subprocess.run(["surfaceCheck","constant/triSurface/spinnerWall.stl"],capture_output=True,text=True).stdout[-700:])

# -- cell 4 -------------------------------------------------------------------------
# Now the case skeleton and a coarse background mesh: 1 mm base cells, offset so the 1 mm gaps are not
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
def w(p,s): open(p,"w").write(s)
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
w("system/controlDict", head%"controlDict"+"application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
w("system/fvSchemes", head%"fvSchemes"+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
w("system/fvSolution", head%"fvSolution"+"solvers{} relaxationFactors{}\n")
# background box: 1 mm cells, offset half a cell so r=22,23 and z=+-4,+-5 fall at cell centres
w("system/blockMeshDict", head%"blockMeshDict"+"""
scale 1;
vertices ( (-0.0235 -0.0235 -0.0055) (0.0235 -0.0235 -0.0055) (0.0235 0.0235 -0.0055) (-0.0235 0.0235 -0.0055)
           (-0.0235 -0.0235  0.0055) (0.0235 -0.0235  0.0055) (0.0235 0.0235  0.0055) (-0.0235 0.0235  0.0055) );
blocks ( hex (0 1 2 3 4 5 6 7) (47 47 11) simpleGrading (1 1 1) );
edges (); boundary ( bg { type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); } );
""")
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:], r.stderr[-500:])

# -- cell 5 -------------------------------------------------------------------------
names = ["ringWall","spinnerWall","ringEndTop","ringEndBottom"]
LVL = 1   # coarse first: 1 mm base -> 0.5 mm at the walls, 2 cells across each 1 mm gap
geom = "\n".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs = "\n".join(f'      {n} {{ level ({LVL} {LVL}); patchInfo {{ type wall; }} }}' for n in names)
w("system/snappyHexMeshDict", head%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 4000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces
  {{
{refs}
  }}
  refinementRegions {{}}
  locationInMesh (0 0 0);   // inside the spinner bore, which is oil
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
  implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90;
  nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; writeFlags (scalarLevels layerSets layerFields);
""")
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:]); print(r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
# Mesh built. Now verify it against the geometry: volume, per-patch areas, and checkMesh.
r=subprocess.run(["checkMesh","-allTopology"],capture_output=True,text=True); out=r.stdout
print(out[out.find("Checking geometry"):][:2500])

# -- cell 7 -------------------------------------------------------------------------
print(out[out.find("Checking patch topology"):out.find("Checking geometry")])
r=subprocess.run(["foamToVTK","-latestTime","-no-internal","-no-boundary"],capture_output=True,text=True)
r2=subprocess.run(["foamToVTK","-latestTime"],capture_output=True,text=True); print(r2.returncode)
print(subprocess.run(["find","VTK","-name","*.vtp","-o","-name","*.vtm","-o","-name","*.vtu"],capture_output=True,text=True).stdout)

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
exp = {"ringWall":1445.1,"spinnerWall":3487.2,"ringEndTop":1661.9,"ringEndBottom":1661.9}
pl = pv.Plotter(off_screen=True, window_size=(900,750))
cols = {"ringWall":"steelblue","spinnerWall":"orange","ringEndTop":"green","ringEndBottom":"red"}
for n in exp:
    m = pv.read(f"VTK/t9_0/boundary/{n}.vtp")
    a = m.compute_cell_sizes(length=False, volume=False)["Area"].sum()*1e6
    print(f"{n:14s} mesh area mm^2 {a:8.1f}   CAD {exp[n]:8.1f}")
    pl.add_mesh(m.clip(normal="y", origin=(0,0,0)), color=cols[n], show_edges=True, line_width=0.4)
pl.camera_position=[(0.07,-0.07,0.05),(0,0,0),(0,0,1)]
pl.screenshot("view.png"); print("ok")

# -- cell 9 -------------------------------------------------------------------------
from IPython.display import Image, display
print(os.path.getsize("view.png"))
display(Image("view.png"))

# -- cell 10 ------------------------------------------------------------------------
# Patch areas all within 0.7 % of CAD. Now a cut through the volume mesh to confirm the 1 mm gaps and 
vol = pv.read("VTK/t9_0/internal.vtu")
sl  = vol.slice(normal="y", origin=(0,0,0))
pl = pv.Plotter(off_screen=True, window_size=(1100,420))
pl.add_mesh(sl, color="lightgrey", show_edges=True, line_width=0.6)
pl.camera_position=[(0,-0.1,0),(0,0,0),(0,0,1)]; pl.camera.parallel_projection=True; pl.camera.parallel_scale=0.026
pl.screenshot("slice.png"); display(Image("slice.png"))
print("min/max cells across: gap cell size m =", (vol.compute_cell_sizes(length=False,area=False)["Volume"].min())**(1/3))

# -- cell 11 ------------------------------------------------------------------------
# Two cells across each 1 mm gap is marginal for a bearing clearance. Refine to level 2 (0.25 mm at th
LVL = 2   # final: 0.25 mm at the walls -> ~4 cells across each 1 mm gap
refs = "\n".join(f'      {n} {{ level ({LVL} {LVL}); patchInfo {{ type wall; }} }}' for n in names)
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"(refinementSurfaces\n  \{\n)(.*?)(\n  \}\n  refinementRegions)", lambda m: m.group(1)+refs+m.group(3), d, flags=re.S)
w("system/snappyHexMeshDict", d)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-600:])

# -- cell 12 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); o=r.stdout
print(o[o.find("Mesh stats"):o.find("Checking patch topology")])
print(o[o.find("Checking geometry"):][:1400])

# -- cell 13 ------------------------------------------------------------------------
# Volume now matches CAD to 5 digits and the mesh is a single region (oil continuous through the gaps)
subprocess.run(["foamToVTK","-latestTime"],capture_output=True,text=True)
for n in exp:
    m=pv.read(f"VTK/t9_0/boundary/{n}.vtp"); a=m.compute_cell_sizes(length=False,volume=False)["Area"].sum()*1e6
    print(f"{n:14s} mesh {a:8.1f}  CAD {exp[n]:8.1f} mm^2")
vol=pv.read("VTK/t9_0/internal.vtu"); sl=vol.slice(normal="y")
pl=pv.Plotter(off_screen=True,window_size=(1200,420))
pl.add_mesh(sl,color="lightgrey",show_edges=True,line_width=0.4)
pl.camera_position=[(0,-0.1,0),(0,0,0),(0,0,1)];pl.camera.parallel_projection=True;pl.camera.parallel_scale=0.0125
pl.screenshot("slice2.png"); display(Image("slice2.png"))

# -- cell 14 ------------------------------------------------------------------------
# Let me check the welded union of the four STLs myself before declaring, since the ring wall and the 
u = pv.PolyData()
for n in exp: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
c = u.clean(tolerance=1e-6, absolute=True)
e = c.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges in union:", e.n_cells, " bodies:", c.split_bodies().n_blocks)

# -- cell 15 ------------------------------------------------------------------------
# The union fails winding: the end discs are almost certainly wound opposite to the ring wall at the t
for n in exp:
    m = pv.read(f"constant/triSurface/{n}.stl").compute_normals(auto_orient_normals=False, consistent_normals=False, cell_normals=True, point_normals=False)
    nm = m.cell_normals; cc = m.cell_centers().points
    if n.startswith("ringEnd"): dot = (nm[:,2]*np.sign(cc[:,2])).mean()          # outward = +-z
    elif n=="ringWall": dot = (nm[:,0]*cc[:,0]+nm[:,1]*cc[:,1]).mean()/0.023     # outward = +r
    else: dot = (nm[:,0]*cc[:,0]+nm[:,1]*cc[:,1]).mean()                          # spinner: mixed, just report
    print(f"{n:14s} mean outward dot = {dot:+.3f}")

# -- cell 16 ------------------------------------------------------------------------
# `ringEndBottom` is wound with its normal into the oil while the other sheets face out — that is the 
# the bottom cap must be wound outward (-z) like the rest of the envelope; Circle's face normal is +z
endBot = (bd.Pos(0,0,-H_ring/2) * bd.Circle(R_ring_i).face())
endBot = bd.Face(endBot.wrapped.Reversed())
patches = {"ringWall":bd.Shell([ringWall]), "spinnerWall":spinWall,
           "ringEndTop":bd.Shell([endTop]), "ringEndBottom":bd.Shell([endBot])}
for n,s in patches.items():
    bd.export_stl(s, f"constant/triSurface/{n}.stl", tolerance=5e-5, angular_tolerance=0.1)
u = pv.PolyData()
for n in patches: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
c = u.clean(tolerance=1e-6, absolute=True)
print("free edges:", c.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False).n_cells)
m = pv.read("constant/triSurface/ringEndBottom.stl").compute_normals(auto_orient_normals=False, consistent_normals=False)
print("ringEndBottom mean nz (want -1):", m.cell_normals[:,2].mean())

# -- cell 17 ------------------------------------------------------------------------
# Now count the same thing the gate counts: directed edges walked twice the same way on the welded uni
f = c.faces.reshape(-1,4)[:,1:]
from collections import Counter
cnt = Counter()
for a,b,d in f:
    for e in ((a,b),(b,d),(d,a)): cnt[e]+=1
bad = sum(v for e,v in cnt.items() if v>1)
print("directed edges traversed more than once:", bad, " triangles:", len(f))

# -- cell 18 ------------------------------------------------------------------------
# Re-run the mesh from the corrected surfaces so the delivered mesh is the one built from the delivere
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode)
o=subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print([l for l in o.splitlines() if "cells:" in l or "Total volume" in l or "Number of regions" in l or "Mesh OK" in l or "skewness" in l or "non-orthogonality Max" in l])
