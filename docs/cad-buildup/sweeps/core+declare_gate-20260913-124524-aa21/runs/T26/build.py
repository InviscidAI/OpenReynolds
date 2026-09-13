"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
from build123d import *
import numpy as np, os

# --- parameters (metres) ---
R_SHAFT = 1.200      # internal radius of shaft (2400 mm dia)
H       = 3.000      # shaft height
R_COL   = 0.100      # central column radius (200 mm dia)
N_TR    = 20         # treads
L_TR    = 1.100      # tread radial length, column surface -> shaft wall
D_TR    = 0.250      # tread depth (tangential)
T_TR    = 0.040      # tread thickness
RISE    = 0.150      # rise per tread
TURN    = 18.0       # deg per tread

shaft  = Pos(0, 0, H/2) * Cylinder(R_SHAFT, H)
column = Pos(0, 0, H/2) * Cylinder(R_COL, H)

# tread solid runs from the axis (x=0) outward so it welds into the column;
# the exposed span outside the column is exactly L_TR = 1.1 m
def tread(i):
    ztop = RISE * (i + 1)
    return Rot(0, 0, TURN * i) * Pos((R_COL + L_TR) / 2, 0, ztop - T_TR / 2) * \
           Box(R_COL + L_TR, D_TR, T_TR)

treads = [tread(i) for i in range(N_TR)]
solid_obstacles = column
for t in treads:
    solid_obstacles = solid_obstacles + t

fluid = shaft - solid_obstacles
print("fluid volume", fluid.volume)
print("check: shaft-col-treads =",
      np.pi*(R_SHAFT**2-R_COL**2)*H - N_TR*L_TR*D_TR*T_TR)
print("bbox", fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Shape is right (volume matches the analytic value up to the small tread haunches that lap onto the c
TOL = 1e-6
groups = {"shaftWall": [], "column": [], "treads": [], "floor": [], "topOpening": []}
for f in fluid.faces():
    gt = f.geom_type
    c = f.center()
    if gt == GeomType.PLANE:
        n = f.normal_at(c)
        if abs(abs(n.Z) - 1) < 1e-6 and abs(c.Z) < TOL:
            groups["floor"].append(f); continue
        if abs(abs(n.Z) - 1) < 1e-6 and abs(c.Z - H) < TOL:
            groups["topOpening"].append(f); continue
        groups["treads"].append(f); continue
    if gt == GeomType.CYLINDER:
        r = f.radius
        if abs(r - R_SHAFT) < 1e-6: groups["shaftWall"].append(f); continue
        if abs(r - R_COL) < 1e-6:   groups["column"].append(f); continue
    groups["treads"].append(f)

for k, v in groups.items():
    print(k, len(v), "area", round(sum(x.area for x in v), 4))
print("total area", round(sum(f.area for f in fluid.faces()), 4))

# -- cell 3 -------------------------------------------------------------------------
# Face groups match the analytic areas (shaft wall 22.62 − 20×0.01 tread ends = 22.42; floor π(1.2²−0.
import subprocess
os.makedirs("constant/triSurface", exist_ok=True)
for k, v in groups.items():
    sh = Compound(children=[Face(f.wrapped) for f in v]) if len(v) > 1 else v[0]
    export_stl(sh, f"constant/triSurface/{k}.stl", tolerance=0.004, angular_tolerance=0.2, ascii_format=True)
    # rename the ascii solid to the patch name so snappy sees the region name
    p = f"constant/triSurface/{k}.stl"
    txt = open(p).read().replace("solid ", f"solid {k} ", 1)
    lines = txt.splitlines()
    lines = [(f"solid {k}" if l.startswith("solid") else (f"endsolid {k}" if l.startswith("endsolid") else l)) for l in lines]
    open(p, "w").write("\n".join(lines) + "\n")
    print(k, os.path.getsize(p)//1024, "KB", sum(1 for l in lines if l.startswith("  facet")) or sum(1 for l in lines if "facet normal" in l), "facets")

# -- cell 4 -------------------------------------------------------------------------
# Check that the five patches weld into one closed surface, and look at it.
import pyvista as pv
pv.start_xvfb() if False else None
names = list(groups)
meshes = {n: pv.read(f"constant/triSurface/{n}.stl") for n in names}
union = meshes[names[0]].copy()
for n in names[1:]:
    union = union.merge(meshes[n])
union = union.clean(tolerance=1e-6)
edges = union.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges in union:", edges.n_cells, " cells:", union.n_cells)

pl = pv.Plotter(off_screen=True, window_size=(800, 800))
cols = dict(shaftWall="lightgray", column="red", treads="tan", floor="steelblue", topOpening="green")
for n in names:
    pl.add_mesh(meshes[n].clip(normal="y", origin=(0,0,0)) if n=="shaftWall" else meshes[n],
                color=cols[n], opacity=0.5 if n=="shaftWall" else 1.0, show_edges=False)
pl.camera_position = [(5, -5, 4.5), (0, 0, 1.5), (0, 0, 1)]
pl.screenshot("look.png")
print("ok")

# -- cell 5 -------------------------------------------------------------------------
# Watertight union (0 free edges). Display the render.
from IPython.display import Image, display
display(Image("look.png"))

# -- cell 6 -------------------------------------------------------------------------
# Verify a safe `locationInMesh` inside the fluid, then write the coarse mesh case.
import math
pt = (1.0*math.cos(math.radians(9)), 1.0*math.sin(math.radians(9)), 1.5)
probe = pv.PolyData(np.array([pt]))
sel = probe.select_enclosed_points(union.extract_surface().triangulate(), tolerance=1e-9)
print(pt, "inside fluid:", bool(sel["SelectedPoints"][0]))

# -- cell 7 -------------------------------------------------------------------------
# Write the OpenFOAM case dictionaries (coarse: base 0.1 m, surface refinement level 1, treads level 2
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
HEAD = ("FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n")

open("system/controlDict","w").write(HEAD%"controlDict"+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;\n"
 "runTimeModifiable true;\n")
open("system/fvSchemes","w").write(HEAD%"fvSchemes"+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(HEAD%"fvSolution"+"solvers{} \n")

NX, NY, NZ = 26, 26, 32
open("system/blockMeshDict","w").write(HEAD%"blockMeshDict"+f"""
scale 1;
vertices ((-1.3 -1.3 -0.1) (1.3 -1.3 -0.1) (1.3 1.3 -0.1) (-1.3 1.3 -0.1)
          (-1.3 -1.3 3.1) (1.3 -1.3 3.1) (1.3 1.3 3.1) (-1.3 1.3 3.1));
blocks (hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1));
edges ();
boundary (background {{ type patch; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3)); }});
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
pat = ["shaftWall","column","treads","floor","topOpening"]
open("system/surfaceFeatureExtractDict","w").write(HEAD%"surfaceFeatureExtractDict"+
  "".join(f'{p}.stl {{ extractionMethod extractFromSurface; includedAngle 150; writeObj no; }}\n' for p in pat))
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:], r.stderr[-300:])
print(sorted(os.listdir("constant/triSurface")))

# -- cell 9 -------------------------------------------------------------------------
LOC = "(0.98769 0.15643 1.5)"
lvl = {"shaftWall":(1,1),"column":(2,2),"treads":(2,2),"floor":(1,1),"topOpening":(1,1)}
geom = "".join(f'{p}.stl {{ type triSurfaceMesh; name {p}; }}\n' for p in pat)
refs = "".join(f'{p} {{ level ({lvl[p][0]} {lvl[p][1]}); patchInfo {{ type wall; }} }}\n' for p in pat)
feats = "".join(f'{{ file "{p}.eMesh"; level 1; }}\n' for p in pat)
open("system/snappyHexMeshDict","w").write(HEAD%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{ {geom} }}
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 2;
    features ( {feats} );
    refinementSurfaces {{ {refs} }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh {LOC};
    allowFreeStandingZoneFaces false;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.5;
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
writeFlags (); mergeTolerance 1e-6;
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
import re
print(out[out.index("Mesh stats"):out.index("Checking geometry")])
print(out[out.index("Checking geometry"):][-1800:])

# -- cell 11 ------------------------------------------------------------------------
# Mesh OK, total volume 13.2305 vs exact 13.2504 (0.15 % low). Check each patch's meshed area against 
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_constant_time_value(0) if False else None
rd.enable_all_patch_arrays()
blk = rd.read()
bnd = blk["boundary"]
cad = {k: sum(f.area for f in v) for k, v in groups.items()}
for n in bnd.keys():
    a = bnd[n].extract_surface().compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{n:11s} mesh {a:8.4f}   cad {cad[n]:8.4f}   diff {100*(a-cad[n])/cad[n]:+.2f}%")

# -- cell 12 ------------------------------------------------------------------------
# Every patch within 0.5 % of its CAD area, so all 20 treads survived snapping. Render a clipped view 
pl = pv.Plotter(off_screen=True, window_size=(900, 900))
cols = dict(shaftWall="lightgray", column="red", treads="tan", floor="steelblue", topOpening="green")
for n in bnd.keys():
    s = bnd[n].extract_surface().clip(normal="y", origin=(0, 0.02, 0))
    pl.add_mesh(s, color=cols[n], show_edges=True, line_width=0.3, edge_color="k")
pl.camera_position = [(6, -5.5, 4.5), (0, 0, 1.5), (0, 0, 1)]
pl.screenshot("mesh.png"); display(Image("mesh.png"))

# -- cell 13 ------------------------------------------------------------------------
# The shaft wall hides the interior; clip the internal volume instead to see column, treads and the ga
vol = blk["internalMesh"].clip(normal="y", origin=(0,0,0))
pl = pv.Plotter(off_screen=True, window_size=(900, 900))
pl.add_mesh(vol, color="whitesmoke", show_edges=True, line_width=0.3, edge_color="k")
for n, c in [("treads","tan"),("column","red")]:
    pl.add_mesh(bnd[n].extract_surface().clip(normal="y", origin=(0,0,0)), color=c, show_edges=True, line_width=0.3)
pl.camera_position = [(0.2, -7, 3.0), (0, 0, 1.5), (0, 0, 1)]
pl.screenshot("cut.png"); display(Image("cut.png"))

# -- cell 14 ------------------------------------------------------------------------
# Camera was on the wrong side of the cut; view the cut plane from +y.
pl = pv.Plotter(off_screen=True, window_size=(900, 900))
pl.add_mesh(vol, color="whitesmoke", show_edges=True, line_width=0.3, edge_color="k")
for n, c in [("treads","tan"),("column","red")]:
    pl.add_mesh(bnd[n].extract_surface().clip(normal="y", origin=(0,0,0)), color=c, show_edges=True, line_width=0.3)
pl.camera_position = [(0.2, 7, 3.0), (0, 0, 1.5), (0, 0, 1)]
pl.screenshot("cut2.png"); display(Image("cut2.png"))

# -- cell 15 ------------------------------------------------------------------------
# Coarse mesh is correct. Refine once: treads and column to level 3 (12.5 mm, ~3 cells across a 40 mm 
lvl = {"shaftWall":(2,2),"column":(2,3),"treads":(3,3),"floor":(2,2),"topOpening":(1,1)}
refs = "".join(f'{p} {{ level ({lvl[p][0]} {lvl[p][1]}); patchInfo {{ type wall; }} }}\n' for p in pat)
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"refinementSurfaces \{\{?.*?\}\n \}", "refinementSurfaces { "+refs+" }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
print(d[d.index("refinementSurfaces"):d.index("resolveFeatureAngle")])
subprocess.run(["blockMesh"], capture_output=True, text=True)
p = subprocess.Popen(["snappyHexMesh","-overwrite"], stdout=open("snappy.log","w"), stderr=subprocess.STDOUT)
print("started", p.pid)

# -- cell 16 ------------------------------------------------------------------------
# `topOpening` is an opening, not a wall — restart with the right patch type before it gets far.
p.kill(); p.wait()
refs = "".join(f'{q} {{ level ({lvl[q][0]} {lvl[q][1]}); patchInfo {{ type {"patch" if q=="topOpening" else "wall"}; }} }}\n' for q in pat)
d = open("system/snappyHexMeshDict").read()
d = re.sub(r"refinementSurfaces \{.*?\n \}", "refinementSurfaces { "+refs+" }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"], capture_output=True, text=True)
p = subprocess.Popen(["snappyHexMesh","-overwrite"], stdout=open("snappy.log","w"), stderr=subprocess.STDOUT)
print("restarted", p.pid, "topOpening ->", "type patch" in d)

# -- cell 17 ------------------------------------------------------------------------
import time
time.sleep(180); print(p.poll()); print(open("snappy.log").read()[-700:])

# -- cell 18 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True); out = r.stdout
print(out[out.index("Mesh stats"):out.index("Overall number")])
print(out[out.index("Checking patch topology"):out.index("Checking faceZone")])
print(out[out.index("Checking geometry"):][-1200:])

# -- cell 19 ------------------------------------------------------------------------
# measured properties of the built geometry, against the request
c0, c1 = treads[0].center(), treads[1].center()
ang = lambda c: math.degrees(math.atan2(c.Y, c.X))
bb0 = treads[0].bounding_box()
print("turn per tread (deg): ", round(ang(c1)-ang(c0), 4), " asked 18")
print("rise per tread (m):   ", round(c1.Z-c0.Z, 5), " asked 0.150")
print("tread thickness (m):  ", round(bb0.max.Z-bb0.min.Z, 5), " asked 0.040")
print("tread depth (m):      ", round(bb0.max.Y-bb0.min.Y, 5), " asked 0.250")
print("tread exposed radial span (m):", round(bb0.max.X-R_COL, 5), " asked 1.100")
print("shaft radius / height (m):", R_SHAFT, H, " column radius:", R_COL)
print("top tread top z:", round(treads[-1].bounding_box().max.Z,4), " n treads:", len(treads))
rd2 = pv.OpenFOAMReader("case.foam"); rd2.enable_all_patch_arrays(); b2 = rd2.read()["boundary"]
for n in b2.keys():
    a = b2[n].extract_surface().compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{n:11s} mesh area {a:8.4f}  cad {cad[n]:8.4f}  {100*(a-cad[n])/cad[n]:+.2f}%")

# -- cell 20 ------------------------------------------------------------------------
# Refined mesh is clean. Final render for the record, then I'll close.
vol2 = rd2.read()["internalMesh"].clip(normal="y", origin=(0,0,0))
pl = pv.Plotter(off_screen=True, window_size=(900,900))
pl.add_mesh(vol2, color="whitesmoke", show_edges=True, line_width=0.2, edge_color="gray")
pl.add_mesh(b2["treads"].extract_surface(algorithm=None).clip(normal="y", origin=(0,0,0)), color="tan")
pl.add_mesh(b2["column"].extract_surface(algorithm=None).clip(normal="y", origin=(0,0,0)), color="red")
pl.camera_position = [(0.2, 7, 3.0), (0,0,1.5), (0,0,1)]
pl.screenshot("final.png"); display(Image("final.png"))
