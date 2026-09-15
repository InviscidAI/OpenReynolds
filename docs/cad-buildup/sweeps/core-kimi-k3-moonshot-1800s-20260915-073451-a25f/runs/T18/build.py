"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll build this in stages: first the cylinder-with-fins solid in build123d (working directly in metr
# First cell: parameters and the fused cylinder solid.
import build123d as bd
import numpy as np

# ---- dimensions (metres) ----
R_BARREL = 0.045          # barrel OD 90 mm
H_BARREL = 0.120          # barrel height 120 mm
N_FINS   = 12
R_FIN    = 0.070          # fin OD 140 mm
T_FIN    = 0.003          # fin thickness 3 mm
GAP_FIN  = 0.006          # air gap between fins 6 mm
R_DOME   = R_BARREL       # hemispherical head closing the top (assumption: hemisphere matching barrel OD)
DUCT_W   = 0.300          # 300 x 300 mm square duct
DUCT_Z0  = -0.150         # 150 mm below cylinder base
DUCT_Z1  = H_BARREL + 0.300  # 300 mm above barrel top

# fin stack: 12 fins of 3 mm with 6 mm gaps = 102 mm; centred on the 120 mm barrel -> 9 mm margins
stack = N_FINS*T_FIN + (N_FINS-1)*GAP_FIN
z0_fin = (H_BARREL - stack)/2
print("fin stack:", stack, "first fin bottom at", z0_fin)

barrel = bd.Cylinder(radius=R_BARREL, height=H_BARREL)  # centred at origin, z in [-60,60]mm... build123d Cylinder is centred
# place barrel base at z=0
barrel = bd.Pos(0, 0, H_BARREL/2) * barrel

fins = []
for i in range(N_FINS):
    zb = z0_fin + i*(T_FIN + GAP_FIN)
    ring = bd.Cylinder(radius=R_FIN, height=T_FIN) - bd.Cylinder(radius=R_BARREL, height=T_FIN+0.01)
    fins.append(bd.Pos(0, 0, zb + T_FIN/2) * ring)
fin_solid = fins[0]
for f in fins[1:]:
    fin_solid += f

dome = bd.Sphere(R_DOME)
dome = dome & (bd.Pos(0,0,R_DOME) * bd.Box(2*R_DOME, 2*R_DOME, 2*R_DOME))  # upper hemisphere
dome = bd.Pos(0, 0, H_BARREL) * dome

cyl = barrel + fin_solid + dome
print("cylinder bounding box:", cyl.bounding_box().min, cyl.bounding_box().max)
print("n faces:", len(cyl.faces()), "volume m^3:", cyl.volume)

# -- cell 2 -------------------------------------------------------------------------
# Small API fix — build123d Vectors expose `.X/.Y/.Z`, not `.x`:
import pyvista as pv
import numpy as np
pv.OFF_SCREEN = True

def to_pv(shape, tol=1e-4):
    verts, tris = shape.tessellate(tol)
    v = np.array([tuple(p) for p in verts])
    f = np.hstack([np.full((len(tris),1),3), np.array(tris)]).ravel()
    return pv.PolyData(v, f)

mesh = to_pv(cyl)
pl = pv.Plotter(off_screen=True, window_size=(700,700))
pl.add_mesh(mesh, color="tan", smooth_shading=True)
pl.add_axes()
pl.view_isometric()
pl.show(screenshot="cyl_check.png")

# -- cell 3 -------------------------------------------------------------------------
# `geom_type` is an enum, not a string. Fixing the comparison:
GT = bd.GeomType
faces = cyl.faces()
groups = {"barrel": [], "fins": [], "head": []}
for f in faces:
    gt = f.geom_type
    bb = f.bounding_box()
    if gt == GT.SPHERE:
        groups["head"].append(f)
    elif gt == GT.CYLINDER:
        d = bb.max.X - bb.min.X
        if abs(d - 2*R_BARREL) < 1e-6:
            groups["barrel"].append(f)
        else:
            groups["fins"].append(f)
    elif gt == GT.PLANE:
        if f.area > 8e-3:
            groups["fins"].append(f)
        else:
            groups["barrel"].append(f)
    else:
        print("unclassified:", gt, f.area)

for k, v in groups.items():
    print(k, len(v), "faces, total area %.6f m^2" % sum(f.area for f in v))

fin_out  = [f for f in groups["fins"] if f.geom_type == GT.CYLINDER]
fin_flat = [f for f in groups["fins"] if f.geom_type == GT.PLANE]
print("\nfin OD (m):", fin_out[0].bounding_box().max.X*2)
z_ann = sorted(round(f.center().Z, 9) for f in fin_flat)
print("n annuli:", len(z_ann), " n outer cyl faces:", len(fin_out))
thick = [z_ann[2*i+1]-z_ann[2*i] for i in range(N_FINS)]
gaps  = [z_ann[2*i+2]-z_ann[2*i+1] for i in range(N_FINS-1)]
print("fin thickness min/max (m): %.6f %.6f" % (min(thick), max(thick)))
print("gap min/max (m): %.6f %.6f" % (min(gaps), max(gaps)))
bar_cyl = [f for f in groups["barrel"] if f.geom_type == GT.CYLINDER]
print("barrel OD (m):", bar_cyl[0].bounding_box().max.X*2, " barrel lateral area %.6f" % sum(f.area for f in bar_cyl))
print("head dome diameter (m):", groups["head"][0].bounding_box().max.X*2, " dome top z:", groups["head"][0].bounding_box().max.Z)

# -- cell 4 -------------------------------------------------------------------------
import os
os.makedirs("constant/triSurface", exist_ok=True)

duct = bd.Pos(0, 0, (DUCT_Z0 + DUCT_Z1)/2) * bd.Box(DUCT_W, DUCT_W, DUCT_Z1 - DUCT_Z0)
duct_faces = {"inlet": [], "outlet": [], "walls": []}
for f in duct.faces():
    n = f.normal_at()
    c = f.center()
    if abs(n.Z) > 0.99:
        duct_faces["inlet" if c.Z < 0 else "outlet"].append(f)
    else:
        duct_faces["walls"].append(f)
for k, v in duct_faces.items():
    print(k, len(v), "faces, area %.6f m^2" % sum(f.area for f in v))

def write_stl(name, facelist):
    comp = bd.Compound(children=facelist)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=3e-5)

write_stl("barrel", groups["barrel"])
write_stl("fins",   groups["fins"])
write_stl("head",   groups["head"])
write_stl("walls",  duct_faces["walls"])
write_stl("inlet",  duct_faces["inlet"])
write_stl("outlet", duct_faces["outlet"])
print(sorted(os.listdir("constant/triSurface")))

# -- cell 5 -------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(800,700))
pl.add_mesh(to_pv(cyl), color="tan", smooth_shading=True)
ductmesh = to_pv(duct, tol=1e-3)
pl.add_mesh(ductmesh, color="lightblue", opacity=0.25, style="wireframe")
pl.add_axes()
pl.view_isometric()
pl.show(screenshot="assembly.png")

# -- cell 6 -------------------------------------------------------------------------
os.makedirs("system", exist_ok=True)

blockmesh = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
scale 1;
vertices
(
    (-0.15 -0.15 {DUCT_Z0})
    ( 0.15 -0.15 {DUCT_Z0})
    ( 0.15  0.15 {DUCT_Z0})
    (-0.15  0.15 {DUCT_Z0})
    (-0.15 -0.15 {DUCT_Z1})
    ( 0.15 -0.15 {DUCT_Z1})
    ( 0.15  0.15 {DUCT_Z1})
    (-0.15  0.15 {DUCT_Z1})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (40 40 76) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet  {{ type patch; faces ((0 3 2 1)); }}
    outlet {{ type patch; faces ((4 5 6 7)); }}
    walls  {{ type wall;  faces ((0 4 7 3) (2 6 5 1) (1 5 4 0) (3 7 6 2)); }}
);
"""
open("system/blockMeshDict","w").write(blockmesh)

snappy = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    barrel { type triSurfaceMesh; file "barrel.stl"; }
    fins   { type triSurfaceMesh; file "fins.stl"; }
    head   { type triSurfaceMesh; file "head.stl"; }
    walls  { type triSurfaceMesh; file "walls.stl"; }
    inlet  { type triSurfaceMesh; file "inlet.stl"; }
    outlet { type triSurfaceMesh; file "outlet.stl"; }
}

castellatedMeshControls
{
    maxLocalCells 1000000;
    maxGlobalCells 4000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        fins   { level (3 3); patchInfo { type wall; } }
        barrel { level (2 2); patchInfo { type wall; } }
        head   { level (2 2); patchInfo { type wall; } }
        walls  { level (0 0); patchInfo { type wall; } }
        inlet  { level (0 0); patchInfo { type patch; } }
        outlet { level (0 0); patchInfo { type patch; } }
    }
    resolveFeatureAngle 30;
    refinementRegions {}
    locationInMesh (0.12 0.12 0.30);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers {}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality -1e30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)

control = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     snappyHexMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
open("system/controlDict","w").write(control)
print("dicts written")

# -- cell 7 -------------------------------------------------------------------------
# Running blockMesh, then launching snappyHexMesh (in background, logging to file, since it may outrun
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-500:], r.stderr[-500:])
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappy","w"), stderr=subprocess.STDOUT)
print("snappy pid:", p.pid)

# -- cell 8 -------------------------------------------------------------------------
# blockMesh is up (121,600 cells). Polling snappyHexMesh:
import time, subprocess
time.sleep(60)
r = subprocess.run(["tail","-n","25","log.snappy"], capture_output=True, text=True)
print(r.stdout)
print("running?" , subprocess.run(["pgrep","-f","snappyHexMesh"],capture_output=True,text=True).stdout.strip())

# -- cell 9 -------------------------------------------------------------------------
# snappyHexMesh wants `fvSchemes`/`fvSolution` present even though it doesn't solve. Adding minimal on
open("system/fvSchemes","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")

open("system/fvSolution","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
SIMPLE {}
relaxationFactors {}
""")

subprocess.run(["pkill","-f","snappyHexMesh"], capture_output=True)
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappy","w"), stderr=subprocess.STDOUT)
print("snappy pid:", p.pid)

# -- cell 10 ------------------------------------------------------------------------
time.sleep(90)
r = subprocess.run(["tail","-n","15","log.snappy"], capture_output=True, text=True)
print(r.stdout)
alive = subprocess.run(["pgrep","-f","snappyHexMesh"],capture_output=True,text=True).stdout.strip()
print("still running:", bool(alive))

# -- cell 11 ------------------------------------------------------------------------
time.sleep(120)
r = subprocess.run(["tail","-n","12","log.snappy"], capture_output=True, text=True)
print(r.stdout)
alive = subprocess.run(["pgrep","-f","snappyHexMesh"],capture_output=True,text=True).stdout.strip()
print("still running:", bool(alive))

# -- cell 12 ------------------------------------------------------------------------
time.sleep(45)
alive = subprocess.run(["pgrep","-f","snappyHexMesh"],capture_output=True,text=True).stdout.strip()
print("still running:", bool(alive))
print(subprocess.run(["tail","-n","6","log.snappy"], capture_output=True, text=True).stdout)

# -- cell 13 ------------------------------------------------------------------------
time.sleep(120)
alive = subprocess.run(["pgrep","-f","snappyHexMesh"],capture_output=True,text=True).stdout.strip()
print("still running:", bool(alive))
print(subprocess.run(["tail","-n","8","log.snappy"], capture_output=True, text=True).stdout)

# -- cell 14 ------------------------------------------------------------------------
time.sleep(150)
alive = subprocess.run(["pgrep","-f","snappyHexMesh"],capture_output=True,text=True).stdout.strip()
print("still running:", bool(alive))
print(subprocess.run(["tail","-n","10","log.snappy"], capture_output=True, text=True).stdout)

# -- cell 15 ------------------------------------------------------------------------
# snappyHexMesh finished cleanly (~323 s). Running checkMesh and looking at patch sizes:
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print(out[-3000:])

# -- cell 16 ------------------------------------------------------------------------
reader = pv.POpenFOAMReader("t18.foam")
reader.cell_to_point_creation = False
mb = reader.read()
print(mb.keys())
grid = mb["internalMesh"]
print("cells:", grid.n_cells)
sl = grid.slice(normal=(1,0,0), origin=(0,0,0))
pl = pv.Plotter(off_screen=True, window_size=(700,900))
pl.add_mesh(sl, color="white", show_edges=False)
pl.view_xy()
pl.camera.zoom(2.2)
pl.show(screenshot="mesh_slice.png")

# -- cell 17 ------------------------------------------------------------------------
# White mesh on white background — restyling the plot:
pl = pv.Plotter(off_screen=True, window_size=(700,900))
pl.set_background("white")
pl.add_mesh(sl, color="lightsteelblue", show_edges=True, edge_color="gray", line_width=0.3)
pl.view_xy()
pl.camera.zoom(2.0)
pl.show(screenshot="mesh_slice.png")

# -- cell 18 ------------------------------------------------------------------------
print("slice cells:", sl.n_cells, "bounds:", sl.bounds)
print("grid bounds:", grid.bounds)

# -- cell 19 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(700,900))
pl.set_background("white")
pl.add_mesh(sl, color="lightsteelblue", show_edges=True, edge_color="dimgray", line_width=0.3)
pl.camera_position = [( -1.1, 0.0, 0.135), (0, 0, 0.135), (0, 1, 0)]  # looking down +x? no: use xz-plane view
# slice is the y-z plane (normal x); camera on +x axis looking at origin
pl.camera_position = [(1.2, 0.0, 0.135), (0, 0, 0.135), (0, 0, 1)]
pl.show(screenshot="mesh_slice.png")

# -- cell 20 ------------------------------------------------------------------------
# The slice shows exactly the intended air mesh: cylinder cut out, 12 fins with meshed air gaps. Zoomi
pl = pv.Plotter(off_screen=True, window_size=(800,800))
pl.set_background("white")
pl.add_mesh(sl, color="lightsteelblue", show_edges=True, edge_color="dimgray", line_width=0.3)
pl.camera_position = [(1.2, 0.10, 0.06), (0.0, 0.10, 0.06), (0, 0, 1)]  # zoom on right fin pack
pl.show(screenshot="fin_zoom.png")

# -- cell 21 ------------------------------------------------------------------------
# Quantitative confirmation that the inter-fin air gaps actually contain mesh cells (point-probing the
# probe points: mid-gap at r=58mm for a few gaps, and inside a fin (should NOT be in mesh)
probe = []
for i in [0, 5, 10]:
    z_gap = 0.009 + i*0.009 + 0.003 + 0.003   # middle of air gap
    probe.append(("gap %d" % i, (0.058, 0.0, z_gap)))
    z_fin = 0.009 + i*0.009 + 0.0015          # inside fin
    probe.append(("fin %d" % i, (0.058, 0.0, z_fin)))
    z_bar = 0.009 + i*0.009 + 0.0045          # barrel surface gap region at r=30mm (inside barrel -> not in mesh)
    probe.append(("barrel-in %d" % i, (0.030, 0.0, z_gap)))

for name, pt in probe:
    cid = grid.find_containing_cell(pt)
    print(name, pt, "-> cell", cid, end="")
    if cid >= 0:
        c = grid.get_cell(cid)
        print("  cell volume %.3e m^3" % grid.compute_cell_sizes(length=False, area=False, volume=True)["Volume"][cid])
    else:
        print("  (not in mesh)")

# -- cell 22 ------------------------------------------------------------------------
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_Orientation
import numpy as np, struct

def face_triangles(face):
    """Triangles of a face from the triangulation cached on its parent solid,
    wound so normals point out of the solid."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    trsf = loc.Transformation()
    nodes = []
    for i in range(1, tri.NbNodes()+1):
        p = tri.Node(i).Transformed(trsf)
        nodes.append((p.X(), p.Y(), p.Z()))
    nodes = np.array(nodes)
    flip = (face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED)
    tris = []
    for i in range(1, tri.NbTriangles()+1):
        a, b, c = tri.Triangle(i).Get()
        tris.append((a-1, c-1, b-1) if flip else (a-1, b-1, c-1))
    return nodes, np.array(tris, dtype=np.int64)

def write_stl_file(fname, facelist):
    all_v, all_t, off = [], [], 0
    for f in facelist:
        v, t = face_triangles(f)
        all_v.append(v); all_t.append(t + off); off += len(v)
    V = np.vstack(all_v); T = np.vstack(all_t)
    with open(fname, "w") as fh:
        fh.write("solid patch\n")
        for a, b, c in T:
            p, q, r = V[a], V[b], V[c]
            n = np.cross(q-p, r-p); ln = np.linalg.norm(n)
            n = n/ln if ln > 0 else n
            fh.write(" facet normal %g %g %g\n  outer loop\n" % tuple(n))
            for pnt in (p, q, r):
                fh.write("   vertex %g %g %g\n" % tuple(pnt))
            fh.write("  endloop\n endfacet\n")
        fh.write("endsolid patch\n")
    return len(T)

# tessellate each parent solid ONCE so shared edges discretise identically
BRepMesh_IncrementalMesh(cyl.wrapped, 2e-5, False, 0.2, True)
BRepMesh_IncrementalMesh(duct.wrapped, 2e-5, False, 0.2, True)

counts = {}
counts["barrel"] = write_stl_file("constant/triSurface/barrel.stl", groups["barrel"])
counts["fins"]   = write_stl_file("constant/triSurface/fins.stl",   groups["fins"])
counts["head"]   = write_stl_file("constant/triSurface/head.stl",   groups["head"])
counts["walls"]  = write_stl_file("constant/triSurface/walls.stl",  duct_faces["walls"])
counts["inlet"]  = write_stl_file("constant/triSurface/inlet.stl",  duct_faces["inlet"])
counts["outlet"] = write_stl_file("constant/triSurface/outlet.stl", duct_faces["outlet"])
print(counts)

# -- cell 23 ------------------------------------------------------------------------
import glob
from collections import Counter

V_all, T_all, off = [], [], 0
for fn in sorted(glob.glob("constant/triSurface/*.stl")):
    m = pv.read(fn)
    V_all.append(m.points); T_all.append(m.faces.reshape(-1,4)[:,1:] + off); off += m.n_points
V = np.vstack(V_all); T = np.vstack(T_all)
print("union:", V.shape, T.shape)

key = np.round(V/1e-9).astype(np.int64)   # 1 nm weld
_, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
Tw = inv[T]
undirected = Counter()
directed = Counter()
for a,b,c in Tw:
    for e in ((a,b),(b,c),(c,a)):
        undirected[frozenset(e)] += 1
        directed[e] += 1
free = sum(1 for v in undirected.values() if v == 1)
same_dir = sum(1 for (a,b) in directed if directed[(a,b)] > 0 and directed[(b,a)] == 0 and directed[(a,b)] > 0 and undirected[frozenset((a,b))] == 2 and directed[(a,b)] >= 1)
# count edges walked twice the same way: directed count >1 with no reverse
same_way = 0
for (a,b), n in directed.items():
    if n >= 2:
        same_way += 1
    elif n == 1 and directed[(b,a)] == 0 and undirected[frozenset((a,b))] == 2:
        pass
# simpler: edges with undirected count 2 but both same direction
both_same = sum(1 for fs, n in undirected.items() if n == 2 and (lambda e: directed[(e[0],e[1])] == 2 or directed[(e[1],e[0])] == 2)(tuple(fs)))
print("free edges:", free)
print("edges walked twice same way:", both_same)
print("edges with >2 faces:", sum(1 for n in undirected.values() if n > 2))
