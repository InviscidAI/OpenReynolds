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
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Turbofan flow-path profile, metres, axis = +X ----
NAC = [(0.00,1.05),(0.30,1.02),(0.60,1.00),(1.60,1.00),(2.20,0.94),(2.60,0.85)]
SPL_OUT = [(2.60,0.62),(2.20,0.72),(1.60,0.78),(1.20,0.70),(1.00,0.60)]
CORE_OUT = [(1.20,0.55),(1.80,0.50),(2.20,0.50),(2.60,0.50),(3.00,0.45)]
HUB = [(3.00,0.25),(2.60,0.40),(2.20,0.42),(1.80,0.38),(1.40,0.40),(1.00,0.40),
       (0.60,0.30),(0.15,0.00),(0.00,0.00)]
PROFILE = NAC + SPL_OUT + CORE_OUT + HUB

p = np.array(PROFILE)
plt.figure(figsize=(11,4))
plt.plot(p[:,0], p[:,1], 'o-')
plt.plot([0,3.0],[0,0],'k--',lw=0.7)
plt.gca().set_aspect('equal'); plt.grid(True); plt.title("fluid half-plane profile (x,r)")
plt.savefig("profile.png", dpi=110)
print(len(PROFILE), os.getcwd())

# -- cell 2 -------------------------------------------------------------------------
from IPython.display import Image, display
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(p[:,0], p[:,1], 'o-'); ax.plot([0,3.0],[0,0],'k--',lw=0.7)
ax.set_aspect('equal'); ax.grid(True); ax.set_title("fluid half-plane profile (x,r)")
fig

# -- cell 3 -------------------------------------------------------------------------
# Profile is a clean C-shaped fluid region. Now revolve it about X and check volume/bbox.
from build123d import *
pts = [Vector(x, 0.0, r) for (x, r) in PROFILE]
face = make_face(Polyline(*pts, close=True))
fluid0 = revolve(face, axis=Axis.X, revolution_arc=360)
print(fluid0.volume, fluid0.bounding_box(), len(fluid0.faces()))

# -- cell 4 -------------------------------------------------------------------------
# Now the blade rows. A blade = lofted twisted rectangle spanning past both annulus walls, polar-patte
def blade_row(xc, r0, r1, chord, thick, ang0, ang1, n):
    """One annular blade row: n twisted plates, spans beyond both walls so the cut is clean."""
    p0 = Plane(origin=(xc, 0, r0), z_dir=(0, 0, 1)).rotated((0, 0, ang0))
    p1 = Plane(origin=(xc, 0, r1), z_dir=(0, 0, 1)).rotated((0, 0, ang1))
    b = loft([p0 * Rectangle(chord, thick), p1 * Rectangle(chord, thick)])
    row = b
    for k in range(1, n):
        row = row + Rot(X=360.0 * k / n) * b
    return row

FAN = blade_row(0.60, 0.24, 1.06, 0.30, 0.050, -35, -58, 12)
print(FAN.volume, len(FAN.solids()), FAN.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
OGV   = blade_row(1.50, 0.72, 1.05, 0.22, 0.040,  12,   4, 10)   # outlet guide vanes, bypass duct
COMP1 = blade_row(1.35, 0.34, 0.60, 0.14, 0.035, -30, -45, 12)   # compressor stage 1
COMP2 = blade_row(1.62, 0.33, 0.59, 0.13, 0.035, -28, -42, 12)   # compressor stage 2
TURB1 = blade_row(2.18, 0.36, 0.56, 0.17, 0.045,  35,  25, 10)   # turbine stage 1
TURB2 = blade_row(2.45, 0.36, 0.56, 0.17, 0.045, -35, -25, 10)   # turbine stage 2
blades = FAN + OGV + COMP1 + COMP2 + TURB1 + TURB2
print(len(blades.solids()), blades.volume)

# -- cell 6 -------------------------------------------------------------------------
fluid = fluid0 - blades
print("fluid volume", fluid.volume, "solids", len(fluid.solids()), "faces", len(fluid.faces()))
print(fluid.bounding_box())

# -- cell 7 -------------------------------------------------------------------------
import math
def planar_x_faces(shp, x0, tol=1e-6):
    out = []
    for f in shp.faces():
        c = f.center()
        n = f.normal_at(c)
        if abs(abs(n.X) - 1) < 1e-6 and abs(c.X - x0) < tol and abs(f.center().X - x0) < tol:
            # confirm every vertex lies on the plane
            if all(abs(v.X - x0) < 1e-6 for v in f.vertices()):
                out.append(f)
    return out

inlet_f = planar_x_faces(fluid, 0.0)
byp_f   = planar_x_faces(fluid, 2.6)
core_f  = planar_x_faces(fluid, 3.0)
for nm, fs, want in [("inlet", inlet_f, math.pi*1.05**2),
                     ("bypass_outlet", byp_f, math.pi*(0.85**2-0.62**2)),
                     ("core_outlet", core_f, math.pi*(0.45**2-0.25**2))]:
    print(nm, len(fs), "area", sum(f.area for f in fs), "analytic", want)
wall_f = [f for f in fluid.faces() if f not in inlet_f + byp_f + core_f]
print("wall faces", len(wall_f), "total", len(fluid.faces()))

# -- cell 8 -------------------------------------------------------------------------
os.makedirs("constant/triSurface", exist_ok=True)
def export_group(faces, name):
    c = Compound(children=[Face(f.wrapped) for f in faces]) if len(faces) > 1 else faces[0]
    path = f"constant/triSurface/{name}.stl"
    export_stl(c, path, tolerance=0.002, angular_tolerance=0.25, ascii_format=True)
    return path

for nm, fs in [("inlet", inlet_f), ("bypass_outlet", byp_f), ("core_outlet", core_f), ("walls", wall_f)]:
    pth = export_group(fs, nm)
    print(nm, os.path.getsize(pth)//1024, "kB")

# -- cell 9 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(1400, 700))
cols = {"walls":"lightsteelblue","inlet":"red","bypass_outlet":"green","core_outlet":"orange"}
meshes = {n: pv.read(f"constant/triSurface/{n}.stl") for n in cols}
for n,m in meshes.items():
    pl.add_mesh(m.clip(normal=(0,1,0), origin=(0,0,0)), color=cols[n], show_edges=False, opacity=1.0)
pl.camera_position = [(4.5,-5.0,2.5),(1.5,0,0.3),(0,0,1)]
img = pl.screenshot("view1.png")
display(Image("view1.png"))

# -- cell 10 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(1500, 700))
for n,m in meshes.items():
    pl.add_mesh(m.clip(normal=(0,1,0), origin=(0,0,0)), color=cols[n], show_edges=True, line_width=0.3)
pl.camera_position = [(2.0, 6.0, 2.0), (1.5, 0, 0.25), (0,0,1)]
pl.screenshot("view2.png"); display(Image("view2.png"))

# -- cell 11 ------------------------------------------------------------------------
# Cut-away looks right: spinner, fan, splitter, core stages, plug. Now check the union of the four pat
names = ["inlet","bypass_outlet","core_outlet","walls"]
with open("constant/triSurface/all.stl","w") as out:
    for n in names:
        out.write(open(f"constant/triSurface/{n}.stl").read())
r = subprocess.run(["surfaceCheck","constant/triSurface/all.stl"], capture_output=True, text=True)
print("\n".join([l for l in r.stdout.splitlines() if any(k in l for k in
      ["Surface","open edge","closed","illegal","Number of","triangles","region"])][-25:]))

# -- cell 12 ------------------------------------------------------------------------
# Separate `export_stl` calls re-tessellate each group independently, so shared edges don't weld. Mesh
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED

BRepMesh_IncrementalMesh(fluid.wrapped, 0.004, False, 0.25, True)

def write_group(faces, name):
    path = f"constant/triSurface/{name}.stl"
    nt = 0
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for f in faces:
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
            if tri is None: continue
            trsf = loc.Transformation()
            P = [tri.Node(i+1).Transformed(trsf) for i in range(tri.NbNodes())]
            rev = f.wrapped.Orientation() == TopAbs_REVERSED
            for i in range(tri.NbTriangles()):
                a, b, c = tri.Triangle(i+1).Get()
                if rev: b, c = c, b
                p1, p2, p3 = P[a-1], P[b-1], P[c-1]
                u = np.array([p2.X()-p1.X(), p2.Y()-p1.Y(), p2.Z()-p1.Z()])
                v = np.array([p3.X()-p1.X(), p3.Y()-p1.Y(), p3.Z()-p1.Z()])
                n = np.cross(u, v); L = np.linalg.norm(n)
                if L < 1e-14: continue
                n /= L
                fh.write(f"facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n outer loop\n")
                for p in (p1, p2, p3):
                    fh.write(f"  vertex {p.X():.8e} {p.Y():.8e} {p.Z():.8e}\n")
                fh.write(" endloop\nendfacet\n")
                nt += 1
        fh.write(f"endsolid {name}\n")
    return nt

groups = [("inlet", inlet_f), ("bypass_outlet", byp_f), ("core_outlet", core_f), ("walls", wall_f)]
for nm, fs in groups:
    print(nm, write_group(fs, nm), "tris")

# -- cell 13 ------------------------------------------------------------------------
# Checking the welded union, then straight into blockMesh + a coarse snappy.
with open("constant/triSurface/all.stl","w") as out:
    for n in names: out.write(open(f"constant/triSurface/{n}.stl").read())
r = subprocess.run(["surfaceCheck","constant/triSurface/all.stl"], capture_output=True, text=True)
print("\n".join([l for l in r.stdout.splitlines() if any(k in l for k in
      ["Triangles","illegal","closed","unconnected"])]))

# -- cell 14 ------------------------------------------------------------------------
import collections
def load_tris(path):
    v=[]
    for line in open(path):
        if line.lstrip().startswith("vertex"):
            v.append([float(t) for t in line.split()[1:]])
    return np.array(v).reshape(-1,3,3)
T = np.vstack([load_tris(f"constant/triSurface/{n}.stl") for n in names])
key = np.round(T.reshape(-1,3), 7)
uniq, inv = np.unique(key, axis=0, return_inverse=True)
tri = inv.reshape(-1,3)
ec = collections.Counter()
for a,b,c in tri:
    for e in ((a,b),(b,c),(c,a)): ec[tuple(sorted(e))]+=1
free=[e for e,k in ec.items() if k!=2]
print("tris",len(tri),"pts",len(uniq),"free/odd edges",len(free))
if free:
    pts=uniq[np.array([e[0] for e in free])]
    print("free edge point bbox x", pts[:,0].min(), pts[:,0].max(), "r", np.hypot(pts[:,1],pts[:,2]).min(), np.hypot(pts[:,1],pts[:,2]).max())

# -- cell 15 ------------------------------------------------------------------------
missing = 0
for f in fluid.faces():
    loc = TopLoc_Location()
    if BRep_Tool.Triangulation_s(f.wrapped, loc) is None: missing += 1
print("faces w/o triangulation:", missing)
# free edges of walls alone
Tw = load_tris("constant/triSurface/walls.stl")
k = np.round(Tw.reshape(-1,3),7); u,iv = np.unique(k,axis=0,return_inverse=True); tw=iv.reshape(-1,3)
ecw=collections.Counter()
for a,b,c in tw:
    for e in ((a,b),(b,c),(c,a)): ecw[tuple(sorted(e))]+=1
print("walls free edges", sum(1 for v in ecw.values() if v!=2))

# -- cell 16 ------------------------------------------------------------------------
# Per-face OCC triangulations don't weld. Switch to gmsh surface meshing on the STEP, which is conform
export_step(fluid, "fluid.step")
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("tf")
gmsh.model.occ.importShapes("fluid.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
print("surfaces", len(surfs))
def tag_group(t):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2,t)
    if abs(x1-x0) < 1e-7:
        if abs(x0) < 1e-7: return "inlet"
        if abs(x0-2.6) < 1e-7: return "bypass_outlet"
        if abs(x0-3.0) < 1e-7: return "core_outlet"
    return "walls"
from collections import defaultdict
grp = defaultdict(list)
for _,t in surfs: grp[tag_group(t)].append(t)
print({k:len(v) for k,v in grp.items()})

# -- cell 17 ------------------------------------------------------------------------
print(gmsh.model.getBoundingBox(-1,-1))

# -- cell 18 ------------------------------------------------------------------------
def tag_group(t):
    x0,_,_,x1,_,_ = gmsh.model.getBoundingBox(2,t)
    if (x1-x0) < 1e-5:
        xm = 0.5*(x0+x1)
        for x,nm in [(0.0,"inlet"),(2.6,"bypass_outlet"),(3.0,"core_outlet")]:
            if abs(xm-x) < 1e-5: return nm
    return "walls"
grp = defaultdict(list)
for _,t in surfs: grp[tag_group(t)].append(t)
print({k:len(v) for k,v in grp.items()})

# -- cell 19 ------------------------------------------------------------------------
# Now a coarse conformal surface mesh in gmsh and per-group STL export.
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.02)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.09)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
for nm, tags in grp.items():
    pg = gmsh.model.addPhysicalGroup(2, tags); gmsh.model.setPhysicalName(2, pg, nm)
gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 2)   # one solid per physical group
gmsh.option.setNumber("Mesh.Binary", 0)
for nm, tags in grp.items():
    gmsh.model.removePhysicalGroups()
    pg = gmsh.model.addPhysicalGroup(2, tags); gmsh.model.setPhysicalName(2, pg, nm)
    gmsh.write(f"constant/triSurface/{nm}.stl")
    print(nm, os.path.getsize(f"constant/triSurface/{nm}.stl")//1024,"kB")

# -- cell 20 ------------------------------------------------------------------------
T = np.vstack([load_tris(f"constant/triSurface/{n}.stl") for n in names])
key = np.round(T.reshape(-1,3), 7)
uniq, inv = np.unique(key, axis=0, return_inverse=True); tri = inv.reshape(-1,3)
ec = collections.Counter()
for a,b,c in tri:
    for e in ((a,b),(b,c),(c,a)): ec[tuple(sorted(e))]+=1
print("tris",len(tri),"free edges",sum(1 for v in ec.values() if v!=2))

# -- cell 21 ------------------------------------------------------------------------
# Watertight. Now the case: blockMesh background box + snappy.
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
def w(p,s): open(p,"w").write(s)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
w("system/controlDict", HDR%"controlDict"+"""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;
writePrecision 8; runTimeModifiable true;
""")
w("system/fvSchemes", HDR%"fvSchemes"+"gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
w("system/fvSolution", HDR%"fvSolution"+"solvers{} \n")
NX,NY,NZ = 40,29,29
w("system/blockMeshDict", HDR%"blockMeshDict"+f"""
scale 1;
vertices ((-0.1 -1.15 -1.15)(3.1 -1.15 -1.15)(3.1 1.15 -1.15)(-0.1 1.15 -1.15)
          (-0.1 -1.15 1.15)(3.1 -1.15 1.15)(3.1 1.15 1.15)(-0.1 1.15 1.15));
blocks (hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1));
edges ();
boundary (background {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }});
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:])

# -- cell 22 ------------------------------------------------------------------------
w("system/surfaceFeatureExtractDict", HDR%"surfaceFeatureExtractDict"+ "".join(
 f'{n}.stl {{ extractionMethod extractFromSurface; includedAngle 150; subsetFeatures{{nonManifoldEdges no; openEdges yes;}} writeObj no; }}\n' for n in names))
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:], sorted(os.listdir("constant/extendedFeatureEdgeMesh"))[:6] if os.path.isdir("constant/extendedFeatureEdgeMesh") else os.listdir("constant/triSurface"))

# -- cell 23 ------------------------------------------------------------------------
def snappy_dict(wall_lvl=(1,2), feat_lvl=2, layers=False):
    geo = "".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}\n' for n in names)
    ref = "".join(f'    {n} {{ level ({wall_lvl[0]} {wall_lvl[1] if n=="walls" else wall_lvl[0]}); patchInfo {{ type {"patch" if n!="walls" else "wall"}; }} }}\n' for n in names)
    feats = "".join(f'      {{ file "{n}.eMesh"; level {feat_lvl}; }}\n' for n in names)
    return HDR%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers {str(layers).lower()};
geometry {{
{geo}}}
castellatedMeshControls {{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 40; allowFreeStandingZoneFaces true;
  features (
{feats}  );
  refinementSurfaces {{
{ref}  }}
  refinementRegions {{}}
  locationInMesh (1.0 0.0 0.88);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 6;
  nFeatureSnapIter 12; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.4;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #includeEtc "caseDicts/meshQualityDict" }}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
"""
w("system/snappyHexMeshDict", snappy_dict())
import time
t0=time.time()
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, time.time()-t0)
print(r.stdout[-1500:])

# -- cell 24 ------------------------------------------------------------------------
print(r.stderr[-2000:])

# -- cell 25 ------------------------------------------------------------------------
d = open("system/snappyHexMeshDict").read().replace(
 'meshQualityControls { #includeEtc "caseDicts/meshQualityDict" }',
 'meshQualityControls { #includeEtc "caseDicts/meshQualityDict" nSmoothScale 4; errorReduction 0.75; }')
w("system/snappyHexMeshDict", d)
t0=time.time(); r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, round(time.time()-t0), r.stdout[-700:], r.stderr[-500:])

# -- cell 26 ------------------------------------------------------------------------
subprocess.run(["blockMesh"], capture_output=True, text=True)
t0=time.time(); r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, round(time.time()-t0))
print(r.stdout[-900:]); print("ERR:", r.stderr[-800:])

# -- cell 27 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-40:]))

# -- cell 28 ------------------------------------------------------------------------
# Mesh OK; volume 5.639 vs CAD 5.642 m³ (0.05%). Let me look at it — patch surfaces and a y=0 slice.
r=subprocess.run(["foamToVTK","-constant","-ascii"],capture_output=True,text=True); print(r.returncode)
import glob; print(glob.glob("VTK/**/*.vt*", recursive=True)[:8])

# -- cell 29 ------------------------------------------------------------------------
grid = pv.read("VTK/t25_1/internal.vtu")
sl = grid.slice(normal=(0,1,0), origin=(0,0,0))
pl = pv.Plotter(off_screen=True, window_size=(1600,700))
pl.add_mesh(sl, color="white", show_edges=True, line_width=0.4)
pl.camera_position=[(1.5,-8,0.0),(1.5,0,0.0),(0,0,1)]
pl.screenshot("slice.png"); display(Image("slice.png"))
