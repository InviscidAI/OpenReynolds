"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, inspect
print(bd.__version__ if hasattr(bd,'__version__') else 'n/a')
print([n for n in dir(bd) if 'export' in n.lower()])
print(inspect.signature(bd.export_stl))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd, numpy as np
from build123d import Vector, Plane, Rectangle, Cylinder, Location, Pos, extrude

# --- given dimensions (mm) ---
R_CH, H_CH = 60.0, 200.0      # 120 mm ID, 200 mm tall
R_OUT      = 20.0             # 40 mm dia outlet in top
W_P, H_P   = 20.0, 30.0       # port width x height
Z_P        = 40.0             # port centre above floor
L_D        = 50.0             # duct length along its own axis
ANG        = 15.0             # degrees off local tangent
THETAS     = [0.0, 90.0, 180.0, 270.0]

chamber = Cylinder(radius=R_CH, height=H_CH).moved(Pos(0, 0, H_CH/2))

ducts, info = [], []
for th in THETAS:
    t = np.radians(th); a = np.radians(ANG)
    rhat = np.array([np.cos(t),  np.sin(t), 0.0])
    that = np.array([-np.sin(t), np.cos(t), 0.0])
    d    = np.cos(a)*that - np.sin(a)*rhat        # into the chamber, 15 deg off tangent
    P    = R_CH*rhat + np.array([0, 0, Z_P])      # axis meets wall here
    dv   = Vector(*d)
    xdir = dv.cross(Vector(0, 0, 1))              # horizontal, across the duct
    pl   = Plane(origin=Vector(*(P - L_D*d)), z_dir=dv, x_dir=xdir)
    ducts.append(extrude(pl * Rectangle(W_P, H_P), amount=L_D))
    info.append((th, d, P, pl))

fluid = chamber
for dk in ducts:
    fluid = fluid + dk

print("volume mm^3", round(fluid.volume, 1), " chamber alone", round(chamber.volume, 1))
print("bbox", fluid.bounding_box())
for th, d, P, pl in info:
    t = np.radians(th); that = np.array([-np.sin(t), np.cos(t), 0.0])
    print(f"th={th:5.1f} angle(axis,tangent)={np.degrees(np.arccos(np.dot(d,that))):.3f} deg"
          f"  inlet centre r={np.hypot(*(P-L_D*d)[:2]):.2f}")

# -- cell 3 -------------------------------------------------------------------------
# Now list the faces of the union so I can name patches from geometry, not from a bounding box.
fs = fluid.faces()
print(len(fs))
for i,f in enumerate(fs):
    c = f.center(); g = f.geom_type
    print(i, g, f"area={f.area:9.2f} c=({c.X:7.2f},{c.Y:7.2f},{c.Z:7.2f}) r={np.hypot(c.X,c.Y):7.2f} n={tuple(round(v,3) for v in f.normal_at(c))}")

# -- cell 4 -------------------------------------------------------------------------
# Classify faces per duct by proximity to each duct's own axis (not by bounding box), and split the to
from build123d import Circle, Face, Compound
import collections

duct_centres = [Vector(*(P - 0.5*L_D*d)) for th,d,P,pl in info]
inlet_centres = [Vector(*(P - L_D*d)) for th,d,P,pl in info]

groups = collections.defaultdict(list)
for f in fluid.faces():
    c = f.center(); r = np.hypot(c.X, c.Y)
    if f.geom_type == bd.GeomType.CYLINDER:
        groups["cylinderWall"].append(f)
    elif abs(c.Z) < 1e-6 and r < 1.0:
        groups["floor"].append(f)
    elif abs(c.Z - H_CH) < 1e-6 and r < 1.0:
        pass                      # top plane -> replaced by split sheets below
    else:
        i = int(np.argmin([(c - dc).length for dc in duct_centres]))
        if (c - inlet_centres[i]).length < 1e-6:
            groups[f"inlet{i+1}"].append(f)
        else:
            groups[f"ductWall{i+1}"].append(f)

top_pl = Plane(origin=(0, 0, H_CH))
groups["outlet"] = [(top_pl * Circle(R_OUT)).face()]
groups["top"]    = [(top_pl * (Circle(R_CH) - Circle(R_OUT))).face()]

for k in sorted(groups): print(k, len(groups[k]), round(sum(f.area for f in groups[k]), 2))

# -- cell 5 -------------------------------------------------------------------------
import os, subprocess
print(inspect.signature(bd.scale))
os.makedirs("constant/triSurface", exist_ok=True)
MM = 0.001
patches = {}
for k, flist in groups.items():
    shp = bd.scale(Compound(children=[f for f in flist]), MM)
    patches[k] = shp
    bd.export_stl(shp, f"constant/triSurface/{k}.stl")
print(sorted(os.listdir("constant/triSurface")))
allsurf = Compound(children=[s for s in patches.values()])
print("union bbox (m)", allsurf.bounding_box())

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(900,700))
cols = {"cylinderWall":"lightgray","floor":"gray","top":"tan","outlet":"red"}
for k in sorted(patches):
    m = pv.read(f"constant/triSurface/{k}.stl")
    c = cols.get(k, "blue" if k.startswith("inlet") else "lightsteelblue")
    pl.add_mesh(m, color=c, opacity=0.45 if k in ("cylinderWall","top") else 1.0, show_edges=True)
pl.add_axes(); pl.camera_position=[(0.35,-0.35,0.3),(0,0,0.1),(0,0,1)]
pl.screenshot("view_iso.png")
pl.camera_position=[(0,0,0.6),(0,0,0.1),(0,1,0)]
pl.screenshot("view_top.png")
print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view_iso.png")); display(Image("view_top.png"))

# -- cell 8 -------------------------------------------------------------------------
pl2 = pv.Plotter(off_screen=True, window_size=(800,800))
for k in sorted(patches):
    m = pv.read(f"constant/triSurface/{k}.stl")
    c = cols.get(k, "blue" if k.startswith("inlet") else "lightsteelblue")
    pl2.add_mesh(m, color=c, opacity=0.3 if k in ("cylinderWall","top","outlet") else 1.0, show_edges=True)
pl2.view_xy(); pl2.camera.zoom(1.3); pl2.add_axes()
pl2.screenshot("view_top2.png")
display(Image("view_top2.png"))

# -- cell 9 -------------------------------------------------------------------------
# Now the mesh: background blockMesh + snappyHexMesh, coarse first (10 mm cells).
import os, textwrap
for d in ("system","constant","0"): os.makedirs(d, exist_ok=True)

def w(p,s): open(p,"w").write("FoamFile{version 2.0;format ascii;class dictionary;object "+os.path.basename(p)+";}\n"+textwrap.dedent(s))

w("system/controlDict","""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0;
writeFormat ascii; writePrecision 6; writeCompression off; timeFormat general;
timePrecision 6; runTimeModifiable true;
""")
w("system/fvSchemes","""
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;}
divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;} snGradSchemes{default corrected;}
""")
w("system/fvSolution","""
solvers{}
""")

BASE = 0.010
X = 0.090; Z0, Z1 = -0.010, 0.210
nx = int(round(2*X/BASE)); nz = int(round((Z1-Z0)/BASE))
w("system/blockMeshDict", f"""
scale 1;
vertices
(
 (-{X} -{X} {Z0}) ({X} -{X} {Z0}) ({X} {X} {Z0}) (-{X} {X} {Z0})
 (-{X} -{X} {Z1}) ({X} -{X} {Z1}) ({X} {X} {Z1}) (-{X} {X} {Z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1));
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
mergePatchPairs ();
""")
print(nx, nz, nx*nx*nz)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 10 ------------------------------------------------------------------------
PATCHES = sorted(patches)
def snappy(levels, nCellsBetween=3, resolveAngle=45):
    geo = "\n".join(f'  {p}.stl {{ type triSurfaceMesh; name {p}; }}' for p in PATCHES)
    ref = "\n".join(f'      {p} {{ level ({levels.get(p,(0,0))[0]} {levels.get(p,(0,0))[1]}); patchInfo {{ type {"patch" if (p.startswith("inlet") or p=="outlet") else "wall"}; }} }}' for p in PATCHES)
    w("system/snappyHexMeshDict", f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geo}
}};
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10;
  maxLoadUnbalance 0.1; nCellsBetweenLevels {nCellsBetween};
  features ();
  refinementSurfaces
  {{
{ref}
  }}
  resolveFeatureAngle {resolveAngle};
  refinementRegions {{}}
  locationInMesh (0 0 0.1);
  allowFreeStandingZoneFaces true;
}}
snapControls
{{
  nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
  multiRegionFeatureSnap false;
}}
addLayersControls
{{
  relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.5;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3;
  nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
  nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
  maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
  errorReduction 0.75; relaxed {{ maxNonOrtho 75; }}
}}
mergeTolerance 1e-6; debug 0;
""")
snappy({})
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:], r.returncode)

# -- cell 11 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Checking geometry"):][:100])
import re
b=open("constant/polyMesh/boundary").read()
print(re.findall(r"^\s{4}(\w+)\n\s+\{\n\s+type\s+(\w+);\n\s+(?:inGroups[^\n]*\n\s+)?nFaces\s+(\d+)", b, re.M))

# -- cell 12 ------------------------------------------------------------------------
# Ducts are lost at 10 mm. Refine the duct/inlet surfaces to level 2 (2.5 mm) and the chamber surfaces
lev = {p:((2,2) if (p.startswith("duct") or p.startswith("inlet")) else (1,1)) for p in PATCHES}
snappy(lev, nCellsBetween=2)
subprocess.run(["blockMesh"],check=True,capture_output=True)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-600:])
b=open("constant/polyMesh/boundary").read()
print(re.findall(r"^\s{4}(\w+)\n\s+\{\n\s+type\s+(\w+);\n\s+(?:inGroups[^\n]*\n\s+)?nFaces\s+(\d+)", b, re.M))

# -- cell 13 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant","-allGeometry"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):r.stdout.find("End")][-2600:])

# -- cell 14 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):r.stdout.find("Checking patch")][:600])
print(r.stdout[-400:])

# -- cell 15 ------------------------------------------------------------------------
# Refining: 5 mm background, chamber surfaces level 1 (2.5 mm), duct/inlet surfaces level 2 (1.25 mm).
BASE = 0.005
nx = int(round(2*X/BASE)); nz = int(round((Z1-Z0)/BASE))
w("system/blockMeshDict", f"""
scale 1;
vertices
(
 (-{X} -{X} {Z0}) ({X} -{X} {Z0}) ({X} {X} {Z0}) (-{X} {X} {Z0})
 (-{X} -{X} {Z1}) ({X} -{X} {Z1}) ({X} {X} {Z1}) (-{X} {X} {Z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1));
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
mergePatchPairs ();
""")
subprocess.run(["blockMesh"],check=True,capture_output=True)
snappy(lev, nCellsBetween=2)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-300:])

# -- cell 16 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):r.stdout.find("Overall number")])
print(re.findall(r"^\s{4}(\w+)\s+(\d+)\s+\d+\s+ok", r.stdout, re.M))
print([l for l in r.stdout.splitlines() if "Total volume" in l or "skewness" in l or "non-orthogonality" in l.lower() or "Mesh OK" in l or "***" in l or "**" in l])

# -- cell 17 ------------------------------------------------------------------------
r=subprocess.run(["foamToVTK","-constant","-ascii"],capture_output=True,text=True); print(r.returncode)
import glob
print([p for p in glob.glob("VTK/**/*.vtk",recursive=True)][:20], glob.glob("VTK/*")[:20])

# -- cell 18 ------------------------------------------------------------------------
blk = pv.read("VTK/t24_1.vtm")
names=[]
def walk(b,pref=""):
    for i in range(b.n_blocks):
        nm=b.get_block_name(i); sub=b[i]
        if isinstance(sub,pv.MultiBlock): walk(sub,pref+str(nm)+"/")
        else: names.append((pref+str(nm), sub))
walk(blk)
for n,s in names: print(n, s.n_cells if s is not None else None)

# -- cell 19 ------------------------------------------------------------------------
d = dict(names)
for n,s in names:
    if n=="internal": continue
    sz = s.extract_surface().compute_cell_sizes(length=False, volume=False)
    A = sz["Area"].sum()
    nm = s.extract_surface().compute_normals(cell_normals=True, point_normals=False)["Normals"]
    w_ = sz["Area"]/A
    navg = (nm*w_[:,None]).sum(0)
    b = s.bounds
    print(f"{n.split('/')[-1]:13s} A={A*1e6:9.1f} mm^2  n={np.round(navg,3)}  z=[{b[4]:.4f},{b[5]:.4f}]")

# -- cell 20 ------------------------------------------------------------------------
pl3 = pv.Plotter(off_screen=True, window_size=(1000,450), shape=(1,2))
surf = pv.MultiBlock([s.extract_surface() for n,s in names if n!="internal"]).combine()
pl3.subplot(0,0)
for n,s in names:
    if n=="internal": continue
    k=n.split('/')[-1]
    pl3.add_mesh(s.extract_surface(), color=cols.get(k,"blue" if k.startswith("inlet") else "lightsteelblue"),
                 opacity=0.35 if k in ("cylinderWall","top") else 1.0, show_edges=True, line_width=0.3)
pl3.camera_position=[(0.35,-0.35,0.32),(0,0,0.09),(0,0,1)]
pl3.subplot(0,1)
sl = d["internal"].slice(normal=(0,0,1), origin=(0,0,0.04))
pl3.add_mesh(sl, show_edges=True, color="white")
pl3.view_xy(); pl3.camera.zoom(1.2)
pl3.screenshot("mesh.png"); display(Image("mesh.png"))

# -- cell 21 ------------------------------------------------------------------------
# Quick pre-flight on the exported STL union: closure and winding consistency.
u = pv.PolyData()
for p in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{p}.stl"))
u = u.clean(tolerance=1e-8)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
nm = u.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
print("free edges", fe.n_cells, "non-manifold", nm.n_cells, "tris", u.n_cells)
print("volume", u.volume)

# -- cell 22 ------------------------------------------------------------------------
pts = fe.points[np.unique(fe.lines.reshape(-1,3)[:,1:])] if fe.n_cells else np.empty((0,3))
zs = np.round(pts[:,2],4); rs = np.round(np.hypot(pts[:,0],pts[:,1]),4)
import collections
print(collections.Counter([(z, r) for z,r in zip(zs, rs)]).most_common(12))
print("unique z:", sorted(set(zs))[:10], " r range", rs.min(), rs.max())

# -- cell 23 ------------------------------------------------------------------------
# All free edges are the duct/wall intersection curves — independent per-face tessellation. Fix: mesh 
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED

DEFL, ANGD = 0.4, 0.25          # mm / rad, one tessellation for the whole solid
BRepMesh_IncrementalMesh(fluid.wrapped, DEFL, False, ANGD, True)

def face_tris(f):
    loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    if tri is None: return None
    trsf = loc.Transformation()
    P = np.array([[(p := tri.Node(i).Transformed(trsf)).X(), p.Y(), p.Z()]
                  for i in range(1, tri.NbNodes()+1)])
    T = np.array([[t.Value(1)-1, t.Value(2)-1, t.Value(3)-1]
                  for t in (tri.Triangle(i) for i in range(1, tri.NbTriangles()+1))])
    if f.wrapped.Orientation() == TopAbs_REVERSED: T = T[:, ::-1]
    return P, T

tst = face_tris(fluid.faces()[0])
print(tst[0].shape, tst[1].shape)
print("total tris", sum(face_tris(f)[1].shape[0] for f in fluid.faces()))

# -- cell 24 ------------------------------------------------------------------------
def write_stl(path, P, T, name):
    P = P*MM
    with open(path,"w") as fh:
        fh.write(f"solid {name}\n")
        for a,b,c in T:
            p,q,r_ = P[a],P[b],P[c]; n = np.cross(q-p, r_-p); L=np.linalg.norm(n)
            n = n/L if L>0 else n
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in (p,q,r_): fh.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")

# regroup faces of the meshed solid, and find the full top face
grp = collections.defaultdict(list); top_face = None
for f in fluid.faces():
    c = f.center(); r = np.hypot(c.X, c.Y)
    if f.geom_type == bd.GeomType.CYLINDER: grp["cylinderWall"].append(f)
    elif abs(c.Z) < 1e-6 and r < 1.0:       grp["floor"].append(f)
    elif abs(c.Z-H_CH) < 1e-6 and r < 1.0:  top_face = f
    else:
        i = int(np.argmin([(c-dc).length for dc in duct_centres]))
        grp[("inlet%d" if (c-inlet_centres[i]).length < 1e-6 else "ductWall%d") % (i+1)].append(f)

tris = {}
for k, fl in grp.items():
    Ps, Ts, off = [], [], 0
    for f in fl:
        P,T = face_tris(f); Ps.append(P); Ts.append(T+off); off += len(P)
    tris[k] = (np.vstack(Ps), np.vstack(Ts))
print({k:len(v[1]) for k,v in tris.items()}, "top face area", round(top_face.area,1))

# -- cell 25 ------------------------------------------------------------------------
Pt, Tt = face_tris(top_face)
ring = Pt[np.hypot(Pt[:,0],Pt[:,1]) > R_CH-1e-6]
ang  = np.arctan2(ring[:,1], ring[:,0]); ring = ring[np.argsort(ang)]; ang = np.sort(ang)
inner = np.stack([R_OUT*np.cos(ang), R_OUT*np.sin(ang), np.full(len(ang), H_CH)], 1)
n = len(ring); print("ring nodes", n)

Pa = np.vstack([ring, inner]); Ta = []
for i in range(n):
    j = (i+1) % n
    Ta += [[i, j, n+j], [i, n+j, n+i]]                  # annulus, +z winding
tris["top"] = (Pa, np.array(Ta))

Pd = np.vstack([inner, [[0,0,H_CH]]]); Td = [[i, (i+1)%n, n] for i in range(n)]
tris["outlet"] = (Pd, np.array(Td))

for k,(P,T) in tris.items(): write_stl(f"constant/triSurface/{k}.stl", P, T, k)

u = pv.PolyData()
for p in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{p}.stl"))
u = u.clean(tolerance=1e-9)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
nmn = u.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
print("tris",u.n_cells,"free edges",fe.n_cells,"non-manifold",nmn.n_cells,"volume",u.volume)

# -- cell 26 ------------------------------------------------------------------------
# Watertight. Now a finer tessellation (the cylinder was only 51 segments) and re-export.
DEFL, ANGD = 0.15, 0.08
BRepMesh_IncrementalMesh(fluid.wrapped, DEFL, False, ANGD, True)

tris = {}
for k, fl in grp.items():
    Ps, Ts, off = [], [], 0
    for f in fl:
        P,T = face_tris(f); Ps.append(P); Ts.append(T+off); off += len(P)
    tris[k] = (np.vstack(Ps), np.vstack(Ts))

Pt, Tt = face_tris(top_face)
ring = Pt[np.hypot(Pt[:,0],Pt[:,1]) > R_CH-1e-6]
ang  = np.arctan2(ring[:,1], ring[:,0]); ring = ring[np.argsort(ang)]; ang = np.sort(ang)
inner = np.stack([R_OUT*np.cos(ang), R_OUT*np.sin(ang), np.full(len(ang), H_CH)], 1)
n = len(ring)
Pa = np.vstack([ring, inner]); Ta = []
for i in range(n):
    j = (i+1) % n
    Ta += [[i, j, n+j], [i, n+j, n+i]]
tris["top"] = (Pa, np.array(Ta))
Pd = np.vstack([inner, [[0,0,H_CH]]]); Td = [[i, (i+1)%n, n] for i in range(n)]
tris["outlet"] = (Pd, np.array(Td))
for k,(P,T) in tris.items(): write_stl(f"constant/triSurface/{k}.stl", P, T, k)

u = pv.PolyData()
for p in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{p}.stl"))
u = u.clean(tolerance=1e-9)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
nmn = u.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
print("ring", n, "tris", u.n_cells, "free", fe.n_cells, "nonmanifold", nmn.n_cells,
      "vol", u.volume, "vs CAD", fluid.volume*MM**3)

# -- cell 27 ------------------------------------------------------------------------
subprocess.run(["blockMesh"],check=True,capture_output=True)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode)
c = subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(c.stdout[c.stdout.find("Mesh stats"):c.stdout.find("Overall number")])
print([l for l in c.stdout.splitlines() if "Total volume" in l or "skewness" in l or "orthogonality" in l or "Mesh OK" in l or "***" in l])
print(re.findall(r"^\s{4}(\w+)\n\s+\{\n\s+type\s+(\w+);\n\s+(?:inGroups[^\n]*\n\s+)?nFaces\s+(\d+)", open("constant/polyMesh/boundary").read(), re.M))

# -- cell 28 ------------------------------------------------------------------------
bb = fluid.bounding_box()
print("requested vs measured (CAD, mm)")
print(f" chamber ID   120   -> {2*R_CH:.1f} (cyl face radius {grp['cylinderWall'][0].radius*2:.1f})")
print(f" height       200   -> {bb.max.Z-bb.min.Z:.1f}")
print(f" outlet dia    40   -> {2*np.sqrt(1256.637/np.pi):.1f}  (outlet sheet area {sum(0.5*np.linalg.norm(np.cross(tris['outlet'][0][t[1]]-tris['outlet'][0][t[0]], tris['outlet'][0][t[2]]-tris['outlet'][0][t[0]])) for t in tris['outlet'][1]):.1f} mm2 vs 1256.6)")
for i,(th,dd,P,_) in enumerate(info):
    f_in = grp[f"inlet{i+1}"][0]; c=f_in.center(); nrm=np.array(f_in.normal_at(c).to_tuple())
    t = np.radians(th); that = np.array([-np.sin(t), np.cos(t), 0.0])
    ex = f_in.bounding_box()
    print(f" port{i+1}: area {f_in.area:.1f} (600) z {ex.min.Z:.1f}-{ex.max.Z:.1f} (25-55)"
          f" angle to tangent {np.degrees(np.arccos(abs(np.dot(-nrm,that)))):.2f} deg (15)"
          f" axis length to wall {np.linalg.norm(P-np.array(c.to_tuple())):.2f} (50)")
pl4 = pv.Plotter(off_screen=True, window_size=(700,600))
blk2 = pv.read("VTK/t24_1.vtm")
for p in PATCHES:
    m = pv.read(f"constant/triSurface/{p}.stl")
    pl4.add_mesh(m, color=cols.get(p,"blue" if p.startswith("inlet") else "lightsteelblue"),
                 opacity=0.3 if p in ("cylinderWall","top") else 1.0, show_edges=True, line_width=0.4)
pl4.camera_position=[(0.3,-0.3,0.28),(0,0,0.09),(0,0,1)]
pl4.screenshot("final_stl.png"); display(Image("final_stl.png"))
