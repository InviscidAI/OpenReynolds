"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os
print(bd.__version__ if hasattr(bd,'__version__') else 'ok')
print(os.getcwd(), os.listdir('.'))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd
from build123d import Pos, Rot

# --- all dimensions in metres (request given in mm) ---
R_SHAFT = 1.200      # 2400 mm internal diameter
H       = 3.000      # 3000 mm tall
R_COL   = 0.100      # 200 mm diameter column
N_TREAD = 20
T_LEN   = 1.100      # radial length, column surface -> shaft wall
T_DEP   = 0.250
T_THK   = 0.040
RISE    = 0.150
TWIST   = 18.0       # degrees per tread

shaft  = bd.Cylinder(R_SHAFT, H).moved(Pos(0, 0, H/2))
column = bd.Cylinder(R_COL,  H).moved(Pos(0, 0, H/2))

# tread cutting tools: oversized radially (they are trimmed by column and shaft wall,
# leaving exactly T_LEN of exposed radial run), top tread poked through the ceiling plane
treads = []
for i in range(1, N_TREAD + 1):
    ztop = RISE * i
    over = 0.05 if i == N_TREAD else 0.0
    box = bd.Box(R_SHAFT + 0.30, T_DEP, T_THK + over, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MAX))
    treads.append(Rot(0, 0, TWIST * i) * Pos(-0.15, 0, ztop + over) * box)

fluid = shaft - column
for t in treads:
    fluid = fluid - t
fluid = fluid & bd.Cylinder(R_SHAFT, H).moved(Pos(0, 0, H/2))  # trim the poked-through bit
print("volume m^3", fluid.volume, " bbox", fluid.bounding_box())
print("faces", len(fluid.faces()), "solids", len(fluid.solids()))

# -- cell 3 -------------------------------------------------------------------------
treads = []
for i in range(1, N_TREAD + 1):
    ztop = RISE * i
    over = 0.05 if i == N_TREAD else 0.0
    box = bd.Box(R_SHAFT + 0.30, T_DEP, T_THK + over, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MAX))
    treads.append(Rot(0, 0, TWIST * i) * Pos(0, 0, ztop + over) * box)   # inner end at the axis -> trimmed by column

fluid = shaft - column
for t in treads:
    fluid = fluid - t
fluid = fluid & bd.Cylinder(R_SHAFT, H).moved(Pos(0, 0, H/2))
print("volume m^3", fluid.volume)
exp = 3.141592653589793*(R_SHAFT**2-R_COL**2)*H - N_TREAD*((R_SHAFT*T_DEP - 0.5*3.141592653589793*R_COL**2)*T_THK)
print("expected  ", exp, " solids", len(fluid.solids()), "faces", len(fluid.faces()))

# -- cell 4 -------------------------------------------------------------------------
from collections import defaultdict
import math
groups = defaultdict(list)
for f in fluid.faces():
    gt = f.geom_type
    c = f.center()
    if gt == bd.GeomType.CYLINDER:
        r = f.radius
        if abs(r - R_SHAFT) < 1e-6: groups["shaftWall"].append(f)
        elif abs(r - R_COL) < 1e-6: groups["column"].append(f)
        else: groups["?cyl%.4f" % r].append(f)
    elif gt == bd.GeomType.PLANE:
        n = f.normal_at(c)
        if abs(n.Z) > 0.999 and abs(c.Z) < 1e-9: groups["floor"].append(f)
        elif abs(n.Z) > 0.999 and abs(c.Z - H) < 1e-9: groups["topOpening"].append(f)
        else: groups["treads"].append(f)
    else: groups["?"+str(gt)].append(f)
for k,v in groups.items(): print(k, len(v), "area %.4f" % sum(f.area for f in v))

# -- cell 5 -------------------------------------------------------------------------
import pyvista as pv, numpy as np
def to_pv(faces, tol=0.004):
    meshes=[]
    for f in faces:
        v,t = f.tessellate(tol)
        pts=np.array([[p.X,p.Y,p.Z] for p in v]); tri=np.array(t)
        meshes.append(pv.PolyData(pts, np.hstack([np.full((len(tri),1),3),tri]).ravel()))
    return meshes
cols={"shaftWall":"lightblue","column":"orange","treads":"green","floor":"gray","topOpening":"red"}
p=pv.Plotter(off_screen=True, window_size=(800,800))
for k,c in cols.items():
    for m in to_pv(groups[k]): p.add_mesh(m, color=c, opacity=0.25 if k=="shaftWall" else 1.0)
p.camera_position=[(5,-5,4.5),(0,0,1.5),(0,0,1)]
p.screenshot("geom.png"); print("ok")

# -- cell 6 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 7 -------------------------------------------------------------------------
import inspect, build123d as bd
print(inspect.signature(bd.export_stl))

# -- cell 8 -------------------------------------------------------------------------
import os
os.makedirs("constant/triSurface", exist_ok=True)
PATCHES = ["shaftWall","column","treads","floor","topOpening"]
for k in PATCHES:
    comp = bd.Compound(children=[bd.Solid() if False else f for f in groups[k]])
    bd.export_stl(comp, f"constant/triSurface/{k}.stl", tolerance=0.002, angular_tolerance=0.2, ascii_format=True)
    # rewrite solid name so snappy sees the patch name
    p=f"constant/triSurface/{k}.stl"; s=open(p).read().replace("solid ",f"solid {k} ",1)
    lines=s.splitlines()
    lines[0]=f"solid {k}"; lines[-1]=f"endsolid {k}"
    open(p,"w").write("\n".join(lines)+"\n")
    print(k, os.path.getsize(p)//1024, "kB", sum(1 for l in lines if l.strip().startswith("facet")), "tris")

# -- cell 9 -------------------------------------------------------------------------
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
import numpy as np

TESS_LIN, TESS_ANG = 0.004, 0.25   # m, rad
BRepMesh_IncrementalMesh(fluid.wrapped, TESS_LIN, False, TESS_ANG, True)

def face_tris(f):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    if tri is None: return None
    trsf = loc.Transformation()
    pts = []
    for i in range(1, tri.NbNodes()+1):
        p = tri.Node(i).Transformed(trsf); pts.append((p.X(),p.Y(),p.Z()))
    pts = np.array(pts)
    idx = []
    rev = f.wrapped.Orientation() == 1  # TopAbs_REVERSED
    for i in range(1, tri.NbTriangles()+1):
        a,b,c = tri.Triangle(i).Get()
        idx.append((a-1,c-1,b-1) if rev else (a-1,b-1,c-1))
    return pts, np.array(idx)

tot=0
for k in PATCHES:
    tot += sum(len(face_tris(f)[1]) for f in groups[k])
print("total triangles", tot)

# -- cell 10 ------------------------------------------------------------------------
def write_stl(name, faces):
    path = f"constant/triSurface/{name}.stl"
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        n=0
        for f in faces:
            pts, idx = face_tris(f)
            for t in idx:
                p0,p1,p2 = pts[t[0]],pts[t[1]],pts[t[2]]
                nv = np.cross(p1-p0,p2-p0); L=np.linalg.norm(nv)
                if L < 1e-14: continue
                nv/=L
                fh.write("  facet normal %.6e %.6e %.6e\n   outer loop\n" % tuple(nv))
                for p in (p0,p1,p2): fh.write("    vertex %.9e %.9e %.9e\n" % tuple(p))
                fh.write("   endloop\n  endfacet\n"); n+=1
        fh.write(f"endsolid {name}\n")
    return n
for k in PATCHES: print(k, write_stl(k, groups[k]))

# -- cell 11 ------------------------------------------------------------------------
import pyvista as pv
u = pv.PolyData()
for k in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{k}.stl"))
u = u.clean(tolerance=1e-7)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
nm = u.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
print("cells",u.n_cells,"free edges",fe.n_cells,"non-manifold",nm.n_cells, "volume", u.volume)

# -- cell 12 ------------------------------------------------------------------------
import os, subprocess
for d in ("system","constant/triSurface","0"): os.makedirs(d, exist_ok=True)

def foam(cls, obj, body, loc="system"):
    hdr = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj))
    open(f"{loc}/{obj}", "w").write("/*--------*- C++ -*----------*/\n"+hdr+body+"\n")

foam("dictionary","controlDict", """
application     snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 8; runTimeModifiable true;
""")
foam("dictionary","fvSchemes","ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
foam("dictionary","fvSolution","solvers{}\n")

L = 1.25
foam("dictionary","blockMeshDict", f"""
scale 1;
vertices
(
    (-{L} -{L} -0.05) ({L} -{L} -0.05) ({L} {L} -0.05) (-{L} {L} -0.05)
    (-{L} -{L} 3.05) ({L} -{L} 3.05) ({L} {L} 3.05) (-{L} {L} 3.05)
);
blocks ( hex (0 1 2 3 4 5 6 7) (25 25 31) simpleGrading (1 1 1) );
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-500:])

# -- cell 13 ------------------------------------------------------------------------
geom = "\n".join(f'    {k}.stl {{ type triSurfaceMesh; name {k}; }}' for k in PATCHES)
refs = "\n".join(
  f'        {k} {{ level ({lv} {lv}); patchInfo {{ type {ty}; }} }}'
  for k,lv,ty in [("shaftWall",1,"wall"),("column",2,"wall"),("treads",2,"wall"),
                  ("floor",1,"wall"),("topOpening",1,"patch")])
foam("dictionary","snappyHexMeshDict", f"""
castellatedMesh true; snap true; addLayers false;

geometry
{{
{geom}
}};

castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10;
    maxLoadUnbalance 0.1; nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{refs}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}};
    locationInMesh (0.7 0.0 1.45);
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3;
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1;
    nSmoothScale 4; errorReduction 0.75;
}}

mergeTolerance 1e-6;
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-2500:])

# -- cell 14 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 15 ------------------------------------------------------------------------
open("case.foam","w").write("")
import pyvista as pv
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
rd.disable_all_cell_arrays(); m = rd.read()
bnd = m["boundary"]
print(list(bnd.keys()))
p = pv.Plotter(off_screen=True, window_size=(900,900))
cols={"shaftWall":"lightblue","column":"orange","treads":"green","floor":"gray","topOpening":"red"}
for k,c in cols.items():
    if k in bnd.keys(): p.add_mesh(bnd[k], color=c, show_edges=True, line_width=0.3,
                                   opacity=0.15 if k=="shaftWall" else 1.0)
p.camera_position=[(5,-5,4.5),(0,0,1.5),(0,0,1)]
p.screenshot("mesh.png"); from IPython.display import Image, display; display(Image("mesh.png"))

# -- cell 16 ------------------------------------------------------------------------
import numpy as np, math
horiz=[]
for f in groups["treads"]:
    if f.geom_type==bd.GeomType.PLANE and abs(f.normal_at(f.center()).Z)>0.999:
        v=np.array([[p.X,p.Y,p.Z] for p in f.vertices()]); horiz.append((f,v))
# pair up by z: tops and bottoms
zs=sorted(set(round(v[0][2],4) for f,v in horiz))
print("distinct tread face heights:", zs)
print("rise between consecutive tops (mm):",
      [round((zs[i+2]-zs[i])*1000,2) for i in range(0,len(zs)-2,2)][:5])
print("thickness (mm):", [round((zs[i+1]-zs[i])*1000,2) for i in range(0,len(zs)-1,2)][:5])
# angle of each top face
ang=[]
for f,v in horiz:
    c=v.mean(axis=0); d=v[:,:2]-c[:2]
    u,s,vt=np.linalg.svd(d-d.mean(0)); ax=vt[0]
    a=math.degrees(math.atan2(c[1],c[0]))%360
    ang.append((round(v[0][2],4), round(a,2)))
ang=sorted(set(ang))
print("centroid angle per tread face:", ang[:6])
print("radial extent of one tread top: r_min=%.4f r_max=%.4f"%(
      np.hypot(*horiz[0][1][:,:2].T).min(), np.hypot(*horiz[0][1][:,:2].T).max()))

# -- cell 17 ------------------------------------------------------------------------
tops=[(f,v) for f,v in horiz if abs(round(v[0][2],4)*1000 % 150) < 1e-6]  # top faces at multiples of 150 mm
res=[]
for f,v in tops:
    c=f.center(); a=math.degrees(math.atan2(c.Y,c.X))%360
    # depth: extent perpendicular to the radial centreline
    u=np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
    perp=np.array([-u[1],u[0]]); w=v[:,:2]@perp
    res.append((round(c.Z,3), round(a,3), round((w.max()-w.min())*1000,2)))
res.sort()
for r_ in res[:5]+res[-2:]: print("z=%.3f m  angle=%7.3f deg  depth=%.1f mm"%r_)
print("turn per tread (deg):", [round(res[i+1][1]-res[i][1],3) for i in range(4)])

# -- cell 18 ------------------------------------------------------------------------
refs = "\n".join(
  f'        {k} {{ level ({lv} {lv}); patchInfo {{ type {ty}; }} }}'
  for k,lv,ty in [("shaftWall",1,"wall"),("column",2,"wall"),("treads",3,"wall"),
                  ("floor",1,"wall"),("topOpening",1,"patch")])
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"refinementSurfaces\n    \{.*?\n    \}", "refinementSurfaces\n    {\n"+refs+"\n    }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:])

# -- cell 19 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
out=r.stdout
import re
print(re.search(r"Checking patch topology.*?Checking faceZone", out, re.S).group()[:900])
print(re.search(r"Checking geometry.*", out, re.S).group()[-1300:])

# -- cell 20 ------------------------------------------------------------------------
u = pv.PolyData()
for k in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{k}.stl"))
u = u.clean(tolerance=1e-8)
fe=u.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
nm=u.extract_feature_edges(boundary_edges=False,feature_edges=False,manifold_edges=False,non_manifold_edges=True)
print("free edges",fe.n_cells,"non-manifold",nm.n_cells,"vol",u.volume)

# -- cell 21 ------------------------------------------------------------------------
pts=fe.points
print(fe.bounds)
import numpy as np
zs=np.unique(np.round(pts[:,2],3)); print("z of free edges:", zs[:20], len(zs))
r=np.hypot(pts[:,0],pts[:,1]); print("r range %.4f %.4f"%(r.min(),r.max()))

# -- cell 22 ------------------------------------------------------------------------
pf = face_tris(groups["floor"][0])[0]; pw = face_tris(groups["shaftWall"][0])[0]
bf = pf[np.abs(np.hypot(pf[:,0],pf[:,1])-1.2)<1e-6]
bw = pw[(np.abs(pw[:,2])<1e-9)]
print(len(bf), len(bw))
af=np.sort(np.degrees(np.arctan2(bf[:,1],bf[:,0]))%360); aw=np.sort(np.degrees(np.arctan2(bw[:,1],bw[:,0]))%360)
print(af[:8]); print(aw[:8])

# -- cell 23 ------------------------------------------------------------------------
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopExp import TopExp
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
m_ = TopTools_IndexedDataMapOfShapeListOfShape()
TopExp.MapShapesAndAncestors_s(fluid.wrapped, TopAbs_EDGE, TopAbs_FACE, m_)
cnt = {}
for i in range(1, m_.Extent()+1):
    n = m_.FindFromIndex(i).Extent(); cnt[n] = cnt.get(n, 0) + 1
print("edges by #adjacent faces:", cnt, "solid valid:", fluid.is_valid)

# -- cell 24 ------------------------------------------------------------------------
from OCP.BRepTools import BRepTools
BRepTools.Clean_s(fluid.wrapped)                     # drop any per-face triangulation left by earlier exports
BRepMesh_IncrementalMesh(fluid.wrapped, 0.003, False, 0.20, True)   # one shared tessellation for the whole solid
for k in PATCHES: print(k, write_stl(k, groups[k]))
u = pv.PolyData()
for k in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{k}.stl"))
u = u.clean(tolerance=1e-8)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges", fe.n_cells, "vol", u.volume, "vs CAD", fluid.volume)

# -- cell 25 ------------------------------------------------------------------------
BRepTools.Clean_s(fluid.wrapped)
BRepMesh_IncrementalMesh(fluid.wrapped, 0.001, False, 0.12, True)
for k in PATCHES: print(k, write_stl(k, groups[k]))
u = pv.PolyData()
for k in PATCHES: u = u.merge(pv.read(f"constant/triSurface/{k}.stl"))
u = u.clean(tolerance=1e-8)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges", fe.n_cells, "STL vol", u.volume, "CAD vol", fluid.volume)

# -- cell 26 ------------------------------------------------------------------------
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:])
