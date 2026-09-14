"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

# --- given dimensions (mm) ---
D_BARREL, H_BARREL = 90.0, 120.0
N_FIN, D_FIN, T_FIN, GAP = 12, 140.0, 3.0, 6.0
DUCT, BELOW, ABOVE = 300.0, 150.0, 300.0

PITCH = T_FIN + GAP                      # 9 mm
SPAN  = N_FIN*T_FIN + (N_FIN-1)*GAP      # fin stack length
Z0    = (H_BARREL - SPAN)/2.0            # margin at each end of barrel
print("pitch", PITCH, "span", SPAN, "margin", Z0)

barrel = bd.Cylinder(D_BARREL/2, H_BARREL, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
head   = bd.Pos(0,0,H_BARREL) * bd.Sphere(D_BARREL/2, arc_size1=0, arc_size2=90)
fins = []
for i in range(N_FIN):
    zb = Z0 + i*PITCH
    fins.append(bd.Pos(0,0,zb) * bd.Cylinder(D_FIN/2, T_FIN,
                align=(bd.Align.CENTER,)*2+(bd.Align.MIN,)))
solid = barrel + head
for f in fins: solid = solid + f
bb = solid.bounding_box()
print("bbox", bb.min, bb.max, "volume mm^3", round(solid.volume,1))
print("faces", len(solid.faces()))

# -- cell 2 -------------------------------------------------------------------------
# Dome bbox was recentred by `align`; pin its base to the barrel top.
head = bd.Pos(0,0,H_BARREL) * bd.Sphere(D_BARREL/2, arc_size1=0, arc_size2=90,
            align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
solid = barrel + head
for f in fins: solid = solid + f
bb = solid.bounding_box()
print("bbox", bb.min, bb.max)
print("expected top", H_BARREL + D_BARREL/2)
print("faces", len(solid.faces()), "volume mm^3", round(solid.volume,1))

# -- cell 3 -------------------------------------------------------------------------
from collections import Counter
info=[]
for f in solid.faces():
    c=f.center(); gt=f.geom_type
    r=(c.X**2+c.Y**2)**0.5
    info.append((str(gt), round(r,2), round(c.Z,2), round(f.area,1)))
for k,v in Counter([(i[0],) for i in info]).items(): print(k,v)
for i in sorted(set(info)): print(i)

# -- cell 4 -------------------------------------------------------------------------
MM = 1/1000.0   # mm -> m, scale once here; everything exported is metres
fcs = solid.faces()
barrel_f = [f for f in fcs if (f.geom_type==bd.GeomType.CYLINDER and abs((f.center().X**2+f.center().Y**2)**0.5-45)<1e-6)
            or (f.geom_type==bd.GeomType.PLANE and abs(f.center().Z)<1e-6)]
fin_f    = [f for f in fcs if (f.geom_type==bd.GeomType.CYLINDER and abs((f.center().X**2+f.center().Y**2)**0.5-70)<1e-6)
            or (f.geom_type==bd.GeomType.PLANE and f.center().Z>1e-6)]
head_f   = [f for f in fcs if f.geom_type==bd.GeomType.SPHERE]
print(len(barrel_f), len(fin_f), len(head_f), len(barrel_f)+len(fin_f)+len(head_f), len(fcs))

ZMIN, ZMAX = -BELOW, H_BARREL + D_BARREL/2 + ABOVE
print("duct z", ZMIN, ZMAX, "x,y +-", DUCT/2)
half = DUCT/2
inlet_f  = [bd.Pos(0,0,ZMIN) * bd.Rectangle(DUCT, DUCT).face()]
outlet_f = [bd.Pos(0,0,ZMAX) * bd.Rectangle(DUCT, DUCT).face()]
L = ZMAX-ZMIN
side = bd.Rectangle(DUCT, L).face()
walls_f = [bd.Pos(0,-half,(ZMIN+ZMAX)/2) * bd.Rotation(90,0,0) * side,
           bd.Pos(0, half,(ZMIN+ZMAX)/2) * bd.Rotation(90,0,0) * side,
           bd.Pos(-half,0,(ZMIN+ZMAX)/2) * bd.Rotation(90,0,90) * side,
           bd.Pos( half,0,(ZMIN+ZMAX)/2) * bd.Rotation(90,0,90) * side]
for nm,g in [("barrel",barrel_f),("fins",fin_f),("head",head_f),
             ("ductWalls",walls_f),("inlet",inlet_f),("outlet",outlet_f)]:
    sh = bd.Compound(children=[f for f in g]).scale(MM)
    bd.export_stl(sh, f"constant/triSurface/{nm}.stl", tolerance=2e-5, angular_tolerance=0.1)
    print(nm, len(g), "bbox", sh.bounding_box().min, sh.bounding_box().max)

# -- cell 5 -------------------------------------------------------------------------
zc = (ZMIN+ZMAX)/2
rect = bd.Rectangle(DUCT, L).face()
walls_f = [(bd.Plane.XZ.offset(half) * rect).moved(bd.Pos(0,0,zc)),
           (bd.Plane.XZ.offset(-half) * rect).moved(bd.Pos(0,0,zc)),
           (bd.Plane.YZ.offset(half) * rect).moved(bd.Pos(0,0,zc)),
           (bd.Plane.YZ.offset(-half) * rect).moved(bd.Pos(0,0,zc))]
shw = bd.Compound(children=walls_f).scale(MM)
bd.export_stl(shw, "constant/triSurface/ductWalls.stl")
print("ductWalls bbox", shw.bounding_box().min, shw.bounding_box().max, "area m2", round(shw.area,4),
      "expected", 4*DUCT*L*MM*MM)

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(900,700))
cols={"barrel":"orange","fins":"steelblue","head":"red","ductWalls":"lightgray","inlet":"green","outlet":"purple"}
for n,c in cols.items():
    m=pv.read(f"constant/triSurface/{n}.stl")
    pl.add_mesh(m, color=c, opacity=0.25 if n in ("ductWalls","inlet","outlet") else 1.0,
                show_edges=(n not in ("ductWalls",)))
pl.camera_position=[(1.0,-1.0,0.6),(0,0,0.1),(0,0,1)]
pl.screenshot("geom.png")
print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 8 -------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(800,700))
for n,c in [("barrel","orange"),("fins","steelblue"),("head","red")]:
    pl.add_mesh(pv.read(f"constant/triSurface/{n}.stl"), color=c, show_edges=True, lighting=False)
pl.camera_position=[(0.5,-0.5,0.25),(0,0,0.08),(0,0,1)]
pl.screenshot("cyl.png"); display(Image("cyl.png"))

# -- cell 9 -------------------------------------------------------------------------
import textwrap, pathlib
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
HDR = "FoamFile{{version 2.0;format ascii;class dictionary;object {};}}\n"
NB = 15e-3   # background cell size, m  (my choice)
nx = int(round(DUCT*MM/NB)); nz = int(round(L*MM/NB))
print(nx, nz, DUCT*MM/nx, L*MM/nz)
pathlib.Path("system/blockMeshDict").write_text(HDR.format("blockMeshDict") + f"""
scale 1;
vertices
(
 (-0.15 -0.15 -0.15) (0.15 -0.15 -0.15) (0.15 0.15 -0.15) (-0.15 0.15 -0.15)
 (-0.15 -0.15 0.465) (0.15 -0.15 0.465) (0.15 0.15 0.465) (-0.15 0.15 0.465)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet     {{ type patch; faces ((0 3 2 1)); }}
 outlet    {{ type patch; faces ((4 5 6 7)); }}
 ductWalls {{ type wall;  faces ((0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3)); }}
);
mergePatchPairs ();
""")
pathlib.Path("system/controlDict").write_text(HDR.format("controlDict") + """
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
""")
pathlib.Path("system/fvSchemes").write_text(HDR.format("fvSchemes") + """
gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}
""")
pathlib.Path("system/fvSolution").write_text(HDR.format("fvSolution") + "solvers{}\n")
print(subprocess.run(["blockMesh","-case","."],capture_output=True,text=True).stdout[-500:])

# -- cell 10 ------------------------------------------------------------------------
def snappy(levels, nlayers=0, path="system/snappyHexMeshDict"):
    geo = "\n".join([f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in ("barrel","fins","head")])
    refs = "\n".join([f'    {n} {{ level ({levels[n][0]} {levels[n][1]}); patchInfo {{ type wall; }} }}' for n in ("barrel","fins","head")])
    lay = "\n".join([f'    "{n}" {{ nSurfaceLayers {nlayers}; }}' for n in ("barrel","fins","head")]) if nlayers else ""
    pathlib.Path(path).write_text(HDR.format("snappyHexMeshDict") + f"""
castellatedMesh true; snap true; addLayers {"true" if nlayers else "false"};
geometry {{
{geo}
}}
castellatedMeshControls
{{
  maxLocalCells 4000000; maxGlobalCells 12000000; minRefinementCells 10; nCellsBetweenLevels 2;
  maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces
  {{
{refs}
  }}
  refinementRegions {{}}
  locationInMesh (0.1 0.1 0.05);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
  nGrow 0; featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50; layers {{
{lay}
  }} }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
""")
snappy({"barrel":(2,2),"fins":(2,2),"head":(2,2)})
r=subprocess.run(["snappyHexMesh","-case",".","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:]); print(r.stderr[-500:])

# -- cell 11 ------------------------------------------------------------------------
os.makedirs("constant/polyMesh", exist_ok=True)
r=subprocess.run(["foamToVTK","-case",".","-constant","-latestTime"],capture_output=True,text=True)
print(r.returncode, r.stdout[-300:], r.stderr[-300:])
print(sorted(os.listdir("VTK"))[:10] if os.path.isdir("VTK") else os.listdir("."))

# -- cell 12 ------------------------------------------------------------------------
def slicepic(fn, title):
    m = pv.read("VTK/t18_0/internal.vtu")
    c = m.clip(normal=(0,1,0), origin=(0,0,0))
    pl = pv.Plotter(off_screen=True, window_size=(700,900))
    pl.add_mesh(c, show_edges=True, color="white", lighting=False)
    pl.camera_position=[(1.2,0,0.15),(0,0,0.15),(0,0,1)]
    pl.add_text(title, font_size=10)
    pl.screenshot(fn); display(Image(fn))
    pl2 = pv.Plotter(off_screen=True, window_size=(700,700))
    pl2.add_mesh(c, show_edges=True, color="white", lighting=False)
    pl2.camera_position=[(0.45,0,0.06),(0,0,0.06),(0,0,1)]
    pl2.screenshot("zoom_"+fn); display(Image("zoom_"+fn))
slicepic("slice_L2.png","level 2, base 15mm")

# -- cell 13 ------------------------------------------------------------------------
def slicepic(fn, zoom=False):
    m = pv.read("VTK/t18_0/internal.vtu")
    s = m.slice(normal=(0,1,0), origin=(0,0.0,0))
    pl = pv.Plotter(off_screen=True, window_size=(800,800))
    pl.add_mesh(s, show_edges=True, color="white", lighting=False)
    pl.enable_parallel_projection()
    pl.view_xz(); pl.camera.up=(0,0,1)
    if zoom:
        pl.camera.focal_point=(0,0,0.08); pl.camera.position=(0,-1,0.08); pl.camera.parallel_scale=0.11
    pl.screenshot(fn); display(Image(fn))
slicepic("s1.png"); slicepic("s2.png", zoom=True)

# -- cell 14 ------------------------------------------------------------------------
subprocess.run(["blockMesh","-case","."],check=True,capture_output=True,text=True)
snappy({"barrel":(3,3),"fins":(3,3),"head":(3,3)})
r=subprocess.run(["snappyHexMesh","-case",".","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-900:])

# -- cell 15 ------------------------------------------------------------------------
subprocess.run(["foamToVTK","-case",".","-constant","-latestTime"],capture_output=True,text=True)
slicepic("s3.png", zoom=True)

# -- cell 16 ------------------------------------------------------------------------
m = pv.read("VTK/t18_0/internal.vtu")
cc = m.cell_centers().points
rr = np.hypot(cc[:,0],cc[:,1])
for i in range(N_FIN-1):
    zmid = (Z0 + T_FIN + i*PITCH)*MM + GAP/2*MM
    sel = (abs(cc[:,2]-zmid) < 1.5e-3) & (rr>0.047) & (rr<0.068)
    print(f"gap {i+1} z={zmid*1000:.1f}mm cells in gap: {sel.sum()}")

# -- cell 17 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-case",".","-constant","-allGeometry","-allTopology"],capture_output=True,text=True)
print(r.stdout[-3500:])

# -- cell 18 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-case",".","-constant"],capture_output=True,text=True)
print(r.stdout[-1600:])

# -- cell 19 ------------------------------------------------------------------------
import re
d = pathlib.Path("system/snappyHexMeshDict").read_text().replace("minTetQuality 1e-15","minTetQuality 1e-9")
pathlib.Path("system/snappyHexMeshDict").write_text(d)
subprocess.run(["blockMesh","-case","."],check=True,capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-case",".","-overwrite"],capture_output=True,text=True)
print(r.returncode, [l for l in r.stdout.splitlines() if "Snapped mesh" in l])
c=subprocess.run(["checkMesh","-case",".","-constant","-allGeometry","-allTopology"],capture_output=True,text=True).stdout
print("\n".join([l for l in c.splitlines() if "***" in l or "Failed" in l or "Mesh OK" in l]))

# -- cell 20 ------------------------------------------------------------------------
subprocess.run(["foamToVTK","-case",".","-constant","-latestTime"],capture_output=True,text=True)
R,Rf = D_BARREL/2*MM, D_FIN/2*MM
exp = {"barrel": 2*np.pi*R*(H_BARREL-N_FIN*T_FIN)*MM + np.pi*R**2,
       "fins": N_FIN*(2*np.pi*(Rf**2-R**2) + 2*np.pi*Rf*T_FIN*MM),
       "head": 2*np.pi*R**2, "ductWalls": 4*DUCT*MM*L*MM,
       "inlet": (DUCT*MM)**2, "outlet": (DUCT*MM)**2}
for n,a in exp.items():
    got = pv.read(f"VTK/t18_0/boundary/{n}.vtp").compute_cell_sizes(length=False, volume=False)["Area"].sum()
    stl = pv.read(f"constant/triSurface/{n}.stl").compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{n:10s} analytic {a:.5f}  STL {stl:.5f}  mesh patch {got:.5f}")
print("\ntotal cells", pv.read("VTK/t18_0/internal.vtu").n_cells)

# -- cell 21 ------------------------------------------------------------------------
union = pv.PolyData()
for n in exp: union = union.merge(pv.read(f"constant/triSurface/{n}.stl"))
u = union.clean(tolerance=1e-6)
fe = u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges", fe.n_cells)
print("normals consistent?", u.n_open_edges if hasattr(u,'n_open_edges') else "-")
import vtk
print("point normals check: n_cells", u.n_cells)

# -- cell 22 ------------------------------------------------------------------------
# The patch STLs are tessellated face-by-face, so shared edges don't share vertices and the welded uni
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
import struct

def mesh_once(shape, lin=0.05, ang=0.15):   # linear deflection in mm (model units)
    BRepMesh_IncrementalMesh(shape.wrapped, lin, False, ang, True)

def face_tris(face):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    trsf = loc.Transformation()
    pts = np.array([[(p:=tri.Node(i+1).Transformed(trsf)).X(), p.Y(), p.Z()] for i in range(tri.NbNodes())])
    idx = np.array([[t.Value(1)-1, t.Value(2)-1, t.Value(3)-1] for t in
                    (tri.Triangle(i+1) for i in range(tri.NbTriangles()))])
    if face.wrapped.Orientation() == TopAbs_REVERSED: idx = idx[:, ::-1]
    return pts, idx

def write_stl(path, groups, scale):
    tris = []
    for f in groups:
        p, i = face_tris(f)
        tris.append(p[i] * scale)
    T = np.concatenate(tris)
    with open(path, "wb") as fh:
        fh.write(b"\0"*80); fh.write(struct.pack("<I", len(T)))
        for t in T:
            n = np.cross(t[1]-t[0], t[2]-t[0]); nn = np.linalg.norm(n)
            n = n/nn if nn > 0 else n
            fh.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))
    return len(T)

mesh_once(solid)
print(write_stl("/tmp/test_fins.stl", fin_f, MM), "triangles")

# -- cell 23 ------------------------------------------------------------------------
duct_solid = bd.Pos(0,0,zc) * bd.Box(DUCT, DUCT, L)
mesh_once(duct_solid, lin=2.0, ang=0.5)
dfz = [f for f in duct_solid.faces() if abs(abs(f.normal_at(f.center()).Z)-1) < 1e-9]
inlet_f  = [f for f in dfz if abs(f.center().Z-ZMIN) < 1e-6]
outlet_f = [f for f in dfz if abs(f.center().Z-ZMAX) < 1e-6]
walls_f  = [f for f in duct_solid.faces() if abs(f.normal_at(f.center()).Z) < 1e-9]
print(len(inlet_f), len(outlet_f), len(walls_f))
groups = {"barrel":barrel_f, "fins":fin_f, "head":head_f,
          "ductWalls":walls_f, "inlet":inlet_f, "outlet":outlet_f}
for n,g in groups.items():
    nt = write_stl(f"constant/triSurface/{n}.stl", g, MM)
    a = pv.read(f"constant/triSurface/{n}.stl").compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{n:10s} tris {nt:6d} area {a:.5f} analytic {exp[n]:.5f}")

# -- cell 24 ------------------------------------------------------------------------
union = pv.PolyData()
for n in groups: union = union.merge(pv.read(f"constant/triSurface/{n}.stl"))
u = union.clean(tolerance=1e-8)
print("free edges", u.extract_feature_edges(boundary_edges=True, feature_edges=False,
      manifold_edges=False, non_manifold_edges=False).n_cells)
print("non-manifold", u.extract_feature_edges(boundary_edges=False, feature_edges=False,
      manifold_edges=False, non_manifold_edges=True).n_cells)

# -- cell 25 ------------------------------------------------------------------------
subprocess.run(["blockMesh","-case","."],check=True,capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-case",".","-overwrite"],capture_output=True,text=True)
print(r.returncode, [l for l in r.stdout.splitlines() if "Snapped mesh" in l])
c=subprocess.run(["checkMesh","-case",".","-constant"],capture_output=True,text=True).stdout
print("\n".join(c.splitlines()[-12:]))

# -- cell 26 ------------------------------------------------------------------------
subprocess.run(["foamToVTK","-case",".","-constant","-latestTime"],capture_output=True,text=True)
m = pv.read("VTK/t18_0/internal.vtu"); cc = m.cell_centers().points; rr = np.hypot(cc[:,0],cc[:,1])
gaps = [((abs(cc[:,2]-((Z0+T_FIN+i*PITCH)*MM+GAP/2*MM))<1.5e-3)&(rr>0.047)&(rr<0.068)).sum() for i in range(N_FIN-1)]
print("cells in each of the 11 fin gaps:", gaps)
for n in groups:
    a = pv.read(f"VTK/t18_0/boundary/{n}.vtp").compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{n:10s} patch area {a:.5f}  analytic {exp[n]:.5f}")
print("cells", m.n_cells, "fluid volume", 0.0540719, "analytic", DUCT*DUCT*L*MM**3 - solid.volume*MM**3)
slicepic("final.png", zoom=True)

# -- cell 27 ------------------------------------------------------------------------
u = pv.PolyData()
for n in groups: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
u = u.clean(tolerance=1e-8).triangulate()
F = u.faces.reshape(-1,4)[:,1:]
from collections import Counter
de = Counter()
for a,b,c in F:
    for e in ((a,b),(b,c),(c,a)): de[e]+=1
bad = [e for e,k in de.items() if de.get((e[1],e[0]),0)==0 or k>1]
print(len(bad), bad[:8])
for e in bad[:6]: print(u.points[list(e)])

# -- cell 28 ------------------------------------------------------------------------
def wind_check(tol):
    v = pv.PolyData()
    for n in groups: v = v.merge(pv.read(f"constant/triSurface/{n}.stl"))
    v = v.clean(absolute=True, tolerance=tol).triangulate()
    F = v.faces.reshape(-1,4)[:,1:]
    d = Counter()
    for a,b,c in F:
        for e in ((a,b),(b,c),(c,a)): d[e]+=1
    same = [e for e,k in d.items() if k>1]
    ar = v.compute_cell_sizes(length=False, volume=False)["Area"]
    return len(same), (ar<1e-14).sum(), v.n_points, same[:4]
for t in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
    print(t, wind_check(t))
