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
D_BAR, H_BAR = 90.0, 120.0
N_FIN, D_FIN, T_FIN, GAP = 12, 140.0, 3.0, 6.0
DUCT, BELOW, ABOVE = 300.0, 150.0, 300.0
R_BAR, R_FIN = D_BAR/2, D_FIN/2
PITCH = T_FIN + GAP
span = N_FIN*T_FIN + (N_FIN-1)*GAP
z0 = (H_BAR - span)/2.0
print("fin span", span, "start z", z0, "end", z0+span)

barrel = bd.Cylinder(R_BAR, H_BAR, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
head   = bd.Pos(0,0,H_BAR) * bd.Sphere(R_BAR, arc_size1=0)
fins = None
for i in range(N_FIN):
    f = bd.Pos(0,0,z0+i*PITCH) * bd.Cylinder(R_FIN, T_FIN, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
    fins = f if fins is None else fins + f
solid = barrel + head + fins
print("solid bbox", solid.bounding_box())
print("vol", solid.volume)

# -- cell 2 -------------------------------------------------------------------------
head = bd.Pos(0,0,H_BAR) * bd.Sphere(R_BAR, arc_size1=0, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
solid = barrel + head + fins
print("head bbox", head.bounding_box())
print("solid bbox", solid.bounding_box(), solid.volume)
Z_IN, Z_OUT = -BELOW, H_BAR + R_BAR + ABOVE
duct = bd.Box(DUCT, DUCT, Z_OUT-Z_IN, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Pos(0,0,Z_IN))
fluid = duct - solid
print("duct z", Z_IN, Z_OUT, "fluid vol", fluid.volume, "faces", len(fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
for i,f in enumerate(fluid.faces()):
    c = f.center()
    print(i, f.geom_type, "area=%.1f"%f.area, "c=(%.1f,%.1f,%.1f)"%(c.X,c.Y,c.Z),
          "n=(%.2f,%.2f,%.2f)"%tuple(f.normal_at(c)) if f.geom_type==bd.GeomType.PLANE else
          ("r=%.1f"%f.radius if hasattr(f,'radius') else ""))

# -- cell 4 -------------------------------------------------------------------------
import math, subprocess, os
tol=1e-6
groups={"inlet":[],"outlet":[],"duct":[],"barrel":[],"fins":[],"head":[]}
for f in fluid.faces():
    c=f.center(); g=f.geom_type
    if g==bd.GeomType.PLANE:
        n=f.normal_at(c)
        if abs(n.Z)>0.99:
            if abs(c.Z-Z_IN)<1e-6: groups["inlet"].append(f)
            elif abs(c.Z-Z_OUT)<1e-6: groups["outlet"].append(f)
            elif abs(c.Z)<1e-6: groups["barrel"].append(f)   # barrel bottom disc
            else: groups["fins"].append(f)                    # fin annular faces
        else: groups["duct"].append(f)
    elif g==bd.GeomType.CYLINDER:
        groups["fins" if abs(f.radius-R_FIN)<1e-6 else "barrel"].append(f)
    elif g==bd.GeomType.SPHERE:
        groups["head"].append(f)
    else: raise RuntimeError(g)
for k,v in groups.items(): print(k, len(v), "area=%.1f"%sum(x.area for x in v))

# -- cell 5 -------------------------------------------------------------------------
os.makedirs("constant/triSurface", exist_ok=True)
MM=0.001
for k,v in groups.items():
    comp = bd.Compound(children=[bd.Solid(x.wrapped) if False else x for x in v]).scale(MM)
    bd.export_stl(comp, f"constant/triSurface/{k}.stl", tolerance=2e-4, angular_tolerance=0.2)
    print(k, os.path.getsize(f"constant/triSurface/{k}.stl"))
print(subprocess.run(["ls","-l","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
# STLs exported per-face compound may not share edge discretisation. I'll retessellate the whole fluid
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
import numpy as np

LIN_DEF, ANG_DEF = 0.4, 0.25   # mm, rad
BRepMesh_IncrementalMesh(fluid.wrapped, LIN_DEF, False, ANG_DEF, True)

def face_tris(face):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    trsf = loc.Transformation()
    pts=[]
    for i in range(1, tri.NbNodes()+1):
        p = tri.Node(i).Transformed(trsf); pts.append((p.X(),p.Y(),p.Z()))
    pts=np.array(pts)
    rev = face.wrapped.Orientation()==TopAbs_REVERSED
    tl=[]
    for i in range(1, tri.NbTriangles()+1):
        a,b,c = tri.Triangle(i).Get()
        tl.append((a-1,c-1,b-1) if rev else (a-1,b-1,c-1))
    return pts, np.array(tl)

def write_stl(path, name, faces, scale=0.001):
    n=0
    with open(path,"w") as fh:
        fh.write(f"solid {name}\n")
        for f in faces:
            P,T = face_tris(f); P=P*scale
            for t in T:
                v=P[list(t)]
                nrm=np.cross(v[1]-v[0], v[2]-v[0]); L=np.linalg.norm(nrm)
                nrm = nrm/L if L>0 else np.array([0,0,1.])
                fh.write("facet normal %g %g %g\n outer loop\n"%tuple(nrm))
                for q in v: fh.write("  vertex %.9g %.9g %.9g\n"%tuple(q))
                fh.write(" endloop\nendfacet\n"); n+=1
        fh.write(f"endsolid {name}\n")
    return n

for k,v in groups.items():
    print(k, write_stl(f"constant/triSurface/{k}.stl", k, v), "tris")

# -- cell 7 -------------------------------------------------------------------------
from collections import Counter
import glob
def read_stl(p):
    V=[];T=[]
    for line in open(p):
        if line.strip().startswith("vertex"):
            V.append(tuple(round(float(x),9) for x in line.split()[1:]))
    for i in range(0,len(V),3): T.append(V[i:i+3])
    return T
tris=[]
for p in sorted(glob.glob("constant/triSurface/*.stl")): tris+=read_stl(p)
dedges=Counter(); uedges=Counter()
for t in tris:
    for a,b in ((0,1),(1,2),(2,0)):
        dedges[(t[a],t[b])]+=1; uedges[tuple(sorted((t[a],t[b])))]+=1
print("tris",len(tris),"free edges",sum(1 for e,c in uedges.items() if c!=2),
      "same-dir dup",sum(1 for e,c in dedges.items() if c>1))

# -- cell 8 -------------------------------------------------------------------------
print([e for e,c in uedges.items() if c!=2])
print([ (e,c) for e,c in dedges.items() if c>1])

# -- cell 9 -------------------------------------------------------------------------
def write_stl(path, name, faces, scale=0.001, amin=1e-14):
    n=0
    with open(path,"w") as fh:
        fh.write(f"solid {name}\n")
        for f in faces:
            P,T = face_tris(f); P=P*scale
            for t in T:
                v=P[list(t)]
                nrm=np.cross(v[1]-v[0], v[2]-v[0]); L=np.linalg.norm(nrm)
                if L<amin: continue          # skip degenerate (sphere pole) triangles
                nrm=nrm/L
                fh.write("facet normal %g %g %g\n outer loop\n"%tuple(nrm))
                for q in v: fh.write("  vertex %.9g %.9g %.9g\n"%tuple(q))
                fh.write(" endloop\nendfacet\n"); n+=1
        fh.write(f"endsolid {name}\n")
    return n
for k,v in groups.items(): print(k, write_stl(f"constant/triSurface/{k}.stl", k, v), "tris")
tris=[]
for p in sorted(glob.glob("constant/triSurface/*.stl")): tris+=read_stl(p)
dedges=Counter(); uedges=Counter()
for t in tris:
    for a,b in ((0,1),(1,2),(2,0)):
        dedges[(t[a],t[b])]+=1; uedges[tuple(sorted((t[a],t[b])))]+=1
print("free edges",sum(1 for e,c in uedges.items() if c!=2),"same-dir",sum(1 for e,c in dedges.items() if c>1))

# -- cell 10 ------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pl=pv.Plotter(off_screen=True, window_size=(900,900))
for k,c in [("barrel","silver"),("fins","orange"),("head","red"),("duct","lightblue")]:
    m=pv.read(f"constant/triSurface/{k}.stl")
    pl.add_mesh(m.clip(normal='y', origin=(0,0,0)) if k!="duct" else m, color=c,
                opacity=0.2 if k=="duct" else 1.0, show_edges=(k!="duct"))
pl.camera_position=[(0.7,-0.7,0.5),(0,0,0.15),(0,0,1)]
pl.screenshot("look.png"); print("ok")

# -- cell 11 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look.png"))

# -- cell 12 ------------------------------------------------------------------------
# Shape confirmed (12 fins, dome, duct). Now the coarse mesh: blockMesh box = the duct itself (so inle
import textwrap, subprocess, os
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
HDR='FoamFile{version 2.0;format ascii;class dictionary;object %s;}\n'
open("system/controlDict","w").write(HDR%"controlDict"+
 "application snappyHexMesh;startFrom startTime;startTime 0;stopAt endTime;endTime 1;deltaT 1;writeControl timeStep;writeInterval 1;\n")
NX,NY,NZ = 20,20,41   # ~15 mm base cells
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+f"""
scale 1;
vertices ((-0.15 -0.15 -0.15)(0.15 -0.15 -0.15)(0.15 0.15 -0.15)(-0.15 0.15 -0.15)
          (-0.15 -0.15 0.465)(0.15 -0.15 0.465)(0.15 0.15 0.465)(-0.15 0.15 0.465));
blocks (hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1));
edges ();
boundary
(
  inlet {{ type patch; faces ((0 3 2 1)); }}
  outlet{{ type patch; faces ((4 5 6 7)); }}
  duct  {{ type wall;  faces ((0 1 5 4)(2 3 7 6)(1 2 6 5)(3 0 4 7)); }}
);
""")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-600:])

# -- cell 13 ------------------------------------------------------------------------
SNAP = """
castellatedMesh true; snap true; addLayers false;
geometry { barrel.stl{type triSurfaceMesh; name barrel;}
           fins.stl  {type triSurfaceMesh; name fins;}
           head.stl  {type triSurfaceMesh; name head;}
           refBox {type searchableBox; min (-0.09 -0.09 -0.03); max (0.09 0.09 0.20);} }
castellatedMeshControls
{
 maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; nCellsBetweenLevels 2;
 maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
 features ();
 refinementSurfaces { barrel{level (%(L)d %(L)d);} fins{level (%(L)d %(L)d);} head{level (%(L)d %(L)d);} }
 refinementRegions { refBox { mode inside; levels ((1e15 %(LB)d)); } }
 locationInMesh (0.1 0.1 0.3);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
               nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers{} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-9; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(HDR%"snappyHexMeshDict"+SNAP%{"L":2,"LB":1})
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:])

# -- cell 14 ------------------------------------------------------------------------
print(r.returncode); print(r.stderr[-2000:]); print(len(r.stdout))

# -- cell 15 ------------------------------------------------------------------------
open("system/fvSchemes","w").write(HDR%"fvSchemes"+"ddtSchemes{default steadyState;}gradSchemes{default Gauss linear;}divSchemes{default none;}laplacianSchemes{default Gauss linear corrected;}interpolationSchemes{default linear;}snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(HDR%"fvSolution"+"solvers{}\n")
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1200:], r.stderr[-500:])

# -- cell 16 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-2200:])

# -- cell 17 ------------------------------------------------------------------------
open("system/snappyHexMeshDict","w").write(HDR%"snappyHexMeshDict"+SNAP%{"L":3,"LB":2})
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-800:])

# -- cell 18 ------------------------------------------------------------------------
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-900:], r.stderr[-300:])

# -- cell 19 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
import re
print("\n".join([l for l in r.stdout.splitlines() if re.search(r"patch|faces |Total volume|skew|non-ortho|Failed|OK\.|\*\*\*|inlet|outlet|duct|barrel|fins|head",l)])[-2500:])

# -- cell 20 ------------------------------------------------------------------------
f=pv.read("constant/triSurface/fins.stl"); b=pv.read("constant/triSurface/barrel.stl"); h=pv.read("constant/triSurface/head.stl")
print("fin OD (m)", f.bounds[1]-f.bounds[0], " barrel OD", b.bounds[1]-b.bounds[0])
zf=np.unique(np.round(f.points[:,2],6)); print("fin z-planes (mm)", np.round(zf*1000,3))
print("fin thickness", np.round(np.diff(zf*1000)[0::2],3), "gaps", np.round(np.diff(zf*1000)[1::2],3))
print("barrel z range mm", b.bounds[4]*1000, b.bounds[5]*1000, "head top mm", h.bounds[5]*1000)
print("duct 300x300:", pv.read("constant/triSurface/duct.stl").bounds)
print("fluid volume mesh vs CAD:", 0.0540759, fluid.volume*1e-9)

# -- cell 21 ------------------------------------------------------------------------
import pyvista as pv
m=pv.read("constant/polyMesh" if False else ".")  if False else None
of=pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").close()
rd=pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
mesh=rd.read()["internalMesh"]
sl=mesh.slice(normal='y', origin=(0,0,0))
pl=pv.Plotter(off_screen=True, window_size=(700,1100)); pl.add_mesh(sl, show_edges=True, color='white')
pl.camera_position=[(1.2,0,0.16),(0,0,0.16),(0,0,1)]; pl.screenshot("mesh_slice.png")
display(Image("mesh_slice.png"))

# -- cell 22 ------------------------------------------------------------------------
pl=pv.Plotter(off_screen=True, window_size=(600,1100))
pl.add_mesh(sl, show_edges=True, color='lightgray', edge_color='black', line_width=0.4)
pl.view_xz(); pl.reset_camera(); pl.screenshot("mesh_slice.png")
display(Image("mesh_slice.png"))

# -- cell 23 ------------------------------------------------------------------------
import pyvista as pv, glob, numpy as np
u=None
for p in sorted(glob.glob("constant/triSurface/*.stl")):
    m=pv.read(p); u=m if u is None else u.merge(m)
u=u.clean()
pt=pv.PolyData(np.array([[0.1,0.1,0.3]]))
sel=pt.select_enclosed_points(u.triangulate(), tolerance=1e-9, check_surface=False)
print("inside?", sel['SelectedPoints'])
print("n_open_edges", u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).n_cells)
print("dist to surf", np.linalg.norm(u.points-np.array([0.1,0.1,0.3]),axis=1).min())
