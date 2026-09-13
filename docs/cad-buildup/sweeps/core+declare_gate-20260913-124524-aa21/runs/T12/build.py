"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll set up the parameters and check the blade pitch before building anything.
import numpy as np, os, subprocess
from build123d import *

mm = 1e-3
R_BP, T_BP = 60*mm, 4*mm          # backplate (reference only)
R_HUB, H_HUB = 12*mm, 18*mm       # hub rises 18 mm from backplate top
Z0 = T_BP                          # backplate top surface, z = 4 mm
Z_SHR = Z0 + 14*mm                 # shroud, 14 mm above backplate  -> 18 mm
Z_TOP = Z0 + H_HUB                 # hub top / inlet plane -> 22 mm
R1, R2 = 16*mm, 58*mm              # blade leading / trailing radius
T_BL, N_BL = 3*mm, 7               # blade thickness, count
WRAP = np.radians(30.0)            # backward sweep, against rotation

def camber(n=60):
    r = np.linspace(R1, R2, n)
    th = -WRAP*(r-R1)/(R2-R1)      # negative = backward (against +theta rotation)
    return r, th

r, th = camber()
print("wrap deg =", np.degrees(th[0]-th[-1]), " r1,r2 mm =", r[0]/mm, r[-1]/mm)
print("pitch at R1 (mm) =", 2*np.pi*R1/N_BL/mm, " passage gap =", (2*np.pi*R1/N_BL-T_BL)/mm)
print("passage gap at R2 (mm) =", (2*np.pi*R2/N_BL-T_BL)/mm)

# -- cell 2 -------------------------------------------------------------------------
# Narrowest passage is 11.4 mm, so cells around 1.2–1.5 mm. A 2 mm root fillet would need ~0.4 mm cell
R2E = 60*mm   # blade extended past the 58 mm outlet so the outlet cylinder trims it

def blade_face(n=80):
    rr = np.linspace(R1, R2E, n)
    tt = -WRAP*(rr-R1)/(R2-R1)
    x, y = rr*np.cos(tt), rr*np.sin(tt)
    dx, dy = np.gradient(x), np.gradient(y)
    L = np.hypot(dx, dy); nx, ny = -dy/L, dx/L
    a = [(x[i]+1.5*mm*nx[i], y[i]+1.5*mm*ny[i]) for i in range(n)]
    b = [(x[i]-1.5*mm*nx[i], y[i]-1.5*mm*ny[i]) for i in range(n)]
    return Polygon(*(a + b[::-1]), align=None)

bf = blade_face()
blade0 = extrude(bf, amount=Z_SHR-Z0).moved(Location((0,0,Z0)))
blades = Compound([blade0.moved(Rot(0,0,360/N_BL*k)) for k in range(N_BL)])
print("blade vol mm^3", blade0.volume/mm**3, "  z-range", blade0.bounding_box().min.Z/mm, blade0.bounding_box().max.Z/mm)

# -- cell 3 -------------------------------------------------------------------------
# The extrude went in −Z; I'll place it explicitly rather than rely on the face normal.
bf = blade_face()
bl_raw = extrude(bf, amount=Z_SHR-Z0)
bb = bl_raw.bounding_box(); print("raw z", bb.min.Z/mm, bb.max.Z/mm)
blade0 = Pos(0, 0, Z0 - bb.min.Z) * bl_raw
bb = blade0.bounding_box(); print("placed z", bb.min.Z/mm, bb.max.Z/mm)
blades = Compound([Rot(0, 0, 360/N_BL*k) * blade0 for k in range(N_BL)])
print("n solids", len(blades.solids()), " total vol mm^3", blades.volume/mm**3)

# -- cell 4 -------------------------------------------------------------------------
# Now the fluid volume: the impeller disc plus the axial eye annulus, minus hub and blades.
R_EYE = R1   # shroud eye inner radius = blade leading-edge radius
disc = Pos(0,0,(Z0+Z_SHR)/2) * Cylinder(R2, Z_SHR-Z0)
eye  = Pos(0,0,(Z_SHR+Z_TOP)/2) * Cylinder(R_EYE, Z_TOP-Z_SHR)
hub  = Pos(0,0,(Z0+Z_TOP)/2) * Cylinder(R_HUB, Z_TOP-Z0)
fluid = (disc + eye) - hub - blades
print("solids:", len(fluid.solids()), " vol mm^3:", fluid.volume/mm**3)
bb = fluid.bounding_box(); print("bbox mm", bb.min, bb.max)
print("faces:", len(fluid.faces()))

# -- cell 5 -------------------------------------------------------------------------
# Fluid is a single solid. Now classify faces by their actual surface geometry (plane z, cylinder axis
def classify(f):
    t = f.geom_type
    c = f.center()
    if t == GeomType.PLANE:
        n = f.normal_at(c)
        if abs(abs(n.Z)-1) < 1e-6:
            if abs(c.Z-Z0) < 1e-7:    return "backplate"
            if abs(c.Z-Z_TOP) < 1e-7: return "inlet"
            if abs(c.Z-Z_SHR) < 1e-7: return "shroud"
        return "blades"
    if t == GeomType.CYLINDER:
        rad = (f.center() - Vector(0,0,c.Z)).length
        ax = f.axis_of_rotation if hasattr(f,'axis_of_rotation') else None
        if abs(rad-R2) < 1e-6 and c.Z < Z_SHR:  return "outlet"
        if abs(rad-R_HUB) < 1e-6:               return "hub"
        if abs(rad-R_EYE) < 1e-6 and c.Z > Z_SHR: return "shroud"
    return "blades"

groups = {}
for f in fluid.faces():
    groups.setdefault(classify(f), []).append(f)
for k,v in groups.items():
    print(k, len(v), "area mm^2", sum(x.area for x in v)/mm**2)

# -- cell 6 -------------------------------------------------------------------------
# Group areas match the analytic annuli (inlet π(16²−12²)=351.9, hub 2π·12·18=1357.2). Now measure the
V = np.array([[v.X,v.Y,v.Z] for v in blade0.vertices()])
rad = np.hypot(V[:,0],V[:,1]); ang = np.degrees(np.arctan2(V[:,1],V[:,0]))
le = ang[np.abs(rad-R1)<0.4*mm]; te = ang[np.abs(rad-R2)<1.0*mm]
print("LE theta", le.mean(), " TE theta(at r~58)", te.mean(), " wrap deg =", le.mean()-te.mean())
# thickness: distance between the two offset flanks at mid-chord
rm = 37*mm; tm = -WRAP*(rm-R1)/(R2-R1)
p = Vector(rm*np.cos(tm), rm*np.sin(tm), 11*mm)
print("mid-chord thickness mm =", 2*abs(blade0.distance_to(p) if hasattr(blade0,'distance_to') else 0))
print("blade height mm =", (blade0.bounding_box().max.Z-blade0.bounding_box().min.Z)/mm,
      " blade solid vol/(h*t) -> camber len mm =", blade0.volume/(14*mm*3*mm)/mm)

# -- cell 7 -------------------------------------------------------------------------
# The wrap measurement was biased by which flank vertices got caught. Measuring the camber line proper
Vb = V[np.abs(V[:,2]-Z0)<1e-9]                      # bottom face vertices
rb = np.hypot(Vb[:,0],Vb[:,1])
le2 = Vb[np.argsort(rb)[:2]]                        # two leading-edge corners
mid_le = le2.mean(axis=0)
# trailing camber point at r=58 from the blade's outer flank pair
mask = np.abs(rb-R2E)<0.5*mm
te2 = Vb[mask]; mid_te = te2.mean(axis=0)
a = lambda p: np.degrees(np.arctan2(p[1],p[0]))
print("LE camber r,theta =", np.hypot(*mid_le[:2])/mm, a(mid_le))
print("TE camber r,theta =", np.hypot(*mid_te[:2])/mm, a(mid_te))
th58 = -np.degrees(WRAP)*(R2-R1)/(R2-R1)
print("wrap over 16->58 mm, design =", np.degrees(WRAP), " built(16->60) =", a(mid_le)-a(mid_te))
print("built wrap scaled to r=58 =", (a(mid_le)-a(mid_te))*(R2-R1)/(R2E-R1))

# -- cell 8 -------------------------------------------------------------------------
# Let me measure camber angle as the mid of the two flanks within radius bands, which doesn't depend o
rb_ang = np.degrees(np.arctan2(Vb[:,1], Vb[:,0]))
def camber_theta(r0, w=0.6*mm):
    m = np.abs(rb-r0) < w
    return 0.5*(rb_ang[m].max()+rb_ang[m].min()), m.sum()
for r0 in [16.1, 30, 44, 58]:
    t_, n_ = camber_theta(r0*mm); print(f"r={r0:5.1f} mm  camber theta = {t_:7.3f} deg  (n={n_})")
t16,_ = camber_theta(16.1*mm); t58,_ = camber_theta(58*mm)
print("built wrap 16->58 =", t16-t58, "deg   vs requested 30")

# -- cell 9 -------------------------------------------------------------------------
os.makedirs("constant/triSurface", exist_ok=True)
for name, faces in groups.items():
    sh = Compound(children=list(faces)) if len(faces) > 1 else faces[0]
    export_stl(sh, f"constant/triSurface/{name}.stl", tolerance=5e-5, angular_tolerance=0.2)
    print(name, len(faces), os.path.getsize(f"constant/triSurface/{name}.stl"))

# -- cell 10 ------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(900,700))
cols = dict(blades="firebrick", backplate="steelblue", hub="orange", shroud="lightgray", inlet="green", outlet="purple")
for n,c in cols.items():
    m = pv.read(f"constant/triSurface/{n}.stl")
    pl.add_mesh(m, color=c, opacity=0.35 if n=="shroud" else 1.0, show_edges=False)
pl.camera_position = [(0.14,-0.12,0.12),(0,0,0.011),(0,0,1)]
pl.screenshot("view1.png"); print("ok")

# -- cell 11 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view1.png"))

# -- cell 12 ------------------------------------------------------------------------
# Shape is right. Now the coarsest mesh that exists: blockMesh box + snappyHexMesh with the six STL re
import textwrap, shutil
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
head = lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(head("dictionary","controlDict")+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;\n"
 "runTimeModifiable true;\n")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution")+"solvers{}\n")
N = 40
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ( (-0.0605 -0.0605 0.0035) (0.0605 -0.0605 0.0035) (0.0605 0.0605 0.0035) (-0.0605 0.0605 0.0035)
           (-0.0605 -0.0605 0.0225) (0.0605 -0.0605 0.0225) (0.0605 0.0605 0.0225) (-0.0605 0.0605 0.0225) );
blocks ( hex (0 1 2 3 4 5 6 7) ({N} {N} 7) simpleGrading (1 1 1) );
edges (); boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 13 ------------------------------------------------------------------------
names = ["inlet","outlet","blades","backplate","hub","shroud"]
geom = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs = "\n".join(f'    {n} {{ level ({1 if n=="blades" else 1} {1 if n=="blades" else 1}); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in names)
th=np.radians(12.0); pin=(0.035*np.cos(th),0.035*np.sin(th),0.011)
open("system/snappyHexMeshDict","w").write(head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; features (); 
  refinementSurfaces {{
{refs}
  }}
  resolveFeatureAngle 30; refinementRegions {{}};
  locationInMesh ({pin[0]:.6f} {pin[1]:.6f} {pin[2]:.6f}); allowFreeStandingZoneFaces true;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-14; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
print("pin", pin)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:])

# -- cell 14 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):r.stdout.find("Checking geometry")])
print(r.stdout[-900:])

# -- cell 15 ------------------------------------------------------------------------
# Mesh OK, volume 129309 mm³ vs CAD 129330 mm³ (0.02%). The inlet annulus only has 148 faces though, s
N2 = 61
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ( (-0.0605 -0.0605 0.0035) (0.0605 -0.0605 0.0035) (0.0605 0.0605 0.0035) (-0.0605 0.0605 0.0035)
           (-0.0605 -0.0605 0.0225) (0.0605 -0.0605 0.0225) (0.0605 0.0605 0.0225) (-0.0605 0.0605 0.0225) );
blocks ( hex (0 1 2 3 4 5 6 7) ({N2} {N2} 10) simpleGrading (1 1 1) );
edges (); boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""")
subprocess.run(["blockMesh"],check=True,capture_output=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-300:])
c=subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print(c[c.find("Checking patch topology"):c.find("Checking faceZone")])
print([l for l in c.splitlines() if "Total volume" in l or "non-orthogonality" in l.lower() or "skewness" in l or l.strip().startswith("cells:")][:8])
print(c.strip()[-40:])

# -- cell 16 ------------------------------------------------------------------------
# 84k cells, Mesh OK, volume within 0.007% of CAD. Let me look at the mesh patches.
subprocess.run(["foamToVTK","-constant","-excludePatches","()"],capture_output=True,text=True)
import glob; print(glob.glob("VTK/**/*.vt*",recursive=True)[:12])

# -- cell 17 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(1000,800), shape=(1,2))
for j,cam in enumerate([[(0.13,-0.11,0.14),(0,0,0.012),(0,0,1)],[(0.0,-0.001,0.30),(0,0,0.012),(0,1,0)]]):
    pl.subplot(0,j)
    for n,c in cols.items():
        m = pv.read(f"VTK/t12_0/boundary/{n}.vtp")
        if n=="shroud" and j==1: continue
        pl.add_mesh(m, color=c, show_edges=True, edge_color="k", line_width=0.3)
    pl.camera_position = cam
pl.screenshot("mesh.png"); display(Image("mesh.png"))

# -- cell 18 ------------------------------------------------------------------------
# Free edges and self-intersections in the STLs. Let me find which patch file is at fault.
for n in names:
    o=subprocess.run(["surfaceCheck",f"constant/triSurface/{n}.stl"],capture_output=True,text=True).stdout
    key=[l for l in o.splitlines() if "illegal" in l or "open edges" in l or "intersect" in l or "Surface is not closed" in l or "triangles" in l.lower()][:4]
    print(n, key)

# -- cell 19 ------------------------------------------------------------------------
# Every file reports nearly all edges open — the triangles aren't sharing vertices. Let me look at wha
print(open("constant/triSurface/inlet.stl","rb").read(300))
m = pv.read("constant/triSurface/inlet.stl")
print(m.n_points, m.n_cells)
mc = m.clean(tolerance=1e-8)
fe = mc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("after clean:", mc.n_points, " open edges:", fe.n_cells)

# -- cell 20 ------------------------------------------------------------------------
# Binary STL in float32 — the quantization stops shared vertices from welding. I'll re-export in full-
def export_group(name, faces, tol=5e-5, ang=0.2):
    tris = []
    for f in faces:
        vs, ts = f.tessellate(tolerance=tol, angular_tolerance=ang)
        for t in ts:
            tris.append([vs[i] for i in t])
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for a,b,c in tris:
            u, v = b-a, c-a
            n = u.cross(v); n = n.normalized() if n.length > 0 else Vector(0,0,1)
            fh.write(f" facet normal {n.X:.9e} {n.Y:.9e} {n.Z:.9e}\n  outer loop\n")
            for p in (a,b,c): fh.write(f"   vertex {p.X:.9e} {p.Y:.9e} {p.Z:.9e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
    return len(tris)

for n in names: print(n, export_group(n, groups[n]))

# -- cell 21 ------------------------------------------------------------------------
import pyvista as pv
u = pv.PolyData()
for n in names: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(absolute=True, tolerance=1e-9)
fe = uc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("union points", uc.n_points, "cells", uc.n_cells, "free edges", fe.n_cells)

# -- cell 22 ------------------------------------------------------------------------
# Per-face tessellation doesn't share edge nodes. I'll mesh the whole solid once with BRepMesh so adja
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED

BRepMesh_IncrementalMesh(fluid.wrapped, 5e-5, False, 0.2, True)

def export_group2(name, faces):
    tris = []
    for f in faces:
        loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
        if tri is None: print("no tri", name); continue
        trsf = loc.Transformation(); rev = f.wrapped.Orientation() == TopAbs_REVERSED
        P = [tri.Node(i+1).Transformed(trsf) for i in range(tri.NbNodes())]
        for k in range(tri.NbTriangles()):
            a,b,c = tri.Triangle(k+1).Get()
            idx = (a-1,c-1,b-1) if rev else (a-1,b-1,c-1)
            tris.append([P[i] for i in idx])
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for a,b,c in tris:
            u = np.array([b.X()-a.X(), b.Y()-a.Y(), b.Z()-a.Z()]); v = np.array([c.X()-a.X(), c.Y()-a.Y(), c.Z()-a.Z()])
            n = np.cross(u,v); L = np.linalg.norm(n); n = n/L if L>0 else np.array([0,0,1.])
            fh.write(f" facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n  outer loop\n")
            for p in (a,b,c): fh.write(f"   vertex {p.X():.9e} {p.Y():.9e} {p.Z():.9e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
    return len(tris)

for n in names: print(n, export_group2(n, groups[n]))

# -- cell 23 ------------------------------------------------------------------------
u = pv.PolyData()
for n in names: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(absolute=True, tolerance=1e-9)
fe = uc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("union points", uc.n_points, "cells", uc.n_cells, "free edges", fe.n_cells)
print("volume of union:", uc.volume/mm**3, " vs CAD", fluid.volume/mm**3)

# -- cell 24 ------------------------------------------------------------------------
# Volume is right but 2177 free edges persist — let me see where they are.
pts = fe.points
r_ = np.hypot(pts[:,0],pts[:,1]); z_ = pts[:,2]
import collections
print("z hist", collections.Counter(np.round(z_/mm,1)).most_common(8))
print("r range", r_.min()/mm, r_.max()/mm)
print("n free-edge pts", len(pts))

# -- cell 25 ------------------------------------------------------------------------
# Free edges sit entirely on the two annuli at r 12–16. Let me look at them.
pl = pv.Plotter(off_screen=True, window_size=(800,700))
pl.add_mesh(uc, color="lightgray", opacity=0.3)
pl.add_mesh(fe, color="red", line_width=3)
pl.camera_position=[(0.09,-0.08,0.09),(0,0,0.013),(0,0,1)]
pl.screenshot("fe.png"); display(Image("fe.png"))
print(fe.n_cells, uc.n_cells)

# -- cell 26 ------------------------------------------------------------------------
# Three seam circles (r=12 and 16 at the inlet plane, r=12 at the backplate) have duplicate/unshared e
fluid2 = fluid.clean()
print("faces", len(fluid2.faces()), "vol", fluid2.volume/mm**3)
groups2 = {}
for f in fluid2.faces(): groups2.setdefault(classify(f), []).append(f)
print({k:len(v) for k,v in groups2.items()})
BRepMesh_IncrementalMesh(fluid2.wrapped, 5e-5, False, 0.2, True)
for n in names: export_group2(n, groups2[n])
u = pv.PolyData()
for n in names: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(absolute=True, tolerance=1e-9)
fe = uc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges", fe.n_cells, " vol", uc.volume/mm**3)

# -- cell 27 ------------------------------------------------------------------------
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
m = TopTools_IndexedDataMapOfShapeListOfShape()
TopExp.MapShapesAndAncestors_s(fluid2.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
cnt = collections.Counter()
lonely=[]
for i in range(1, m.Extent()+1):
    k = m.FindFromIndex(i).Extent(); cnt[k]+=1
    if k==1: lonely.append(m.FindKey(i))
print(cnt)
for e in lonely[:6]:
    ed = Edge(e); c=ed.center(); print("lonely edge len",ed.length/mm,"center",c, "r",np.hypot(c.X,c.Y)/mm)
