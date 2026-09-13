"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess, numpy as np
from build123d import *

R_S, R_B, L = 20.0, 20.05, 60.0      # mm: shaft radius, bore radius, sealed length
KW_W, KW_D = 12.0, 5.0               # mm: keyway width, depth

bore  = Pos(0,0,0) * Cylinder(R_B, L, align=(Align.CENTER, Align.CENTER, Align.MIN))
shaft = Pos(0,0,0) * Cylinder(R_S, L, align=(Align.CENTER, Align.CENTER, Align.MIN))
key   = Pos(0, R_S - KW_D + 50.0, L/2) * Box(KW_W, 100.0, L)   # slot from y=R_S-KW_D outward
fluid = bore - (shaft - key)

print(type(fluid), fluid.volume)
print("bbox", fluid.bounding_box())
print("n faces", len(fluid.faces()), "n solids", len(fluid.solids()))

# -- cell 2 -------------------------------------------------------------------------
# Section area matches the analytic value exactly (62.64081 mm², 3758.4483 mm³). Face 6 broke the prin
f6 = fluid.faces()[6]
import math
print(f6.geom_type, f6.area, f6.center())
th = math.degrees(math.asin(6.0/R_S))
print("expected shaft face area", 2*math.pi*R_S*L*(360-2*th)/360, " keyway half-angle deg", th)
print("expected bore area", 2*math.pi*R_B*L)
print("expected flank height", math.sqrt(R_S**2-36)-(R_S-KW_D))

# -- cell 3 -------------------------------------------------------------------------
# All seven faces match analytically. Now name each patch on the face itself (by surface type/radius/n
import os, math
from build123d import export_stl, Compound, Shell
os.makedirs("constant/triSurface", exist_ok=True)

groups = {k: [] for k in ["shaft","keywayFloor","keywayFlanks","bore","inletHP","outletLP"]}
for f in fluid.faces():
    c = f.center()
    if f.geom_type == GeomType.CYLINDER:
        r = math.hypot(*[getattr(f.center(), a) for a in "XY"])  # placeholder
        # distinguish by area-consistent radius: sample a point on the surface
        p = f.center()
        rad = math.hypot(p.X, p.Y)
        groups["bore" if abs(rad-R_B) < abs(rad-R_S) else "shaft"].append(f)
    else:
        n = f.normal_at(c)
        if abs(n.Z) > 0.9:
            groups["inletHP" if c.Z < L/2 else "outletLP"].append(f)   # z=0 is the 50 mm (oil) end
        elif abs(n.Y) > 0.9:
            groups["keywayFloor"].append(f)
        else:
            groups["keywayFlanks"].append(f)
print({k: (len(v), round(sum(f.area for f in v),3)) for k,v in groups.items()})

# -- cell 4 -------------------------------------------------------------------------
# Patch grouping is correct by area. Export each group as an STL in metres (scale 0.001 applied once, 
from build123d import Compound, export_stl
import numpy as np
SC = 0.001   # mm -> m, applied once at export
for name, fl in groups.items():
    comp = Compound(children=[f.scale(SC) for f in fl])
    ok = export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-6, angular_tolerance=0.05)
    print(name, ok, os.path.getsize(f"constant/triSurface/{name}.stl"))

# -- cell 5 -------------------------------------------------------------------------
# Rewrite as ASCII with the solid named after the patch, and verify the exported bounds are in metres.
for name, fl in groups.items():
    comp = Compound(children=[f.scale(SC) for f in fl])
    export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-6, angular_tolerance=0.05, ascii_format=True)
    txt = open(f"constant/triSurface/{name}.stl").read().replace("solid \n","solid %s\n"%name)
    lines = txt.splitlines()
    lines[0] = "solid " + name
    lines[-1] = "endsolid " + name
    open(f"constant/triSurface/{name}.stl","w").write("\n".join(lines)+"\n")
    v = np.array([[float(t) for t in l.split()[1:]] for l in lines if l.strip().startswith("vertex")])
    print(name, len(v)//3, "tris  bbox", np.round(v.min(0),5), np.round(v.max(0),5))

# -- cell 6 -------------------------------------------------------------------------
# The floor STL came out with mixed units — `Face.scale` looks in-place, so faces got scaled twice. Ch
print("fluid volume now:", fluid.volume)
f = fluid.faces()[4]
print(f.area, f.center())
try:
    vt, tr = f.tessellate(1e-3)
    print(len(vt), len(tr), vt[0])
except Exception as e:
    print("tess err", e)

# -- cell 7 -------------------------------------------------------------------------
# `fluid` is intact; the scale helper is unreliable. Write the STLs myself from each face's tessellati
SC = 0.001  # mm -> m, applied explicitly here
def write_stl(path, name, faces, tol=1e-3):
    with open(path, "w") as fh:
        fh.write("solid %s\n" % name)
        n = 0
        for f in faces:
            vt, tr = f.tessellate(tol)
            P = np.array([[v.X, v.Y, v.Z] for v in vt]) * SC
            for a, b, c in tr:
                p, q, r = P[a], P[b], P[c]
                nv = np.cross(q-p, r-p); L2 = np.linalg.norm(nv)
                if L2 == 0: continue
                nv /= L2
                fh.write("facet normal %g %g %g\n outer loop\n" % tuple(nv))
                for v in (p, q, r): fh.write("  vertex %.9g %.9g %.9g\n" % tuple(v))
                fh.write(" endloop\nendfacet\n"); n += 1
        fh.write("endsolid %s\n" % name)
    return n

for name, fl in groups.items():
    n = write_stl(f"constant/triSurface/{name}.stl", name, fl)
    v = np.array([[float(t) for t in l.split()[1:]] for l in open(f"constant/triSurface/{name}.stl") if l.strip().startswith("vertex")])
    print(name, n, "tris  bbox", np.round(v.min(0),6), np.round(v.max(0),6))

# -- cell 8 -------------------------------------------------------------------------
# The per-face tessellations are independent, so shared edges may not match — that would show up as fr
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
from OCP.BRepMesh import BRepMesh_IncrementalMesh

BRepMesh_IncrementalMesh(fluid.wrapped, 1e-3, False, 0.1, True)  # deflection 1e-3 mm on whole solid

def face_tris(face):
    loc = TopLoc_Location()
    poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
    trsf = loc.Transformation()
    P = np.array([[(p := poly.Node(i).Transformed(trsf)).X(), p.Y(), p.Z()] for i in range(1, poly.NbNodes()+1)])
    T = []
    rev = face.wrapped.Orientation() == TopAbs_REVERSED
    for i in range(1, poly.NbTriangles()+1):
        a, b, c = poly.Triangle(i).Get()
        T.append((a-1, c-1, b-1) if rev else (a-1, b-1, c-1))
    return P*SC, np.array(T)

tot = 0
for name, fl in groups.items():
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write("solid %s\n"%name); n=0
        for f in fl:
            P, T = face_tris(f)
            for a,b,c in T:
                p,q,r = P[a],P[b],P[c]
                nv = np.cross(q-p, r-p); s = np.linalg.norm(nv)
                if s == 0: continue
                nv/=s
                fh.write("facet normal %g %g %g\n outer loop\n"%tuple(nv))
                for v in (p,q,r): fh.write("  vertex %.9g %.9g %.9g\n"%tuple(v))
                fh.write(" endloop\nendfacet\n"); n+=1
        fh.write("endsolid %s\n"%name)
    tot += n; print(name, n)
print("total tris", tot)

# -- cell 9 -------------------------------------------------------------------------
# Check the union myself before trusting it: weld all patch triangles and count edges used an odd numb
import glob, collections
V=[]; F=[]
for p in sorted(glob.glob("constant/triSurface/*.stl")):
    v=np.array([[float(t) for t in l.split()[1:]] for l in open(p) if l.strip().startswith("vertex")])
    F.append(v.reshape(-1,3,3))
Vt=np.vstack([f.reshape(-1,3) for f in F])
key=np.round(Vt/1e-9).astype(np.int64)
uniq,inv=np.unique(key,axis=0,return_inverse=True)
tris=inv.reshape(-1,3)
ec=collections.Counter()
for t in tris:
    for a,b in ((0,1),(1,2),(2,0)): ec[tuple(sorted((t[a],t[b])))]+=1
print("tris",len(tris),"verts",len(uniq),"free edges",sum(1 for v in ec.values() if v!=2))
print("Euler V-E+F =", len(uniq)-len(ec)+len(tris))

# -- cell 10 ------------------------------------------------------------------------
# Re-emitting the geometry cell self-contained (`A` computed inside), and carrying straight on to the 
import gmsh, math, numpy as np
x = np.linspace(-XK if False else -KW_W/2, KW_W/2, 200001)
A = np.pi*R_B**2 - (np.pi*R_S**2 - np.trapezoid(np.sqrt(R_S**2 - x**2) - (R_S - KW_D), x))

if gmsh.isInitialized(): gmsh.clear()
else: gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("seal")
occ = gmsh.model.occ

NR, NC, NL = 4, 28, 15          # radial cells across clearance, circ. cells per ring block, axial layers
XK, YF = KW_W/2, R_S - KW_D
YS = math.sqrt(R_S**2 - XK**2)
phi1 = math.atan2(YS, XK); phi2 = math.pi - phi1
span = 2*math.pi - (phi2 - phi1)
ang = [phi1 - i*span/4 for i in range(5)]

ctr = occ.addPoint(0, 0, 0)
ps = [occ.addPoint(R_S*math.cos(a), R_S*math.sin(a), 0) for a in ang]
pb = [occ.addPoint(R_B*math.cos(a), R_B*math.sin(a), 0) for a in ang]
Sarc = [occ.addCircleArc(ps[i], ctr, ps[i+1]) for i in range(4)]
Barc = [occ.addCircleArc(pb[i], ctr, pb[i+1]) for i in range(4)]
rad  = [occ.addLine(ps[i], pb[i]) for i in range(5)]
ring = [occ.addPlaneSurface([occ.addCurveLoop([Sarc[i], rad[i+1], -Barc[i], -rad[i]])]) for i in range(4)]
pf0, pf1 = occ.addPoint(-XK, YF, 0), occ.addPoint(XK, YF, 0)
floor  = occ.addLine(pf0, pf1)
flankR = occ.addLine(pf1, ps[0])
flankL = occ.addLine(ps[4], pf0)
topArc = occ.addCircleArc(pb[4], ctr, pb[0])
pocket = occ.addPlaneSurface([occ.addCurveLoop([floor, flankR, rad[0], topArc, -rad[4], flankL])])
occ.synchronize()
print("station angles deg", [round(math.degrees(a),3) for a in ang])
print("2D area total %.9f vs analytic %.9f" % (sum(occ.getMass(2,s) for s in ring)+occ.getMass(2,pocket), A))

# -- cell 11 ------------------------------------------------------------------------
# Cross-section area matches to 12 digits. Now transfinite the ring, extrude 60 mm, and map each later
base = ring + [pocket]
for i in range(4):
    gmsh.model.mesh.setTransfiniteCurve(Sarc[i], NC+1)
    gmsh.model.mesh.setTransfiniteCurve(Barc[i], NC+1)
    gmsh.model.mesh.setTransfiniteSurface(ring[i])
for r in rad: gmsh.model.mesh.setTransfiniteCurve(r, NR+1)
for s in base: gmsh.model.mesh.setRecombine(2, s)
gmsh.model.mesh.setSize([(0, p) for p in (pf0, pf1)], 1.0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0125)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

out = occ.extrude([(2, s) for s in base], 0, 0, L, numElements=[NL], recombine=True)
occ.synchronize()
vols = [t for d, t in out if d == 3]
print("volumes", vols, "total vol %.6f vs %.6f" % (sum(occ.getMass(3,v) for v in vols), A*L))

curve_of = {}
for nm, cs in [("shaft", Sarc), ("bore", Barc+[topArc]), ("keywayFloor", [floor]), ("keywayFlanks", [flankR, flankL])]:
    for c in cs: curve_of[c] = nm
lat = {}
tops = []
for d, t in gmsh.model.getEntities(2):
    if t in base: continue
    cur = [abs(c) for _, c in gmsh.model.getBoundary([(2, t)], combined=False, oriented=False)]
    hit = {curve_of[c] for c in cur if c in curve_of}
    internal = any(c in rad for c in cur)
    if hit: lat.setdefault(hit.pop() if len(hit)==1 else "AMBIG", []).append(t)
    elif internal: pass
    else: tops.append(t)
print("lateral map", {k: len(v) for k, v in lat.items()}, "tops", tops)

# -- cell 12 ------------------------------------------------------------------------
# Volume matches the CAD solid exactly (3758.448304 mm³). Tag physical groups and mesh, scaling to met
for nm, ts in lat.items():
    gmsh.model.addPhysicalGroup(2, ts, name=nm)
gmsh.model.addPhysicalGroup(2, base, name="inletHP")     # z=0 : the 50 mm (oil-side) end
gmsh.model.addPhysicalGroup(2, tops, name="outletLP")    # z=60 mm : the 30 mm end
gmsh.model.addPhysicalGroup(3, vols, name="oil")
gmsh.option.setNumber("Mesh.ScalingFactor", 0.001)       # mm -> m at write
gmsh.model.mesh.generate(3)
gmsh.write("seal.msh")
et, ec_, _ = gmsh.model.mesh.getElements(3)
print([(int(a), len(b)) for a, b in zip(et, ec_)])   # 5=hex, 6=prism, 4=tet
print("nodes", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 13 ------------------------------------------------------------------------
# Write msh2 for `gmshToFoam`, add a minimal case skeleton, and convert.
import os, subprocess, textwrap
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("seal.msh")
os.makedirs("system", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr%"controlDict"+textwrap.dedent("""
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;
timeFormat general; runTimeModifiable true;"""))
open("system/fvSchemes","w").write(hdr%"fvSchemes"+"ddtSchemes{default Euler;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr%"fvSolution"+"solvers{}\n")
r = subprocess.run(["gmshToFoam","seal.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 14 ------------------------------------------------------------------------
# `gmshToFoam` reported 5014 undefined faces — check what actually landed in each patch.
print(open("constant/polyMesh/boundary").read()[-2000:])

# -- cell 15 ------------------------------------------------------------------------
# All 5014 went into named patches (defaultFaces emptied out). Run checkMesh.
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 16 ------------------------------------------------------------------------
# Read the OpenFOAM mesh instead.
import pyvista as pv
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays()
m = rd.read()["internalMesh"]
sl = m.slice(normal='z', origin=(0,0,0.03))
p = pv.Plotter(off_screen=True, shape=(1,2), window_size=[1400,650])
p.subplot(0,0); p.add_mesh(sl, show_edges=True, color='steelblue', line_width=0.3); p.view_xy(); p.camera.zoom(1.4)
p.subplot(0,1)
p.add_mesh(sl.clip_box([-0.008,0.008,0.0145,0.0205,-1,1], invert=False), show_edges=True, color='indianred'); p.view_xy(); p.camera.zoom(1.4)
p.screenshot("view_section.png"); print(m.n_cells)

# -- cell 17 ------------------------------------------------------------------------
from IPython.display import Image, display
print(os.path.getsize("view_section.png"))
display(Image("view_section.png"))

# -- cell 18 ------------------------------------------------------------------------
# Pocket meshed (slicing triangulates, the cells themselves are 8010 hex + 120 prism). Now zoom right 
p = pv.Plotter(off_screen=True, window_size=[1200,500])
p.add_mesh(sl.clip_box([-0.002,0.002,0.0199,0.02010,-1,1], invert=False), show_edges=True, color='seagreen')
p.view_xy(); p.camera.zoom(1.0); p.screenshot("view_gap.png")
display(Image("view_gap.png"))

# -- cell 19 ------------------------------------------------------------------------
# The clip is geometrically right but 4 mm × 0.1 mm renders as a sliver. Plot the same slice in matplo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sub = sl.clip_box([-0.0015,0.0015,0.0199,0.02010,-1,1], invert=False)
fig, ax = plt.subplots(figsize=(11,4))
P = sub.points; 
for c in range(sub.n_cells):
    ids = sub.get_cell(c).point_ids
    poly = P[list(ids)+[ids[0]]]
    ax.plot(poly[:,0]*1e3, poly[:,1]*1e3, 'k-', lw=0.6)
ax.axhline(20.0, color='b', lw=1); ax.axhline(20.05, color='r', lw=1)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title("clearance at top of bore: shaft r=20 (blue), bore r=20.05 (red)")
fig.tight_layout(); fig.savefig("view_gap.png", dpi=110); print(sub.n_cells)

# -- cell 20 ------------------------------------------------------------------------
import matplotlib.pyplot as plt
sub = sl.clip_box([-0.0015,0.0015,0.0199,0.02010,-1,1], invert=False)
fig, ax = plt.subplots(figsize=(11,4))
P = sub.points
for c in range(sub.n_cells):
    ids = list(sub.get_cell(c).point_ids)
    poly = P[ids+[ids[0]]]
    ax.plot(poly[:,0]*1e3, poly[:,1]*1e3, 'k-', lw=0.6)
ax.axhline(20.0, color='b', lw=1); ax.axhline(20.05, color='r', lw=1)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title("clearance: shaft r=20 (blue), bore r=20.05 (red)")
fig.tight_layout(); fig

# -- cell 21 ------------------------------------------------------------------------
# That window is inside the keyway pocket (the keyway is at the top). Look at the bottom of the bore, 
sub = sl.clip_box([-0.0015,0.0015,-0.02010,-0.01999,-1,1], invert=False)
fig, ax = plt.subplots(figsize=(11,4))
P = sub.points
for c in range(sub.n_cells):
    ids = list(sub.get_cell(c).point_ids); poly = P[ids+[ids[0]]]
    ax.plot(poly[:,0]*1e3, poly[:,1]*1e3, 'k-', lw=0.6)
ax.axhline(-20.0, color='b', lw=1); ax.axhline(-20.05, color='r', lw=1)
ax.set_title("radial clearance, bottom: %d cells in window" % sub.n_cells); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
fig.tight_layout(); fig

# -- cell 22 ------------------------------------------------------------------------
# 4 cells across the gap, bounded by r=20 and r=20.05 as required — but at 1 mm chords the facet sag i
NR, NC, NL = 6, 48, 30          # refined: radial cells across clearance, circ. per ring block, axial layers
gmsh.clear(); gmsh.model.add("seal2"); occ = gmsh.model.occ
ctr = occ.addPoint(0,0,0)
ps = [occ.addPoint(R_S*math.cos(a), R_S*math.sin(a), 0) for a in ang]
pb = [occ.addPoint(R_B*math.cos(a), R_B*math.sin(a), 0) for a in ang]
Sarc = [occ.addCircleArc(ps[i], ctr, ps[i+1]) for i in range(4)]
Barc = [occ.addCircleArc(pb[i], ctr, pb[i+1]) for i in range(4)]
rad  = [occ.addLine(ps[i], pb[i]) for i in range(5)]
ring = [occ.addPlaneSurface([occ.addCurveLoop([Sarc[i], rad[i+1], -Barc[i], -rad[i]])]) for i in range(4)]
pf0, pf1 = occ.addPoint(-XK, YF, 0), occ.addPoint(XK, YF, 0)
floor  = occ.addLine(pf0, pf1); flankR = occ.addLine(pf1, ps[0]); flankL = occ.addLine(ps[4], pf0)
topArc = occ.addCircleArc(pb[4], ctr, pb[0])
pocket = occ.addPlaneSurface([occ.addCurveLoop([floor, flankR, rad[0], topArc, -rad[4], flankL])])
occ.synchronize()
base = ring + [pocket]
for i in range(4):
    gmsh.model.mesh.setTransfiniteCurve(Sarc[i], NC+1); gmsh.model.mesh.setTransfiniteCurve(Barc[i], NC+1)
    gmsh.model.mesh.setTransfiniteSurface(ring[i])
for r_ in rad: gmsh.model.mesh.setTransfiniteCurve(r_, NR+1)
for s in base: gmsh.model.mesh.setRecombine(2, s)
gmsh.model.mesh.setSize([(0,p) for p in (pf0,pf1)], 0.6)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.008); gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
out = occ.extrude([(2,s) for s in base], 0, 0, L, numElements=[NL], recombine=True)
occ.synchronize()
vols = [t for d,t in out if d==3]
print("volume %.6f vs %.6f mm3" % (sum(occ.getMass(3,v) for v in vols), A*L))

# -- cell 23 ------------------------------------------------------------------------
curve_of = {}
for nm, cs in [("shaft", Sarc), ("bore", Barc+[topArc]), ("keywayFloor", [floor]), ("keywayFlanks", [flankR, flankL])]:
    for c in cs: curve_of[c] = nm
lat = {}; tops = []
for d, t in gmsh.model.getEntities(2):
    if t in base: continue
    cur = [abs(c) for _, c in gmsh.model.getBoundary([(2,t)], combined=False, oriented=False)]
    hit = {curve_of[c] for c in cur if c in curve_of}
    if len(hit) == 1: lat.setdefault(hit.pop(), []).append(t)
    elif hit: raise RuntimeError("ambiguous lateral %d" % t)
    elif not any(c in rad for c in cur): tops.append(t)
for nm, ts in lat.items(): gmsh.model.addPhysicalGroup(2, ts, name=nm)
gmsh.model.addPhysicalGroup(2, base, name="inletHP")    # z=0, the 50 mm (high-pressure oil) end
gmsh.model.addPhysicalGroup(2, tops, name="outletLP")   # z=60 mm, the 30 mm end
gmsh.model.addPhysicalGroup(3, vols, name="oil")
gmsh.option.setNumber("Mesh.ScalingFactor", 0.001)      # mm -> m at write
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3); gmsh.write("seal.msh")
print({k: len(v) for k,v in lat.items()}, "tops", len(tops))
print([(int(a), len(b)) for a,b in zip(*gmsh.model.mesh.getElements(3)[:2])])

# -- cell 24 ------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam","seal.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:])
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):][:900])
print(r.stdout[r.stdout.find("Checking geometry"):][:1400])

# -- cell 25 ------------------------------------------------------------------------
# Mesh OK at 41250 cells, volume within 0.03 % of the CAD. Now measure the requested dimensions off th
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays(); d = rd.read()
bl = d["boundary"]
print(list(bl.keys()))
def rad(pts): return np.hypot(pts[:,0], pts[:,1])*1e3
for k in bl.keys():
    P = bl[k].points
    print("%-13s faces=%6d  r[mm] %.4f..%.4f  y %.4f..%.4f  x %.4f..%.4f  z %.4f..%.4f"
          % (k, bl[k].n_cells, rad(P).min(), rad(P).max(), P[:,1].min()*1e3, P[:,1].max()*1e3,
             P[:,0].min()*1e3, P[:,0].max()*1e3, P[:,2].min()*1e3, P[:,2].max()*1e3))
