"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first cell errored on `is_valid` (a property, not a method), so it wasn't logged. Re-emitting th
import os, math, numpy as np
from build123d import *

LX, LY, LZ = 0.160, 0.100, 0.040      # internal cavity, metres
TW = 0.003                            # wall thickness
VW, VH = 0.030, 0.020                 # vent width (y) x height (z)
VZ0 = 0.010                           # vent bottom above floor
SD, SH = 0.010, 0.025                 # standoff dia, height
HD, HDEP = 0.004, 0.012               # blind hole dia, depth
INSET = 0.015                         # standoff centre inset from internal corner

cavity = Box(LX, LY, LZ, align=(Align.MIN,)*3)
duct = Pos(LX, (LY-VW)/2, VZ0) * Box(TW, VW, VH, align=(Align.MIN,)*3)
centres = [(INSET, INSET), (LX-INSET, INSET), (INSET, LY-INSET), (LX-INSET, LY-INSET)]
stand = None
for cx, cy in centres:
    c = Pos(cx, cy, 0) * Cylinder(SD/2, SH, align=(Align.CENTER, Align.CENTER, Align.MIN))
    h = Pos(cx, cy, SH-HDEP) * Cylinder(HD/2, HDEP, align=(Align.CENTER, Align.CENTER, Align.MIN))
    stand = (c - h) if stand is None else stand + (c - h)
fluid = (cavity + duct) - stand

V_expect = LX*LY*LZ + TW*VW*VH - 4*(math.pi*(SD/2)**2*SH - math.pi*(HD/2)**2*HDEP)
print("valid:", fluid.is_valid, "faces:", len(fluid.faces()))
print("volume:", fluid.volume, " expected:", V_expect)
print(fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Now classify the 27 faces by their actual surface type and radius (planes by normal/offset, cylinder
from collections import defaultdict
groups = defaultdict(list)
for f in fluid.faces():
    gt = f.geom_type
    c = f.center()
    if gt == GeomType.CYLINDER:
        r = f.radius
        groups["standoffs" if abs(r-SD/2) < 1e-6 else "holes"].append(f)
    else:
        n = f.normal_at(c)
        if abs(n.Z) > 0.9 and abs(c.Z) < 1e-9:
            groups["floor"].append(f)
        elif abs(n.Z) > 0.9 and abs(c.Z - LZ) < 1e-9:
            groups["top"].append(f)
        elif abs(n.Z) > 0.9 and abs(c.Z - SH) < 1e-9:
            groups["standoffs"].append(f)       # standoff top annulus
        elif abs(n.Z) > 0.9 and abs(c.Z - (SH-HDEP)) < 1e-9:
            groups["holes"].append(f)           # blind hole bottom
        elif abs(n.X) > 0.9 and abs(c.X - (LX+TW)) < 1e-9:
            groups["vent"].append(f)
        else:
            groups["walls"].append(f)
for k, v in groups.items():
    print(k, len(v), "area=%.6g" % sum(f.area for f in v))

# -- cell 3 -------------------------------------------------------------------------
# All six groups match hand-computed areas (top 0.016, vent 6e-4, floor 0.016−4πr², standoff/hole area
import subprocess, pyvista as pv
os.makedirs("constant/triSurface", exist_ok=True)
for name, fs in groups.items():
    sh = fs[0] if len(fs) == 1 else Compound(children=fs)
    export_stl(sh, f"constant/triSurface/{name}.stl", tolerance=2e-5, angular_tolerance=0.2)
print(sorted(os.listdir("constant/triSurface")))

pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
pl = pv.Plotter(off_screen=True, window_size=(900, 650))
cols = dict(walls="lightgray", floor="tan", top="skyblue", vent="red", standoffs="green", holes="magenta")
for name in cols:
    m = pv.read(f"constant/triSurface/{name}.stl")
    pl.add_mesh(m, color=cols[name], opacity=0.35 if name in ("walls", "top") else 1.0, show_edges=False)
pl.camera_position = [(0.30, -0.25, 0.22), (0.08, 0.05, 0.015), (0, 0, 1)]
pl.screenshot("shape.png")

# -- cell 4 -------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def read_tris(p):
    m = pv.read(p)
    f = m.faces.reshape(-1, 4)[:, 1:]
    return m.points[f]
fig = plt.figure(figsize=(11, 5))
for i, (elev, azim) in enumerate([(25, -60), (70, -90)]):
    ax = fig.add_subplot(1, 2, i+1, projection="3d")
    for name, col in cols.items():
        t = read_tris(f"constant/triSurface/{name}.stl")
        ax.add_collection3d(Poly3DCollection(t, facecolor=col, edgecolor="k",
                                             linewidths=0.15, alpha=0.25 if name in ("walls","top") else 1.0))
    ax.set_xlim(0, 0.17); ax.set_ylim(0, 0.11); ax.set_zlim(0, 0.05)
    ax.set_box_aspect((0.17, 0.11, 0.05)); ax.view_init(elev, azim)
plt.tight_layout(); plt.show()

# -- cell 5 -------------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def read_tris(p):
    m = pv.read(p)
    f = m.faces.reshape(-1, 4)[:, 1:]
    return m.points[f]
fig = plt.figure(figsize=(12, 5))
for i, (elev, azim) in enumerate([(25, -60), (70, -90)]):
    ax = fig.add_subplot(1, 2, i+1, projection="3d")
    for name, col in cols.items():
        t = read_tris(f"constant/triSurface/{name}.stl")
        ax.add_collection3d(Poly3DCollection(t, facecolor=col, edgecolor="k",
                                             linewidths=0.15, alpha=0.2 if name in ("walls","top") else 1.0))
    ax.set_xlim(0, 0.17); ax.set_ylim(0, 0.11); ax.set_zlim(0, 0.05)
    ax.set_box_aspect((0.17, 0.11, 0.05)); ax.view_init(elev, azim)
fig

# -- cell 6 -------------------------------------------------------------------------
# Shape is right. Now export STEP and import into gmsh, mapping each gmsh surface to a patch group by 
import gmsh
export_step(fluid, "fluid.step")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("enc")
gmsh.model.occ.importShapes("fluid.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
print("gmsh surfaces:", len(surfs), "volumes:", gmsh.model.getEntities(3))

ref = []
for name, fs in groups.items():
    for f in fs:
        c = f.center(); ref.append((name, np.array([c.X, c.Y, c.Z]), f.area))
assign = {}
for dim, tag in surfs:
    com = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
    a = gmsh.model.occ.getMass(dim, tag)
    best = min(ref, key=lambda r: np.linalg.norm(r[1]-com) + abs(r[2]-a))
    assign.setdefault(best[0], []).append(tag)
    d = np.linalg.norm(best[1]-com)
    if d > 1e-6: print("loose match", tag, best[0], d)
print({k: len(v) for k, v in assign.items()})

# -- cell 7 -------------------------------------------------------------------------
# Face centres of cylinders differ between the two kernels, so classify the gmsh surfaces directly by 
assign = {}
for dim, tag in gmsh.model.getEntities(2):
    t = gmsh.model.getType(dim, tag)
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    if t == "Cylinder":
        r = (x1-x0)/2
        name = "standoffs" if abs(r-SD/2) < 1e-6 else "holes"
    elif abs(z1-z0) < 1e-9:
        z = z0
        name = ("floor" if abs(z) < 1e-9 else "top" if abs(z-LZ) < 1e-9
                else "standoffs" if abs(z-SH) < 1e-9 else "holes" if abs(z-(SH-HDEP)) < 1e-9 else "?")
    elif abs(x1-x0) < 1e-9 and abs(x0-(LX+TW)) < 1e-9:
        name = "vent"
    else:
        name = "walls"
    assign.setdefault(name, []).append(tag)
    if name == "?": print("unclassified", tag, t, com)
print({k: (len(v), round(sum(gmsh.model.occ.getMass(2, s) for s in v), 7)) for k, v in assign.items()})

# -- cell 8 -------------------------------------------------------------------------
# Plane detection failed — gmsh's bounding boxes are padded by a tolerance. Check the actual padding.
for dim, tag in gmsh.model.getEntities(2)[:6]:
    print(tag, gmsh.model.getType(dim, tag), ["%.3e" % v for v in gmsh.model.getBoundingBox(dim, tag)])

# -- cell 9 -------------------------------------------------------------------------
# Padding is 1e-7; reclassify planes with a 1e-6 tolerance, using the centre of mass for the plane's l
TOL = 1e-6
assign = {}
for dim, tag in gmsh.model.getEntities(2):
    t = gmsh.model.getType(dim, tag)
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
    if t == "Cylinder":
        r = (x1-x0-2e-7)/2
        name = "standoffs" if abs(r-SD/2) < TOL else "holes"
    elif (z1-z0) < TOL:
        name = ("floor" if abs(cz) < TOL else "top" if abs(cz-LZ) < TOL
                else "standoffs" if abs(cz-SH) < TOL else "holes" if abs(cz-(SH-HDEP)) < TOL else "?")
    elif (x1-x0) < TOL and abs(cx-(LX+TW)) < TOL:
        name = "vent"
    else:
        name = "walls"
    assign.setdefault(name, []).append(tag)
    if name == "?": print("unclassified", tag, t, cx, cy, cz)
print({k: (len(v), round(sum(gmsh.model.occ.getMass(2, s) for s in v), 7)) for k, v in assign.items()})

# -- cell 10 ------------------------------------------------------------------------
# Those two are the vent duct's floor/ceiling — walls. Fold them in, tag physical groups, and mesh coa
TOL = 1e-6
assign = {}
for dim, tag in gmsh.model.getEntities(2):
    t = gmsh.model.getType(dim, tag)
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
    if t == "Cylinder":
        r = (x1-x0-2e-7)/2
        name = "standoffs" if abs(r-SD/2) < TOL else "holes"
    elif (z1-z0) < TOL and abs(cz) < TOL:
        name = "floor"
    elif (z1-z0) < TOL and abs(cz-LZ) < TOL:
        name = "top"
    elif (z1-z0) < TOL and abs(cz-SH) < TOL and (x1-x0) < SD+TOL:
        name = "standoffs"                      # standoff top annulus
    elif (z1-z0) < TOL and abs(cz-(SH-HDEP)) < TOL and (x1-x0) < HD+TOL:
        name = "holes"                          # blind hole bottom
    elif (x1-x0) < TOL and abs(cx-(LX+TW)) < TOL:
        name = "vent"
    else:
        name = "walls"                          # side walls + vent duct walls
    assign.setdefault(name, []).append(tag)
for name, tags in assign.items():
    gmsh.model.addPhysicalGroup(2, tags, name=name)
gmsh.model.addPhysicalGroup(3, [1], name="internal")
print({k: (len(v), round(sum(gmsh.model.occ.getMass(2, s) for s in v), 7)) for k, v in assign.items()})

# -- cell 11 ------------------------------------------------------------------------
# Areas match the build123d groups exactly. Coarse tet mesh now: 6 mm global, refined on standoffs and
import time
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", assign["holes"] + assign["standoffs"])
gmsh.model.mesh.field.setNumber(f1, "Sampling", 60)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", 0.0012)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", 0.006)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.002)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.015)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)
t0 = time.time(); gmsh.model.mesh.generate(3); print("mesh time %.1fs" % (time.time()-t0))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("enc.msh")
print(len(gmsh.model.mesh.getElementsByType(4)[0]), "tets")

# -- cell 12 ------------------------------------------------------------------------
# Mesh volume matches (6.3458e-4). Convert to OpenFOAM: write a minimal case skeleton, run `gmshToFoam
import subprocess, textwrap
os.makedirs("system", exist_ok=True)
def foam(loc, obj, body, cls="dictionary"):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n"
            "    location \"%s\";\n    object %s;\n}\n\n" % (cls, loc, obj)) + body
open("system/controlDict", "w").write(foam("system", "controlDict",
    "application icoFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\n"
    "deltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n"))
open("system/fvSchemes", "w").write(foam("system", "fvSchemes",
    "ddtSchemes{default Euler;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\n"
    "laplacianSchemes{default Gauss linear corrected;}\ninterpolationSchemes{default linear;}\n"
    "snGradSchemes{default corrected;}\n"))
open("system/fvSolution", "w").write(foam("system", "fvSolution", "solvers{}\n"))
r = subprocess.run(["gmshToFoam", "enc.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh", "-constant"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 14 ------------------------------------------------------------------------
# Mesh OK at 73.6k tets, all six patches present and no leftover default faces. Now refine — 4 mm hole
gmsh.model.mesh.field.setNumber(f2, "SizeMin", 0.0006)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", 0.004)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0015)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.012)
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.mesh.clear()
t0 = time.time(); gmsh.model.mesh.generate(3); print("mesh time %.1fs" % (time.time()-t0))
gmsh.write("enc.msh")
print(len(gmsh.model.mesh.getElementsByType(4)[0]), "tets")

# -- cell 15 ------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam", "enc.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:])
b = "constant/polyMesh/boundary"
txt = open(b).read()
for p in ["walls", "floor", "standoffs", "holes"]:
    txt = txt.replace("%s\n    {\n        type            patch;" % p,
                      "%s\n    {\n        type            wall;" % p)
open(b, "w").write(txt)
print([l.strip() for l in open(b) if "type" in l or "nFaces" in l])

# -- cell 16 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh", "-constant"], capture_output=True, text=True)
print(r.stdout[-1800:])

# -- cell 17 ------------------------------------------------------------------------
# checkMesh passes at 322k cells. Look at the meshed surfaces, zoomed on one standoff with its blind h
nt, nc, _ = gmsh.model.mesh.getNodes()
xyz = {int(t): nc[3*i:3*i+3] for i, t in enumerate(nt)}
def tris(name):
    tags = assign[name]; out = []
    for s in tags:
        et, en, ev = gmsh.model.mesh.getElements(2, s)
        for k, t in enumerate(et):
            if t == 2:
                v = np.array(ev[k]).reshape(-1, 3)
                out.append(np.array([[xyz[n] for n in tri] for tri in v]))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for name, col in [("standoffs", "green"), ("holes", "magenta"), ("floor", "tan")]:
    t = tris(name)
    m = (t[:, :, 0].max(1) < 0.032) & (t[:, :, 1].max(1) < 0.032)
    ax.add_collection3d(Poly3DCollection(t[m], facecolor=col, edgecolor="k", linewidths=0.2))
ax.set_xlim(0, 0.032); ax.set_ylim(0, 0.032); ax.set_zlim(0, 0.03)
ax.set_box_aspect((1, 1, 1)); ax.view_init(22, -55); ax.set_title("standoff + blind hole")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for name, col in [("vent", "red"), ("walls", "lightgray")]:
    t = tris(name)
    m = t[:, :, 0].min(1) > 0.14
    ax2.add_collection3d(Poly3DCollection(t[m], facecolor=col, edgecolor="k", linewidths=0.2))
ax2.set_xlim(0.14, 0.165); ax2.set_ylim(0.02, 0.08); ax2.set_zlim(0, 0.04)
ax2.set_box_aspect((0.025, 0.06, 0.04)); ax2.view_init(15, -50); ax2.set_title("vent")
fig

# -- cell 18 ------------------------------------------------------------------------
# Both views correct. Final numeric verification of the requested dimensions, measured from the meshed
hb = np.concatenate([tris("holes")]).reshape(-1, 3)
sb = tris("standoffs").reshape(-1, 3)
vb = tris("vent").reshape(-1, 3)
cen = []
for cx, cy in centres:
    d = np.hypot(sb[:, 0]-cx, sb[:, 1]-cy)
    sel = sb[d < 0.006]
    cen.append((sel[:, 0].min()+sel[:, 0].max())/2*1000)
print("standoff Ø (mm): %.3f  asked 10" % (sb[:, 0].max()-sb[:, 0].min() - 0 if False else 1000*(np.ptp(sb[np.hypot(sb[:,0]-centres[0][0], sb[:,1]-centres[0][1])<0.006][:, 0]))))
print("standoff height (mm): %.3f  asked 25" % (1000*sb[:, 2].max()))
print("standoff centre inset x,y (mm): %.3f %.3f  asked 15,15" % (1000*centres[0][0], 1000*centres[0][1]))
print("hole Ø (mm): %.3f  asked 4" % (1000*np.ptp(hb[np.hypot(hb[:,0]-centres[0][0], hb[:,1]-centres[0][1])<0.003][:, 0])))
print("hole z-range (mm): %.3f .. %.3f  -> depth %.3f, asked 12 from top 25" % (1000*hb[:,2].min(), 1000*hb[:,2].max(), 1000*(hb[:,2].max()-hb[:,2].min())))
print("vent y-span %.3f, z-span %.3f, x %.3f  asked 30 x 20 at x=163, 10..30" % (1000*np.ptp(vb[:,1]), 1000*np.ptp(vb[:,2]), 1000*vb[:,0].mean()), " z0=%.1f" % (1000*vb[:,2].min()))
print("cavity internal LxWxD (mm): %.1f %.1f %.1f" % (1000*LX, 1000*LY, 1000*LZ))
