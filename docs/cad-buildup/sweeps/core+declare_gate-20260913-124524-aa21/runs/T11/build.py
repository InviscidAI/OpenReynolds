"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

W = 80.0          # duct side, mm
TCORE = 20.0      # core thickness (flow dir), mm
AF = 8.0          # cell width across flats, mm
TW = 1.0          # wall thickness, mm
PITCH = AF + TW   # centre-to-centre across flats, mm
RP = PITCH/np.sqrt(3.0)   # lattice circumradius
RC = AF/np.sqrt(3.0)      # cell circumradius (vertex at angle 0)

ni = int(np.ceil(W/(1.5*RP))) + 2
nj = int(np.ceil(W/PITCH)) + 2
cent = []
for i in range(-ni, ni+1):
    for j in range(-nj, nj+1):
        x = 1.5*RP*i
        y = PITCH*j + (PITCH/2 if i % 2 else 0.0)
        if abs(x) < W/2 + RC and abs(y) < W/2 + RC:
            cent.append((x, y))
cent = np.array(cent)
print("cells kept:", len(cent), " RC=%.4f RP=%.4f"%(RC, RP))

fig, ax = plt.subplots(figsize=(6,6))
for x,y in cent:
    ax.add_patch(RegularPolygon((x,y), 6, radius=RC, orientation=0, fill=False, ec='b', lw=0.6))
ax.add_patch(plt.Rectangle((-W/2,-W/2), W, W, fill=False, ec='r', lw=1.5))
ax.set_xlim(-50,50); ax.set_ylim(-50,50); ax.set_aspect('equal')
plt.savefig("lattice.png", dpi=110)
print("ok")

# -- cell 2 -------------------------------------------------------------------------
from IPython.display import display
display(fig)

# -- cell 3 -------------------------------------------------------------------------
# Matplotlib's `RegularPolygon(orientation=0)` puts a vertex at +y, but my lattice assumes a vertex at
def hexpts(cx, cy, r):
    a = np.arange(6)*np.pi/3.0
    return np.stack([cx + r*np.cos(a), cy + r*np.sin(a)], axis=1)

def clip_halfplane(poly, keep):           # keep(p)->bool, edge crossing by linear interp
    out = []
    n = len(poly)
    for i in range(n):
        A, B = poly[i], poly[(i+1) % n]
        ka, kb = keep(A), keep(B)
        if ka: out.append(A)
        if ka != kb:
            fa, fb = keep.f(A), keep.f(B)
            t = fa/(fa-fb)
            out.append(A + t*(B-A))
    return np.array(out) if out else np.zeros((0,2))

class HP:                                  # f>=0 kept
    def __init__(s, f): s.f = f
    def __call__(s, p): return s.f(p) >= 0

def clip_square(poly, h):
    for f in (lambda p: h-p[0], lambda p: h+p[0], lambda p: h-p[1], lambda p: h+p[1]):
        if len(poly) == 0: return poly
        poly = clip_halfplane(poly, HP(f))
    return poly

def area(p):
    if len(p) < 3: return 0.0
    x, y = p[:,0], p[:,1]
    return 0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

AFULL = 1.5*np.sqrt(3)*RC**2
clipped = [(c, area(clip_square(hexpts(c[0], c[1], RC), W/2))) for c in cent]
fr = np.array([a/AFULL for _, a in clipped])
print("full cell area %.3f mm2; fractions sorted:"%AFULL, np.round(np.sort(fr)[:12],3))
KEEP_FRAC = 0.25
kept = np.array([c for (c,a) in clipped if a/AFULL >= KEEP_FRAC])
open_area = sum(a for (c,a) in clipped if a/AFULL >= KEEP_FRAC)
print("kept", len(kept), "of", len(cent), " open area %.2f mm2 = %.1f%% of %.0f"%(open_area, 100*open_area/W**2, W**2))

# -- cell 4 -------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(6,6))
for c in kept:
    p = clip_square(hexpts(c[0], c[1], RC), W/2)
    ax.add_patch(plt.Polygon(p, closed=True, fc='skyblue', ec='b', lw=0.5))
ax.add_patch(plt.Rectangle((-W/2,-W/2), W, W, fill=False, ec='r', lw=1.5))
ax.set_xlim(-45,45); ax.set_ylim(-45,45); ax.set_aspect('equal'); ax.set_title("honeycomb open area (fluid) in 80x80 duct")
fig2

# -- cell 5 -------------------------------------------------------------------------
from build123d import *
hx = RegularPolygon(RC, 6, major_radius=True)
print([tuple(np.round(v.to_tuple(),3)) for v in hx.vertices()])
print("bbox", hx.bounding_box())

# -- cell 6 -------------------------------------------------------------------------
# Volume matches the analytic value to 2e-12 %. Re-running the cell with `is_valid` as a property so i
import time
ZIN, ZOUT = -40.0, 100.0          # duct extent along flow (core occupies 0..TCORE)
L = ZOUT - ZIN
duct = Pos(0, 0, (ZIN+ZOUT)/2) * Box(W, W, L)
slab = Pos(0, 0, TCORE/2) * Box(W, W, TCORE)
prisms = [Pos(float(c[0]), float(c[1]), TCORE/2) *
          extrude(RegularPolygon(RC, 6, major_radius=True), amount=TCORE/2+2, both=True)
          for c in kept]
t = time.time(); walls = slab.cut(*prisms); print("cut walls %.1fs"%(time.time()-t))
t = time.time(); fluid = duct.cut(walls);   print("cut fluid %.1fs"%(time.time()-t))
Vexp = W*W*L - TCORE*(W*W - open_area)
print("fluid volume %.1f mm3 vs analytic %.1f  (diff %.3g%%)"%(fluid.volume, Vexp, 100*(fluid.volume-Vexp)/Vexp))
print("solids:", len(fluid.solids()), " valid:", fluid.is_valid, " faces:", len(fluid.faces()))
print("bbox", fluid.bounding_box())

# -- cell 7 -------------------------------------------------------------------------
# One valid solid, exact volume. Now classify the 556 faces into the four named groups using each face
TOL = 1e-6
def grp(f):
    n = f.normal_at(f.center()); c = f.center()
    if abs(abs(n.Z)-1) < 1e-9:
        if abs(c.Z-ZIN)  < TOL: return "inlet"
        if abs(c.Z-ZOUT) < TOL: return "outlet"
        return "honeycomb"
    if abs(abs(n.X)-1) < 1e-9 and abs(abs(c.X)-W/2) < TOL: return "ductWalls"
    if abs(abs(n.Y)-1) < 1e-9 and abs(abs(c.Y)-W/2) < TOL: return "ductWalls"
    return "honeycomb"

groups = {}
for f in fluid.faces(): groups.setdefault(grp(f), []).append(f)
areas = {k: sum(f.area for f in v) for k, v in groups.items()}

# analytic: boundary-coincident edge length of each clipped cell -> wetted duct perimeter in core
def on_sq(p): return min(abs(abs(p[0])-W/2), abs(abs(p[1])-W/2)) < 1e-9
open_per, cut_per = 0.0, 0.0
for c in kept:
    p = clip_square(hexpts(c[0], c[1], RC), W/2)
    for i in range(len(p)):
        A, B = p[i], p[(i+1) % len(p)]; Ln = np.hypot(*(B-A))
        if on_sq(A) and on_sq(B) and on_sq(0.5*(A+B)): open_per += Ln
        else: cut_per += Ln
ana = {"inlet": W*W, "outlet": W*W,
       "ductWalls": 4*W*(L-TCORE) + open_per*TCORE,
       "honeycomb": cut_per*TCORE + 2*(W*W-open_area)}
for k in ("inlet","outlet","ductWalls","honeycomb"):
    print("%-10s %3d faces  area %10.2f  analytic %10.2f  diff %.2g%%"
          % (k, len(groups[k]), areas[k], ana[k], 100*(areas[k]-ana[k])/ana[k]))

# -- cell 8 -------------------------------------------------------------------------
# All four patch areas match analytic to machine precision. Now scale to metres (OpenFOAM reads number
SC = 0.001                      # mm -> m, applied once, here
fluid_m = scale(fluid, SC)
Wm, ZINm, ZOUTm = W*SC, ZIN*SC, ZOUT*SC
TOLm = 1e-9
def grp_m(f):
    n = f.normal_at(f.center()); c = f.center()
    if abs(abs(n.Z)-1) < 1e-9:
        if abs(c.Z-ZINm)  < TOLm: return "inlet"
        if abs(c.Z-ZOUTm) < TOLm: return "outlet"
        return "honeycombWalls"
    if abs(abs(n.X)-1) < 1e-9 and abs(abs(c.X)-Wm/2) < TOLm: return "ductWalls"
    if abs(abs(n.Y)-1) < 1e-9 and abs(abs(c.Y)-Wm/2) < TOLm: return "ductWalls"
    return "honeycombWalls"

gm = {}
for f in fluid_m.faces(): gm.setdefault(grp_m(f), []).append(f)
print({k: (len(v), round(sum(x.area for x in v), 6)) for k, v in gm.items()})
print("volume m3 %.6g  (expect %.6g)" % (fluid_m.volume, Vexp*SC**3))
export_step(fluid_m, "fluid.step")
for k, v in gm.items(): export_stl(Compound(children=v), k + ".stl")
import os; print(sorted(os.listdir(".")))

# -- cell 9 -------------------------------------------------------------------------
# STEP and per-patch STLs are out. Now gmsh: import the metre STEP, classify each surface by its own d
import gmsh
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("hc")
gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3); surfs = gmsh.model.getEntities(2)
print("volumes", vols, "surfaces", len(surfs))
def gclass(tag):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2, tag); e = 1e-9
    if z1-z0 < e:
        if abs(z0-ZINm)  < 1e-7: return "inlet"
        if abs(z0-ZOUTm) < 1e-7: return "outlet"
        return "honeycombWalls"
    if x1-x0 < e and abs(abs(x0)-Wm/2) < 1e-7: return "ductWalls"
    if y1-y0 < e and abs(abs(y0)-Wm/2) < 1e-7: return "ductWalls"
    return "honeycombWalls"
pg = {}
for d,t in surfs: pg.setdefault(gclass(t), []).append(t)
print({k: len(v) for k,v in pg.items()})

# -- cell 10 ------------------------------------------------------------------------
for d,t in surfs[:6]:
    print(t, np.round(gmsh.model.getBoundingBox(2,t), 9), np.round(gmsh.model.occ.getCenterOfMass(2,t),6))

# -- cell 11 ------------------------------------------------------------------------
# Bounding boxes come back padded by 1e-7; widening the degeneracy tolerance accordingly.
def gclass(tag):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2, tag); e = 1e-6
    if z1-z0 < e:
        if abs(z0-ZINm)  < e: return "inlet"
        if abs(z0-ZOUTm) < e: return "outlet"
        return "honeycombWalls"
    if x1-x0 < e and abs(abs(x0)-Wm/2) < e: return "ductWalls"
    if y1-y0 < e and abs(abs(y0)-Wm/2) < e: return "ductWalls"
    return "honeycombWalls"
pg = {}
for d,t in surfs: pg.setdefault(gclass(t), []).append(t)
print({k: len(v) for k,v in pg.items()})
assert {k: len(v) for k,v in pg.items()} == {"inlet":1, "outlet":1, "ductWalls":4, "honeycombWalls":550}
for k,v in pg.items(): gmsh.model.addPhysicalGroup(2, v, name=k)
gmsh.model.addPhysicalGroup(3, [1], name="internal")
print("physical groups", [(d,t,gmsh.model.getPhysicalName(d,t)) for d,t in gmsh.model.getPhysicalGroups()])

# -- cell 12 ------------------------------------------------------------------------
# Meshing now at 4 mm, coarsest version, straight to disk.
import subprocess, time
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.002)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.004)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
t=time.time(); gmsh.model.mesh.generate(3); print("mesh %.1fs"%(time.time()-t))
gmsh.write("fluid.msh")
print(subprocess.run(["bash","-c","ls -la fluid.msh"],capture_output=True,text=True).stdout)

# -- cell 13 ------------------------------------------------------------------------
# Converting to OpenFOAM: minimal case files, msh2 format, `gmshToFoam`, then `checkMesh`.
import os, subprocess, textwrap
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.write("fluid.msh")
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr%"controlDict"+textwrap.dedent("""
    application simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
    endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
    """))
open("system/fvSchemes","w").write(hdr%"fvSchemes"+"ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr%"fvSolution"+"solvers{}\n")
r = subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 14 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):][:3000])

# -- cell 15 ------------------------------------------------------------------------
import pyvista as pv
pv.global_theme.background = 'white'
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays()
blocks = rd.read(); internal = blocks["internalMesh"]
clip = internal.clip_box((-0.041,0.041,-0.041,0.041,-0.041,0.012), invert=True)
p = pv.Plotter(off_screen=True, window_size=(900,700))
p.add_mesh(clip, show_edges=True, edge_color='gray', color='lightblue', line_width=0.3)
p.camera_position = [(0.22,-0.20,0.16),(0,0,0.02),(0,0,1)]
p.show(screenshot="coarse.png")
from IPython.display import Image; Image("coarse.png")

# -- cell 16 ------------------------------------------------------------------------
# The clip kept the downstream duct (notches at its base are the honeycomb exits). Better check: rende
bnd = blocks["boundary"]
print(list(bnd.keys()))
p = pv.Plotter(off_screen=True, window_size=(900,700))
p.add_mesh(bnd["honeycombWalls"], color='orange', show_edges=True, edge_color='gray', line_width=0.3)
p.add_mesh(bnd["ductWalls"], color='lightsteelblue', opacity=0.25)
p.camera_position = [(0.14,-0.16,0.13),(0,0,0.01),(0,0,1)]
p.show(screenshot="hc.png")
from IPython.display import Image; Image("hc.png")

# -- cell 17 ------------------------------------------------------------------------
# Honeycomb is correct. Now refine: 1.8 mm in the core, 4 mm far field, via a box size field.
gmsh.model.mesh.clear()
f = gmsh.model.mesh.field
f.add("Box", 1)
f.setNumber(1,"VIn",0.0018); f.setNumber(1,"VOut",0.004); f.setNumber(1,"Thickness",0.020)
f.setNumber(1,"XMin",-Wm/2); f.setNumber(1,"XMax",Wm/2)
f.setNumber(1,"YMin",-Wm/2); f.setNumber(1,"YMax",Wm/2)
f.setNumber(1,"ZMin",-0.002); f.setNumber(1,"ZMax",TCORE*SC+0.002)
f.setAsBackgroundMesh(1)
for o in ("Mesh.MeshSizeExtendFromBoundary","Mesh.MeshSizeFromPoints","Mesh.MeshSizeFromCurvature"):
    gmsh.option.setNumber(o, 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0005); gmsh.option.setNumber("Mesh.MeshSizeMax", 0.004)
t=time.time(); gmsh.model.mesh.generate(3); print("mesh %.1f s"%(time.time()-t))
et = gmsh.model.mesh.getElements(3)[1][0]; print("tets:", len(et))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.write("fluid.msh")

# -- cell 18 ------------------------------------------------------------------------
subprocess.run(["bash","-c","rm -rf constant/polyMesh 0"],check=True)
r = subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True); print(r.returncode)
c = subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print(c[c.find("Mesh stats"):][:1200])
print(c[c.find("Checking geometry"):][:2200])

# -- cell 19 ------------------------------------------------------------------------
# Mesh OK at 237 k tets. Final numeric check: patch areas measured on the OpenFOAM mesh itself against
rd2 = pv.OpenFOAMReader("case.foam"); rd2.enable_all_patch_arrays()
bb = rd2.read()["boundary"]
for k in ["inlet","outlet","ductWalls","honeycombWalls"]:
    print("%-15s mesh area %10.6f m2   CAD %10.6f m2" % (k, bb[k].area, sum(x.area for x in gm[k])))
print("\nrequested vs built:")
print("  duct section      80 x 80 mm   ->  %.1f x %.1f mm" % (Wm*1000, Wm*1000))
print("  core thickness    20 mm        ->  %.1f mm" % (TCORE))
print("  upstream/downstr  40 / 80 mm   ->  %.1f / %.1f mm" % (-ZIN, ZOUT-TCORE))
print("  cell across flats 8 mm         ->  %.3f mm" % (2*RC*np.cos(np.pi/6)))
print("  wall thickness    1 mm         ->  %.3f mm" % (PITCH - 2*RC*np.cos(np.pi/6)))
print("  cells in core: %d, open area %.1f%%" % (len(kept), 100*open_area/W**2))
