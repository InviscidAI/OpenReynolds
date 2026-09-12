"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build123d import *
import build123d
print(os.getcwd())
print(getattr(build123d, "__version__", "?"))
print(trace)

# -- cell 2 -------------------------------------------------------------------------
# Now the fluid face: main channel + one loop, in metres, to look at the teardrop before repeating it.
import math
mm = 0.001
W  = 6*mm          # channel width
L  = 120*mm        # overall length
PITCH = 26*mm
XS = [12*mm, 38*mm, 64*mm, 90*mm]

def loop_centerline(xs):
    pts = [(xs+0*mm, 0), (xs-3*mm, 5*mm), (xs+3*mm, 11*mm),
           (xs+14*mm, 12*mm), (xs+21*mm, 6*mm), (xs+22*mm, 0)]
    t0 = (math.cos(math.radians(150)), math.sin(math.radians(150)))
    t1 = (math.cos(math.radians(-70)), math.sin(math.radians(-70)))
    return Spline(*[Vector(p[0], p[1], 0) for p in pts], tangents=(t0, t1))

main_cl = Line((0, 0), (L, 0))
cl1 = loop_centerline(XS[0])
test = trace([main_cl], W) + trace([cl1], W)
print(type(test), test.area, test.bounding_box())

fig, ax = plt.subplots(figsize=(10, 3))
for e in test.edges():
    p = np.array([tuple(e @ (i/60)) for i in range(61)])
    ax.plot(p[:, 0]*1000, p[:, 1]*1000, 'k-', lw=1)
ax.set_aspect('equal'); ax.set_title("main + one loop [mm]")
fig

# -- cell 3 -------------------------------------------------------------------------
# The right-hand turn was tighter than the half-width, so the trace self-intersected. I'll end the loo
def loop_edges(xs):
    q  = (xs+20*mm, 6*mm)                      # end of curved part
    dx = 6*mm/math.tan(math.radians(60))       # straight run down at -60 deg to axis
    b  = (q[0]+dx, 0.0)
    pts = [(xs, 0.0), (xs-3*mm, 6*mm), (xs+5*mm, 12*mm), (xs+14*mm, 12.5*mm), q]
    t0 = (math.cos(math.radians(150)), math.sin(math.radians(150)))
    t1 = (math.cos(math.radians(-60)), math.sin(math.radians(-60)))
    sp = Spline(*[Vector(p[0], p[1], 0) for p in pts], tangents=(t0, t1))
    return [sp, Line(q, b)], b

eds, bpt = loop_edges(XS[0])
test = trace([main_cl], W) + trace(eds, W)
f = test.faces()
print(len(f), test.area, "b =", bpt)
fig, ax = plt.subplots(figsize=(10, 3))
for fc in f:
    for w in [fc.outer_wire()] + fc.inner_wires():
        for e in w.edges():
            p = np.array([tuple(e @ (i/60)) for i in range(61)])
            ax.plot(p[:, 0]*1000, p[:, 1]*1000, 'k-', lw=1)
ax.set_aspect('equal'); ax.set_title(f"{len(f)} face(s)")
fig

# -- cell 4 -------------------------------------------------------------------------
# One face — let me confirm the island is a proper hole and zoom in on the loop.
fc = test.faces()[0]
print("inner wires:", len(fc.inner_wires()))
fig, ax = plt.subplots(figsize=(8, 5))
for w, c in [(fc.outer_wire(), 'k')] + [(w, 'r') for w in fc.inner_wires()]:
    for e in w.edges():
        p = np.array([tuple(e @ (i/80)) for i in range(81)])
        ax.plot(p[:, 0]*1000, p[:, 1]*1000, c+'-', lw=1.2)
ax.set_xlim(0, 45); ax.set_ylim(-5, 18); ax.set_aspect('equal'); ax.grid(alpha=.3)
fig

# -- cell 5 -------------------------------------------------------------------------
# The inner offset spikes where the spline's curvature drops below the half-width. I'll replace the sp
R_LOOP = 7*mm
A_START = 150.0     # loop leaves the axis heading up-left -> returning jet points down-right (+x)
SWEEP   = -205.0    # clockwise over the top

def loop_edges(xs):
    a = Vector(xs, 0, 0)
    t = Vector(math.cos(math.radians(A_START)), math.sin(math.radians(A_START)), 0)
    arc = JernArc(start=a, tangent=t, radius=R_LOOP, arc_size=SWEEP)
    e = arc @ 1
    ang = math.radians(A_START + SWEEP)          # tangent heading at arc end
    dx = -e.Y / math.tan(ang)                    # straight down to the axis
    b = Vector(e.X + dx, 0, 0)
    return [arc, Line(e, b)], a, b, arc

eds, a, b, arc = loop_edges(XS[0])
print("arc end", arc @ 1, "tangent deg", A_START+SWEEP, "b", b)
test = trace([main_cl], W) + trace(eds, W)
fc = test.faces()[0]
print("faces", len(test.faces()), "inner wires", len(fc.inner_wires()))
fig, ax = plt.subplots(figsize=(8, 5))
for w, c in [(fc.outer_wire(), 'k')] + [(w, 'r') for w in fc.inner_wires()]:
    for ed in w.edges():
        p = np.array([tuple(ed @ (i/80)) for i in range(81)])
        ax.plot(p[:, 0]*1000, p[:, 1]*1000, c+'-', lw=1.2)
ax.set_xlim(0, 45); ax.set_ylim(-6, 20); ax.set_aspect('equal'); ax.grid(alpha=.3)
fig

# -- cell 6 -------------------------------------------------------------------------
# Clean teardrop with a proper island. Now the full four-loop face, with measurements of the things th
loops = [loop_edges(x) for x in XS]
face = trace([main_cl], W)
for eds, a, b, arc in loops:
    face = face + trace(eds, W)
face = face.faces()[0]
print("faces:", 1, "islands (inner wires):", len(face.inner_wires()))
bb = face.bounding_box()
print("bbox mm: x %.2f..%.2f  y %.2f..%.2f" % (bb.min.X*1e3, bb.max.X*1e3, bb.min.Y*1e3, bb.max.Y*1e3))
print("overall length mm = %.2f (asked ~120)" % ((bb.max.X-bb.min.X)*1e3))
# width check: straight main channel section, measure the face's y-extent at x=2mm (pure channel)
for xq in [2*mm, 35*mm, 61*mm]:
    sec = face & Plane.YZ.offset(xq)  # section curve
    ys = [v.Y for v in sec.vertices()]
    print("x=%5.1f mm  y-span %.3f..%.3f" % (xq*1e3, min(ys)*1e3, max(ys)*1e3))
print("loop starts mm:", [round(x*1e3,1) for x in XS], " loop ends mm:", [round(b.X*1e3,1) for _,_,b,_ in loops])
print("loop arc radius mm %.2f, return stub angle deg %.1f" % (R_LOOP*1e3, A_START+SWEEP))
fig, ax = plt.subplots(figsize=(13, 3))
for w, c in [(face.outer_wire(), 'k')] + [(w, 'r') for w in face.inner_wires()]:
    for ed in w.edges():
        p = np.array([tuple(ed @ (i/60)) for i in range(61)])
        ax.plot(p[:,0]*1e3, p[:,1]*1e3, c+'-', lw=1)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title("Tesla valve fluid face [mm]")
fig

# -- cell 7 -------------------------------------------------------------------------
# Four loops, four islands, 120.00 mm long, 6.000 mm channel width at the plain section. Now extrude o
THICK = 1*mm           # one cell in z, 2D case
# verify arm width away from junctions: distance from island apex to outer wall at the loop crown
apex_out = face.bounding_box().max.Y
isl_top  = max(v.Y for w in face.inner_wires() for v in w.vertices())
print("loop arm width at crown = %.3f mm (asked 6)" % ((apex_out-isl_top)*1e3))
solid = extrude(face, THICK)
print("solid volume mm^3 %.1f, face area mm^2 %.1f, ratio %.4f mm" %
      (solid.volume*1e9, face.area*1e6, solid.volume/face.area*1e3))
export_step(solid, "tesla.step")
print(os.listdir("."))

# -- cell 8 -------------------------------------------------------------------------
# The crown check compared vertices at different x. Proper section through the loop crown:
xc = XS[0] + R_LOOP*math.cos(math.radians(A_START-90))   # arc centre x
print("arc centre x mm %.2f" % (xc*1e3))
sec = face & Plane.YZ.offset(xc)
ys = sorted(v.Y*1e3 for v in sec.vertices())
print("y crossings at crown [mm]:", [round(y,3) for y in ys])
print("main channel width %.3f, loop arm width %.3f  (asked 6)" % (ys[1]-ys[0], ys[3]-ys[2]))

# -- cell 9 -------------------------------------------------------------------------
# Coarse mesh: import the planar face into gmsh, quad-mesh it at 1.5 mm, extrude one layer in z.
export_step(face, "tesla_face.step")
import gmsh
def build_mesh(h, fname):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tesla")
    gmsh.model.occ.importShapes("tesla_face.step")
    gmsh.model.occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    print("imported surfaces:", surfs)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)          # frontal-delaunay for quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    ext = gmsh.model.occ.extrude([(2, surfs[0][1])], 0, 0, THICK, numElements=[1], recombine=True)
    gmsh.model.occ.synchronize()
    return ext

ext = build_mesh(1.5*mm, None)
print("extrude result:", ext)

# -- cell 10 ------------------------------------------------------------------------
back_tag, vol_tag, front_tag = 1, 1, 30
lat = [t for d, t in ext if d == 2 and t not in (front_tag,)] + []
lat = [t for t in lat if t != back_tag]
# inlet/outlet are the extrusions of the main-channel start/end cross sections,
# whose centres are known from the centreline construction: (0,0) and (L,0).
targets = {"inlet": (0.0, 0.0, THICK/2), "outlet": (L, 0.0, THICK/2)}
found = {}
for t in lat:
    c = np.array(gmsh.model.occ.getCenterOfMass(2, t))
    for name, p in targets.items():
        if np.allclose(c, p, atol=1e-9):
            found[name] = t
print("lateral count", len(lat), "found", found)
for name, t in found.items():
    m = gmsh.model.occ.getMass(2, t)
    nrm = gmsh.model.getNormal(t, [0.5, 0.5])
    print(name, "tag", t, "area mm^2 %.4f (expect %.4f)" % (m*1e6, W*THICK*1e6), "normal", np.round(nrm, 6))

# -- cell 11 ------------------------------------------------------------------------
# Patch groups and the coarse mesh:
walls = [t for t in lat if t not in found.values()]
gmsh.model.addPhysicalGroup(2, [found["inlet"]],  name="inlet")
gmsh.model.addPhysicalGroup(2, [found["outlet"]], name="outlet")
gmsh.model.addPhysicalGroup(2, walls,             name="walls")
gmsh.model.addPhysicalGroup(2, [back_tag],        name="back")
gmsh.model.addPhysicalGroup(2, [front_tag],       name="front")
gmsh.model.addPhysicalGroup(3, [vol_tag],         name="fluid")
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("tesla.msh")
ne = gmsh.model.mesh.getElements(3)
print("3D elements:", sum(len(x) for x in ne[1]))
gmsh.finalize()
print(os.listdir("."))

# -- cell 12 ------------------------------------------------------------------------
# Convert to OpenFOAM: minimal case skeleton, then `gmshToFoam`.
os.makedirs("system", exist_ok=True)
hdr = lambda cls, obj: ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj))
open("system/controlDict", "w").write(hdr("dictionary", "controlDict") + """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
""")
open("system/fvSchemes", "w").write(hdr("dictionary", "fvSchemes") + "\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\nddtSchemes{default steadyState;}\n")
open("system/fvSolution", "w").write(hdr("dictionary", "fvSolution") + "\nsolvers{}\n")
r = subprocess.run(["gmshToFoam", "tesla.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 13 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 14 ------------------------------------------------------------------------
# All 2070 boundary faces are accounted for. Set types: walls→wall, front/back→empty, then checkMesh.
import re
def set_types(path="constant/polyMesh/boundary",
              types={"walls": "wall", "front": "empty", "back": "empty",
                     "inlet": "patch", "outlet": "patch"}):
    txt = open(path).read()
    for name, ty in types.items():
        txt = re.sub(r"(\b%s\b\s*\{\s*type\s+)\w+;" % name, r"\g<1>%s;" % ty, txt)
        txt = re.sub(r"(\b%s\b\s*\{[^}]*?physicalType\s+)\w+;" % name, r"\g<1>%s;" % ty, txt)
    open(path, "w").write(txt)
set_types()
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 15 ------------------------------------------------------------------------
# Coarse mesh is valid 2D (877 hexes, volume matches the face area × 1 mm). Let me look at it.
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
pv.global_theme.background = "white"
m = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam", "w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
mesh = rd.read()["internalMesh"]
p = pv.Plotter(off_screen=True, window_size=(1500, 350))
p.add_mesh(mesh.extract_surface(), show_edges=True, color="lightsteelblue", line_width=1)
p.view_xy(); p.camera.zoom(1.35)
p.screenshot("coarse.png")
from IPython.display import Image
Image("coarse.png")

# -- cell 16 ------------------------------------------------------------------------
# Shape is right. Now refine to 0.5 mm (12 cells across the channel) with one self-contained function.
def make_msh(h, out="tesla.msh"):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tesla")
    gmsh.model.occ.importShapes("tesla_face.step")
    gmsh.model.occ.synchronize()
    s = gmsh.model.getEntities(2)[0][1]
    for o, v in [("Mesh.MeshSizeMin", h), ("Mesh.MeshSizeMax", h), ("Mesh.MeshSizeFromCurvature", 0),
                 ("Mesh.Algorithm", 8), ("Mesh.RecombineAll", 1), ("Mesh.RecombinationAlgorithm", 2)]:
        gmsh.option.setNumber(o, v)
    ex = gmsh.model.occ.extrude([(2, s)], 0, 0, THICK, numElements=[1], recombine=True)
    gmsh.model.occ.synchronize()
    top = ex[0][1]; vol = [t for d, t in ex if d == 3][0]
    lats = [t for d, t in ex if d == 2 and t != top]
    tg = {"inlet": (0.0, 0.0, THICK/2), "outlet": (L, 0.0, THICK/2)}
    io = {}
    for t in lats:
        c = np.array(gmsh.model.occ.getCenterOfMass(2, t))
        for n, pt in tg.items():
            if np.allclose(c, pt, atol=1e-9):
                io[n] = t
    assert set(io) == {"inlet", "outlet"}, io
    wl = [t for t in lats if t not in io.values()]
    gmsh.model.addPhysicalGroup(2, [io["inlet"]], name="inlet")
    gmsh.model.addPhysicalGroup(2, [io["outlet"]], name="outlet")
    gmsh.model.addPhysicalGroup(2, wl, name="walls")
    gmsh.model.addPhysicalGroup(2, [s], name="back")
    gmsh.model.addPhysicalGroup(2, [top], name="front")
    gmsh.model.addPhysicalGroup(3, [vol], name="fluid")
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(out)
    areas = {n: gmsh.model.occ.getMass(2, t) for n, t in io.items()}
    nc = sum(len(x) for x in gmsh.model.mesh.getElements(3)[1])
    gmsh.finalize()
    return nc, areas

print(make_msh(0.5*mm))

# -- cell 17 ------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam", "tesla.msh"], capture_output=True, text=True)
print(r.returncode, [l for l in r.stdout.splitlines() if "hex" in l or "Patch" in l or "total" in l][:12])
set_types()
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-40:]))

# -- cell 18 ------------------------------------------------------------------------
# Mesh OK at 0.5 mm. Final look at the refined mesh with patches coloured.
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays(); rd.set_active_time_value(0.0)
d = rd.read()
p = pv.Plotter(off_screen=True, window_size=(1600, 420))
p.add_mesh(d["internalMesh"].extract_surface(algorithm=None), show_edges=True, color="lightsteelblue", line_width=0.5)
for nm, col in [("inlet", "green"), ("outlet", "red")]:
    p.add_mesh(d["boundary"][nm], color=col, line_width=6, render_lines_as_tubes=True)
p.view_xy(); p.camera.zoom(1.5); p.screenshot("fine.png")
Image("fine.png")
