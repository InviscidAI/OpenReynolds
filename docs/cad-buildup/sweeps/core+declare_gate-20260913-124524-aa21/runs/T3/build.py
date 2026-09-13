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

# --- all dimensions in mm here; converted to metres before meshing ---
W      = 6.0      # channel width
L      = 120.0    # overall length
HW     = W/2.0
R_LOOP = 10.0     # loop centreline radius
SPAN   = 26.0     # x-distance between the two junctions of one loop
CY     = 14.0     # loop circle centre height above axis
X0S    = [10.0, 37.0, 64.0, 91.0]   # upstream junction x of each loop

def tangent_dir(P, C, R, sign):
    v = np.array(C) - np.array(P); d = np.linalg.norm(v)
    th = np.arctan2(v[1], v[0]); al = np.arcsin(R/d)
    return th + sign*al, np.sqrt(d*d - R*R)

def loop_centerline(x0, n=200):
    A = np.array([x0, 0.0]); B = np.array([x0+SPAN, 0.0]); C = np.array([x0+SPAN/2, CY])
    thA, LA = tangent_dir(A, C, R_LOOP, -1)
    thB, LB = tangent_dir(B, C, R_LOOP, -1)
    T1 = A + LA*np.array([np.cos(thA), np.sin(thA)])
    T2 = B + LB*np.array([np.cos(thB), np.sin(thB)])
    a1 = np.arctan2(*(T1-C)[::-1]); a2 = np.arctan2(*(T2-C)[::-1])
    while a2 < a1: a2 += 2*np.pi          # counter-clockwise over the top
    arc = np.stack([C + R_LOOP*np.array([np.cos(t), np.sin(t)]) for t in np.linspace(a1, a2, n)])
    return np.vstack([A, arc, B]), np.degrees(thA), np.degrees(thB), np.degrees(a2-a1)

cl, thA, thB, sweep = loop_centerline(X0S[0])
print(f"entry leg angle {thA:.1f} deg, exit leg angle {thB:.1f} deg, arc sweep {sweep:.1f} deg")
fig, ax = plt.subplots(figsize=(12,3.5))
ax.add_patch(plt.Rectangle((0,-HW), L, W, fc='lightblue', ec='k'))
for x0 in X0S:
    c,_,_,_ = loop_centerline(x0); ax.plot(c[:,0], c[:,1], 'r-')
ax.set_aspect('equal'); ax.set_xlim(-5,125); ax.set_ylim(-8,30); plt.savefig("cl.png", dpi=90)
print("loop extent x:", X0S[-1], X0S[-1]+SPAN, " max y:", cl[:,1].max())

# -- cell 2 -------------------------------------------------------------------------
# The figure needs displaying inline. Let me look at it, with the loops mirrored in x so that the *sha
import numpy as np
import matplotlib.pyplot as plt

def loop_mirrored(x0, n=200):
    c,a,b,s = loop_centerline(x0, n)
    xm = x0 + SPAN/2
    c2 = c.copy(); c2[:,0] = 2*xm - c2[:,0]
    return c2[::-1]   # traced from upstream junction A to downstream junction B

fig, ax = plt.subplots(figsize=(13,4))
ax.add_patch(plt.Rectangle((0,-HW), L, W, fc='lightblue', ec='k'))
for x0 in X0S:
    c = loop_mirrored(x0); ax.plot(c[:,0], c[:,1], 'r-')
    ax.plot(*c[0], 'go'); ax.plot(*c[-1], 'ks')
d0 = loop_mirrored(X0S[0])
print("upstream-junction leg angle:", np.degrees(np.arctan2(*(d0[1]-d0[0])[::-1])))
print("downstream-junction leg angle (into loop):", np.degrees(np.arctan2(*(d0[-2]-d0[-1])[::-1])))
ax.set_aspect('equal'); ax.set_xlim(-5,125); ax.set_ylim(-8,22); fig

# -- cell 3 -------------------------------------------------------------------------
# The centerline is right in character (steep upstream leg, shallow downstream leg) but the arch looks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def thicken(cl, hw=HW):
    d = np.gradient(cl, axis=0); d /= np.linalg.norm(d, axis=1)[:,None]
    n = np.stack([-d[:,1], d[:,0]], axis=1)
    return np.vstack([cl + hw*n, (cl - hw*n)[::-1]])

def show(cy, r, span, n=300):
    global CY, R_LOOP, SPAN
    CY, R_LOOP, SPAN = cy, r, span
    fig, ax = plt.subplots(figsize=(13,4.5))
    ax.add_patch(plt.Rectangle((0,-HW), L, W, fc='0.8', ec='none'))
    for x0 in X0S:
        c = loop_mirrored(x0, n)
        ax.add_patch(plt.Polygon(thicken(c), fc='0.8', ec='none'))
        ax.plot(*thicken(c).T, 'k-', lw=0.6)
    ax.set_aspect('equal'); ax.set_xlim(-5,125); ax.set_ylim(-8,30)
    ax.set_title(f"CY={cy} R={r} SPAN={span}")
    return fig

show(13.0, 8.0, 26.0)

# -- cell 4 -------------------------------------------------------------------------
# The offset-polyline route merges the loop into a blob. Better construction: define the **teardrop is
from build123d import *
import numpy as np

# ---- geometry parameters, millimetres (scaled to metres before meshing) ----
W     = 6.0                  # channel width
L     = 120.0                # overall length
HW    = W/2
RI    = 4.0                  # island nose radius
LTIP  = 7.0                  # island tip offset from nose centre (x)
PITCH = 30.0
XS    = [8.0, 38.0, 68.0, 98.0]   # left end of each island (nose tangency zone)

def hull_face(cx, cy, r, tx, ty):
    """convex hull of circle(r)@(cx,cy) and point (tx,ty): a teardrop."""
    C = np.array([cx, cy]); T = np.array([tx, ty])
    v = T - C; d = np.linalg.norm(v); al = np.arccos(r/d)
    th = np.arctan2(v[1], v[0])
    P1 = C + r*np.array([np.cos(th+al), np.sin(th+al)])
    P2 = C + r*np.array([np.cos(th-al), np.sin(th-al)])
    tri = Polygon(tuple(T), tuple(P1), tuple(P2), align=None)
    return Circle(r).locate(Location((cx, cy))) + tri

def loop_ring(xs):
    cx, cy = xs + RI, HW + RI          # nose circle tangent to channel top wall
    tx, ty = cx + LTIP, HW
    island = hull_face(cx, cy, RI, tx, ty)
    k = (RI + W)/RI                    # homothety = exact 6 mm outward offset
    outer = hull_face(cx, cy, RI + W, cx + k*(tx-cx), cy + k*(ty-cy))
    return outer - island, island

channel = Rectangle(L, W, align=None).locate(Location((0, -HW)))
rings   = [loop_ring(xs)[0] for xs in XS]
fluid2d = channel + rings[0] + rings[1] + rings[2] + rings[3]
print("faces:", len(fluid2d.faces()), " area mm^2:", round(fluid2d.area,1))
print("bbox mm:", fluid2d.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
# Rectangle defaults to centred, so the channel landed at x∈[-60,60]. Fixing the placement:
channel = Pos(L/2, 0) * Rectangle(L, W)
rings   = [loop_ring(xs)[0] for xs in XS]
fluid2d = channel + rings[0] + rings[1] + rings[2] + rings[3]
print("faces:", len(fluid2d.faces()), " area mm^2:", round(fluid2d.area,1))
print("bbox mm:", fluid2d.bounding_box())
f = fluid2d.faces()[0] if len(fluid2d.faces())==1 else None
print("inner wires (islands) in face 0:", len(fluid2d.faces()[0].inner_wires()))

# -- cell 6 -------------------------------------------------------------------------
# One face with exactly four island holes. Let me look at it by sampling the wires.
import matplotlib.pyplot as plt
face = fluid2d.faces()[0]
def wire_xy(w, n=400):
    return np.array([[p.X, p.Y] for p in [w.position_at(t) for t in np.linspace(0,1,n)]])
fig, ax = plt.subplots(figsize=(13,3.2))
ax.fill(*wire_xy(face.outer_wire()).T, fc='lightblue', ec='k', lw=0.8)
for w in face.inner_wires():
    ax.fill(*wire_xy(w).T, fc='white', ec='r', lw=0.8)
ax.set_aspect('equal'); ax.set_xlim(-3,123); ax.set_ylim(-6,20); fig

# -- cell 7 -------------------------------------------------------------------------
# Teardrop islands, round nose upstream, tip downstream. Now extrude one cell thick in z and export a 
THICK = 1.0    # mm, single-cell depth for the 2D (empty) direction
solid = extrude(fluid2d, amount=THICK)
print(type(solid), "volume mm^3:", round(solid.volume,1), " expected:", round(1474.1*THICK,1))
print("bbox:", solid.bounding_box())
print("faces:", len(solid.faces()), " solids:", len(solid.solids()))
export_step(solid, "tesla_mm.step")
import os; print("step bytes:", os.path.getsize("tesla_mm.step"))

# -- cell 8 -------------------------------------------------------------------------
outer = wire_xy(face.outer_wire(), 8000)
for k, w in enumerate(face.inner_wires()):
    ip = wire_xy(w, 1200)
    d = np.sqrt(((ip[:,None,:]-outer[None,:,:])**2).sum(-1)).min(axis=1)
    print(f"island {k}: gap to outer wall  min {d.min():.3f}  median {np.median(d):.3f} mm")
for xq in (30.0, 60.0, 90.0, 118.0):
    ys = outer[np.abs(outer[:,0]-xq) < 0.06][:,1]
    print(f"x={xq}: outer-wall y values {np.unique(np.round(ys,2))}")

# -- cell 9 -------------------------------------------------------------------------
# Width is exactly 6.000 mm everywhere around the islands and in the straight channel. Now the mesh: e
export_step(fluid2d, "tesla_face_mm.step")
import gmsh
CELL = 2.0   # mm, coarse first
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("tesla")
ents = gmsh.model.occ.importShapes("tesla_face_mm.step")
gmsh.model.occ.synchronize()
print("imported:", ents)
surf = [t for d,t in ents if d==2]
out = gmsh.model.occ.extrude([(2,s) for s in surf], 0, 0, THICK, [1], [1], True)
gmsh.model.occ.synchronize()
print("extrude result dims:", [(d,t) for d,t in out][:4], "... n=", len(out))

# -- cell 10 ------------------------------------------------------------------------
top  = out[0][1]; vol = out[1][1]; lats = [t for d,t in out[2:]]
back = surf[0]
inlet, outlet, walls = [], [], []
for t in lats:
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
    if abs(cx - 0.0)   < 1e-6: inlet.append(t)
    elif abs(cx - L)   < 1e-6: outlet.append(t)
    else: walls.append(t)
print("inlet", inlet, "outlet", outlet, "n walls", len(walls))
for t in inlet+outlet:
    print(t, gmsh.model.occ.getCenterOfMass(2,t), gmsh.model.occ.getMass(2,t), "mm^2 (expect 6x1=6)")

# -- cell 11 ------------------------------------------------------------------------
# Inlet and outlet are the 6 mm × 1 mm faces at x=0 and x=120. Tag the groups and generate a coarse me
gmsh.model.addPhysicalGroup(3, [vol], name="internal")
gmsh.model.addPhysicalGroup(2, inlet,  name="inlet")
gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
gmsh.model.addPhysicalGroup(2, walls,  name="walls")
gmsh.model.addPhysicalGroup(2, [back, top], name="frontAndBack")
gmsh.option.setNumber("Mesh.MeshSizeMin", CELL)
gmsh.option.setNumber("Mesh.MeshSizeMax", CELL)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 8)       # frontal-delaunay for quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
gmsh.model.mesh.generate(3)
import numpy as np
et, en, _ = gmsh.model.mesh.getElements(3)
print("3D element types/counts:", [(int(a), len(b)) for a,b in zip(et,en)])
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("tesla.msh")

# -- cell 12 ------------------------------------------------------------------------
# The mesh exists (573 hexes, all type 5). Convert it to an OpenFOAM case, scale mm→m, and run checkMe
import subprocess, os, shutil
for d in ("system","constant","0"):
    os.makedirs(d, exist_ok=True)
open("system/controlDict","w").write("""FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
""")
open("system/fvSchemes","w").write("""FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;}
divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;} snGradSchemes{default corrected;}
""")
open("system/fvSolution","w").write("""FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}
solvers{} 
""")
r = subprocess.run(["gmshToFoam","tesla.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:], r.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 14 ------------------------------------------------------------------------
# All 1392 boundary faces landed in named patches. Now scale mm→m, make the 2D direction `empty`, and 
r = subprocess.run(["transformPoints","scale=(0.001 0.001 0.001)"], capture_output=True, text=True)
print(r.returncode, r.stdout[-200:])
b = open("constant/polyMesh/boundary").read()
b = b.replace("""    frontAndBack
    {
        type            patch;
        physicalType    patch;""", """    frontAndBack
    {
        type            empty;
        physicalType    empty;""")
b = b.replace("""    walls
    {
        type            patch;
        physicalType    patch;""", """    walls
    {
        type            wall;
        physicalType    wall;""")
open("constant/polyMesh/boundary","w").write(b)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1800:])

# -- cell 15 ------------------------------------------------------------------------
# checkMesh passes but the points are still in mm. Correct argument form, then look at the mesh.
r = subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:], r.stderr[-200:])
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "bounding box" in l or "Total volume" in l])

# -- cell 16 ------------------------------------------------------------------------
# In metres now (0.12 m × 0.006 m channel). Let me look at the actual mesh.
import pyvista as pv
pv.start_xvfb() if False else None
pv.OFF_SCREEN = True
rd = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").write("")
m = pv.OpenFOAMReader("case.foam").read()["internalMesh"]
p = pv.Plotter(off_screen=True, window_size=(1500,400))
p.add_mesh(m, show_edges=True, color="lightblue", line_width=1)
p.view_xy(); p.camera.zoom(1.3)
p.screenshot("mesh_coarse.png")
from IPython.display import Image
Image("mesh_coarse.png")

# -- cell 17 ------------------------------------------------------------------------
# Topology is right. Now regenerate at a finer size (0.5 mm ⇒ 12 cells across the 6 mm channel) and re
import gmsh
CELL_FINE = 0.5   # mm
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("tesla_fine")
ents = gmsh.model.occ.importShapes("tesla_face_mm.step")
gmsh.model.occ.synchronize()
surf = [t for d,t in ents if d==2]
out = gmsh.model.occ.extrude([(2,s) for s in surf], 0, 0, THICK, [1], [1], True)
gmsh.model.occ.synchronize()
top = out[0][1]; vol = out[1][1]; lats = [t for d,t in out[2:]]; back = surf[0]
inlet, outlet, walls = [], [], []
for t in lats:
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
    if abs(cx) < 1e-6: inlet.append(t)
    elif abs(cx - L) < 1e-6: outlet.append(t)
    else: walls.append(t)
gmsh.model.addPhysicalGroup(3, [vol], name="internal")
gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
gmsh.model.addPhysicalGroup(2, walls, name="walls")
gmsh.model.addPhysicalGroup(2, [back, top], name="frontAndBack")
for o,v in [("Mesh.MeshSizeMin",CELL_FINE),("Mesh.MeshSizeMax",CELL_FINE),
            ("Mesh.MeshSizeFromCurvature",0),("Mesh.Algorithm",8),
            ("Mesh.RecombineAll",1),("Mesh.RecombinationAlgorithm",2),("Mesh.MshFileVersion",2.2)]:
    gmsh.option.setNumber(o,v)
gmsh.model.mesh.generate(3)
et, en, _ = gmsh.model.mesh.getElements(3)
print("3D elements:", [(int(a), len(b)) for a,b in zip(et,en)], " (type 5 = hex, 6 = prism)")
gmsh.write("tesla.msh")
print("inlet",inlet,"outlet",outlet,"walls",len(walls))

# -- cell 18 ------------------------------------------------------------------------
# All-hex, 6481 cells. Re-convert, re-scale, re-tag, re-check.
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","tesla.msh"], capture_output=True, text=True); print("gmshToFoam", r.returncode)
b = open("constant/polyMesh/boundary").read()
b = b.replace("""    frontAndBack
    {
        type            patch;
        physicalType    patch;""", """    frontAndBack
    {
        type            empty;
        physicalType    empty;""")
b = b.replace("""    walls
    {
        type            patch;
        physicalType    patch;""", """    walls
    {
        type            wall;
        physicalType    wall;""")
open("constant/polyMesh/boundary","w").write(b)
subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"], capture_output=True, text=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1500:])

# -- cell 19 ------------------------------------------------------------------------
# Mesh OK. Now export each named patch to its own STL (in metres) and take a closer look at one loop.
import pyvista as pv
rdr = pv.OpenFOAMReader("case.foam"); rdr.enable_all_patch_arrays()
blocks = rdr.read()
bnd = blocks["boundary"]
print(bnd.keys())
for name in bnd.keys():
    s = bnd[name].extract_surface().triangulate()
    s.save(f"{name}.stl")
    print(name, s.n_cells, "tris  bbox", np.round(s.bounds,4))

# -- cell 20 ------------------------------------------------------------------------
m = pv.OpenFOAMReader("case.foam").read()["internalMesh"]
p = pv.Plotter(off_screen=True, window_size=(1600,500))
p.add_mesh(m, show_edges=True, color="lightsteelblue", line_width=1)
p.view_xy(); p.camera.tight(padding=0.02)
p.screenshot("mesh_fine.png")
p2 = pv.Plotter(off_screen=True, window_size=(900,700))
p2.add_mesh(m.clip_box([0.0,0.036,-0.004,0.02,-1,1], invert=True), show_edges=True, color="lightsteelblue")
p2.view_xy(); p2.camera.tight(padding=0.05)
p2.screenshot("loop_zoom.png")
from IPython.display import Image, display
display(Image("mesh_fine.png")); display(Image("loop_zoom.png"))

# -- cell 21 ------------------------------------------------------------------------
# One last set of numbers, measured from the mesh that is on disk rather than from the CAD:
inl = bnd["inlet"]; outl = bnd["outlet"]
print("inlet area  m^2:", inl.extract_surface().area, "-> width", inl.extract_surface().area/0.001*1000, "mm")
print("outlet area m^2:", outl.extract_surface().area, "-> width", outl.extract_surface().area/0.001*1000, "mm")
print("domain length m:", m.bounds[1]-m.bounds[0], " depth (2D dir) m:", m.bounds[5]-m.bounds[4])
print("cells:", m.n_cells, " all hex:", set(m.celltypes))
# main-channel width from the wall patch in a straight stretch (x = 30,60,90 mm)
wp = bnd["walls"].points
for xq in (0.030, 0.060, 0.090):
    ys = wp[np.abs(wp[:,0]-xq) < 2e-4][:,1]
    print(f"x={xq*1000:.0f} mm: wall y = {np.unique(np.round(ys,5))*1000} mm -> width {1000*(ys.max()-ys.min()):.3f} mm")
print("islands (holes) in fluid face:", len(face.inner_wires()))
