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
from build123d import *

W_FLAT = 8.0      # cell width across flats (fluid)
T_WALL = 1.0      # wall thickness
PITCH  = W_FLAT + T_WALL          # 9 mm across flats, centre-to-centre
S      = 80.0     # duct section
R_CELL = W_FLAT/np.sqrt(3)        # circumradius of fluid hexagon
R_P    = PITCH/np.sqrt(3)
dx, dy = 1.5*R_P, np.sqrt(3)*R_P  # 7.794, 9.0

centres = []
for i in range(-8, 9):
    for j in range(-8, 9):
        x = dx*i
        y = dy*(j + 0.5*(i % 2))
        if abs(x) < S/2 + PITCH and abs(y) < S/2 + PITCH:
            centres.append((x, y))
print(len(centres), R_CELL, dx, dy)
plt.figure(figsize=(5,5))
for (x,y) in centres:
    a = np.linspace(0,2*np.pi,7)
    plt.plot(x+R_CELL*np.cos(a), y+R_CELL*np.sin(a),'b-',lw=.6)
plt.gca().add_patch(plt.Rectangle((-S/2,-S/2),S,S,fill=False,ec='r'))
plt.gca().set_aspect('equal'); plt.savefig("lattice.png",dpi=90)

# -- cell 2 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("lattice.png"))

# -- cell 3 -------------------------------------------------------------------------
import time
t0=time.time()
CLIP = S - T_WALL            # 79 mm: keep >=0.5 mm wall between fluid and duct wall
box_clip = Rectangle(CLIP, CLIP)
hex0 = RegularPolygon(radius=R_CELL, side_count=6, major_radius=True)
A_full = hex0.area
kept = []
for (x,y) in centres:
    h = Pos(x,y) * hex0
    c = h & box_clip
    if c.area >= 0.25*A_full:
        kept.append(c)
print("A_full", A_full, "kept", len(kept), time.time()-t0)

# -- cell 4 -------------------------------------------------------------------------
open_face = kept[0]
for c in kept[1:]:
    open_face = open_face + c
wall_face = Rectangle(S, S) - open_face
print("open area", open_face.area, "wall area", wall_face.area, "sum", open_face.area+wall_face.area)
print("min kept area", min(c.area for c in kept), "porosity", open_face.area/(S*S))

# -- cell 5 -------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,7))
for e in wall_face.edges():
    pts = [e@ (t/40) for t in range(41)]
    ax.plot([p.X for p in pts],[p.Y for p in pts],'k-',lw=0.7)
ax.set_aspect('equal'); ax.set_title("honeycomb wall cross-section (mm)")
fig

# -- cell 6 -------------------------------------------------------------------------
L_UP, L_CORE, L_DN = 40.0, 20.0, 80.0
honey = extrude(Plane.XY * wall_face, amount=L_CORE)          # z 0..20
duct  = Pos(0,0,(L_CORE+L_DN-L_UP)/2) * Box(S, S, L_UP+L_CORE+L_DN)
fluid = duct - honey
print("bbox", fluid.bounding_box())
print("fluid vol", fluid.volume, "expected", (L_UP+L_CORE+L_DN)*S*S - L_CORE*wall_face.area)
print("solids", len(fluid.solids()), "faces", len(fluid.faces()))

# -- cell 7 -------------------------------------------------------------------------
import os, subprocess
export_step(fluid, "fluid_mm.step")
print(os.path.getsize("fluid_mm.step"))
import gmsh
gmsh.initialize()
gmsh.model.add("hc")
gmsh.model.occ.importShapes("fluid_mm.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
print("vols", len(gmsh.model.getEntities(3)), "surfs", len(surfs))

# -- cell 8 -------------------------------------------------------------------------
tol = 1e-6
groups = {"inlet":[], "outlet":[], "ductwalls":[], "honeycombwalls":[]}
Z0, Z1 = -L_UP, L_CORE+L_DN
for (d,t) in surfs:
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(d,t)
    if z1-z0 < tol and abs(z0-Z0) < tol:   groups["inlet"].append(t)
    elif z1-z0 < tol and abs(z0-Z1) < tol: groups["outlet"].append(t)
    elif (x1-x0 < tol and abs(abs(x0)-S/2) < tol) or (y1-y0 < tol and abs(abs(y0)-S/2) < tol):
        groups["ductwalls"].append(t)
    else: groups["honeycombwalls"].append(t)
print({k:len(v) for k,v in groups.items()})

# -- cell 9 -------------------------------------------------------------------------
for name, tags in groups.items():
    g = gmsh.model.addPhysicalGroup(2, tags); gmsh.model.setPhysicalName(2, g, name)
gv = gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)])
gmsh.model.setPhysicalName(3, gv, "internal")

H_CORE, H_DUCT = 3.0, 5.0
gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", H_CORE); gmsh.model.mesh.field.setNumber(1, "VOut", H_DUCT)
for k,v in dict(XMin=-41,XMax=41,YMin=-41,YMax=41,ZMin=-8,ZMax=28).items():
    gmsh.model.mesh.field.setNumber(1, k, v)
gmsh.model.mesh.field.setNumber(1, "Thickness", 15)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
import time; t0=time.time()
gmsh.model.mesh.generate(3)
gmsh.write("fluid.msh")
print("time", time.time()-t0, "nodes", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 10 ------------------------------------------------------------------------
import subprocess, os, textwrap
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.write("fluid.msh")
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(head%"controlDict"+
  "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head%"fvSchemes"+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(head%"fvSolution"+"solvers{} \n")
r = subprocess.run(["gmshToFoam","fluid.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 12 ------------------------------------------------------------------------
b = open("constant/polyMesh/boundary").read()
for p in ("ductwalls","honeycombwalls"):
    i = b.index(p); j = b.index("type", i)
    b = b[:j] + b[j:].replace("patch;", "wall;", 1)
    k = b.index("physicalType", i); b = b[:k] + b[k:].replace("patch;", "wall;", 1)
open("constant/polyMesh/boundary","w").write(b)
r1 = subprocess.run(["transformPoints","scale=(0.001 0.001 0.001)"], capture_output=True, text=True)
r2 = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r1.returncode, r2.stdout[-2500:])

# -- cell 13 ------------------------------------------------------------------------
r1 = subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"], capture_output=True, text=True)
print(r1.returncode, r1.stdout[-400:], r1.stderr[-300:])
els = sorted(e.length for e in fluid.edges())
print("min edges mm:", els[:12])
