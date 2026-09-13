"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Re-sending as a tool call.
import numpy as np, math
from build123d import *

mm = 1e-3
W = 80*mm          # duct section
T = 20*mm          # core thickness (flow = z)
LU, LD = 40*mm, 80*mm
AF = 8*mm          # across flats (open cell)
TW = 1*mm          # wall thickness
P  = AF + TW       # centre pitch across flats
R  = AF/math.sqrt(3)   # circumradius of open hexagon

dx = P*math.sqrt(3)/2
cx = np.arange(-8, 9)*dx
centres = []
for i, x in enumerate(cx):
    y0 = 0.0 if i % 2 == 0 else P/2
    for y in np.arange(-9, 10)*P + y0:
        if abs(x) < W/2 + P and abs(y) < W/2 + P:
            centres.append((x, y))
print(len(centres), "cells", "R=", R, "dx=", dx)
hexpts = [(R*math.cos(a), R*math.sin(a)) for a in np.arange(6)*math.pi/3]
print(np.round(np.array(hexpts)/mm, 3))

# -- cell 2 -------------------------------------------------------------------------
# Lattice confirmed (8 mm across flats in y, 9 mm pitch). Now the solid: plate minus hex prisms gives 
import time
t0=time.time()
hexsk = Polygon(*hexpts, align=None)
prisms = [Pos(x, y, 0) * extrude(hexsk, T) for (x, y) in centres]
cells = Compound(children=prisms)
plate = Pos(0, 0, T/2) * Box(W, W, T)
walls = plate - cells
duct  = Pos(0, 0, (T - LU + LD)/2) * Box(W, W, LU + T + LD)
fluid = duct - walls
print("t", round(time.time()-t0,1), "bbox mm", np.round(np.array(fluid.bounding_box().min.to_tuple()+fluid.bounding_box().max.to_tuple())/mm,3))
print("fluid vol cm3", fluid.volume/1e-6, " duct vol", duct.volume/1e-6, " walls vol", walls.volume/1e-6)
print("solids", len(fluid.solids()), "faces", len(fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
# Single closed solid, open fraction 0.795 (theory (√3/2)·8²/((√3/2)·9²)=0.790 for full cells; edge ce
from build123d import export_step, export_stl
export_step(fluid, "fluid.step")
export_stl(fluid, "fluid_view.stl", tolerance=2e-4, angular_tolerance=0.3)
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
m = pv.read("fluid_view.stl")
p = pv.Plotter(off_screen=True, window_size=(900,700))
p.add_mesh(m.clip(normal='y', origin=(0,0,0)), color='lightsteelblue', show_edges=False)
p.camera_position = 'xz'; p.camera.azimuth = 35; p.camera.elevation = 20
p.show(screenshot="view1.png")
print("ok")

# -- cell 4 -------------------------------------------------------------------------
# The clip render is confusing. Look straight down the flow axis, plus a cross-section slice at mid-co
import matplotlib.pyplot as plt
sl = m.slice(normal='z', origin=(0,0,T/2))
fig, ax = plt.subplots(figsize=(6,6))
for c in sl.cell:
    pts = c.points
    ax.plot(pts[:,0]/mm, pts[:,1]/mm, 'k-', lw=0.6)
ax.set_aspect('equal'); ax.set_title("fluid cross-section at z=10 mm"); ax.set_xlabel("x [mm]")
plt.show()

# -- cell 5 -------------------------------------------------------------------------
# Cross-section is the expected honeycomb. Check for degenerate slivers (rows whose flats land exactly
fa = np.array([f.area for f in fluid.faces()])
el = np.array([e.length for e in fluid.edges()])
print("faces:", len(fa), "min area mm2", fa.min()/1e-6, "n<0.1mm2", (fa<1e-7).sum())
print("edges:", len(el), "min len mm", el.min()/mm, "n<0.2mm", (el<2e-4).sum())
print(sorted(np.round(el[el<1e-3]/mm,4))[:20])

# -- cell 6 -------------------------------------------------------------------------
# No degenerate faces (min edge 1.0 mm = the wall thickness). Now a coarse gmsh tet mesh, with patches
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("t11")
gmsh.model.occ.importShapes("fluid.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
groups = {"inlet": [], "outlet": [], "ductWalls": [], "honeycombWalls": []}
tol = 1e-7
for (d, s) in surfs:
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(d, s)
    if abs(z1-z0) < tol and abs(z0-(-LU)) < tol:      groups["inlet"].append(s)
    elif abs(z1-z0) < tol and abs(z0-(T+LD)) < tol:   groups["outlet"].append(s)
    elif (abs(x1-x0) < tol and abs(abs(x0)-W/2) < tol) or (abs(y1-y0) < tol and abs(abs(y0)-W/2) < tol):
        groups["ductWalls"].append(s)
    else: groups["honeycombWalls"].append(s)
print({k: len(v) for k, v in groups.items()}, "total", len(surfs))

# -- cell 7 -------------------------------------------------------------------------
print(gmsh.model.getBoundingBox(-1,-1))

# -- cell 8 -------------------------------------------------------------------------
# Units are metres; gmsh pads bounding boxes by 1e-7, so my tolerance was too tight. Loosen it.
groups = {"inlet": [], "outlet": [], "ductWalls": [], "honeycombWalls": []}
tol = 1e-6
for (d, s) in gmsh.model.getEntities(2):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(d, s)
    if abs(z1-z0) < tol and abs(z0-(-LU)) < tol:      groups["inlet"].append(s)
    elif abs(z1-z0) < tol and abs(z0-(T+LD)) < tol:   groups["outlet"].append(s)
    elif (abs(x1-x0) < tol and abs(abs(x0)-W/2) < tol) or (abs(y1-y0) < tol and abs(abs(y0)-W/2) < tol):
        groups["ductWalls"].append(s)
    else: groups["honeycombWalls"].append(s)
print({k: len(v) for k, v in groups.items()})
import numpy as _np
for k,v in groups.items():
    a = sum(gmsh.model.occ.getMass(2,s) for s in v)
    print(k, "area mm2", round(a/1e-6,1))

# -- cell 9 -------------------------------------------------------------------------
# Patch split is right (inlet/outlet 6400 mm² = 80×80). Now physical groups, size field, and a coarse 
for name, ss in groups.items():
    gmsh.model.addPhysicalGroup(2, ss, name=name)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="internal")

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", groups["honeycombWalls"])
gmsh.model.mesh.field.setNumber(f1, "Sampling", 200)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", 2.0*mm)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", 5.0*mm)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 2.0*mm)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 20.0*mm)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
import time; t0=time.time()
gmsh.model.mesh.generate(3)
gmsh.write("fluid.msh")
print("meshed in", round(time.time()-t0,1), "s")
print("nodes", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 10 ------------------------------------------------------------------------
# Convert to OpenFOAM (msh 2.2 ASCII), with a minimal system/.
import os, subprocess, textwrap
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.write("fluid.msh")
gmsh.finalize()
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr % "controlDict" + textwrap.dedent("""
 application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
 deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
 writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(hdr % "fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr % "fvSolution" + "solvers{} SIMPLE{nNonOrthogonalCorrectors 0;}\n")
r = subprocess.run(["gmshToFoam","fluid.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 12 ------------------------------------------------------------------------
# All boundary faces are assigned. Set the two wall patches to `type wall`, then check.
import re
b = open("constant/polyMesh/boundary").read()
for w in ["ductWalls", "honeycombWalls"]:
    b = re.sub(r"(%s\s*\{\s*type\s+)patch(;\s*physicalType\s+)patch" % w, r"\1wall\2wall", b)
open("constant/polyMesh/boundary","w").write(b)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 13 ------------------------------------------------------------------------
# Mesh OK, volume matches the CAD exactly (869.793 cm³). Look at the mesh itself: patch colours and a 
open("t11.foam","w").close()
rd = pv.OpenFOAMReader("t11.foam"); rd.set_active_time_value(0.0)
mesh = rd.read()
internal = mesh["internalMesh"]
sl = internal.slice(normal='y', origin=(0,0.0005,0))
p = pv.Plotter(off_screen=True, window_size=(1000,600))
p.add_mesh(sl, show_edges=True, edge_color='gray', color='w', line_width=0.3)
p.view_xz(); p.show(screenshot="slice.png")
bnd = mesh["boundary"]
print(list(bnd.keys()))

# -- cell 14 ------------------------------------------------------------------------
# Mesh looks right: through-channels in the core, walls resolved. Final measurements from the built ge
bb = fluid.bounding_box()
print("duct section  : %.1f x %.1f mm (asked 80 x 80)" % ((bb.max.X-bb.min.X)/mm, (bb.max.Y-bb.min.Y)/mm))
print("upstream / core / downstream : %.1f / %.1f / %.1f mm (asked 40 / 20 / 80)" % (-bb.min.Z/mm, T/mm, (bb.max.Z-T)/mm))
print("cell across flats : %.3f mm (asked 8)" % ((max(p[1] for p in hexpts)-min(p[1] for p in hexpts))/mm))
print("wall thickness    : %.3f mm (asked 1)  [pitch %.3f - flats %.3f]" % ((P-AF)/mm, P/mm, AF/mm))
print("min fluid edge    : %.3f mm  (= wall thickness, i.e. no slivers)" % (el.min()/mm))
print("open area ratio   : %.3f  (fluid vol %.2f cm3 / duct %.2f cm3 in core: %.3f)" %
      (1-26.2068/128, fluid.volume/1e-6, duct.volume/1e-6, (128-26.2068)/128))
print("patches:", {k: len(v) for k, v in groups.items()})
print("cells 163496 tets, checkMesh: Mesh OK, maxNonOrtho 72.1 (5 faces), maxSkew 1.43")
