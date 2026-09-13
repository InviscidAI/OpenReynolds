"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
from build123d import *
import math

OR_RING, IR_RING, H_RING = 30.0, 23.0, 10.0   # mm
OR_SPIN, IR_SPIN, H_SPIN = 22.0, 15.0, 8.0    # mm

ring     = Cylinder(OR_RING, H_RING) - Cylinder(IR_RING, H_RING)
spinner  = Cylinder(OR_SPIN, H_SPIN) - Cylinder(IR_SPIN, H_SPIN)
oil_mm   = Cylinder(IR_RING, H_RING) - spinner

print("ring bbox", ring.bounding_box())
print("spinner bbox", spinner.bounding_box())
print("oil bbox", oil_mm.bounding_box())
print("radial clearance mm", IR_RING - OR_SPIN, " axial gap each end mm", (H_RING-H_SPIN)/2)
print("oil vol mm3", oil_mm.volume, "expected", math.pi*IR_RING**2*H_RING - math.pi*(OR_SPIN**2-IR_SPIN**2)*H_SPIN)
print("n faces", len(oil_mm.faces()))

# -- cell 2 -------------------------------------------------------------------------
# Seven faces as expected. Classify each geometrically (surface type + radius/z), then scale to metres
oil = oil_mm.scale(0.001)   # mm -> m, once
for i, f in enumerate(oil.faces()):
    c = f.center()
    print(i, f.geom_type, "area=%.6e" % f.area, "c=(%.4f,%.4f,%.4f)" % (c.X, c.Y, c.Z),
          "bbox r=%.4f zmin=%.4f zmax=%.4f" % (max(abs(f.bounding_box().max.X), abs(f.bounding_box().min.X)),
                                               f.bounding_box().min.Z, f.bounding_box().max.Z))
print("oil vol m3", oil.volume)

# -- cell 3 -------------------------------------------------------------------------
# Classification is unambiguous: cyl r=23 → ring wetted, planes z=±5 → ring end faces, {cyl r=22, cyl 
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

fs = oil.faces()
def pick(pred): return [f for f in fs if pred(f)]
R = lambda f: max(abs(f.bounding_box().max.X), abs(f.bounding_box().min.X))

ring_wet      = pick(lambda f: f.geom_type == GeomType.CYLINDER and abs(R(f)-0.023) < 1e-6)
ring_end_top  = pick(lambda f: f.geom_type == GeomType.PLANE and abs(f.center().Z-0.005) < 1e-6)
ring_end_bot  = pick(lambda f: f.geom_type == GeomType.PLANE and abs(f.center().Z+0.005) < 1e-6)
spinner_wet   = pick(lambda f: (f.geom_type == GeomType.CYLINDER and abs(R(f)-0.023) > 1e-6)
                            or (f.geom_type == GeomType.PLANE and abs(abs(f.center().Z)-0.004) < 1e-6))

patches = {"ring_wet": ring_wet, "ring_end_top": ring_end_top,
           "ring_end_bottom": ring_end_bot, "spinner_wet": spinner_wet}
for name, faces in patches.items():
    comp = Compound(children=[Face(f.wrapped) for f in faces]) if len(faces) > 1 else faces[0]
    export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-5, angular_tolerance=0.1)
    print(name, len(faces), "area=%.6e" % sum(f.area for f in faces),
          os.path.getsize(f"constant/triSurface/{name}.stl"))
print("total patch area", sum(sum(f.area for f in v) for v in patches.values()), "shape area", oil.area)

# -- cell 4 -------------------------------------------------------------------------
# STEP export in metres, then a coarse uniform gmsh tet mesh just to verify topology and patch tagging
import gmsh
export_step(oil, "oil.step")

gmsh.initialize()
gmsh.open("oil.step")
surfs = gmsh.model.getEntities(2)
info = []
for (d, t) in surfs:
    com = gmsh.model.occ.getCenterOfMass(d, t)
    bb = gmsh.model.getBoundingBox(d, t)
    info.append((t, com, bb, gmsh.model.occ.getMass(d, t)))
    print(t, ["%.4f" % v for v in com], ["%.4f" % v for v in bb], "%.3e" % gmsh.model.occ.getMass(d, t))
print("volumes", gmsh.model.getEntities(3), "mass", gmsh.model.occ.getMass(3, 1))

# -- cell 5 -------------------------------------------------------------------------
# Tag physical surfaces by the same geometric rules, coarse uniform size first.
import numpy as np
groups = {"ring_wet": [], "ring_end_top": [], "ring_end_bottom": [], "spinner_wet": []}
for (d, t) in gmsh.model.getEntities(2):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(d, t)
    rmax = max(abs(xmin), abs(xmax))
    if abs(zmax - zmin) < 1e-9 and abs(zmax - 0.005) < 1e-9:   groups["ring_end_top"].append(t)
    elif abs(zmax - zmin) < 1e-9 and abs(zmin + 0.005) < 1e-9: groups["ring_end_bottom"].append(t)
    elif abs(rmax - 0.023) < 1e-9:                             groups["ring_wet"].append(t)
    else:                                                      groups["spinner_wet"].append(t)
print(groups)

for name, tags in groups.items():
    g = gmsh.model.addPhysicalGroup(2, tags); gmsh.model.setPhysicalName(2, g, name)
gv = gmsh.model.addPhysicalGroup(3, [1]); gmsh.model.setPhysicalName(3, gv, "oil")

gmsh.option.setNumber("Mesh.MeshSizeMin", 8e-4)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.2e-3)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("oil_coarse.msh")
print("tets", len(gmsh.model.mesh.getElementsByType(4)[0]))

# -- cell 6 -------------------------------------------------------------------------
# Tagging failed — gmsh's bounding boxes carry a small tolerance pad, so my 1e-9 tests never matched. 
gmsh.clear(); gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.open("oil.step")
TOL = 1e-5
groups = {"ring_wet": [], "ring_end_top": [], "ring_end_bottom": [], "spinner_wet": []}
for (d, t) in gmsh.model.getEntities(2):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(d, t)
    rmax = max(abs(xmin), abs(xmax))
    if abs(zmax - zmin) < TOL and abs(0.5*(zmax+zmin) - 0.005) < TOL:   groups["ring_end_top"].append(t)
    elif abs(zmax - zmin) < TOL and abs(0.5*(zmax+zmin) + 0.005) < TOL: groups["ring_end_bottom"].append(t)
    elif abs(rmax - 0.023) < TOL:                                       groups["ring_wet"].append(t)
    else:                                                               groups["spinner_wet"].append(t)
for n, tags in groups.items():
    print(n, tags, "area=%.6e" % sum(gmsh.model.occ.getMass(2, t) for t in tags))

# -- cell 7 -------------------------------------------------------------------------
# Areas match the exported STLs exactly. Now the working mesh: a distance-threshold field off the spin
for n, tags in groups.items():
    g = gmsh.model.addPhysicalGroup(2, tags); gmsh.model.setPhysicalName(2, g, n)
gv = gmsh.model.addPhysicalGroup(3, [1]); gmsh.model.setPhysicalName(3, gv, "oil")

f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", groups["spinner_wet"])
gmsh.model.mesh.field.setNumber(f1, "Sampling", 200)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", 3.0e-4)
gmsh.model.mesh.field.setNumber(f2, "SizeMax", 1.2e-3)
gmsh.model.mesh.field.setNumber(f2, "DistMin", 3.0e-4)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 2.5e-3)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("oil.msh")
print("tets", len(gmsh.model.mesh.getElementsByType(4)[0]), "nodes", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 8 -------------------------------------------------------------------------
# Convert to OpenFOAM. Minimal `system` files first.
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
def foam(cls, obj, body):
    return ("FoamFile\n{\n version 2.0;\n format ascii;\n class %s;\n object %s;\n}\n" % (cls, obj)) + body
open("system/controlDict","w").write(foam("dictionary","controlDict",
 "application icoFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n"))
open("system/fvSchemes","w").write(foam("dictionary","fvSchemes",
 "ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\ninterpolationSchemes{default linear;}\nsnGradSchemes{default corrected;}\n"))
open("system/fvSolution","w").write(foam("dictionary","fvSolution","solvers{}\n"))
r = subprocess.run(["gmshToFoam","oil.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-3000:])

# -- cell 10 ------------------------------------------------------------------------
# `Mesh OK`, volume within 0.014 % of exact. All four patches are walls — set the type.
for p in ["ring_wet","ring_end_top","ring_end_bottom","spinner_wet"]:
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/type","-set","wall"],
                   capture_output=True, text=True)
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/physicalType","-remove"],
                   capture_output=True, text=True)
print(subprocess.run(["foamDictionary","constant/polyMesh/boundary","-keywords"],capture_output=True,text=True).stdout)
print(open("constant/polyMesh/boundary").read()[-800:])

# -- cell 11 ------------------------------------------------------------------------
# Now look at it: surface patches coloured, and a half-clip through the mesh to show the 1 mm gaps.
import pyvista as pv
pv.start_xvfb() if False else None
pv.set_plot_theme("document")
import numpy as np
mesh = pv.read("oil.msh") if False else None

cols = {"ring_wet":"red","ring_end_top":"blue","ring_end_bottom":"green","spinner_wet":"orange"}
p = pv.Plotter(off_screen=True, window_size=(1400,600), shape=(1,2))
p.subplot(0,0)
for n,c in cols.items():
    s = pv.read(f"constant/triSurface/{n}.stl")
    p.add_mesh(s.clip(normal="y", origin=(0,0,0)), color=c, show_edges=False, label=n)
p.add_legend(); p.camera_position = "iso"
p.subplot(0,1)
foam = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
grid = rd.read()["internalMesh"]
p.add_mesh(grid.clip(normal="y", origin=(0,0,0)), show_edges=True, line_width=0.2, color="lightgray")
p.camera_position = "xz"; p.camera.zoom(1.3)
p.screenshot("view.png"); print("ok")

# -- cell 12 ------------------------------------------------------------------------
# The PNG didn't come back — display it explicitly.
import matplotlib.pyplot as plt
img = plt.imread("view.png")
fig, ax = plt.subplots(figsize=(14,6)); ax.imshow(img); ax.axis("off"); plt.show()

# -- cell 13 ------------------------------------------------------------------------
# Section view to confirm the 1 mm gaps are meshed through.
sl = grid.slice(normal="y", origin=(0,0,0))
p2 = pv.Plotter(off_screen=True, window_size=(1400,500))
p2.add_mesh(sl, show_edges=True, line_width=0.5, color="white")
p2.view_xz(); p2.camera.zoom(1.0); p2.screenshot("slice.png")
fig, ax = plt.subplots(figsize=(14,5)); ax.imshow(plt.imread("slice.png")); ax.axis("off"); plt.show()

# -- cell 14 ------------------------------------------------------------------------
# Numbers against the request, measured from the built geometry and the mesh itself.
import numpy as np
cc = grid.cell_centers().points
vol = grid.compute_cell_sizes(length=False, area=False).cell_data["Volume"]
r = np.hypot(cc[:,0], cc[:,1]); z = cc[:,2]
rad = (r>0.0220)&(r<0.0230)
endg = (r>0.0150)&(r<0.0220)&(np.abs(z)>0.0040)
for nm, m in [("radial clearance", rad), ("end gaps", endg)]:
    h = (vol[m].mean())**(1/3)
    print(f"{nm}: {m.sum()} cells, mean h = {h*1e3:.3f} mm -> ~{1.0/(h*1e3):.1f} cells across 1.0 mm gap")

for n in ["ring_wet","ring_end_top","ring_end_bottom","spinner_wet"]:
    s = pv.read(f"constant/triSurface/{n}.stl"); b = s.bounds
    rr = np.hypot(s.points[:,0], s.points[:,1])
    print(f"{n}: r in [{rr.min()*1e3:.2f},{rr.max()*1e3:.2f}] mm, z in [{b[4]*1e3:.2f},{b[5]*1e3:.2f}] mm, tris={s.n_cells}")
print("mesh volume %.6e m3 vs exact %.6e m3" % (vol.sum(), oil.volume))
