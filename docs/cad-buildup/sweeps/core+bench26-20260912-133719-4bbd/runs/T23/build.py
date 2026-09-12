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

CUBE = 0.040          # cube edge [m]
R_HOLLOW = 0.016      # spherical hollow radius
R_BALL   = 0.015      # ball radius
R_HOLE   = 0.012      # face hole radius

hollow = Sphere(R_HOLLOW)
ball   = Sphere(R_BALL)
cyl_z  = Cylinder(R_HOLE, CUBE)                      # axis z, -20..+20 mm
cyl_x  = Rot(0, 90, 0) * Cylinder(R_HOLE, CUBE)
cyl_y  = Rot(90, 0, 0) * Cylinder(R_HOLE, CUBE)

water = (hollow + cyl_x + cyl_y + cyl_z) - ball
print("volume [mm^3]", water.volume*1e9, " expected ~20729")
print("bbox [mm]", [round(v*1000,3) for v in water.bounding_box().min], [round(v*1000,3) for v in water.bounding_box().max])
print("faces", len(water.faces()), "solids", len(water.solids()))

# -- cell 2 -------------------------------------------------------------------------
# Solid is one piece, volume matches the hand estimate (the 70 mm³ deficit is the x–y cylinder mutual 
for i, f in enumerate(water.faces()):
    c = f.center()
    print(i, str(f.geom_type), "area_mm2=%.1f"%(f.area*1e6),
          "center_mm=(%.2f,%.2f,%.2f)"%(c.X*1e3,c.Y*1e3,c.Z*1e3),
          "|c|=%.2f"%(c.length*1e3), "n=%s"%(f.normal_at(),))

# -- cell 3 -------------------------------------------------------------------------
# Faces classify cleanly by surface identity: 8 spherical patches at |p|=16 mm (the hollow wall surviv
os.makedirs("constant/triSurface", exist_ok=True)
AX = {0:"x", 1:"y", 2:"z"}

def classify(f):
    c = f.center(); n = f.normal_at(c)
    if f.geom_type == GeomType.SPHERE:
        return "ball" if abs(c.length - R_BALL) < 1e-6 else "hollow"
    v = np.array([c.X, c.Y, c.Z]); nv = np.array([n.X, n.Y, n.Z])
    if f.geom_type == GeomType.PLANE:
        i = int(np.argmax(abs(nv)))
        return "opening_%s%s" % (AX[i], "p" if nv[i] > 0 else "m")
    # cylinder: axis is the coordinate axis orthogonal to the radial normal
    w = v * (abs(nv) < 0.5); i = int(np.argmax(abs(w)))
    return "hole_%s%s" % (AX[i], "p" if w[i] > 0 else "m")

groups = {}
for f in water.faces():
    groups.setdefault(classify(f), []).append(f)
for name, fs in sorted(groups.items()):
    print(name, len(fs), "area_mm2=%.1f" % (sum(x.area for x in fs) * 1e6))
    export_stl(Compound(children=fs), f"constant/triSurface/{name}.stl", tolerance=2e-5, angular_tolerance=0.15)
print(sorted(os.listdir("constant/triSurface")))

# -- cell 4 -------------------------------------------------------------------------
# `pv` was bound only in the cell that errored, so I re-emit the render self-contained.
import pyvista as pv
export_step(water, "water.step")
pl = pv.Plotter(off_screen=True, window_size=(900, 700))
for name in sorted(groups):
    m = pv.read(f"constant/triSurface/{name}.stl").clip(normal="y", origin=(0, 0, 0))
    if m.n_points == 0: continue
    c = {"ball": "red", "hollow": "blue"}.get(name, "green" if name.startswith("hole") else "orange")
    pl.add_mesh(m, color=c, show_edges=True, line_width=1)
pl.camera_position = "yz"; pl.camera.azimuth = 35; pl.camera.elevation = 20
pl.screenshot("look1.png"); print("done")

# -- cell 5 -------------------------------------------------------------------------
# Screenshot written but not shown; display it.
from IPython.display import Image, display
display(Image("look1.png"))

# -- cell 6 -------------------------------------------------------------------------
# Shape reads correctly (red ball inside, green hole bores). Now import the STEP into gmsh and classif
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("cage")
gmsh.model.occ.importShapes("water.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
print(len(surfs), "surfaces; volumes:", gmsh.model.getEntities(3))
for d, t in surfs:
    a = gmsh.model.occ.getMass(2, t) * 1e6
    c = np.array(gmsh.model.occ.getCenterOfMass(2, t)) * 1e3
    print(t, gmsh.model.getType(2, t), "A=%.1f" % a, "COM=(%.2f,%.2f,%.2f)" % tuple(c))

# -- cell 7 -------------------------------------------------------------------------
# Classification rule is clear. Set physical groups, a distance-based size field refining the 1 mm she
LC_GAP, LC_MAX = 0.5e-3, 2.0e-3          # coarse pass

def gname(t):
    a = gmsh.model.occ.getMass(2, t) * 1e6
    c = np.array(gmsh.model.occ.getCenterOfMass(2, t))
    if a > 2000: return "ball"
    if a < 100:  return "hollow"
    i = int(np.argmax(abs(c))); s = "p" if c[i] > 0 else "m"
    return ("hole_" if a > 500 else "opening_") + AX[i] + s

pg = {}
for d, t in gmsh.model.getEntities(2):
    pg.setdefault(gname(t), []).append(t)
for name, tags in sorted(pg.items()):
    gmsh.model.addPhysicalGroup(2, tags, name=name)
gmsh.model.addPhysicalGroup(3, [1], name="water")
print({k: len(v) for k, v in sorted(pg.items())})

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", pg["ball"] + pg["hollow"])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", LC_GAP)
gmsh.model.mesh.field.setNumber(2, "SizeMax", LC_MAX)
gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0e-3)
gmsh.model.mesh.field.setNumber(2, "DistMax", 6.0e-3)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("water.msh")
print("nodes", len(gmsh.model.mesh.getNodes()[0]), "tets", len(gmsh.model.mesh.getElementsByType(4)[0]))

# -- cell 8 -------------------------------------------------------------------------
# Convert to OpenFOAM. Need a minimal case skeleton for the utilities first.
os.makedirs("system", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr % "controlDict" + """
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;
runTimeModifiable true;
""")
open("system/fvSchemes","w").write(hdr % "fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr % "fvSolution" + "solvers{} \n")
r = subprocess.run(["gmshToFoam","water.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-3000:])

# -- cell 10 ------------------------------------------------------------------------
# Mesh OK; all 38368 boundary faces are in the 14 named patches (the `".*"` row is the group sum, defa
def nodes_of(name):
    tags = [t for (d, t) in gmsh.model.getEntities(2) if gname(t) == name]
    P = [gmsh.model.mesh.getNodes(2, t, includeBoundary=True)[1].reshape(-1, 3) for t in tags]
    return np.vstack(P) * 1e3   # mm

b, h = nodes_of("ball"), nodes_of("hollow")
print("ball radius  mm: %.4f..%.4f  (asked 15)" % (np.linalg.norm(b,axis=1).min(), np.linalg.norm(b,axis=1).max()))
print("hollow radius mm: %.4f..%.4f (asked 16)" % (np.linalg.norm(h,axis=1).min(), np.linalg.norm(h,axis=1).max()))
print("shell gap mm: %.4f (asked 1)" % (np.linalg.norm(h,axis=1).mean()-np.linalg.norm(b,axis=1).mean()))
for ax,i in (("x",0),("y",1),("z",2)):
    P = nodes_of("hole_%sp"%ax); j=[k for k in range(3) if k!=i]
    print("hole_%sp: bore radius mm %.4f..%.4f (asked 12), axial extent %.3f..%.3f"%(ax,
        np.linalg.norm(P[:,j],axis=1).min(), np.linalg.norm(P[:,j],axis=1).max(), P[:,i].min(), P[:,i].max()))
    Q = nodes_of("opening_%sp"%ax)
    print("opening_%sp: plane %s=%.4f..%.4f (asked 20), max radius %.4f (asked 12)"%(ax,ax,
        Q[:,i].min(), Q[:,i].max(), np.linalg.norm(Q[:,j],axis=1).max()))

# -- cell 11 ------------------------------------------------------------------------
# All named dimensions exact. Refine so the 1 mm shell gets ~3 cells across (0.30 mm) and regenerate.
LC_GAP_F, LC_MAX_F = 0.30e-3, 1.5e-3      # final pass: ~3.3 cells across the 1 mm shell
gmsh.model.mesh.field.setNumber(2, "SizeMin", LC_GAP_F)
gmsh.model.mesh.field.setNumber(2, "SizeMax", LC_MAX_F)
gmsh.model.mesh.field.setNumber(2, "DistMin", 1.2e-3)
gmsh.model.mesh.field.setNumber(2, "DistMax", 5.0e-3)
gmsh.model.mesh.clear()
gmsh.model.mesh.generate(3)
gmsh.write("water.msh")
print("nodes", len(gmsh.model.mesh.getNodes()[0]), "tets", len(gmsh.model.mesh.getElementsByType(4)[0]))

# -- cell 12 ------------------------------------------------------------------------
subprocess.run(["rm","-rf","constant/polyMesh"], check=True)
r = subprocess.run(["gmshToFoam","water.msh"], capture_output=True, text=True); print("gmshToFoam", r.returncode)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1800:])

# -- cell 13 ------------------------------------------------------------------------
# Mesh OK. Verify the shell is actually resolved (cell size in the gap) and take a final look at a cli
nt, ec = gmsh.model.mesh.getElementsByType(4)
crd = gmsh.model.mesh.getNodes()[1].reshape(-1,3); tag2i = {t:i for i,t in enumerate(gmsh.model.mesh.getNodes()[0])}
idx = np.array([tag2i[t] for t in ec]).reshape(-1,4)
cen = crd[idx].mean(axis=1)*1e3; r = np.linalg.norm(cen,axis=1)
sel = (r>15)&(r<16)
vol = np.abs(np.einsum('ij,ij->i', np.cross(crd[idx][:,1]-crd[idx][:,0], crd[idx][:,2]-crd[idx][:,0]), crd[idx][:,3]-crd[idx][:,0]))/6*1e9
h = (vol[sel].mean()*6/np.sqrt(2))**(1/3)
print("cells in 1 mm shell:", sel.sum(), " mean edge length mm %.3f -> ~%.1f cells across gap" % (h, 1.0/h))
m = pv.read("water.msh") if False else None
pl = pv.Plotter(off_screen=True, window_size=(900,700))
for name in sorted(groups):
    s = pv.read(f"constant/triSurface/{name}.stl").clip(normal="y", origin=(0,0,0))
    if s.n_points==0: continue
    pl.add_mesh(s, color={"ball":"red","hollow":"blue"}.get(name,"green" if name.startswith("hole") else "orange"), lighting=False)
pl.camera_position="yz"; pl.camera.azimuth=35; pl.camera.elevation=20
pl.screenshot("look2.png"); display(Image("look2.png"))
