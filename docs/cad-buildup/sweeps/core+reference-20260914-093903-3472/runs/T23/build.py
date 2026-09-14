"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np
from build123d import Pos, Rot

# Dimensions from the request (mm; scaled to metres before export)
R_HOLLOW, R_BALL, R_HOLE, L_CUBE = 16.0, 15.0, 12.0, 40.0

hollow = bd.Sphere(R_HOLLOW)
ball   = bd.Sphere(R_BALL)
cx = Rot(0, 90, 0) * bd.Cylinder(R_HOLE, L_CUBE)   # hole axis along x, spans -20..20
cy = Rot(90, 0, 0) * bd.Cylinder(R_HOLE, L_CUBE)
cz = bd.Cylinder(R_HOLE, L_CUBE)

water_mm = (hollow + cx + cy + cz) - ball          # water = hollow+holes, ball removed
print("valid", water_mm.is_valid, "solids", len(water_mm.solids()), "vol mm^3", round(water_mm.volume, 2))
print(water_mm.bounding_box())
for i, f in enumerate(water_mm.faces()):
    bc = f.bounding_box().center()
    print(i, f.geom_type, round(f.area, 3), [round(v, 3) for v in bc])

# -- cell 2 -------------------------------------------------------------------------
MM = 0.001
water = water_mm.scale(MM)     # metres from here on
print("vol m^3", water.volume, water.bounding_box())

AX = {"xp": (0, +1), "xn": (0, -1), "yp": (1, +1), "yn": (1, -1), "zp": (2, +1), "zn": (2, -1)}
def side(c):                                   # which face-direction a feature points to
    t = c.to_tuple()
    return max(AX, key=lambda k: AX[k][1] * t[AX[k][0]])

patches = {}
for f in water.faces():
    c = f.bounding_box().center()
    if f.geom_type == bd.GeomType.SPHERE:
        key = "ball" if abs(f.radius - R_BALL*MM) < 1e-7 else "hollow"
    elif f.geom_type == bd.GeomType.CYLINDER:
        key = "hole_" + side(c)
    else:
        key = "open_" + side(c)
    patches.setdefault(key, []).append(f)

for k, fl in sorted(patches.items()):
    rr = sorted({round(x.radius, 6) for x in fl if x.geom_type != bd.GeomType.PLANE})
    print(f"{k:9s} nfaces={len(fl)} area={sum(x.area for x in fl):.6e} radii={rr}")
print("total patches:", len(patches))

# -- cell 3 -------------------------------------------------------------------------
# 14 named patch groups, areas and radii exactly as specified. Now export one STL per patch (metres) p
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)
for k, fl in patches.items():
    bd.export_stl(bd.Compound(children=[bd.Solid(f.wrapped) if False else f for f in fl]),
                  f"constant/triSurface/{k}.stl", tolerance=2e-5, angular_tolerance=0.2)
bd.export_step(water, "water.step")
print(sorted(os.listdir("constant/triSurface")))
print(subprocess.run(["ls","-l","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# Per-face independent tessellation gives huge, non-conformal STLs. Better: triangulate the whole soli
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED

LIN_DEF = 2e-4   # 0.2 mm chord tolerance
BRepMesh_IncrementalMesh(water.wrapped, LIN_DEF, False, 0.35, True)

def face_tris(f):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    trsf = loc.Transformation()
    pts = np.array([[p.X(), p.Y(), p.Z()] for p in
                    (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes()+1))])
    idx = np.array([[t.Value(1)-1, t.Value(2)-1, t.Value(3)-1]
                    for t in (tri.Triangle(i) for i in range(1, tri.NbTriangles()+1))])
    if f.wrapped.Orientation() == TopAbs_REVERSED:
        idx = idx[:, [0, 2, 1]]
    return pts, idx

def write_stl(path, name, chunks):
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for pts, idx in chunks:
            for a, b, c in idx:
                p, q, r = pts[a], pts[b], pts[c]
                n = np.cross(q-p, r-p); ln = np.linalg.norm(n)
                n = n/ln if ln > 0 else np.array([0., 0., 1.])
                fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
                for v in (p, q, r):
                    fh.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")

ntri = 0
for k, fl in patches.items():
    chunks = [face_tris(f) for f in fl]
    ntri += sum(len(i) for _, i in chunks)
    write_stl(f"constant/triSurface/{k}.stl", k, chunks)
print("total triangles", ntri)
print(subprocess.run(["du","-sh","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 5 -------------------------------------------------------------------------
# Triangulation was cached from the earlier `export_stl`. Clean it and re-mesh.
from OCP.BRepTools import BRepTools
BRepTools.Clean_s(water.wrapped)
BRepMesh_IncrementalMesh(water.wrapped, LIN_DEF, False, 0.35, True)
ntri = 0
for k, fl in patches.items():
    chunks = [face_tris(f) for f in fl]
    n = sum(len(i) for _, i in chunks); ntri += n
    write_stl(f"constant/triSurface/{k}.stl", k, chunks)
    print(k, n)
print("total triangles", ntri)
print(subprocess.run(["du","-sh","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
# Refine the tessellation and check the welded union is closed (edge-count test) before meshing.
LIN_DEF2 = 5e-5
BRepTools.Clean_s(water.wrapped)
BRepMesh_IncrementalMesh(water.wrapped, LIN_DEF2, False, 0.2, True)
all_tris, ntri = [], 0
for k, fl in patches.items():
    chunks = [face_tris(f) for f in fl]
    ntri += sum(len(i) for _, i in chunks)
    write_stl(f"constant/triSurface/{k}.stl", k, chunks)
    all_tris += [(p[i], p[j], p[l]) for p, idx in chunks for i, j, l in idx]
print("total triangles", ntri)

from collections import Counter
key = lambda v: tuple(np.round(v, 9))
ec = Counter()
for t in all_tris:
    ks = [key(v) for v in t]
    for a, b in zip(ks, ks[1:]+ks[:1]):
        ec[frozenset((a, b))] += 1
print("edges used once (free):", sum(1 for v in ec.values() if v == 1),
      " used twice:", sum(1 for v in ec.values() if v == 2),
      " other:", sum(1 for v in ec.values() if v > 2))

# -- cell 7 -------------------------------------------------------------------------
bad = [(sorted(e), c) for e, c in ec.items() if c != 2]
for e, c in bad:
    print(c, [tuple(round(x,6) for x in v) for v in e])

# -- cell 8 -------------------------------------------------------------------------
# Two degenerate zero-area triangles at the ball's poles. Filter them out and re-check.
def write_stl(path, name, chunks):          # redefine: drop degenerate triangles
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for pts, idx in chunks:
            for a, b, c in idx:
                p, q, r = pts[a], pts[b], pts[c]
                n = np.cross(q-p, r-p); ln = np.linalg.norm(n)
                if ln < 1e-16:
                    continue
                n = n/ln
                fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
                for v in (p, q, r):
                    fh.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")

all_tris, ntri = [], 0
for k, fl in patches.items():
    chunks = [face_tris(f) for f in fl]
    write_stl(f"constant/triSurface/{k}.stl", k, chunks)
    keep = [(p[i], p[j], p[l]) for p, idx in chunks for i, j, l in idx
            if np.linalg.norm(np.cross(p[j]-p[i], p[l]-p[i])) > 1e-16]
    ntri += len(keep); all_tris += keep
ec = Counter()
for t in all_tris:
    ks = [key(v) for v in t]
    for a, b in zip(ks, ks[1:]+ks[:1]):
        ec[frozenset((a, b))] += 1
print("triangles", ntri, "| free edges", sum(1 for v in ec.values() if v == 1),
      "| >2:", sum(1 for v in ec.values() if v > 2))

# -- cell 9 -------------------------------------------------------------------------
# Meshing now: gmsh on the STEP, surfaces classified into the same 14 groups by area + direction, coar
import gmsh
gmsh.initialize()
gmsh.model.add("water")
gmsh.merge("water.step")
surfs = gmsh.model.getEntities(2)
AREA = {"ball": 2.827433e-3, "hollow": 1.696022e-5, "hole": 6.656880e-4, "open": 4.523893e-4}
groups = {}
for dim, tag in surfs:
    a = gmsh.model.occ.getMass(dim, tag)
    com = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
    bb = np.array(gmsh.model.getBoundingBox(dim, tag)); bc = (bb[:3]+bb[3:])/2
    kind = min(AREA, key=lambda k: abs(AREA[k]-a)/AREA[k])
    name = kind if kind in ("ball", "hollow") else f"{kind}_{side(bd.Vector(*bc))}"
    groups.setdefault(name, []).append(tag)
    print(f"{tag:3d} a={a:.3e} -> {name}")
print(sorted(groups), len(groups))

# -- cell 10 ------------------------------------------------------------------------
for name, tags in groups.items():
    gmsh.model.addPhysicalGroup(2, tags, name=name)
vols = [t for d, t in gmsh.model.getEntities(3)]
gmsh.model.addPhysicalGroup(3, vols, name="internal")

df = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(df, "SurfacesList", groups["ball"])
th = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(th, "InField", df)
gmsh.model.mesh.field.setNumber(th, "SizeMin", 5.0e-4)   # ~2 cells across the 1 mm gap
gmsh.model.mesh.field.setNumber(th, "SizeMax", 2.0e-3)
gmsh.model.mesh.field.setNumber(th, "DistMin", 1.2e-3)
gmsh.model.mesh.field.setNumber(th, "DistMax", 5.0e-3)
gmsh.model.mesh.field.setAsBackgroundMesh(th)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.model.mesh.generate(3)
gmsh.write("water.msh")
print("nodes", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 11 ------------------------------------------------------------------------
# Mesh volume 2.0612e-5 m³ vs exact 2.0659e-5 (−0.23%, faceting). Convert to OpenFOAM.
import shutil, textwrap
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.SaveAll", 0)
gmsh.write("water.msh")
gmsh.finalize()

os.makedirs("system", exist_ok=True)
head = lambda cls, obj: ("FoamFile\n{ version 2.0; format ascii; class %s; object %s; }\n" % (cls, obj))
open("system/controlDict","w").write(head("dictionary","controlDict") + textwrap.dedent("""
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
"""))
open("system/fvSchemes","w").write(head("dictionary","fvSchemes") + "ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution") + "solvers{}\n")
r = subprocess.run(["gmshToFoam","water.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-2500:], r.stderr[-1500:])

# -- cell 12 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-3500:])

# -- cell 13 ------------------------------------------------------------------------
# Re-emitting the render with its imports inside so it belongs to the script, and adding a patch-colou
import pyvista as pv
pv.OFF_SCREEN = True
open("case.foam", "w").close()
rd = pv.OpenFOAMReader("case.foam")
rd.enable_all_patch_arrays()
blk = rd.read()
internal = blk["internalMesh"]
clip = internal.clip(normal="z", origin=(0, 0, 1e-6))
p = pv.Plotter(off_screen=True, window_size=(900, 850))
p.add_mesh(clip, show_edges=True, line_width=0.4, color="lightsteelblue")
p.camera_position = "xy"
p.show(screenshot="clip.png")
print(internal.n_cells, [b for b in blk["boundary"].keys()])

# -- cell 14 ------------------------------------------------------------------------
# Now measure the built geometry from the mesh itself: ball radius, hollow radius, gap, hole radius, a
b = blk["boundary"]
def radii(name, axis=None):
    pts = b[name].points
    return np.linalg.norm(pts, axis=1)
print("ball   r  min/max mm", radii("ball").min()*1e3, radii("ball").max()*1e3)
print("hollow r  min/max mm", radii("hollow").min()*1e3, radii("hollow").max()*1e3)
print("gap (hollow-ball) mm", radii("hollow").min()*1e3 - radii("ball").max()*1e3)
for h in ("hole_xp", "hole_yp", "hole_zp"):
    q = b[h].points; ax = "xyz".index(h[-2])
    perp = np.delete(q, ax, axis=1)
    print(h, "wall radius mm", np.linalg.norm(perp, axis=1).max()*1e3,
          "axial extent mm", q[:, ax].min()*1e3, q[:, ax].max()*1e3)
o = b["open_xp"].points
print("open_xp plane x mm", o[:,0].min()*1e3, o[:,0].max()*1e3,
      "disc radius mm", np.linalg.norm(o[:,1:], axis=1).max()*1e3)
cc = internal.cell_centers().points; rc = np.linalg.norm(cc, axis=1)
print("cells with 15<r<16 mm:", int(((rc>0.015)&(rc<0.016)).sum()))
