"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# First, inspect the STEP file: number of solids, declared units, validity, bounding box.
import build123d as bd
from build123d import import_step
import subprocess, os

path = "/home/qiuzi/.openreynolds-buildup/work/T5-20260914-164511-cdec/geometry/assembly.step"
os.chdir("/home/qiuzi/.openreynolds-buildup/work/T5-20260914-164511-cdec/t5")

# Peek at the file header for units
out = subprocess.run(["grep","-m","5","-i","SI_UNIT\\|LENGTH_UNIT\\|CONVERSION_BASED_UNIT", path],
                     capture_output=True, text=True)
print(out.stdout[:1500])

shape = import_step(path)
print(type(shape))
if isinstance(shape, bd.Compound):
    print("children:", len(shape.children))
    for c in shape.children:
        print(c, )
solids = shape.solids() if hasattr(shape, "solids") else []
print("solids:", len(solids))
bb = shape.bounding_box()
print("bbox:", bb.min, bb.max, "size:", bb.size)

# -- cell 2 -------------------------------------------------------------------------
# `is_valid` is a property here. Re-run.
for i, s in enumerate(shape.solids()):
    bb = s.bounding_box()
    print(f"solid {i}: vol={s.volume:.4f} m3, size={tuple(round(v,3) for v in bb.size)}, "
          f"valid={s.is_valid}, faces={len(s.faces())}, center={tuple(round(v,3) for v in s.center())}")

# -- cell 3 -------------------------------------------------------------------------
# Check what tessellation/helper methods build123d offers before guessing further.
methods = [m for m in dir(bd.Shape)]
print(methods)
import inspect
print(inspect.signature(bd.Shape.tessellate) if hasattr(bd.Shape,'tessellate') else "no tessellate")

# -- cell 4 -------------------------------------------------------------------------
verts, tris = shape.solids()[0].tessellate(0.5)
print(type(verts[0]), verts[0] if hasattr(verts[0],'to_tuple') else None)
print(verts[0])
print(tris[:3])

# -- cell 5 -------------------------------------------------------------------------
import build123d as bd
for (i, j) in [(0,1),(0,2),(1,2)]:
    inter = shape.solids()[i] & shape.solids()[j]
    if inter is None:
        print(i,j,"no overlap", flush=True)
    else:
        print(i,j, "vols:", [s.volume for s in inter.solids()], flush=True)

# -- cell 6 -------------------------------------------------------------------------
import build123d as bd
bb = shape.bounding_box()
m = 0.01
L = tuple(v + 2*m for v in bb.size)
ext = bd.Pos(bb.min.X - m, bb.min.Y - m, bb.min.Z - m) * bd.Box(L[0], L[1], L[2])
complement = ext - shape
sizes = sorted(((s.volume, s) for s in complement.solids()), key=lambda t: t[0], reverse=True)
print("solids:", [(round(v,1), s.bounding_box().size) for v, s in sizes])

# -- cell 7 -------------------------------------------------------------------------
import build123d as bd
ss = shape.solids()
print([mm for mm in dir(bd) if 'union' in mm.lower() or mm=='add'])

# -- cell 8 -------------------------------------------------------------------------
import build123d as bd
comp = bd.Compound(children=shape.solids())
ext_len = tuple(v + 2*m for v in bb.size)
ext = bd.Pos(bb.min.X - m, bb.min.Y - m, bb.min.Z - m) * bd.Box(ext_len[0], ext_len[1], ext_len[2])
complement = ext - comp
res = sorted(complement.solids(), key=lambda s: s.volume, reverse=True)
print("n solids:", len(res))
for s in res:
    print(round(s.volume,2), tuple(round(x,3) for x in s.bounding_box().size))

# -- cell 9 -------------------------------------------------------------------------
import build123d as bd
ext_len = tuple(v + 2*m for v in bb.size)
comp = ext
for s in shape.solids():
    comp = comp - s
res = sorted(comp.solids(), key=lambda s: s.volume, reverse=True)
print("n solids:", len(res))
for s in res:
    print(round(s.volume,3), tuple(round(x,3) for x in s.bounding_box().size))

# -- cell 10 ------------------------------------------------------------------------
import build123d as bd
test = ext.copy() if hasattr(ext,'copy') else bd.Pos(bb.min.X - m, bb.min.Y - m, bb.min.Z - m) * bd.Box(*(v + 2*m for v in bb.size))
t = test - shape.solids()[2]
print("ext vol:", test.volume, "after minus solid2:", t.volume, "expected:", test.volume - 26455.7)

# -- cell 11 ------------------------------------------------------------------------
# build123d's subtraction on these solids is behaving oddly; check for a defeature API in OCP, then pl
import OCP
import importlib
names = dir(OCP)
print([n for n in names if 'Defeature' in n or 'defeat' in n.lower()])
from OCP import BRepAlgoAPI
print([n for n in dir(OCP.BRepAlgoAPI) if 'Def' in n or 'Feat' in n])

# -- cell 12 ------------------------------------------------------------------------
# `BRepAlgoAPI_Defeaturing` exists — fillet removal is feasible. First fix the boolean: run the cut di
import build123d as bd
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
comp = bd.Compound(children=shape.solids())
ext_box = bd.Pos(bb.min.X - m, bb.min.Y - m, bb.min.Z - m) * bd.Box(*(v + 2*m for v in bb.size))
op = BRepAlgoAPI_Cut(ext_box.wrapped, comp.wrapped)
cut = bd.Solid(op.Shape()) if op.IsDone() else None
if cut is not None:
    ss = bd.Compound(children=[cut]).solids()
    print("cut ok; volumes:")
    for s in sorted(ss, key=lambda s: s.volume, reverse=True):
        print(round(s.volume,3), tuple(round(x,3) for x in s.bounding_box().size))
else:
    print("cut failed")

# -- cell 13 ------------------------------------------------------------------------
# The cut yields one solid — check if it has two shells (outer boundary plus an enclosed inner void, w
shells = cut.shells()
print("n shells:", len(shells))
for s in shells:
    print("shell faces:", len(s.faces()), "area:", s.area, "bbox:", tuple(round(x,2) for x in s.bounding_box().size))

# -- cell 14 ------------------------------------------------------------------------
# No enclosed void — the cavity communicates. Check pairwise containment via OCP Common, and render th
import build123d as bd
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
ss = shape.solids()
for (i,j) in [(0,1),(0,2),(1,2)]:
    op = BRepAlgoAPI_Common(ss[i].wrapped, ss[j].wrapped)
    kind = op.Shape().ShapeType().name if op.IsDone() else "FAIL"
    c = bd.Compound(op.Shape())
    ivols = [s.volume for s in c.solids()]
    print(i,j,kind, ivols)

# -- cell 15 ------------------------------------------------------------------------
# No overlaps confirmed — the assembly encloses an air gap between the parts. Render the complement (b
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
pv.OFF_SCREEN = True
def to_poly(shp, tol):
    verts, tris = shp.tessellate(tol)
    v = np.array([(p.X,p.Y,p.Z) for p in verts])
    f = np.hstack([np.full((len(tris),1),3), np.array(tris, dtype=np.int64)])
    mesh = pv.PolyData(v, f)
    try:
        return mesh.reconstruct_surface()
    except Exception:
        return mesh

plotter = pv.Plotter(off_screen=True)
for i, s in enumerate(shape.solids()):
    plotter.add_mesh(to_poly(s, 0.5), color=["tomato","steelblue","limegreen"][i], opacity=0.6)
# section plane normal to Y through center
plotter.add_mesh(to_poly(cut, 0.5), color="cadetblue", opacity=1.0, show_edges=False)
plotter.view_yz()
img = plotter.screenshot()
plotter.close()
plt.figure(figsize=(8,8)); plt.imshow(img); plt.axis("off"); plt.show()

# -- cell 16 ------------------------------------------------------------------------
# The complement view shows the air region is between duct/chamber/dome solids. I need section views a
import build123d as bd
import numpy as np
for i, s in enumerate(shape.solids()):
    print(f"--- solid {i}")
    for f in s.faces():
        if f.geom_type == "PLANE":
            n = f.normal_at()
            c = f.center()
            print(f"  plane area={f.area:7.2f} normal=({n.X:5.2f},{n.Y:5.2f},{n.Z:5.2f}) center=({c.X:7.2f},{c.Y:7.2f},{c.Z:7.2f})")

# -- cell 17 ------------------------------------------------------------------------
from collections import Counter
for i, s in enumerate(shape.solids()):
    types = Counter(f.geom_type for f in s.faces())
    print(i, types)

# -- cell 18 ------------------------------------------------------------------------
import build123d as bd
from build123d import GeomType
for i, s in enumerate(shape.solids()):
    print(f"--- solid {i}")
    for f in s.faces():
        if f.geom_type == GeomType.PLANE and f.area > 0.5:
            n = f.normal_at()
            c = f.center()
            print(f"  plane area={f.area:7.2f} n=({n.X:5.2f},{n.Y:5.2f},{n.Z:5.2f}) c=({c.X:7.2f},{c.Y:7.2f},{c.Z:7.2f})")

# -- cell 19 ------------------------------------------------------------------------
# Need to actually see the flow passage. Cut the complement with section planes x=0 and z=10 and rende
import pyvista as pv, numpy as np
import matplotlib.pyplot as plt
pv.OFF_SCREEN = True
def to_mesh(shp, tol=0.2):
    verts, tris = shp.tessellate(tol)
    v = np.array([(p.X,p.Y,p.Z) for p in verts])
    f = np.hstack([np.full((len(tris),1),3), np.array(tris, dtype=np.int64)])
    return pv.PolyData(v, f)

img_plot = pv.Plotter(off_screen=True)
m = to_mesh(cut, 0.3)
img_plot.add_mesh(m.clip(normal='x', origin=(0,0,15)), color="cadetblue")
for i, s in enumerate(shape.solids()):
    img_plot.add_mesh(to_mesh(s,0.2), color="indianred", opacity=0.4)
img_plot.view_yz()
img1 = img_plot.screenshot()
img_plot.close()
plt.figure(figsize=(9,9)); plt.imshow(img1); plt.axis("off"); plt.show()
