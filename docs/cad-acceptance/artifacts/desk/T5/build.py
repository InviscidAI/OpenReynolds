"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os
os.chdir("/work/accept-t5")
print(os.getcwd())
print(os.listdir("."))

from build123d import *

step_path = "/work/.accept/assembly.step"
imported = import_step(step_path)
print(type(imported))
print(imported)

# -- cell 2 -------------------------------------------------------------------------
from build123d import export_stl
export_stl(imported, "/tmp/assembly_preview.stl")
print("exported preview")

# also check children labels
for c in imported.children:
    print(c.label, type(c), len(c.solids()) if hasattr(c,'solids') else None)

# -- cell 3 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None

mesh = pv.read("/tmp/assembly_preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(900,700))
plotter.add_mesh(mesh, color="lightblue", show_edges=False)
plotter.camera_position = 'iso'
plotter.screenshot("/tmp/assembly_view1.png")
print("done")

# -- cell 4 -------------------------------------------------------------------------
import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import importlib
import pyvista as pv
importlib.reload(pv)

pv.OFF_SCREEN = True

mesh = pv.read("/tmp/assembly_preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(900,700))
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.camera_position = 'iso'
plotter.screenshot("/tmp/assembly_view1.png")
print("done")

# -- cell 5 -------------------------------------------------------------------------
from PIL import Image
img = Image.open("/tmp/assembly_view1.png")
print(img.size)
img

# -- cell 6 -------------------------------------------------------------------------
base, emitter, body = [c for c in imported.children]
print(base.label, base.solids()[0].volume)
print(emitter.label, emitter.solids()[0].volume)
print(body.label, body.solids()[0].volume)

# bounding boxes of each
for c in imported.children:
    print(c.label, c.bounding_box())

# -- cell 7 -------------------------------------------------------------------------
# Let's look at cross-sections of the whole assembly at various z-heights to understand internal cavities.
import numpy as np

for z in [0.5, 2, 5, 10, 15, 18, 19.5, 20.5, 22, 25, 28, 30]:
    plane_normal = (0,0,1)
    try:
        section = imported.section(Plane(origin=(0,0,z), normal=(0,0,1)))
        print(z, "faces:", len(section.faces()) if hasattr(section,'faces') else None, "type", type(section))
    except Exception as e:
        print(z, "ERR", e)

# -- cell 8 -------------------------------------------------------------------------
import subprocess
result = subprocess.run(["grep", "-i", "-n", "section", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"], capture_output=True, text=True)
print(result.stdout[:3000])

# -- cell 9 -------------------------------------------------------------------------
import inspect
from build123d import section
print(inspect.signature(section))
print(section.__doc__)

# -- cell 10 ------------------------------------------------------------------------
# Check pairwise interference between the three solids
s_base = base.solids()[0]
s_body = body.solids()[0]
s_emit = emitter.solids()[0]

for (na,sa),(nb,sb) in [(("base",s_base),("body",s_body)), (("body",s_body),("emitter",s_emit)), (("base",s_base),("emitter",s_emit))]:
    inter = sa & sb
    vol = inter.volume if inter is not None else 0.0
    print(na, nb, "intersection volume:", vol)

# -- cell 11 ------------------------------------------------------------------------
import matplotlib.pyplot as plt

zs = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 18.0, 19.5, 21.0, 25.0, 28.0, 30.0]
fig, axes = plt.subplots(3,4, figsize=(16,12))
for ax, z in zip(axes.flat, zs):
    plane = Plane(origin=(0,0,z), z_dir=(0,0,1))
    sk = section(imported, section_by=plane)
    for f in sk.faces():
        try:
            pts = f.outer_wire().vertices()
            xs = [p.X for p in pts]
            ys = [p.Y for p in pts]
            ax.fill(xs, ys, alpha=0.5)
        except Exception as e:
            pass
    ax.set_title(f"z={z}")
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("/tmp/sections.png", dpi=100)
print("saved")

# -- cell 12 ------------------------------------------------------------------------
from IPython.display import Image
Image("/tmp/sections.png")

# -- cell 13 ------------------------------------------------------------------------
# Union of the three solids
fused = s_base + s_body + s_emit
print("num solids in fused:", len(fused.solids()))
print("fused volume:", fused.volume)
print("sum of individual volumes:", s_base.volume + s_body.volume + s_emit.volume)
print("is_valid:", fused.is_valid)

# -- cell 14 ------------------------------------------------------------------------
# investigate fuse odd result - try explicit boolean union via topology
u1 = s_base.fuse(s_body)
print("u1 solids:", len(u1.solids()), "volume:", u1.volume)

u2 = u1.fuse(s_emit)
print("u2 solids:", len(u2.solids()), "volume:", u2.volume)

# -- cell 15 ------------------------------------------------------------------------
print(type(u1))
ub = s_body.fuse(s_emit)
print("body+emit solids:", len(ub.solids()), "volume:", ub.volume, "expected", s_body.volume+s_emit.volume)

ua = s_base.fuse(s_emit)
print("base+emit solids:", len(ua.solids()), "volume:", ua.volume, "expected", s_base.volume+s_emit.volume)

# -- cell 16 ------------------------------------------------------------------------
u3 = ub.fuse(s_base)
print("(body+emit)+base solids:", len(u3.solids()), "volume:", u3.volume, "expected:", s_base.volume+s_body.volume+s_emit.volume)

# -- cell 17 ------------------------------------------------------------------------
# Build overall bounding box slightly padded, subtract the union of the three solids,
# and look at what disjoint air regions result (to search for an enclosed internal cavity).
bb = imported.bounding_box()
pad = 2.0
box = Box(bb.size.X + 2*pad, bb.size.Y + 2*pad, bb.size.Z + 2*pad)
box = Pos(bb.center()) * box

air = box - s_base - s_body - s_emit
print("num solids in air:", len(air.solids()))
for i, s in enumerate(air.solids()):
    print(i, "volume:", s.volume, "bbox:", s.bounding_box())

# -- cell 18 ------------------------------------------------------------------------
# Re-verify without padding, using exact bbox, and check solid count / volumes precisely.
box_tight = Pos(bb.center()) * Box(bb.size.X, bb.size.Y, bb.size.Z)
air_tight = box_tight - s_base - s_body - s_emit
print("num solids (tight box):", len(air_tight.solids()))
for i, s in enumerate(air_tight.solids()):
    print(i, "volume:", s.volume)

# Also check shell count on each original solid (does any part enclose an internal void itself?)
for name, s in [("base", s_base), ("body", s_body), ("emit", s_emit)]:
    shells = s.shells()
    print(name, "num shells:", len(shells))

# -- cell 19 ------------------------------------------------------------------------
with open("/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/prep/README.md") as f:
    print(f.read())

# -- cell 20 ------------------------------------------------------------------------
with open("/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/prep/README.md") as f:
    content = f.read()
print(content[2000:6500])

# -- cell 21 ------------------------------------------------------------------------
print(content[6500:9500])

# -- cell 22 ------------------------------------------------------------------------
print(content[9500:13000])

# -- cell 23 ------------------------------------------------------------------------
# Repair the STEP using gmsh's healShapes recipe (sewFaces=False, as the README found
# sewing is what silently destroys a multi-solid assembly), then checkpoint to a file
# so later cells don't redo this. Units: the file is unmeasured yet -- check the
# bounding box after import (already ~54 mm across) confirms millimetres, so we tell
# gmsh the source unit explicitly rather than guess after the fact.
import gmsh, os

os.makedirs("prep", exist_ok=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setString("Geometry.OCCTargetUnit", "M")   # STEP declares mm -> convert to m on import
gmsh.model.add("prep")
gmsh.model.occ.importShapes("/work/.accept/assembly.step")
gmsh.model.occ.synchronize()
before = len(gmsh.model.getEntities(3))

gmsh.model.occ.healShapes(
    fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
    sewFaces=False,
    makeSolids=True,
)
gmsh.model.occ.synchronize()
after = len(gmsh.model.getEntities(3))
nfaces = len(gmsh.model.getEntities(2))
print(f"solids {before} -> {after}, surfaces {nfaces}")

gmsh.write("prep/healed.step")
gmsh.finalize()

# -- cell 24 ------------------------------------------------------------------------
# Check what the healed.step file actually contains dimensionally, since the README
# warns OCCTargetUnit="M" can leave the writer also emitting metre-scale numbers under
# a millimetre declaration -- a hazard that survives gmsh.finalize().
part_check = import_step("prep/healed.step")
print("num solids:", len(part_check.solids()))
print("bbox:", part_check.bounding_box())

# -- cell 25 ------------------------------------------------------------------------
# Load the checkpointed, repaired STEP and convert to metres (file numbers are mm).
part = import_step("prep/healed.step").scale(1e-3)
print(len(part.solids()), "solids")
print("volume (m^3):", part.volume)
print("bbox (m):", part.bounding_box())

# -- cell 26 ------------------------------------------------------------------------
from OCP.BRepAdaptor import BRepAdaptor_Surface

cyl_radii = []
tor_radii = []
for face in part.faces():
    surf = BRepAdaptor_Surface(face.wrapped)
    kind = str(surf.GetType()).rsplit("_", 1)[-1]
    if kind == "Cylinder":
        cyl_radii.append(surf.Cylinder().Radius())
    elif kind == "Torus":
        tor_radii.append(surf.Torus().MinorRadius())

import numpy as np
cyl_radii = np.array(sorted(cyl_radii))
tor_radii = np.array(sorted(tor_radii))
print("num cylindrical faces:", len(cyl_radii))
print("cyl radii (mm):", np.round(cyl_radii*1000, 4))
print("num toroidal faces:", len(tor_radii))
print("tor minor radii (mm):", np.round(tor_radii*1000, 4))
