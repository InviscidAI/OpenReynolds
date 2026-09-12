"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by importing the STEP file and finding out what it actually contains — solid count, units
import subprocess, os
p = "/home/qiuzi/.openreynolds-buildup/work/T6-T6/geometry/customer_part.step"
print(os.getcwd())
print(os.path.getsize(p), "bytes")
with open(p) as f:
    head = [next(f) for _ in range(40)]
print("".join(head))

# -- cell 2 -------------------------------------------------------------------------
# Small file, one MANIFOLD_SOLID_BREP. Let me check the declared unit and import it.
txt = open(p).read()
import re
for kw in ["UNIT","CONVERSION"]:
    for line in txt.splitlines():
        if kw in line:
            print(line)
    print("---")

# -- cell 3 -------------------------------------------------------------------------
# The cell errored so it wasn't accepted. Re-emitting the import self-contained, and probing the cylin
from build123d import *
import numpy as np

STEP_PATH = "/home/qiuzi/.openreynolds-buildup/work/T6-T6/geometry/customer_part.step"
shp = import_step(STEP_PATH)
solid = shp.solids()[0]
print("solids:", len(shp.solids()), "faces:", len(solid.faces()), "valid:", solid.is_valid)
print("bbox:", solid.bounding_box())
print("volume:", solid.volume)

cyl = [f for f in solid.faces() if f.geom_type == GeomType.CYLINDER][0]
from OCP.BRepAdaptor import BRepAdaptor_Surface
ad = BRepAdaptor_Surface(cyl.wrapped)
c = ad.Cylinder()
ax = c.Axis()
print("bore radius:", c.Radius())
print("bore axis loc:", ax.Location().X(), ax.Location().Y(), ax.Location().Z())
print("bore axis dir:", ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())
print("cyl face area:", cyl.area, " -> implied length:", cyl.area/(2*np.pi*c.Radius()))

# -- cell 4 -------------------------------------------------------------------------
# The solid is a 200×100×100 block with a single through-bore, r=20, axis along +x at (y,z)=(50,50). T
BB_L, BB_W, BB_H = 200.0, 100.0, 100.0   # mm, from the measured bounding box
MM_TO_M = 0.001                           # STEP declares a bare LENGTH_UNIT; extents imply mm

block = Pos(BB_L/2, BB_W/2, BB_H/2) * Box(BB_L, BB_W, BB_H)
fluid_mm = block - solid
print("fluid solids:", len(fluid_mm.solids()), "valid:", fluid_mm.solids()[0].is_valid)
print("fluid bbox:", fluid_mm.bounding_box())
import numpy as np
print("fluid volume  : %.3f mm^3" % fluid_mm.volume)
print("pi*r^2*L      : %.3f mm^3" % (np.pi*20.0**2*200.0))
for f in fluid_mm.faces():
    print("  ", f.geom_type, "area=%.2f" % f.area, "center=", tuple(round(v,2) for v in f.center()))

# -- cell 5 -------------------------------------------------------------------------
# Exact match to πr²L. Now scale once to metres and export the fluid STEP that the mesher will read.
fluid = scale(fluid_mm, MM_TO_M)
print("fluid bbox (m):", fluid.bounding_box())
print("fluid volume: %.6e m^3   expected %.6e" % (fluid.volume, np.pi*0.02**2*0.2))
bb = fluid.bounding_box()
print("length x = %.4f m, bore diameter = %.4f m" % (bb.size.X, bb.size.Y))
export_step(fluid, "fluid.step")
import os; print("wrote fluid.step", os.path.getsize("fluid.step"), "bytes")

# -- cell 6 -------------------------------------------------------------------------
# Let me look at the fluid solid before meshing it.
import pyvista as pv
pv.start_xvfb() if False else None
verts, tris = fluid.tessellate(0.0005)
pts = np.array([(v.X, v.Y, v.Z) for v in verts])
faces = np.hstack([np.hstack(([3], t)) for t in tris]).astype(np.int64)
mesh = pv.PolyData(pts, faces)
pl = pv.Plotter(off_screen=True, window_size=(900, 500))
pl.add_mesh(mesh, color="lightblue", show_edges=True, edge_color="gray")
pl.add_axes(); pl.camera_position = "iso"
pl.screenshot("fluid_shape.png")
print("tessellated tris:", len(tris))

# -- cell 7 -------------------------------------------------------------------------
# The screenshot didn't come back; let me check whether it rendered at all and display it explicitly.
import os
from IPython.display import Image, display
print(os.path.exists("fluid_shape.png"), os.path.getsize("fluid_shape.png") if os.path.exists("fluid_shape.png") else None)
if os.path.exists("fluid_shape.png"):
    display(Image("fluid_shape.png"))

# -- cell 8 -------------------------------------------------------------------------
# The fluid is a straight circular duct: L = 0.2 m, bore r = 0.02 m, axis +x at (y,z) = (0.05, 0.05). 
import os, textwrap, numpy as np
from math import sqrt

R  = 0.02    # measured bore radius, m
L  = 0.2     # measured bore length, m
CY, CZ = 0.05, 0.05   # measured bore axis location, m

for d in ("system", "constant/polyMesh", "constant/triSurface"):
    os.makedirs(d, exist_ok=True)

HEAD = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""

open("system/controlDict","w").write(HEAD.format(cls="dictionary",obj="controlDict") + textwrap.dedent("""
application     none;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
"""))

# O-grid cross-section points (y,z), angles 45,135,225,315
P = 0.40*R                # inner square half-width
Q = R/sqrt(2.0)           # outer diagonal points on the circle
ang = [45,135,225,315]
inner = [(CY+P*np.sign(np.cos(np.radians(a))), CZ+P*np.sign(np.sin(np.radians(a)))) for a in ang]
outer = [(CY+R*np.cos(np.radians(a)), CZ+R*np.sin(np.radians(a))) for a in ang]
print("inner corners:", [tuple(round(v,5) for v in c) for c in inner])
print("outer corners:", [tuple(round(v,5) for v in c) for c in outer], " |r-R| =",
      max(abs(np.hypot(y-CY,z-CZ)-R) for y,z in outer))
print("Q check:", round(Q,6))

# -- cell 9 -------------------------------------------------------------------------
# Now the blockMeshDict itself — vertices, 5 blocks, arc edges on the measured circle, and the three p
# exit 0 (0 s)
# FoamFile
# {
# version     2.0;
# format      ascii;
# class       dictionary;
# object      blockMeshDict;
# }
# vertices
# (
# (0.00000000 0.05800000 0.05800000)
NC, NR, NX = 6, 4, 20   # coarse first: core, radial, axial

verts = [(0.0, y, z) for (y,z) in inner] + [(0.0, y, z) for (y,z) in outer] + \
        [(L,   y, z) for (y,z) in inner] + [(L,   y, z) for (y,z) in outer]

core  = [0,1,2,3, 8,9,10,11]
# segment k spans inner/outer corner k (angle A) to k+1 (angle A+90): [in(B),in(A),out(A),out(B)]
sides = []
for k in range(4):
    a, b = k, (k+1) % 4
    sides.append([b, a, 4+a, 4+b, 8+b, 8+a, 12+a, 12+b])

def vs(v): return "    (%.8f %.8f %.8f)" % v
blocks  = ["    hex (%s) (%d %d %d) simpleGrading (1 1 1)" % (" ".join(map(str,core)), NC, NC, NX)]
blocks += ["    hex (%s) (%d %d %d) simpleGrading (1 1 1)" % (" ".join(map(str,s)), NC, NR, NX) for s in sides]

# arc midpoints at 90,180,270,0 deg on the measured circle
arcmid = [(CY+R*np.cos(np.radians(a)), CZ+R*np.sin(np.radians(a))) for a in (90,180,270,0)]
edges = []
for k in range(4):
    a, b = 4+k, 4+((k+1) % 4)
    my, mz = arcmid[k]
    edges.append("    arc %d %d (%.8f %.8f %.8f)" % (a, b, 0.0, my, mz))
    edges.append("    arc %d %d (%.8f %.8f %.8f)" % (a+8, b+8, L, my, mz))

inlet_f  = ["        (3 2 1 0)"] + ["        (%d %d %d %d)" % (s[3], s[2], s[1], s[0]) for s in sides]
outlet_f = ["        (8 9 10 11)"] + ["        (%d %d %d %d)" % (s[4], s[5], s[6], s[7]) for s in sides]
wall_f   = ["        (%d %d %d %d)" % (s[2], s[3], s[7], s[6]) for s in sides]

dict_txt = HEAD.format(cls="dictionary", obj="blockMeshDict") + "\nscale 1;\n\nvertices\n(\n" + \
    "\n".join(vs(v) for v in verts) + "\n);\n\nblocks\n(\n" + "\n".join(blocks) + \
    "\n);\n\nedges\n(\n" + "\n".join(edges) + "\n);\n\nboundary\n(\n" + \
    "    inlet\n    {\n        type patch;\n        faces\n        (\n" + "\n".join(inlet_f) + "\n        );\n    }\n" + \
    "    outlet\n    {\n        type patch;\n        faces\n        (\n" + "\n".join(outlet_f) + "\n        );\n    }\n" + \
    "    walls\n    {\n        type wall;\n        faces\n        (\n" + "\n".join(wall_f) + "\n        );\n    }\n" + \
    ");\n\nmergePatchPairs\n(\n);\n"
open("system/blockMeshDict","w").write(dict_txt)
print(dict_txt[:700])
print("... blocks:", len(blocks), "edges:", len(edges))

# -- cell 10 ------------------------------------------------------------------------
# File is correct — the earlier print was just display interleaving. Run blockMesh.
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-2500:])
print("STDERR", r.stderr[-1500:])

# -- cell 11 ------------------------------------------------------------------------
# Mesh built, bbox matches the fluid exactly. Now checkMesh and compare mesh volume / patch areas with
# user exit 0 (0 s)
# Checking geometry...
# Overall domain bounding box (0 0.03 0.03) (0.2 0.07 0.07)
# Mesh has 3 geometric directions (1 1 1)
# Boundary openness (-8.07098e-17 -6.94722e-19 -1.1e-17) OK.
# Max cell openness = 2.2657e-16 OK.
# Max aspect ratio = 4.60428 OK.
# Minimum face area = 4.58e-06. Max face area = 2.667e-05.  Face are not geometrically closed OK.
# Minimum face area = 4.58e-06. Max face area = 2.667e-05.  Max cell volume = 2.667e-07. Min cell volu
# Max cell volume = 2.667e-07. Min cell volume = 4.58e-08.  Total volume = 0.00025066.
# Total volume = 0.00025066.  Cell determinant (wet) = 3.02
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
i = out.find("Checking geometry")
print(out[i:i+3000])

# -- cell 12 ------------------------------------------------------------------------
# Mesh OK at 2640 cells and the volume matches to 0.27%. Now refine: finer core/radial/axial counts an
NC, NR, NX = 12, 8, 60          # refined
WALL_GRADING = 0.5              # last/first cell size ratio, inner -> wall

blocks  = ["    hex (%s) (%d %d %d) simpleGrading (1 1 1)" % (" ".join(map(str,core)), NC, NC, NX)]
blocks += ["    hex (%s) (%d %d %d) simpleGrading (1 %g 1)" % (" ".join(map(str,s)), NC, NR, NX, WALL_GRADING)
           for s in sides]

dict_txt = HEAD.format(cls="dictionary", obj="blockMeshDict") + "\nscale 1;\n\nvertices\n(\n" + \
    "\n".join(vs(v) for v in verts) + "\n);\n\nblocks\n(\n" + "\n".join(blocks) + \
    "\n);\n\nedges\n(\n" + "\n".join(edges) + "\n);\n\nboundary\n(\n" + \
    "    inlet\n    {\n        type patch;\n        faces\n        (\n" + "\n".join(inlet_f) + "\n        );\n    }\n" + \
    "    outlet\n    {\n        type patch;\n        faces\n        (\n" + "\n".join(outlet_f) + "\n        );\n    }\n" + \
    "    walls\n    {\n        type wall;\n        faces\n        (\n" + "\n".join(wall_f) + "\n        );\n    }\n" + \
    ");\n\nmergePatchPairs\n(\n);\n"
open("system/blockMeshDict","w").write(dict_txt)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[r.stdout.find("Mesh Information"):][:600])

# -- cell 13 ------------------------------------------------------------------------
# checkMesh on the refined mesh, and measure the patch areas via the VTK OpenFOAM reader against πr² a
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print(out[out.find("Checking geometry"):][:1400])

open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam")
rd.enable_all_patch_arrays()
blk = rd.read()
bnd = blk["boundary"]
for name in bnd.keys():
    a = bnd[name].compute_cell_sizes(length=False, area=True, volume=False)["Area"].sum()
    print("%-8s area = %.6e m^2" % (name, a))
print("pi*r^2   = %.6e" % (np.pi*R**2))
print("2*pi*r*L = %.6e" % (2*np.pi*R*L))

# -- cell 14 ------------------------------------------------------------------------
# Patch areas match (inlet −0.29%, walls −0.07%: chordal deficit of the faceted circle). The checkMesh
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
txt = r.stdout
i = txt.find("Checking geometry")
print("found at", i)
print(txt[i:])

# -- cell 15 ------------------------------------------------------------------------
print("rc", r.returncode, "len", len(r.stdout))
print(r.stdout[-2000:])
print("ERR:", r.stderr[-1000:])

# -- cell 16 ------------------------------------------------------------------------
# checkMesh now wants `fvSchemes` (the reader's touch of the case changed what it constructs). Add min
open("system/fvSchemes","w").write(HEAD.format(cls="dictionary",obj="fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open("system/fvSolution","w").write(HEAD.format(cls="dictionary",obj="fvSolution") + """
solvers {}
""")
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
t = r.stdout
print("rc", r.returncode)
print(t[t.find("Checking geometry"):])

# -- cell 17 ------------------------------------------------------------------------
# Volume deficit −0.286% matches the exact inscribed-48-gon value (2π/48)²/6 = 0.2856%, so the wall is
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays()
blk = rd.read()
internal = blk["internalMesh"]
pl = pv.Plotter(off_screen=True, window_size=(1100, 450), shape=(1,2))
pl.subplot(0,0)
pl.add_mesh(internal.extract_surface(), color="lightsteelblue", show_edges=True, line_width=0.4)
pl.add_axes(); pl.camera_position = "iso"
pl.subplot(0,1)
sl = internal.slice(normal=(1,0,0), origin=(0.1,0.05,0.05))
pl.add_mesh(sl, color="white", show_edges=True, line_width=0.8)
pl.view_yz(); pl.add_text("x=0.1 m section", font_size=8)
pl.screenshot("mesh.png")
from IPython.display import Image, display
display(Image("mesh.png"))
print("cells:", internal.n_cells)
