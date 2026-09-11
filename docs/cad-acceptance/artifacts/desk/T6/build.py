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
os.chdir("/work/accept-t6")
print(os.getcwd())
print(os.listdir("."))
print(os.path.exists("/work/.accept/customer_part.step"))

# -- cell 2 -------------------------------------------------------------------------
import subprocess
# Cheap look at the STEP file header before importing - unit declaration, etc.
result = subprocess.run(["grep", "-a", "-i", "-E", "FILE_SCHEMA|FILE_NAME|FILE_DESCRIPTION|SI_UNIT|CONVERSION_BASED_UNIT|LENGTH_UNIT|NAMED_UNIT", "/work/.accept/customer_part.step"],
                         capture_output=True, text=True)
print(result.stdout[:3000])
print("----STDERR----")
print(result.stderr[:1000])

# -- cell 3 -------------------------------------------------------------------------
import subprocess
result = subprocess.run(["grep", "-a", "-n", "-E", "#426|#42[0-9]|CONVERSION_BASED_UNIT|MEASURE_WITH_UNIT|LENGTH_MEASURE", "/work/.accept/customer_part.step"],
                         capture_output=True, text=True)
print(result.stdout[:3000])

# -- cell 4 -------------------------------------------------------------------------
import subprocess
result = subprocess.run(["grep", "-a", "-n", "SI_UNIT", "/work/.accept/customer_part.step"],
                         capture_output=True, text=True)
print(result.stdout)

# -- cell 5 -------------------------------------------------------------------------
from build123d import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import display

STEP_PATH = "/work/.accept/customer_part.step"
imported = import_step(STEP_PATH)

solids = imported.solids()
bb = imported.bounding_box()
print("num solids:", len(solids))
print("bbox min", bb.min, "max", bb.max, "size", bb.size)
print("is_valid:", imported.is_valid)
print("volume:", imported.volume)
print("area:", imported.area)

verts, tris = imported.tessellate(tolerance=0.5, angular_tolerance=0.3)
verts_np = np.array([(v.X, v.Y, v.Z) for v in verts])
tris_np = np.array(tris)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts_np[tris_np], alpha=0.6, edgecolor='k', linewidths=0.1)
ax.add_collection3d(mesh)
ax.set_xlim(verts_np[:, 0].min(), verts_np[:, 0].max())
ax.set_ylim(verts_np[:, 1].min(), verts_np[:, 1].max())
ax.set_zlim(verts_np[:, 2].min(), verts_np[:, 2].max())
ax.set_box_aspect((np.ptp(verts_np[:, 0]), np.ptp(verts_np[:, 1]), np.ptp(verts_np[:, 2])))
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title("Imported customer_part.step (raw declared units, unscaled)")
display(fig)

# -- cell 6 -------------------------------------------------------------------------
from build123d import *
import numpy as np

# Inspect faces of the imported solid to understand the bore geometry:
# is there a cylindrical face, what is its axis/radius, and are there planar
# end-caps at x=0 and x=200 (i.e. is the bore open at both ends)?
faces = imported.faces()
print("total faces:", len(faces))

for f in faces:
    surf_type = f.geom_type
    print(surf_type, "area=%.2f" % f.area, "center=", tuple(round(c,2) for c in f.center()))

# -- cell 7 -------------------------------------------------------------------------
from build123d import *

# Get precise cylinder geometry (axis position, direction, radius) for the bore face
faces = imported.faces()
cyl_face = [f for f in faces if f.geom_type == GeomType.CYLINDER][0]

# build123d Face wraps OCP; get the underlying surface via topods / BRep tools
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface

adaptor = BRepAdaptor_Surface(cyl_face.wrapped)
cyl = adaptor.Cylinder()
loc = cyl.Location()
ax = cyl.Axis()
print("radius:", cyl.Radius())
print("axis location:", loc.X(), loc.Y(), loc.Z())
print("axis direction:", ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z())

# Also confirm end faces (planes with holes) are at x=0 and x=200
end_faces = [f for f in faces if f.geom_type == GeomType.PLANE and f.area < 9000]
for f in end_faces:
    print("end face center:", f.center(), "area:", f.area, "normal:", f.normal_at())

# -- cell 8 -------------------------------------------------------------------------
from build123d import *

# The block's bounding box exactly bounds a rectangular stock, and the solid
# is that stock minus a straight cylindrical bore (radius 20, axis along X at
# y=50, z=50, from x=0 to x=200), confirmed above via BRepAdaptor_Surface.
# The fluid domain is that missing cylinder: (bounding box) - (customer solid).
BBOX_L, BBOX_W, BBOX_H = 200.0, 100.0, 100.0  # raw units as declared in the file
stock = Pos(BBOX_L/2, BBOX_W/2, BBOX_H/2) * Box(BBOX_L, BBOX_W, BBOX_H)

fluid_raw = stock - imported
print("fluid solids:", len(fluid_raw.solids()))
print("fluid volume:", fluid_raw.volume)

import math
expected_vol = math.pi * 20.0**2 * 200.0
print("expected cylinder volume:", expected_vol)
print("bbox of fluid:", fluid_raw.bounding_box().min, fluid_raw.bounding_box().max)

# -- cell 9 -------------------------------------------------------------------------
import subprocess
print(subprocess.run(["cat", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/prep/README.md"],
                      capture_output=True, text=True).stdout)

# -- cell 10 ------------------------------------------------------------------------
import subprocess
result = subprocess.run(["grep", "-n", "-A3", "def scale", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"],
                         capture_output=True, text=True)
print(result.stdout[:2000])

# -- cell 11 ------------------------------------------------------------------------
import subprocess
result = subprocess.run(["grep", "-n", "-i", "scale", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"],
                         capture_output=True, text=True)
print(result.stdout[:3000])

# -- cell 12 ------------------------------------------------------------------------
from build123d import *

# Scale the fluid domain from the file's raw (millimetre) units to metres.
# The bounding box was 200x100x100 raw units; as millimetres that is a
# sensible 200mm x 100mm x 100mm block with a 40mm-diameter bore -- a
# plausible bench-top part. As metres it would be a 200m block, which is not.
MM_TO_M = 0.001
fluid_m = scale(fluid_raw, by=MM_TO_M)

bb = fluid_m.bounding_box()
print("fluid bbox (m):", bb.min, bb.max, "size:", bb.size)
print("fluid volume (m^3):", fluid_m.volume)

import math
expected_vol_m3 = math.pi * 0.02**2 * 0.2
print("expected cylinder volume (m^3):", expected_vol_m3)
print("num solids:", len(fluid_m.solids()))

# -- cell 13 ------------------------------------------------------------------------
# Scale confirmed: fluid domain is 0.2 m long, 0.04 m diameter bore (radius 0.02 m), volume matches a 
import os
os.makedirs("prep", exist_ok=True)
export_step(fluid_m, "prep/fluid.step")
print("wrote prep/fluid.step", os.path.getsize("prep/fluid.step"), "bytes")

# -- cell 14 ------------------------------------------------------------------------
from build123d import *

# Work from the checkpointed file, not the live session object, per the prep recipe.
fluid = import_step("prep/fluid.step")
print("solids:", len(fluid.solids()), "faces:", len(fluid.faces()))

faces = fluid.faces()
X_MIN, X_MAX = 0.0, 0.2
TOL = 1e-6

inlet  = faces.filter_by_position(Axis.X, X_MIN - TOL, X_MIN + TOL)
outlet = faces.filter_by_position(Axis.X, X_MAX - TOL, X_MAX + TOL)
walls  = [f for f in faces if f not in list(inlet) + list(outlet)]

print("inlet faces:", len(inlet), [f.geom_type for f in inlet])
print("outlet faces:", len(outlet), [f.geom_type for f in outlet])
print("wall faces:", len(walls), [f.geom_type for f in walls])

for name, group in [("inlet", inlet), ("outlet", outlet), ("walls", walls)]:
    for f in group:
        c = f.center(CenterOf.MASS)
        print(name, f.geom_type, "area=%.6g" % f.area, "centroid=", tuple(round(v,5) for v in c))

# -- cell 15 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["cat", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/README.md"],
                      capture_output=True, text=True).stdout)

# -- cell 16 ------------------------------------------------------------------------
import subprocess, sys
result = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/domain_probe.py",
     "prep/fluid.step", "--suggest", "--json"],
    capture_output=True, text=True)
print("STDOUT:", result.stdout[:3000])
print("STDERR:", result.stderr[:2000])

# -- cell 17 ------------------------------------------------------------------------
# The checkpoint STEP was written without declaring its unit as metres, so gmsh (which honours the STE
from build123d import *

export_step(fluid_m, "prep/fluid.step", unit=Unit.M)

# Reload and confirm the checkpoint now declares/reads back consistently in metres.
fluid = import_step("prep/fluid.step")
bb = fluid.bounding_box()
print("reloaded bbox (m):", bb.min, bb.max, "size:", bb.size)
print("num solids:", len(fluid.solids()), "num faces:", len(fluid.faces()))

faces = fluid.faces()
X_MIN, X_MAX = 0.0, 0.2
TOL = 1e-6
inlet  = faces.filter_by_position(Axis.X, X_MIN - TOL, X_MIN + TOL)
outlet = faces.filter_by_position(Axis.X, X_MAX - TOL, X_MAX + TOL)
walls  = [f for f in faces if f not in list(inlet) + list(outlet)]

for name, group in [("inlet", inlet), ("outlet", outlet), ("walls", walls)]:
    for f in group:
        c = f.center(CenterOf.MASS)
        print(name, f.geom_type, "area=%.6g" % f.area, "centroid=", tuple(round(v,6) for v in c))

# -- cell 18 ------------------------------------------------------------------------
# Let me stop guessing between tools and just read the actual STEP file bytes to see what raw coordina
import subprocess
result = subprocess.run(["grep", "-a", "-n", "-E", "CARTESIAN_POINT|SI_UNIT|LENGTH_UNIT|CONVERSION_BASED", "prep/fluid.step"],
                         capture_output=True, text=True)
print(result.stdout)

# -- cell 19 ------------------------------------------------------------------------
# The reload confusion is now clear: build123d's `import_step` returns raw stored numbers unconverted 
from build123d import *

faces = fluid_m.faces()
print("num faces:", len(faces))

X_MIN, X_MAX = 0.0, 0.2
TOL = 1e-6
inlet  = faces.filter_by_position(Axis.X, X_MIN - TOL, X_MIN + TOL)
outlet = faces.filter_by_position(Axis.X, X_MAX - TOL, X_MAX + TOL)
walls  = [f for f in faces if f not in list(inlet) + list(outlet)]

for name, group in [("inlet", inlet), ("outlet", outlet), ("walls", walls)]:
    for f in group:
        c = f.center(CenterOf.MASS)
        print(name, f.geom_type, "area(m^2)=%.6g" % f.area, "centroid(m)=", tuple(round(v,6) for v in c))

# -- cell 20 ------------------------------------------------------------------------
# Good — inlet, outlet, wall centroids all in metres as expected. Now export the patches to STL via gm
import sys
sys.path.insert(0, "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox")
import cad_convert

key = lambda fs: [tuple(f.center(CenterOf.MASS)) for f in fs]

CLMAX_M = 0.006  # metres, coarse first pass: ~21 facets around the 0.04 m diameter bore

report = cad_convert.export_patches(
    "prep/fluid.step",
    {"inlet":  {"role": "inlet",  "faces": key(inlet)},
     "outlet": {"role": "outlet", "faces": key(outlet)},
     "walls":  {"role": "wall",   "faces": key(walls)}},
    "constant/triSurface",
    clmax=CLMAX_M,
    location_in_mesh=[0.1, 0.05, 0.05],
)
print(cad_convert.render(report))

# -- cell 21 ------------------------------------------------------------------------
# Bounding box `0.2 x 0.0399... x 0.04 m` matches the expected physical size (length 0.2 m, bore diame
import subprocess
result = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/cad_audit.py",
     "constant/triSurface", "--json"],
    capture_output=True, text=True)
print("STDOUT:", result.stdout[:4000])
print("STDERR:", result.stderr[:2000])

# -- cell 22 ------------------------------------------------------------------------
# Surface is closed, manifold, consistently oriented, disjoint patches. Now verify the `locationInMesh
import subprocess
result = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/domain_probe.py",
     "constant/triSurface", "--point", "0.1", "0.05", "0.05", "--json"],
    capture_output=True, text=True)
print("STDOUT:", result.stdout[:3000])
print("STDERR:", result.stderr[:2000])

# -- cell 23 ------------------------------------------------------------------------
# Location confirmed inside the fluid with 0.0197641 m clearance in a passage 0.0395544 m wide (matche
import subprocess, os

os.makedirs("system", exist_ok=True)
r = subprocess.run(["cp",
    "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/blockMeshDict",
    "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/surfaceFeatureExtractDict",
    "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/snappyHexMeshDict",
    "system/"], capture_output=True, text=True)
print(r.returncode, r.stdout, r.stderr)
print(os.listdir("system"))

# -- cell 24 ------------------------------------------------------------------------
import subprocess
for f in ["system/blockMeshDict", "system/surfaceFeatureExtractDict", "system/snappyHexMeshDict"]:
    print("="*20, f, "="*20)
    print(open(f).read())
