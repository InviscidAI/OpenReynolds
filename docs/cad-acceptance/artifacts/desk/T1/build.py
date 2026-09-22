"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox"
print(subprocess.run(["grep", "-n", "-i", "-A", "8", "class CenterArc", f"{toolbox}/b123d_api.md"], capture_output=True, text=True).stdout)
print(subprocess.run(["grep", "-n", "-i", "-A", "8", "class RadiusArc", f"{toolbox}/b123d_api.md"], capture_output=True, text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
print(subprocess.run(["grep", "-n", "-i", "CenterArc", f"{toolbox}/b123d_api.md"], capture_output=True, text=True).stdout)
print(subprocess.run(["grep", "-n", "-i", "RadiusArc", f"{toolbox}/b123d_api.md"], capture_output=True, text=True).stdout)

# -- cell 3 -------------------------------------------------------------------------
from build123d import *
import math

# Parameters (mm as given in the request; will convert to metres at export)
WIDTH_MM = 10.0          # passage width
LEG_LEN_MM = 120.0       # length of each straight leg
R_MID_MM = 15.0          # centreline bend radius
R_IN_MM  = R_MID_MM - WIDTH_MM/2   # 10 mm inner wall radius
R_OUT_MM = R_MID_MM + WIDTH_MM/2   # 20 mm outer wall radius

BEND_CENTER = (0.0, R_MID_MM)   # (0, 15)

e1 = Line((-LEG_LEN_MM, -WIDTH_MM/2), (0.0, -WIDTH_MM/2))          # outer, lower leg
e2 = CenterArc(BEND_CENTER, R_OUT_MM, -90, 180)                     # outer bend
e3 = Line((0.0, WIDTH_MM/2 + R_MID_MM), (-LEG_LEN_MM, WIDTH_MM/2 + R_MID_MM))  # outer, upper leg
e4 = Line((-LEG_LEN_MM, WIDTH_MM/2 + R_MID_MM), (-LEG_LEN_MM, R_MID_MM + WIDTH_MM/2 - WIDTH_MM))  # placeholder, fixed below

print("e1", e1 @ 0, e1 @ 1)
print("e2", e2 @ 0, e2 @ 1)
print("e3", e3 @ 0, e3 @ 1)

# -- cell 4 -------------------------------------------------------------------------
from build123d import *

# Parameters (millimetres, per the request; converted to metres only at STL export)
WIDTH_MM     = 10.0     # passage width, constant
LEG_LEN_MM   = 120.0    # length of each straight leg
R_MID_MM     = 15.0     # centreline bend radius
R_IN_MM      = R_MID_MM - WIDTH_MM/2   # inner wall radius of bend = 10
R_OUT_MM     = R_MID_MM + WIDTH_MM/2   # outer wall radius of bend = 20
LOWER_Y_MM   = 0.0                      # lower leg centreline
UPPER_Y_MM   = 2*R_MID_MM               # upper leg centreline = 30
BEND_CENTER  = (0.0, R_MID_MM)          # (0, 15)

# lower leg: outer wall at LOWER_Y - W/2, inner wall at LOWER_Y + W/2 (inner = towards bend centre)
# upper leg: outer wall at UPPER_Y + W/2, inner wall at UPPER_Y - W/2
y_lo_outer = LOWER_Y_MM - WIDTH_MM/2   # -5
y_lo_inner = LOWER_Y_MM + WIDTH_MM/2   # +5
y_up_inner = UPPER_Y_MM - WIDTH_MM/2   # 25
y_up_outer = UPPER_Y_MM + WIDTH_MM/2   # 35
X0 = 0.0
X1 = -LEG_LEN_MM

e1 = Line((X1, y_lo_outer), (X0, y_lo_outer))                       # outer wall, lower leg
e2 = CenterArc(BEND_CENTER, R_OUT_MM, -90, 180)                      # outer wall, bend
e3 = Line((X0, y_up_outer), (X1, y_up_outer))                        # outer wall, upper leg
e4 = Line((X1, y_up_outer), (X1, y_up_inner))                        # outlet cap
e5 = Line((X1, y_up_inner), (X0, y_up_inner))                        # inner wall, upper leg
e6 = CenterArc(BEND_CENTER, R_IN_MM, 90, -180)                       # inner wall, bend
e7 = Line((X0, y_lo_inner), (X1, y_lo_inner))                        # inner wall, lower leg
e8 = Line((X1, y_lo_inner), (X1, y_lo_outer))                        # inlet cap

for name, e in [("e1",e1),("e2",e2),("e3",e3),("e4",e4),("e5",e5),("e6",e6),("e7",e7),("e8",e8)]:
    print(name, e @ 0, "->", e @ 1)

# -- cell 5 -------------------------------------------------------------------------
# All eight edges chain correctly end-to-end and close the loop. Now assemble into a wire, make the 2D
THICK_MM = 2.0   # slab thickness in Z (one background cell thick, for the 2D convention)

wire = Wire([e1, e2, e3, e4, e5, e6, e7, e8])
face = Face(wire)
print("face area (mm^2):", face.area)
print("wire is closed:", wire.is_closed)

solid = extrude(face, amount=THICK_MM)
print("solid volume (mm^3):", solid.volume, " expected:", face.area*THICK_MM)
print("bbox:", solid.bounding_box())

# -- cell 6 -------------------------------------------------------------------------
# The face area (2871.24 mm²) matches the analytic value exactly: 2×(120×10) straight legs + π/2×(20²−
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# quick 2D plot of the profile wire to eyeball the U-bend shape
pts = []
for e in [e1, e2, e3, e4, e5, e6, e7, e8]:
    n = 40 if e.geom_type == "CIRCLE" else 2
    for i in range(n):
        p = e @ (i/(n-1) if n > 1 else 0)
        pts.append((p.X, p.Y))
xs, ys = zip(*pts)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(xs, ys, '-o', ms=2)
ax.set_aspect('equal')
ax.set_title("U-bend duct centreline profile (mm)")
plt.savefig("profile_check.png", dpi=120)
print("saved profile_check.png")

# -- cell 7 -------------------------------------------------------------------------
import matplotlib.pyplot as plt

pts = []
for e in [e1, e2, e3, e4, e5, e6, e7, e8]:
    n = 40 if e.geom_type == "CIRCLE" else 2
    for i in range(n):
        p = e @ (i/(n-1) if n > 1 else 0)
        pts.append((p.X, p.Y))
xs, ys = zip(*pts)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(xs, ys, '-o', ms=2)
ax.set_aspect('equal')
ax.set_title("U-bend duct centreline profile (mm)")
fig

# -- cell 8 -------------------------------------------------------------------------
for name, e in [("e1",e1),("e2",e2),("e3",e3),("e4",e4),("e5",e5),("e6",e6),("e7",e7),("e8",e8)]:
    print(name, e.geom_type)

# -- cell 9 -------------------------------------------------------------------------
pts = []
for e in [e1, e2, e3, e4, e5, e6, e7, e8]:
    n = 40 if e.geom_type == GeomType.CIRCLE else 2
    for i in range(n):
        p = e @ (i/(n-1) if n > 1 else 0)
        pts.append((p.X, p.Y))
xs, ys = zip(*pts)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(xs, ys, '-o', ms=2)
ax.set_aspect('equal')
ax.set_title("U-bend duct centreline profile (mm)")
fig

# -- cell 10 ------------------------------------------------------------------------
faces = solid.faces()
print("total faces:", len(faces))
for f in faces:
    c = f.center()
    n = f.normal_at(c)
    print(f"center=({c.X:.2f},{c.Y:.2f},{c.Z:.2f}) normal=({n.X:.2f},{n.Y:.2f},{n.Z:.2f}) area={f.area:.2f}")

# -- cell 11 ------------------------------------------------------------------------
from build123d import Axis, GeomType

all_faces = solid.faces()

# front/back: planar faces normal to Z (the two flat 2D faces of the slab, "empty" in OpenFOAM's 2D convention)
z_faces = all_faces.filter_by(Axis.Z)
print("z_faces:", len(z_faces), [f.center() for f in z_faces])

# inlet/outlet: planar faces normal to X, located at the open end x = -120
x_faces = all_faces.filter_by(Axis.X)
print("x_faces (normal to X):", len(x_faces))
for f in x_faces:
    print("  center", f.center(), "area", f.area)

# walls: everything not in z_faces or x_faces (4 straight sides + 2 curved bend faces)
wall_faces = all_faces - z_faces - x_faces
print("wall_faces:", len(wall_faces))
for f in wall_faces:
    print("  center", f.center(), "area", f.area, "geom", f.geom_type)

# -- cell 12 ------------------------------------------------------------------------
# Face groups are exactly as expected: 2 z-normal faces (front/back), 2 x-normal faces at x=-120 (inle
from build123d import ShapeList, export_step

inlet_face  = [f for f in x_faces if abs(f.center().Y - 0.0)  < 1e-6][0]
outlet_face = [f for f in x_faces if abs(f.center().Y - 30.0) < 1e-6][0]
front_face, back_face = z_faces[0], z_faces[1]   # z=0 and z=THICK, both "empty"

print("inlet centre :", inlet_face.center(), "area", inlet_face.area)
print("outlet centre:", outlet_face.center(), "area", outlet_face.area)

# sanity: inlet/outlet area should equal width(10mm) x thickness(2mm) = 20 mm^2
print("expected cap area (mm^2):", WIDTH_MM*THICK_MM)

# save the solid (still in mm) as a STEP cache before tessellation, as the recipe advises
export_step(solid, "duct_mm.step")
print("wrote duct_mm.step")

# -- cell 13 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["cat", f"{toolbox}/templates/prep/README.md"], capture_output=True, text=True).stdout)

# -- cell 14 ------------------------------------------------------------------------
print(subprocess.run(["grep", "-n", "-i", "-B2", "-A6", "def scale", f"{toolbox}/b123d_api.md"], capture_output=True, text=True).stdout)

# -- cell 15 ------------------------------------------------------------------------
print(subprocess.run(["grep", "-n", "-i", "scale", f"{toolbox}/b123d_api.md"], capture_output=True, text=True).stdout[:3000])

# -- cell 16 ------------------------------------------------------------------------
# Good — there's a top-level `scale()` function meant exactly for this mm→m fix. Let's rebuild the geo
from build123d import scale

MM_TO_M = 0.001
solid_m = scale(solid, by=MM_TO_M)
bb = solid_m.bounding_box()
print("bbox (m):", bb)
print("volume (m^3):", solid_m.volume, " expected:", solid.volume * MM_TO_M**3)

# -- cell 17 ------------------------------------------------------------------------
# Scaling is exact and volume checks out. Now re-derive the face selectors on this metre-scale solid (
all_faces_m = solid_m.faces()

z_faces_m = all_faces_m.filter_by(Axis.Z)                 # front/back, "empty"
x_faces_m = all_faces_m.filter_by(Axis.X)                 # inlet/outlet caps at x = -0.12
wall_faces_m = all_faces_m - z_faces_m - x_faces_m         # 4 straight + 2 curved bend walls

inlet_face_m  = [f for f in x_faces_m if f.center().Y < 0.015][0]   # y centre near 0
outlet_face_m = [f for f in x_faces_m if f.center().Y > 0.015][0]   # y centre near 0.030
front_face_m  = [f for f in z_faces_m if f.center().Z < 0.001][0]   # z = 0
back_face_m   = [f for f in z_faces_m if f.center().Z > 0.001][0]   # z = THICK

print("inlet  centre", inlet_face_m.center(), "area", inlet_face_m.area)
print("outlet centre", outlet_face_m.center(), "area", outlet_face_m.area)
print("front  centre", front_face_m.center(), "area", front_face_m.area)
print("back   centre", back_face_m.center(), "area", back_face_m.area)
print("n wall faces:", len(wall_faces_m))
for f in wall_faces_m:
    print("  wall centre", f.center(), "area", f.area, f.geom_type)

WIDTH_M, THICK_M = WIDTH_MM*MM_TO_M, THICK_MM*MM_TO_M
print("expected inlet/outlet area (m^2):", WIDTH_M*THICK_M)

# -- cell 18 ------------------------------------------------------------------------
# Right — that cell depended on names from a cell that had errored (the export before I fixed the STEP
import sys
sys.path.insert(0, toolbox)
import cad_convert

# checkpoint the tagged, metre-scale B-rep before tessellation
export_step(solid_m, "duct_m.step", unit=Unit.M)
print("wrote duct_m.step (unit declared as metres)")

key = lambda fs: [tuple(f.center(CenterOf.MASS)) for f in fs]

# a point well inside the fluid: mid-width, mid-thickness, in the lower leg
LOC_IN_MESH = [-0.06, 0.0, THICK_M/2.0]

# clmax: about half the finest wall cell size intended (10 mm passage -> ~1 mm cells -> 0.5 mm facets)
CLMAX_M = 0.0005

groups = {
    "inlet":  {"role": "inlet",  "faces": key([inlet_face_m])},
    "outlet": {"role": "outlet", "faces": key([outlet_face_m])},
    "wall":   {"role": "wall",   "faces": key(wall_faces_m)},
    "front":  {"role": "wall",   "faces": key([front_face_m])},
    "back":   {"role": "wall",   "faces": key([back_face_m])},
}

report = cad_convert.export_patches(
    "duct_m.step",
    groups,
    "constant/triSurface",
    clmax=CLMAX_M,
    location_in_mesh=LOC_IN_MESH,
)
print(cad_convert.render(report))

# -- cell 19 ------------------------------------------------------------------------
# Patch export succeeded and areas match: inlet/outlet = 2e-5 m² each (10mm × 2mm), wall = 0.00114849 
import subprocess
result = subprocess.run(
    ["python3", f"{toolbox}/cad_audit.py", "constant/triSurface", "--json"],
    capture_output=True, text=True
)
print(result.stdout[-4000:])
print("STDERR:", result.stderr[-2000:])

# -- cell 20 ------------------------------------------------------------------------
# The exported surface audits clean: watertight union, manifold, consistent normals, no degenerate or 
import subprocess
result = subprocess.run(
    ["python3", f"{toolbox}/domain_probe.py", "constant/triSurface",
     "--point", "-0.06", "0.0", "0.001", "--suggest", "--json"],
    capture_output=True, text=True
)
print(result.stdout)
print("STDERR:", result.stderr[-2000:])

# -- cell 21 ------------------------------------------------------------------------
# The seed point is confirmed inside the fluid, with the domain measuring 2 mm wide there (the slab th
print(subprocess.run(["cat", f"{toolbox}/templates/snappy/README.md"], capture_output=True, text=True).stdout)

# -- cell 22 ------------------------------------------------------------------------
import subprocess
for f in ["blockMeshDict", "surfaceFeatureExtractDict", "snappyHexMeshDict"]:
    print("="*20, f, "="*20)
    print(subprocess.run(["cat", f"{toolbox}/templates/snappy/{f}"], capture_output=True, text=True).stdout)

# -- cell 23 ------------------------------------------------------------------------
print(subprocess.run(["cat", f"{toolbox}/templates/snappy/blockMeshDict"], capture_output=True, text=True).stdout)

# -- cell 24 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["cat", f"{toolbox}/templates/snappy/snappyHexMeshDict"], capture_output=True, text=True).stdout)

# -- cell 25 ------------------------------------------------------------------------
print(subprocess.run(["sed", "-n", "40,75p", f"{toolbox}/templates/snappy/snappyHexMeshDict"], capture_output=True, text=True).stdout)
