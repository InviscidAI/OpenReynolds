"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Looking at this task: a 2D U-bend duct (10mm wide passage, two 120mm horizontal legs, 180° bend with
import numpy as np
from build123d import *

# Parameters (metres) -- passage width, leg length, centerline bend radius
W = 0.010      # passage width
L = 0.120      # leg length (horizontal)
R = 0.015      # bend centerline radius
r_in = R - W/2 # inner wall radius of bend
r_out = R + W/2 # outer wall radius of bend

# Bend center: lower leg centerline at y=0, upper leg centerline at y=2R
C = (L, R)

# Boundary points (counterclockwise loop)
P1 = (0.0,  -W/2)      # inlet outer corner
P2 = (L,    -W/2)      # outer wall, end of lower leg
P3 = (L,    2*R + W/2) # outer wall, start of upper leg
P4 = (0.0,  2*R + W/2) # outlet outer corner
P5 = (0.0,  2*R - W/2) # outlet inner corner
P6 = (L,    2*R - W/2) # inner wall, start of upper leg (from bend)
P7 = (L,    W/2)       # inner wall, end of bend (start of lower leg inner)
P8 = (0.0,  W/2)       # inlet inner corner

E1 = Line(P1, P2)                                   # wall_outer: lower leg
E2 = RadiusArc(P2, P3, r_out)                        # wall_outer: bend (outer)
E3 = Line(P3, P4)                                   # wall_outer: upper leg
E4 = Line(P4, P5)                                   # outlet
E5 = Line(P5, P6)                                   # wall_inner: upper leg
E6 = RadiusArc(P6, P7, -r_in)                        # wall_inner: bend (inner)
E7 = Line(P7, P8)                                   # wall_inner: lower leg
E8 = Line(P8, P1)                                   # inlet

wire = Wire([E1, E2, E3, E4, E5, E6, E7, E8])
face = make_face(wire)

print("Face area (m^2):", face.area)
print("Perimeter check (m):", wire.length)

show_object = None
try:
    from ocp_vscode import show
except Exception:
    pass

face

# -- cell 2 -------------------------------------------------------------------------
# The measured area (0.0019 m²) doesn't match the expected area (width × centerline length = 0.010 × (
bbox = face.bounding_box()
print("bounding box min:", bbox.min)
print("bounding box max:", bbox.max)
# expected max.X should be ~ L + r_out = 0.120+0.020 = 0.140 if bend bulges outward (+x)

# -- cell 3 -------------------------------------------------------------------------
# The arcs bulge inward instead of outward — wrong curvature direction. Let me rebuild using `CenterAr
import numpy as np
from build123d import *

# Parameters (metres)
W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2
C = (L, R)

P1 = (0.0,  -W/2)
P2 = (L,    -W/2)
P3 = (L,    2*R + W/2)
P4 = (0.0,  2*R + W/2)
P5 = (0.0,  2*R - W/2)
P6 = (L,    2*R - W/2)
P7 = (L,    W/2)
P8 = (0.0,  W/2)

E1 = Line(P1, P2)
E2 = CenterArc(C, r_out, -90, 180)   # outer arc, CCW from -90 to +90 through 0 (bulge +x)
E3 = Line(P3, P4)
E4 = Line(P4, P5)
E5 = Line(P5, P6)
E6 = CenterArc(C, r_in, 90, -180)    # inner arc, CW from 90 to -90 through 0 (bulge +x)
E7 = Line(P7, P8)
E8 = Line(P8, P1)

print("E2 endpoints:", E2 @ 0, E2 @ 1)
print("E6 endpoints:", E6 @ 0, E6 @ 1)

wire = Wire([E1, E2, E3, E4, E5, E6, E7, E8])
face = make_face(wire)
bbox = face.bounding_box()
print("area:", face.area, "expected:", W*(2*L + np.pi*R))
print("bbox min:", bbox.min, "max:", bbox.max)

# -- cell 4 -------------------------------------------------------------------------
# Area now matches expected (0.0028712 m²) and bbox max X = 0.140 confirms the bend bulges outward cor
import matplotlib.pyplot as plt
from build123d import Vector

# sample the wire boundary as points for plotting
def sample_edge(e, n=40):
    return [e.position_at(t/n) for t in range(n+1)]

pts = []
for e in wire.edges():
    pts.extend(sample_edge(e))

xs = [p.X for p in pts]
ys = [p.Y for p in pts]

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(xs, ys, '-')
ax.set_aspect('equal')
ax.set_title('U-bend duct centerline-offset boundary (m)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.savefig('shape_check.png', dpi=120)
print("saved shape_check.png")

# -- cell 5 -------------------------------------------------------------------------
# The U-bend shape looks correct: two 120mm horizontal legs, 10mm passage, bend swelling to the outer 
import numpy as np

# Width: perpendicular distance between lower-leg outer edge (E1) and inner edge (E7)
p_on_E1 = np.array([E1 @ 0.5])[0]
p_on_E7 = np.array([E7 @ 0.5])[0]
width_measured = abs(p_on_E1.Y - p_on_E7.Y)  # both edges horizontal, so Y difference = width

# Leg lengths
leg1_len = E1.length
leg2_len = E5.length  # upper leg inner edge, should also be 120mm

# Bend arc radii (measured from geometry, via radius property of the arc edges)
r_out_meas = E2.radius
r_in_meas = E6.radius
centerline_radius_meas = (r_out_meas + r_in_meas) / 2
bend_width_meas = r_out_meas - r_in_meas

# Bend angle: total turning angle from arc parametrization
ang0 = E2.param_at_point(E2 @ 0) if hasattr(E2, "param_at_point") else None
# simpler: use tangent directions at start/end of outer arc
t_start = E2.tangent_at(0)
t_end = E2.tangent_at(1)
cosang = t_start.dot(t_end)
bend_angle_deg = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
# tangent-to-tangent angle for a 180 deg turn where tangents are reversed -> arccos(-1)=180, but if collinear same dir could be 0; check via cross too

print(f"width measured = {width_measured*1000:.3f} mm  (requested 10 mm)")
print(f"leg1 length = {leg1_len*1000:.3f} mm (requested 120 mm)")
print(f"leg2 length = {leg2_len*1000:.3f} mm (requested 120 mm)")
print(f"outer wall radius = {r_out_meas*1000:.3f} mm (expected 20 mm)")
print(f"inner wall radius = {r_in_meas*1000:.3f} mm (expected 10 mm)")
print(f"centerline radius measured = {centerline_radius_meas*1000:.3f} mm (requested 15 mm)")
print(f"bend passage width from radii = {bend_width_meas*1000:.3f} mm (requested 10 mm)")
print(f"bend turning angle (tangent method) = {bend_angle_deg:.2f} deg (requested 180 deg)")

# -- cell 6 -------------------------------------------------------------------------
# All measured properties match the request exactly (10mm width, 120mm legs, 15mm centerline radius, 1
import gmsh
import numpy as np

W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2

gmsh.initialize()
gmsh.model.add("ubend")

occ = gmsh.model.occ

# points
p1 = occ.addPoint(0.0,  -W/2, 0)
p2 = occ.addPoint(L,    -W/2, 0)
p3 = occ.addPoint(L,    2*R+W/2, 0)
p4 = occ.addPoint(0.0,  2*R+W/2, 0)
p5 = occ.addPoint(0.0,  2*R-W/2, 0)
p6 = occ.addPoint(L,    2*R-W/2, 0)
p7 = occ.addPoint(L,    W/2, 0)
p8 = occ.addPoint(0.0,  W/2, 0)
pc = occ.addPoint(L, R, 0)  # bend center

# curves
E1 = occ.addLine(p1, p2)                     # wall_outer, lower leg
E2 = occ.addCircleArc(p2, pc, p3)            # wall_outer, bend (outer)
E3 = occ.addLine(p3, p4)                     # wall_outer, upper leg
E4 = occ.addLine(p4, p5)                     # outlet
E5 = occ.addLine(p5, p6)                     # wall_inner, upper leg
E6 = occ.addCircleArc(p6, pc, p7)            # wall_inner, bend (inner)
E7 = occ.addLine(p7, p8)                     # wall_inner, lower leg
E8 = occ.addLine(p8, p1)                     # inlet

Lstart = occ.addLine(p2, p7)  # internal separator: leg/bend boundary near inlet side
Lend   = occ.addLine(p3, p6)  # internal separator: leg/bend boundary near outlet side

occ.synchronize()

loop1 = occ.addCurveLoop([E1, Lstart, E7, E8])          # lower leg block (need correct signs)
loop2 = occ.addCurveLoop([E2, Lend, -E6, -Lstart])      # bend block
loop3 = occ.addCurveLoop([E3, E4, E5, -Lend])           # upper leg block

occ.synchronize()

s1 = occ.addPlaneSurface([loop1])
s2 = occ.addPlaneSurface([loop2])
s3 = occ.addPlaneSurface([loop3])

occ.synchronize()

area_total = 0.0
for s in (s1, s2, s3):
    m = gmsh.model.occ.getMass(2, s)
    print("surface", s, "area", m)
    area_total += m

print("total area:", area_total, "expected:", W*(2*L+np.pi*R))

# -- cell 7 -------------------------------------------------------------------------
# I made a sign error in the bend loop (negated E6 incorrectly). Let me fix the curve loop orientation
import gmsh
import numpy as np

gmsh.finalize()
gmsh.initialize()
gmsh.model.add("ubend")

W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2

occ = gmsh.model.occ

p1 = occ.addPoint(0.0,  -W/2, 0)
p2 = occ.addPoint(L,    -W/2, 0)
p3 = occ.addPoint(L,    2*R+W/2, 0)
p4 = occ.addPoint(0.0,  2*R+W/2, 0)
p5 = occ.addPoint(0.0,  2*R-W/2, 0)
p6 = occ.addPoint(L,    2*R-W/2, 0)
p7 = occ.addPoint(L,    W/2, 0)
p8 = occ.addPoint(0.0,  W/2, 0)
pc = occ.addPoint(L, R, 0)

E1 = occ.addLine(p1, p2)
E2 = occ.addCircleArc(p2, pc, p3)
E3 = occ.addLine(p3, p4)
E4 = occ.addLine(p4, p5)
E5 = occ.addLine(p5, p6)
E6 = occ.addCircleArc(p6, pc, p7)
E7 = occ.addLine(p7, p8)
E8 = occ.addLine(p8, p1)

Lstart = occ.addLine(p2, p7)
Lend   = occ.addLine(p3, p6)

occ.synchronize()

loop1 = occ.addCurveLoop([E1, Lstart, E7, E8])
loop2 = occ.addCurveLoop([E2, Lend, E6, -Lstart])
loop3 = occ.addCurveLoop([E3, E4, E5, -Lend])

occ.synchronize()

s1 = occ.addPlaneSurface([loop1])
s2 = occ.addPlaneSurface([loop2])
s3 = occ.addPlaneSurface([loop3])

occ.synchronize()

area_total = 0.0
for s in (s1, s2, s3):
    m = gmsh.model.occ.getMass(2, s)
    print("surface", s, "area", m)
    area_total += m

expected_bend = 0.5*np.pi*(r_out**2 - r_in**2)
print("bend area expected:", expected_bend)
print("total area:", area_total, "expected:", W*(2*L+np.pi*R))

# -- cell 8 -------------------------------------------------------------------------
# Found the bug: gmsh's `addCircleArc(start, center, end)` sweeps counterclockwise from start to end, 
import gmsh
import numpy as np

gmsh.finalize()
gmsh.initialize()
gmsh.model.add("ubend")

W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2

occ = gmsh.model.occ

p1 = occ.addPoint(0.0,  -W/2, 0)
p2 = occ.addPoint(L,    -W/2, 0)
p3 = occ.addPoint(L,    2*R+W/2, 0)
p4 = occ.addPoint(0.0,  2*R+W/2, 0)
p5 = occ.addPoint(0.0,  2*R-W/2, 0)
p6 = occ.addPoint(L,    2*R-W/2, 0)
p7 = occ.addPoint(L,    W/2, 0)
p8 = occ.addPoint(0.0,  W/2, 0)
pc = occ.addPoint(L, R, 0)

E1 = occ.addLine(p1, p2)
E2 = occ.addCircleArc(p2, pc, p3)   # CCW -90 -> +90 through 0 (bulge +x): OK
E3 = occ.addLine(p3, p4)
E4 = occ.addLine(p4, p5)
E5 = occ.addLine(p5, p6)
E6 = occ.addCircleArc(p7, pc, p6)   # define p7->p6 so CCW -90->+90 through 0 (bulge +x)
E7 = occ.addLine(p7, p8)
E8 = occ.addLine(p8, p1)

Lstart = occ.addLine(p2, p7)
Lend   = occ.addLine(p3, p6)

occ.synchronize()

loop1 = occ.addCurveLoop([E1, Lstart, E7, E8])
loop2 = occ.addCurveLoop([E2, Lend, -E6, -Lstart])   # p6->p7 needs reverse of E6(p7->p6)
loop3 = occ.addCurveLoop([E3, E4, E5, -Lend])

occ.synchronize()

s1 = occ.addPlaneSurface([loop1])
s2 = occ.addPlaneSurface([loop2])
s3 = occ.addPlaneSurface([loop3])

occ.synchronize()

area_total = 0.0
for s in (s1, s2, s3):
    m = gmsh.model.occ.getMass(2, s)
    print("surface", s, "area", m)
    area_total += m

expected_bend = 0.5*np.pi*(r_out**2 - r_in**2)
print("bend area expected:", expected_bend)
print("total area:", area_total, "expected:", W*(2*L+np.pi*R))

# -- cell 9 -------------------------------------------------------------------------
# Areas now match exactly (bend block 0.00047124 m², total 0.0028712 m², matching the analytic expecta
import gmsh

# coarse structured mesh: cells across width, and along legs/bend
n_w = 4          # cells across width (10mm) -> 2.5mm cells
n_leg = 48       # cells along each 120mm leg -> 2.5mm cells
n_bend = 19      # cells along bend arc (180deg, R=15mm, arc len ~47.1mm) -> ~2.5mm cells

# width-direction curves (need n_w+1 points), consistent across blocks
for c in (Lstart, Lend, E8, E4):
    gmsh.model.mesh.setTransfiniteCurve(c, n_w+1)

# length-direction curves for legs (need n_leg+1 points)
for c in (E1, E7, E3, E5):
    gmsh.model.mesh.setTransfiniteCurve(c, n_leg+1)

# length-direction curves for bend arcs (need n_bend+1 points)
for c in (E2, E6):
    gmsh.model.mesh.setTransfiniteCurve(c, n_bend+1)

# transfinite + recombine each surface for structured quads
for s in (s1, s2, s3):
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

gmsh.model.mesh.generate(2)

# report
nnodes = gmsh.model.mesh.getNodes()[0].size
print("2D mesh nodes:", nnodes)
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
for t, tags in zip(elem_types, elem_tags):
    print("elem type", t, "count", len(tags))

gmsh.write("ubend_2d_check.msh")
print("wrote ubend_2d_check.msh")

# -- cell 10 ------------------------------------------------------------------------
# No meshio available — I'll extract nodes/elements directly from the gmsh Python API instead (which i
import numpy as np
import pyvista as pv

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = node_coords.reshape(-1, 3)
tag_to_idx = {t: i for i, t in enumerate(node_tags)}

elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
quads = None
for t, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
    if t == 3:  # quad
        quads = nodes.reshape(-1, 4)

quad_idx = np.vectorize(tag_to_idx.get)(quads)

n = quad_idx.shape[0]
faces = np.hstack([np.full((n,1), 4), quad_idx]).astype(np.int64).flatten()
grid = pv.PolyData(node_coords, faces)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(grid, show_edges=True, color="lightblue")
pl.view_xy()
pl.camera.parallel_projection = True
pl.screenshot("ubend_2d_mesh.png", window_size=(900,500))
print("saved ubend_2d_mesh.png")

# -- cell 11 ------------------------------------------------------------------------
# Let me check if the screenshot actually rendered correctly despite the EGL warnings (OSMesa should s
import os
print(os.path.exists("ubend_2d_mesh.png"), os.path.getsize("ubend_2d_mesh.png"))

# -- cell 12 ------------------------------------------------------------------------
# Time to commit to disk. I'll extrude the 3-block 2D mesh into a single-cell-thick 3D mesh (for the 2
import gmsh
import numpy as np

dz = 0.0025  # thickness = one in-plane cell size (10mm/4), single layer

# capture ordered boundary curves for each surface BEFORE extrude (for correspondence)
b1 = gmsh.model.getBoundary([(2, s1)], oriented=True)
b2 = gmsh.model.getBoundary([(2, s2)], oriented=True)
b3 = gmsh.model.getBoundary([(2, s3)], oriented=True)
print("s1 boundary:", b1)
print("s2 boundary:", b2)
print("s3 boundary:", b3)

# extrude each surface individually, 1 layer, recombine -> hex cells
ext1 = gmsh.model.occ.extrude([(2, s1)], 0, 0, dz, numElements=[1], recombine=True)
ext2 = gmsh.model.occ.extrude([(2, s2)], 0, 0, dz, numElements=[1], recombine=True)
ext3 = gmsh.model.occ.extrude([(2, s3)], 0, 0, dz, numElements=[1], recombine=True)

gmsh.model.occ.synchronize()
print("ext1:", ext1)
print("ext2:", ext2)
print("ext3:", ext3)

# -- cell 13 ------------------------------------------------------------------------
# Now I'll map each lateral surface to the curve it came from (matching the `getBoundary` order to the
import gmsh

# lateral surface tag -> originating curve tag, from the printed correspondence
lateral_map = {
    4: 1,   # E1 (wall_outer, lower leg)
    5: 7,   # E7 (wall_inner, lower leg)
    6: 8,   # E8 (inlet)
    7: 9,   # Lstart (internal, shared s1/s2)
    9: 2,   # E2 (wall_outer, bend)
    10: 6,  # E6 (wall_inner, bend)
    11: 9,  # Lstart (internal, dup)
    12: 10, # Lend (internal, shared s2/s3)
    14: 3,  # E3 (wall_outer, upper leg)
    15: 4,  # E4 (outlet)
    16: 5,  # E5 (wall_inner, upper leg)
    17: 10, # Lend (internal, dup)
}

curve_names = {1:"wall_outer",7:"wall_inner",8:"inlet",2:"wall_outer",6:"wall_inner",
               10:None,9:None,3:"wall_outer",4:"outlet",5:"wall_inner"}

n_before = len(gmsh.model.occ.getEntities(2))
print("surfaces before dedup:", n_before)

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

n_after = len(gmsh.model.occ.getEntities(2))
print("surfaces after dedup:", n_after)
print("volumes:", gmsh.model.occ.getEntities(3))
print("all surfaces:", gmsh.model.occ.getEntities(2))

# -- cell 14 ------------------------------------------------------------------------
# `removeAllDuplicates` over-fragmented the geometry. I'll restart clean and instead use a proper bool
import gmsh
import numpy as np

gmsh.finalize()
gmsh.initialize()
gmsh.model.add("ubend")

W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2
dz = 0.0025

occ = gmsh.model.occ

p1 = occ.addPoint(0.0,  -W/2, 0)
p2 = occ.addPoint(L,    -W/2, 0)
p3 = occ.addPoint(L,    2*R+W/2, 0)
p4 = occ.addPoint(0.0,  2*R+W/2, 0)
p5 = occ.addPoint(0.0,  2*R-W/2, 0)
p6 = occ.addPoint(L,    2*R-W/2, 0)
p7 = occ.addPoint(L,    W/2, 0)
p8 = occ.addPoint(0.0,  W/2, 0)
pc = occ.addPoint(L, R, 0)

E1 = occ.addLine(p1, p2)
E2 = occ.addCircleArc(p2, pc, p3)
E3 = occ.addLine(p3, p4)
E4 = occ.addLine(p4, p5)
E5 = occ.addLine(p5, p6)
E6 = occ.addCircleArc(p7, pc, p6)
E7 = occ.addLine(p7, p8)
E8 = occ.addLine(p8, p1)

Lstart = occ.addLine(p2, p7)
Lend   = occ.addLine(p3, p6)

occ.synchronize()

loop1 = occ.addCurveLoop([E1, Lstart, E7, E8])
loop2 = occ.addCurveLoop([E2, Lend, -E6, -Lstart])
loop3 = occ.addCurveLoop([E3, E4, E5, -Lend])

occ.synchronize()

s1 = occ.addPlaneSurface([loop1])
s2 = occ.addPlaneSurface([loop2])
s3 = occ.addPlaneSurface([loop3])

occ.synchronize()

n_w = 4
n_leg = 48
n_bend = 19

for c in (Lstart, Lend, E8, E4):
    gmsh.model.mesh.setTransfiniteCurve(c, n_w+1)
for c in (E1, E7, E3, E5):
    gmsh.model.mesh.setTransfiniteCurve(c, n_leg+1)
for c in (E2, E6):
    gmsh.model.mesh.setTransfiniteCurve(c, n_bend+1)

for s in (s1, s2, s3):
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

ext1 = occ.extrude([(2, s1)], 0, 0, dz, numElements=[1], recombine=True)
ext2 = occ.extrude([(2, s2)], 0, 0, dz, numElements=[1], recombine=True)
ext3 = occ.extrude([(2, s3)], 0, 0, dz, numElements=[1], recombine=True)
occ.synchronize()

vol1 = [t for d,t in ext1 if d==3][0]
vol2 = [t for d,t in ext2 if d==3][0]
vol3 = [t for d,t in ext3 if d==3][0]
print("volumes to fuse:", vol1, vol2, vol3)

fused, fuse_map = occ.fuse([(3, vol1)], [(3, vol2), (3, vol3)])
occ.synchronize()
print("fused result:", fused)
print("volumes now:", gmsh.model.occ.getEntities(3))
print("surfaces now:", gmsh.model.occ.getEntities(2))

# -- cell 15 ------------------------------------------------------------------------
# Fused into a single volume with 12 boundary surfaces. Let's inspect each by centroid, area, and norm
import numpy as np

surfs = gmsh.model.occ.getEntities(2)
info = []
for d, t in surfs:
    com = gmsh.model.occ.getCenterOfMass(d, t)
    mass = gmsh.model.occ.getMass(d, t)
    bbox = gmsh.model.getBoundingBox(d, t)
    info.append((t, com, mass, bbox))
    print(f"surf {t}: COM={tuple(round(c,5) for c in com)}, area={mass:.6f}, bbox={tuple(round(b,5) for b in bbox)}")

# -- cell 16 ------------------------------------------------------------------------
# A cleaner approach: instead of extruding each surface separately then fusing (which produced boolean
import gmsh
import numpy as np

gmsh.finalize()
gmsh.initialize()
gmsh.model.add("ubend")

W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2
dz = 0.0025

occ = gmsh.model.occ

p1 = occ.addPoint(0.0,  -W/2, 0)
p2 = occ.addPoint(L,    -W/2, 0)
p3 = occ.addPoint(L,    2*R+W/2, 0)
p4 = occ.addPoint(0.0,  2*R+W/2, 0)
p5 = occ.addPoint(0.0,  2*R-W/2, 0)
p6 = occ.addPoint(L,    2*R-W/2, 0)
p7 = occ.addPoint(L,    W/2, 0)
p8 = occ.addPoint(0.0,  W/2, 0)
pc = occ.addPoint(L, R, 0)

E1 = occ.addLine(p1, p2)
E2 = occ.addCircleArc(p2, pc, p3)
E3 = occ.addLine(p3, p4)
E4 = occ.addLine(p4, p5)
E5 = occ.addLine(p5, p6)
E6 = occ.addCircleArc(p7, pc, p6)
E7 = occ.addLine(p7, p8)
E8 = occ.addLine(p8, p1)

Lstart = occ.addLine(p2, p7)
Lend   = occ.addLine(p3, p6)

occ.synchronize()

loop1 = occ.addCurveLoop([E1, Lstart, E7, E8])
loop2 = occ.addCurveLoop([E2, Lend, -E6, -Lstart])
loop3 = occ.addCurveLoop([E3, E4, E5, -Lend])

occ.synchronize()

s1 = occ.addPlaneSurface([loop1])
s2 = occ.addPlaneSurface([loop2])
s3 = occ.addPlaneSurface([loop3])

occ.synchronize()

n_w = 4
n_leg = 48
n_bend = 19

for c in (Lstart, Lend, E8, E4):
    gmsh.model.mesh.setTransfiniteCurve(c, n_w+1)
for c in (E1, E7, E3, E5):
    gmsh.model.mesh.setTransfiniteCurve(c, n_leg+1)
for c in (E2, E6):
    gmsh.model.mesh.setTransfiniteCurve(c, n_bend+1)

for s in (s1, s2, s3):
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

# extrude all three surfaces TOGETHER in one call
ext = occ.extrude([(2, s1), (2, s2), (2, s3)], 0, 0, dz, numElements=[1], recombine=True)
occ.synchronize()

print("extrude result:", ext)
print("volumes:", gmsh.model.occ.getEntities(3))
print("surfaces:", gmsh.model.occ.getEntities(2))

# -- cell 17 ------------------------------------------------------------------------
# Now surfaces 5 and 10 appear as shared laterals between adjacent volumes (no duplication) — exactly 
import numpy as np

all_surfs = gmsh.model.occ.getEntities(2)
for d, t in all_surfs:
    com = gmsh.model.occ.getCenterOfMass(d, t)
    mass = gmsh.model.occ.getMass(d, t)
    bbox = gmsh.model.getBoundingBox(d, t)
    # which volumes is this surface adjacent to?
    ups, downs = gmsh.model.getAdjacencies(d, t)
    print(f"surf {t}: COM=({com[0]:.5f},{com[1]:.5f},{com[2]:.5f}) area={mass:.6f} "
          f"bbox_x=({bbox[0]:.4f},{bbox[3]:.4f}) bbox_y=({bbox[1]:.4f},{bbox[4]:.4f}) "
          f"bbox_z=({bbox[2]:.4f},{bbox[5]:.4f})")

# -- cell 18 ------------------------------------------------------------------------
# The area values (independent of any curved-surface bounding-box imprecision) cleanly classify all 16
# - front (z=0): 1,2,3 · back (z=dz): 8,12,16
# - wall_outer (area=leg·dz or π·r_out·dz): 4,9,13 · wall_inner: 6,11,15
# - inlet (x=0, lower leg): 7 · outlet (x=0, upper leg): 14
# - internal shared interfaces (not a patch): 5,10
# Let's verify 5 and 10 are indeed shared between two volumes, then add physical groups and generate t
for t in (5, 10):
    ups, downs = gmsh.model.getAdjacencies(2, t)
    print(f"surf {t} adjacent volumes:", ups)

wall_outer = [4, 9, 13]
wall_inner = [6, 11, 15]
inlet = [7]
outlet = [14]
front = [1, 2, 3]
back = [8, 12, 16]

pg_wall_outer = gmsh.model.addPhysicalGroup(2, wall_outer)
gmsh.model.setPhysicalName(2, pg_wall_outer, "wall_outer")
pg_wall_inner = gmsh.model.addPhysicalGroup(2, wall_inner)
gmsh.model.setPhysicalName(2, pg_wall_inner, "wall_inner")
pg_inlet = gmsh.model.addPhysicalGroup(2, inlet)
gmsh.model.setPhysicalName(2, pg_inlet, "inlet")
pg_outlet = gmsh.model.addPhysicalGroup(2, outlet)
gmsh.model.setPhysicalName(2, pg_outlet, "outlet")
pg_front = gmsh.model.addPhysicalGroup(2, front)
gmsh.model.setPhysicalName(2, pg_front, "front")
pg_back = gmsh.model.addPhysicalGroup(2, back)
gmsh.model.setPhysicalName(2, pg_back, "back")

pg_vol = gmsh.model.addPhysicalGroup(3, [1, 2, 3])
gmsh.model.setPhysicalName(3, pg_vol, "fluid")

print("physical groups:", gmsh.model.getPhysicalGroups())

# -- cell 19 ------------------------------------------------------------------------
# Confirmed: surfaces 5 and 10 are each shared by exactly two volumes (proper internal interfaces). Ph
import subprocess

gmsh.model.mesh.generate(3)

node_tags = gmsh.model.mesh.getNodes()[0]
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
print("3D mesh nodes:", len(node_tags))
for t, tags in zip(elem_types, elem_tags):
    print("volume elem type", t, "count", len(tags))

gmsh.write("ubend.msh")
print("wrote ubend.msh")

# -- cell 20 ------------------------------------------------------------------------
# 460 hexahedra, 1161 nodes (exactly 581×2 as expected for a clean single-layer extrusion — confirms n
import subprocess, os

os.makedirs("constant", exist_ok=True)
result = subprocess.run(["gmshToFoam", "ubend.msh"], capture_output=True, text=True)
print(result.stdout[-3000:])
print("STDERR:", result.stderr[-2000:])

# -- cell 21 ------------------------------------------------------------------------
# Need a minimal OpenFOAM case structure (system/controlDict etc.) before any utility will run. Let me
import os

os.makedirs("system", exist_ok=True)
os.makedirs("constant", exist_ok=True)

controlDict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
with open("system/controlDict", "w") as f:
    f.write(controlDict)

fvSchemes = """FoamFile
{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
with open("system/fvSchemes", "w") as f:
    f.write(fvSchemes)

fvSolution = """FoamFile
{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  fvSolution;
}
solvers {}
"""
with open("system/fvSolution", "w") as f:
    f.write(fvSolution)

print("minimal case files written")
print(os.listdir("system"))

# -- cell 22 ------------------------------------------------------------------------
import subprocess

result = subprocess.run(["gmshToFoam", "ubend.msh"], capture_output=True, text=True)
print(result.stdout[-3000:])
print("STDERR:", result.stderr[-2000:])

# -- cell 23 ------------------------------------------------------------------------
# The conversion reported "1158 undefined faces added to defaultFaces" — that number exactly equals th
with open("constant/polyMesh/boundary") as f:
    print(f.read())

# -- cell 24 ------------------------------------------------------------------------
# The boundary file shows correct patches with exact expected face counts (front/back=460 each, wall_o
import re

with open("constant/polyMesh/boundary") as f:
    content = f.read()

def set_type(content, patch_name, new_type):
    pattern = re.compile(
        r"(" + re.escape(patch_name) + r"\s*\{[^}]*?type\s+)(\w+)(;)"
    )
    return pattern.sub(lambda m: m.group(1) + new_type + m.group(3), content)

for p in ("front", "back"):
    content = set_type(content, p, "empty")
for p in ("wall_outer", "wall_inner"):
    content = set_type(content, p, "wall")
# inlet/outlet remain 'patch'

with open("constant/polyMesh/boundary", "w") as f:
    f.write(content)

print(content)

# -- cell 25 ------------------------------------------------------------------------
# Now run checkMesh to validate the mesh.
import subprocess

result = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(result.stdout[-5000:])
print("STDERR:", result.stderr[-2000:])

# -- cell 26 ------------------------------------------------------------------------
# 8 faces are broken (open cells + non-orthogonality + wrong orientation), exactly matching the width-
import gmsh
import numpy as np

gmsh.finalize()
gmsh.initialize()
gmsh.model.add("ubend_single")

W = 0.010
L = 0.120
R = 0.015
r_in = R - W/2
r_out = R + W/2
dz = 0.0025

occ = gmsh.model.occ

p1 = occ.addPoint(0.0,  -W/2, 0)
p2 = occ.addPoint(L,    -W/2, 0)
p3 = occ.addPoint(L,    2*R+W/2, 0)
p4 = occ.addPoint(0.0,  2*R+W/2, 0)
p5 = occ.addPoint(0.0,  2*R-W/2, 0)
p6 = occ.addPoint(L,    2*R-W/2, 0)
p7 = occ.addPoint(L,    W/2, 0)
p8 = occ.addPoint(0.0,  W/2, 0)
pc = occ.addPoint(L, R, 0)

E1 = occ.addLine(p1, p2)          # wall_outer, lower leg
E2 = occ.addCircleArc(p2, pc, p3) # wall_outer, bend (bulges +x, verified earlier)
E3 = occ.addLine(p3, p4)          # wall_outer, upper leg
E4 = occ.addLine(p4, p5)          # outlet
E5 = occ.addLine(p5, p6)          # wall_inner, upper leg
E6 = occ.addCircleArc(p7, pc, p6) # wall_inner, bend
E7 = occ.addLine(p7, p8)          # wall_inner, lower leg
E8 = occ.addLine(p8, p1)          # inlet

loop = occ.addCurveLoop([E1, E2, E3, E4, E5, E6, E7, E8])
occ.synchronize()
s = occ.addPlaneSurface([loop])
occ.synchronize()

area = gmsh.model.occ.getMass(2, s)
print("single face area:", area, "expected:", W*(2*L+np.pi*R))

# -- cell 27 ------------------------------------------------------------------------
# Sign issue again — E6 was defined p7→p6 but the single loop needs p6→p7. Fix by negating E6 in the l
gmsh.model.occ.remove([(2, s)], recursive=False)
loop = gmsh.model.occ.addCurveLoop([E1, E2, E3, E4, E5, -E6, E7, E8])
gmsh.model.occ.synchronize()
s = gmsh.model.occ.addPlaneSurface([loop])
gmsh.model.occ.synchronize()

area = gmsh.model.occ.getMass(2, s)
print("single face area:", area, "expected:", W*(2*L + np.pi*R))
bbox = gmsh.model.getBoundingBox(2, s)
print("bbox:", bbox)
