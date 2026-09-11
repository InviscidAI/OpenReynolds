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
print(subprocess.run(["grep", "-n", "-i", "-E", "offset|sweep|thicken", 
                       "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"],
                      capture_output=True, text=True).stdout[:6000])

# -- cell 2 -------------------------------------------------------------------------
import subprocess
res = subprocess.run(["sed", "-n", "40,100p",
                       "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"],
                      capture_output=True, text=True)
print(res.stdout)

# -- cell 3 -------------------------------------------------------------------------
from build123d import *

# Quick test: does offset(side=BOTH) on an open edge give a closed face (a "band") directly?
test_line = Edge.make_line((0,0,0),(20,0,0))
try:
    band = offset(test_line, amount=3, side=Side.BOTH)
    print(type(band), band)
except Exception as e:
    print("error:", e)

# -- cell 4 -------------------------------------------------------------------------
edges = band.edges()
print(len(edges))
for e in edges:
    print(e, e.start_point(), e.end_point())

# -- cell 5 -------------------------------------------------------------------------
res = subprocess.run(["grep", "-n", "-i", "spline\|make_spline\|tangent", 
                       "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"],
                      capture_output=True, text=True)
print(res.stdout)

# -- cell 6 -------------------------------------------------------------------------
res = subprocess.run(["grep", "-n", "Arc\\|CenterArc\\|RadiusArc", 
                       "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.md"],
                      capture_output=True, text=True)
print(res.stdout)

# -- cell 7 -------------------------------------------------------------------------
import numpy as np

# geometry test for one teardrop loop centerline using CenterArc
W = 6.0      # channel width, mm (will convert to metres at export)
Lneck = 6.0  # gap between branch and rejoin points along x (mm) -- how close the loop's neck is
H = 12.0     # how far the loop bulges from the main centerline (mm), to the far point of the loop

# circle passing through branch (xc-d,0) and rejoin (xc+d,0), bulging up to yc+R = H
d = Lneck/2.0
# choose yc (center height) and solve R from d^2 + yc^2 = R^2, and yc+R = H
# R = H - yc  -> d^2 + yc^2 = (H-yc)^2 = H^2 -2H yc + yc^2  -> d^2 = H^2 -2H yc -> yc = (H^2-d^2)/(2H)
yc = (H**2 - d**2) / (2*H)
R = H - yc
print("yc=",yc,"R=",R,"check R^2=",R**2,"d^2+yc^2=",d**2+yc**2)

ang1 = np.degrees(np.arctan2(0-yc, -d - 0))   # branch point relative angle
ang2 = np.degrees(np.arctan2(0-yc, d - 0))    # rejoin point relative angle
ang1 = ang1 % 360
ang2 = ang2 % 360
print("ang1 (branch)=",ang1,"ang2 (rejoin)=",ang2)

# short way (through bottom) difference:
short_diff = (ang2 - ang1) % 360
print("short_diff (ccw from ang1 to ang2) =", short_diff)
# long way (through top) is going clockwise (negative) from ang1 by (360-short_diff)
long_arc_size = -(360 - short_diff)
print("long_arc_size (clockwise, through top) =", long_arc_size)

# -- cell 8 -------------------------------------------------------------------------
# exit 0 (1 s)
# edge start: Vector: (X=30.0000000000000004 begin display commands with the actual raw text of your r
from build123d import *
import numpy as np

xc = 30.0  # arbitrary test location along x for branch/rejoin center
center_pt = (xc, yc, 0)
loop_edge = CenterArc(center=center_pt, radius=R, start_angle=ang1, arc_size=long_arc_size)

branch_pt = (xc-d, 0, 0)
rejoin_pt = (xc+d, 0, 0)
print("edge start:", loop_edge.edges()[0].start_point())
print("edge end:  ", loop_edge.edges()[0].end_point())
print("expected branch:", branch_pt, "expected rejoin:", rejoin_pt)

# -- cell 9 -------------------------------------------------------------------------
import matplotlib.pyplot as plt

e = loop_edge.edges()[0]
ts = np.linspace(0, 1, 200)
pts = np.array([tuple(e.position_at(t)) for t in ts])

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(pts[:,0], pts[:,1], '-')
ax.plot([xc-d, xc+d], [0,0], 'ro')
ax.plot([xc-15, xc+15], [0,0], 'k--', lw=0.5)
ax.set_aspect('equal')
ax.set_title("single teardrop loop centerline")
plt.savefig("test_loop.png", dpi=110)
plt.show()
print("saved")

# -- cell 10 ------------------------------------------------------------------------
fig

# -- cell 11 ------------------------------------------------------------------------
# check tangent directions at branch and rejoin to gauge merge angle with main channel (should be shallow-ish)
e = loop_edge.edges()[0]
tang_start = e.tangent_at(0)
tang_end = e.tangent_at(1)
print("tangent at branch (t=0):", tang_start)
print("tangent at rejoin (t=1):", tang_end)
import numpy as np
ang_start_deg = np.degrees(np.arctan2(tang_start.Y, tang_start.X))
ang_end_deg = np.degrees(np.arctan2(tang_end.Y, tang_end.X))
print("angle vs +x axis: branch=", ang_start_deg, "rejoin=", ang_end_deg)

# -- cell 12 ------------------------------------------------------------------------
# thicken the loop centerline into a channel band of width W, and inspect it
band_loop = offset(loop_edge, amount=W/2, side=Side.BOTH)
print(type(band_loop), len(band_loop.edges()))
face_loop = make_face(band_loop.edges())
print("loop band face area (mm^2):", face_loop.area)

fig2, ax2 = plt.subplots(figsize=(5,5))
for f in face_loop.faces():
    for w in [f.outer_wire()] + f.inner_wires():
        pts_w = np.array([tuple(w.position_at(t)) for t in np.linspace(0,1,300)])
        ax2.plot(pts_w[:,0], pts_w[:,1], '-')
ax2.set_aspect('equal')
ax2.set_title("thickened teardrop loop (band, W=6mm)")
fig2

# -- cell 13 ------------------------------------------------------------------------
res = subprocess.run(["cat", 
   "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/prep/README.md"],
   capture_output=True, text=True)
print(res.stdout)

# -- cell 14 ------------------------------------------------------------------------
res = subprocess.run(["cat", 
   "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/README.md"],
   capture_output=True, text=True)
print(res.stdout)

# -- cell 15 ------------------------------------------------------------------------
from build123d import *
import numpy as np

# ---------- parameters (mm, will scale to metres once, right before export) ----------
W        = 6.0     # channel width, mm  (request: 6 mm)
TOTAL_L  = 120.0    # overall main-channel length, mm (request: ~120 mm)
LEAD_IN  = 20.0     # straight lead length before first loop, mm
LEAD_OUT = 14.0     # straight lead length after last loop, mm
N_LOOPS  = 4
UNIT_L   = (TOTAL_L - LEAD_IN - LEAD_OUT) / N_LOOPS   # spacing between loop centres, mm
NECK     = 6.0      # gap between a loop's branch and rejoin points on the centreline, mm
BULGE_H  = 12.0     # how far each loop bulges from the centreline, mm

def teardrop_loop_edge(xc, sign=1.0):
    """Centreline edge of one teardrop bypass loop, centred at x=xc on the main
    centreline (y=0). sign=+1 bulges up (+y), sign=-1 bulges down (-y)."""
    d  = NECK / 2.0
    yc = (BULGE_H**2 - d**2) / (2 * BULGE_H)
    R  = BULGE_H - yc
    ang1 = np.degrees(np.arctan2(0 - yc, -d)) % 360
    ang2 = np.degrees(np.arctan2(0 - yc,  d)) % 360
    short_diff = (ang2 - ang1) % 360
    long_arc_size = -(360 - short_diff)     # go the long way, through the far side
    center_pt = (xc, sign * yc, 0)
    edge = CenterArc(center=center_pt, radius=R, start_angle=sign*0 + (ang1 if sign > 0 else -ang1),
                      arc_size=long_arc_size if sign > 0 else -long_arc_size)
    return edge.edges()[0]

# quick sanity check for the mirrored (sign=-1) branch before using it in the assembly
e_plus  = teardrop_loop_edge(30.0, sign=1.0)
e_minus = teardrop_loop_edge(30.0, sign=-1.0)
print("plus  start/end:", e_plus.start_point(), e_plus.end_point())
print("minus start/end:", e_minus.start_point(), e_minus.end_point())

# -- cell 16 ------------------------------------------------------------------------
# visualize both to confirm mirroring bulges the correct direction
fig3, ax3 = plt.subplots(figsize=(5,5))
for e, lbl in [(e_plus, "sign=+1 (bulge up)"), (e_minus, "sign=-1 (bulge down)")]:
    ts = np.linspace(0,1,200)
    pts = np.array([tuple(e.position_at(t)) for t in ts])
    ax3.plot(pts[:,0], pts[:,1], '-', label=lbl)
ax3.axhline(0, color='k', lw=0.5, ls='--')
ax3.legend()
ax3.set_aspect('equal')
fig3

# -- cell 17 ------------------------------------------------------------------------
# Both loop orientations mirror correctly. Now assemble the full 2D outline: main channel + 4 alternat
from build123d import *

# ---------- loop centre positions along the main channel, alternating sides ----------
loop_xc = [LEAD_IN + UNIT_L*(i + 0.5) for i in range(N_LOOPS)]
loop_sign = [1.0, -1.0, 1.0, -1.0]
print("loop centres (x, mm):", loop_xc, " unit spacing:", UNIT_L)

# ---------- main channel centreline, thickened to width W ----------
main_edge = Edge.make_line((0, 0, 0), (TOTAL_L, 0, 0))
main_band = make_face(offset(main_edge, amount=W/2, side=Side.BOTH).edges())

# ---------- 4 loop bands ----------
loop_bands = []
for xc, s in zip(loop_xc, loop_sign):
    le = teardrop_loop_edge(xc, sign=s)
    band = make_face(offset(le, amount=W/2, side=Side.BOTH).edges())
    loop_bands.append(band)

# ---------- union everything into one planar face ----------
valve_face_mm = main_band
for b in loop_bands:
    valve_face_mm = valve_face_mm + b

print("valve face (mm units): area=", valve_face_mm.area, "faces:", len(valve_face_mm.faces()))
bb = valve_face_mm.bounding_box()
print("bounding box (mm):", bb.min, bb.max)

# -- cell 18 ------------------------------------------------------------------------
bb_main = main_band.bounding_box()
print("main_band bbox (mm):", bb_main.min, bb_main.max)
print("main_band area:", main_band.area, "expected:", TOTAL_L*W)

# -- cell 19 ------------------------------------------------------------------------
edges_main = main_band.edges()
print(len(edges_main))
for e in edges_main:
    print(e.geom_type, e.start_point(), e.end_point())

# -- cell 20 ------------------------------------------------------------------------
test_line2 = Edge.make_line((0,0,0),(20,0,0))
for k in [Kind.ARC, Kind.INTERSECTION, Kind.TANGENT]:
    b = offset(test_line2, amount=3, side=Side.BOTH, kind=k)
    es = b.edges()
    types = [e.geom_type for e in es]
    bbox = b.bounding_box()
    print(k, "-> n_edges=",len(es), types, "bbox:", bbox.min, bbox.max)

# -- cell 21 ------------------------------------------------------------------------
from build123d import *

# main channel: a flat rectangle (flat, perpendicular end caps -- needed for inlet/outlet)
main_band = Pos(TOTAL_L/2, 0, 0) * Rectangle(TOTAL_L, W)
bb_main = main_band.bounding_box()
print("main_band (rectangle) bbox:", bb_main.min, bb_main.max, " area:", main_band.area,
      " expected area:", TOTAL_L*W)

# rebuild the full valve outline: main rectangle + 4 teardrop loop bands, unioned
loop_bands = []
for xc, s in zip(loop_xc, loop_sign):
    le = teardrop_loop_edge(xc, sign=s)
    band = make_face(offset(le, amount=W/2, side=Side.BOTH).edges())
    loop_bands.append(band)

valve_face_mm = main_band
for b in loop_bands:
    valve_face_mm = valve_face_mm + b

print("valve_face_mm: n_faces=", len(valve_face_mm.faces()), "area=", valve_face_mm.area)
bb = valve_face_mm.bounding_box()
print("overall bbox (mm):", bb.min, bb.max)

# -- cell 22 ------------------------------------------------------------------------
# Overall bounding box now exactly `[0,120] x [-15,15]`, confirming: main channel length = 120 mm (mat
import matplotlib.pyplot as plt
import numpy as np

fig4, ax4 = plt.subplots(figsize=(14,4))
for f in valve_face_mm.faces():
    for w in [f.outer_wire()] + f.inner_wires():
        pts_w = np.array([tuple(w.position_at(t)) for t in np.linspace(0,1,600)])
        ax4.plot(pts_w[:,0], pts_w[:,1], '-', color='C0')
ax4.set_aspect('equal')
ax4.set_title(f"Tesla valve outline (mm): L={TOTAL_L}, W={W}, 4 teardrop loops")
fig4

# -- cell 23 ------------------------------------------------------------------------
from build123d import *

# ---------- convert to metres (build123d numbers above were mm; scale once, here) ----------
MM_TO_M = 0.001
valve_face_m = scale(valve_face_mm, by=MM_TO_M)
bb_m = valve_face_m.bounding_box()
print("valve_face_m bbox (m):", bb_m.min, bb_m.max)

DEPTH_M = 0.002   # slab thickness for the "2D" extrusion, metres (one background cell in Z)
valve_solid = extrude(valve_face_m, amount=DEPTH_M)
print("solid volume (m^3):", valve_solid.volume, " expected approx:", valve_face_mm.area*1e-6*DEPTH_M)
bbs = valve_solid.bounding_box()
print("solid bbox (m):", bbs.min, bbs.max)

# -- cell 24 ------------------------------------------------------------------------
# Volume matches the planar area × depth exactly, and the bounding box confirms 120 mm length in metre
faces = valve_solid.faces()

X_TOL = 1e-6
inlet  = faces.filter_by_position(Axis.X, -X_TOL, X_TOL)
outlet = faces.filter_by_position(Axis.X, 0.12 - X_TOL, 0.12 + X_TOL)
# front/back are the flat z-normal planar faces at z=0 and z=DEPTH_M -- these will be
# declared "empty" directly in blockMeshDict, not tessellated as a snappy patch.
frontback = faces.filter_by(Axis.Z).filter_by(GeomType.PLANE)
walls = [f for f in faces if f not in list(inlet) + list(outlet) + list(frontback)]

print("inlet faces:", len(inlet), " outlet faces:", len(outlet),
      " frontback faces:", len(frontback), " wall faces:", len(walls),
      " total faces:", len(faces))

# sanity: inlet/outlet should each be a flat rectangle W x DEPTH
for name, grp in [("inlet", inlet), ("outlet", outlet)]:
    a = sum(f.area for f in grp)
    print(name, "area (m^2):", a, " expected W*DEPTH =", (W*MM_TO_M)*DEPTH_M)

# -- cell 25 ------------------------------------------------------------------------
import os, sys
os.makedirs("prep", exist_ok=True)
os.makedirs("case/constant/triSurface", exist_ok=True)

sys.path.insert(0, "/work/accept-t3/.toolbox")
sys.path.insert(0, "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox")
import cad_convert

export_step(valve_solid, "prep/valve.step", unit=Unit.M)

key = lambda fs: [tuple(f.center(CenterOf.MASS)) for f in fs]

# clmax: about half the finest wall cell size for the coarse pass; background dx ~ 0.0015 m.
CLMAX_M = 0.00075

frontback_list = list(frontback)
walls_list = list(walls)

report = cad_convert.export_patches(
    "prep/valve.step",
    {"inlet":     {"role": "inlet",  "faces": key(inlet)},
     "outlet":    {"role": "outlet", "faces": key(outlet)},
     "walls":     {"role": "wall",   "faces": key(walls_list)},
     "frontBack": {"role": "wall",   "faces": key(frontback_list)}},
    "case/constant/triSurface",
    clmax=CLMAX_M,
    location_in_mesh=[0.005, 0.0, 0.001],
)
print(cad_convert.render(report))

# -- cell 26 ------------------------------------------------------------------------
# Export succeeded, every B-rep face assigned, bounding box 0.12 × 0.03 × 0.002 m confirms 120 mm leng
import subprocess
res = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/cad_audit.py",
     "case", "--json"],
    capture_output=True, text=True)
print(res.stdout[-4000:])
print("STDERR", res.stderr[-2000:])

# -- cell 27 ------------------------------------------------------------------------
res = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/cad_audit.py",
     "--help"], capture_output=True, text=True)
print(res.stdout)
