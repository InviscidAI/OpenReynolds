"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# First cell: define dimensions (metres), build the Tesla-valve fluid solid analytically (main channel
import numpy as np
from build123d import *
import matplotlib.pyplot as plt

# --- parameters (metres) ---
W = 0.006      # channel width
L = 0.120      # overall length
T = 0.003      # extrusion thickness (plane-2D slab)
NLOOPS = 4
ALPHA = np.deg2rad(55)   # branch leave angle
NPTS = 240               # centerline samples per loop
LOOP_PITCH = 0.024
X0 = 0.012               # first loop station
L1 = 0.014               # leave connector length
R_LOOP = 0.006           # loop arc radius
GAMMA = np.deg2rad(200)  # loop turn angle (clockwise)

# main channel box (x from 0..L, y -W/2..W/2)
main = Pos(L/2, 0, T/2) * Box(L, W, T)

def loop_wire(xs):
    """centerline of one teardrop loop leaving top wall at (xs, W/2)"""
    a = np.array([xs, W/2])
    d1 = np.array([np.cos(ALPHA), np.sin(ALPHA)])
    p1 = a + L1*d1
    n0 = np.array([np.sin(ALPHA), -np.cos(ALPHA)])  # right normal (clockwise turn)
    center = p1 + R_LOOP*n0
    pts = [tuple(a), tuple(p1)]
    for i in range(1, 60):  # arc samples (clockwise)
        th = GAMMA*i/60
        ang = np.arctan2(p1[1]-center[1], p1[0]-center[0]) - th
        pts.append((center[0]+R_LOOP*np.cos(ang), center[1]+R_LOOP*np.sin(ang)))
    d_end = np.array([np.cos(ALPHA-GAMMA), np.sin(ALPHA-GAMMA)])
    p_end = pts[-1]
    # extend return connector so it penetrates the main channel
    l2 = (p_end[1] + W) / abs(d_end[1])
    pts.append((p_end[0] + l2*d_end[0], p_end[1] + l2*d_end[1]))
    return Wire(Edge.make_spline([Vector(*p, 0) for p in pts]))

loops = []
for k in range(NLOOPS):
    xs = X0 + k*LOOP_PITCH
    w = loop_wire(xs)
    a = w.position_at(0)
    d = w.tangent_at(0)
    pln = Plane(origin=a, x_dir=(0,0,1), z_dir=Vector(*d,0))
    sec = Rectangle(T, W).locate(Location(pln))  # width T along z, height W in-plane
    loops.append(sweep(sec, path=w, transition=Transition.ROUND))

fluid = main
for lp in loops:
    fluid = fluid + lp

print("volume (m^3):", fluid.volume)
print("bbox:", fluid.bounding_box().min, fluid.bounding_box().max)
print("expected union volume approx box+loops:",
      L*W*T, "+", sum(lp.volume for lp in loops))

# 2D outline of the top face for a visual check
top = fluid.faces().filter_by(Plane.ZY.reference) if False else None
f = max(fluid.faces(), key=lambda ff: ff.center().Z)
xs_, ys_ = [], []
for w_ in [Wire(f.edges()[0])]:
    pass
# gather boundary sampled points of all horizontal-ish top edges
pts2 = []
topface = [fc for fc in fluid.faces() if abs(fc.normal_at().Z) > 0.999]
big = max(topface, key=lambda fc: fc.area)
for e in big.edges():
    try:
        pts2.append([Vector(*p) for p in e.sample(20)])
    except Exception as ex:
        print("edge sample fail", ex)
fig, ax = plt.subplots(figsize=(16,4))
for seg in pts2:
    pts2a = np.array(seg)
    ax.plot(pts2a[:,0], pts2a[:,1], 'b.-', ms=2)
ax.set_aspect('equal'); ax.grid(True)
plt.savefig("outline.png", dpi=120); plt.show()

# -- cell 2 -------------------------------------------------------------------------
print("main vol:", main.volume, "bbox:", main.bounding_box())
for i, lp in enumerate(loops):
    bb = lp.bounding_box()
    print(i, "loop vol", round(lp.volume,12), "bbox y", round(bb.min.Y,5), round(bb.max.Y,5))

# -- cell 3 -------------------------------------------------------------------------
# union attempt with explicit target: connectors stop at mid-channel depth (y=0)
def loop_wire2(xs):
    a = np.array([xs, W/2]); d1 = np.array([np.cos(ALPHA), np.sin(ALPHA)])
    p1 = a + L1*d1
    n0 = np.array([np.sin(ALPHA), -np.cos(ALPHA)])
    center = p1 + R_LOOP*n0
    pts = [tuple(a), tuple(p1)]
    thg = np.linspace(GAMMA/60, GAMMA, 60)
    base = np.arctan2(p1[1]-center[1], p1[0]-center[0])
    arcs = [(center[0]+R_LOOP*np.cos(base-t), center[1]+R_LOOP*np.sin(base-t)) for t in thg]
    pts.extend(arcs)
    d_end = np.array([np.cos(ALPHA-GAMMA), np.sin(ALPHA-GAMMA)])
    l2 = pts[-1][1]/abs(d_end[1])   # stop at y=0
    pts.append((pts[-1][0]+l2*d_end[0], pts[-1][1]+l2*d_end[1]))
    return Wire(Edge.make_spline([Vector(*p,0) for p in pts]))

test = loop_wire2(X0)
pln = Plane(origin=test.position_at(0), x_dir=(0,0,1), z_dir=Vector(*test.tangent_at(0),0))
sec = Rectangle(T, W).locate(Location(pln))
lp0 = sweep(sec, path=test, transition=Transition.ROUND)
u = main + lp0
bb = u.bounding_box()
print("lp0 vol:", lp0.volume, "u vol:", u.volume, "bbox:", bb)
print("sum vols:", main.volume + lp0.volume)

# -- cell 4 -------------------------------------------------------------------------
b1 = main + Pos(0.05,0,0.0015)*Box(0.01,0.01,0.001)
print("box+box vol:", b1.volume, main.volume+0.01*0.01*0.001)
lp0s = lp0.solid() if hasattr(lp0, 'solid') else None
print("solid vol:", lp0s.volume if lp0s else None)
u3 = main + lp0s
print("main+solid:", u3.volume, "vs sum", main.volume + lp0s.volume)
u4 = Part(main) + lp0 if isinstance(lp0, Part) else None
print("part+part:", u4.volume if u4 else 'n/a')

# -- cell 5 -------------------------------------------------------------------------
print("lp0s type:", type(lp0s), "valid:", lp0s.is_valid)
lp_far = lp0s.moved(Location((0.0, 1.0, 0)))
u5 = main + lp_far
print("disjoint union vol:", u5.volume, "expected:", main.volume+lp_far.volume)
print("u5 bbox:", u5.bounding_box())
