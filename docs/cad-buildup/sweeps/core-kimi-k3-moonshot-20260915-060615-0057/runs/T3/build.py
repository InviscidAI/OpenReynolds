"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll design a 2D Tesla valve profile first: a straight 6 mm main channel with four teardrop loops (a
import numpy as np
import matplotlib.pyplot as plt

# ---- Parameters (mm; geometry built in mm, mesh scaled to metres later) ----
W = 6.0          # channel width
L = 120.0        # overall length
P = 27.0         # unit pitch
X0 = 6.0         # x of first upstream junction J1
THICK = 3.0      # extrusion thickness for the planar (2D) model

# Loop centerline waypoints, local coords: J1=(0,0) upstream, J2=(P,0) downstream.
# Tangent leaving J1 ~165 deg (hooks back upstream) so forward flow skips the loop
# and reverse-flow loop jet rejoins nearly head-on; tangent arriving J2 ~210 deg
# (mouth faces upstream, catches reverse flow). Loop bulges to +y; mirror for down loops.
LOOP_PTS = np.array([
    (0.0, 0.0),
    (-4.0, 1.2),
    (-5.5, 6.0),
    (-2.0, 11.5),
    (6.0, 15.0),
    (15.0, 15.5),
    (23.0, 13.0),
    (29.5, 7.5),
    (32.0, 3.5),
    (P, 0.0),
])

def catmull_rom(pts, n_per_seg=40):
    pts = np.asarray(pts, float)
    ext = np.vstack([2*pts[0]-pts[1], pts, 2*pts[-1]-pts[-2]])
    out = []
    for i in range(len(pts)-1):
        p0,p1,p2,p3 = ext[i], ext[i+1], ext[i+2], ext[i+3]
        t = np.linspace(0,1,n_per_seg,endpoint=False)[:,None]
        out.append(0.5*((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t**2 + (-p0+3*p1-3*p2+p3)*t**3))
    out.append(pts[-1][None,:])
    return np.vstack(out)

def offsets(cl, w):
    d = np.gradient(cl, axis=0)
    n = np.hypot(d[:,0], d[:,1])
    nrm = np.stack([-d[:,1]/n, d[:,0]/n], axis=1)
    return cl + nrm*w/2, cl - nrm*w/2

fig, ax = plt.subplots(figsize=(14,5))
# main channel
ax.add_patch(plt.Rectangle((0,-W/2), L, W, fill=False, ec='k', lw=1))
for i in range(4):
    sgn = 1 if i%2==0 else -1
    pts = LOOP_PTS.copy(); pts[:,1] *= sgn; pts[:,0] += X0 + i*P
    cl = catmull_rom(pts)
    a, b = offsets(cl, W)
    ax.plot(cl[:,0], cl[:,1], 'r-', lw=0.8)
    ax.plot(a[:,0], a[:,1], 'b-', lw=0.8)
    ax.plot(b[:,0], b[:,1], 'b-', lw=0.8)
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title('Tesla valve layout (mm)')
plt.tight_layout(); plt.show()

# report junction tangent angles from the sampled centerline
cl0 = catmull_rom(LOOP_PTS)
d = np.gradient(cl0, axis=0)
ang = np.degrees(np.arctan2(d[:,1], d[:,0]))
print("leaving J1 angle (deg):", ang[3])
print("arriving J2 angle (deg):", ang[-4])
print("loop y extent:", cl0[:,1].min(), cl0[:,1].max(), " x extent:", cl0[:,0].min(), cl0[:,0].max())

# -- cell 2 -------------------------------------------------------------------------
# curvature and clearance checks on one loop centerline
d1 = np.gradient(cl0, axis=0); d2 = np.gradient(d1, axis=0)
curv = np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]) / (np.hypot(d1[:,0],d1[:,1])**3 + 1e-12)
rad = 1.0/np.where(curv>1e-9, curv, np.inf)
print("min centerline bend radius (mm):", rad.min(), " (need > W/2 =", W/2, ")")

fig, axs = plt.subplots(1, 2, figsize=(14,5))
a0, b0 = offsets(cl0, W)
for ax, (lo, hi, ttl) in zip(axs, [((-10,6),(-6,8),'J1 (upstream junction)'),
                                   ((20,36),(-5,9),'J2 (downstream junction)')]):
    (xlo,xhi),(ylo,yhi) = lo, hi
    ax.plot(cl0[:,0], cl0[:,1], 'r-', lw=0.8)
    ax.plot(a0[:,0], a0[:,1], 'b-'); ax.plot(b0[:,0], b0[:,1], 'b-')
    ax.add_patch(plt.Rectangle((xlo,-W/2), xhi-xlo, W, fill=False, ec='k'))
    ax.set_xlim(xlo,xhi); ax.set_ylim(ylo,yhi); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(ttl)
plt.tight_layout(); plt.show()

# -- cell 3 -------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

W = 6.0; L = 123.0; P = 27.0; X0 = 9.0; THICK = 3.0   # mm

def u(a):  # unit vector, angle in degrees
    a = np.radians(a); return np.array([np.cos(a), np.sin(a)])

def arc_pts(c, r, a0, a1, n=60):  # angles deg, swept a0 -> a1
    ang = np.radians(np.linspace(a0, a1, n))
    return c[None,:] + r*np.stack([np.cos(ang), np.sin(ang)], axis=1)

def loop_centerline():
    J1 = np.array([0.0, 0.0]); J2 = np.array([P, 0.0])
    # piece 1: straight lead-out at 165 deg, 3 mm
    S1 = J1 + 3.0*u(165.0)
    # piece 2: CW arc R=5, heading 165 -> 60  (phi: 255 -> 150)
    C1 = S1 + 5.0*u(75.0)
    E1 = C1 + 5.0*u(150.0)
    # piece 4 (computed first to size piece 3): J2 hook, CW, heading 300 -> 210
    Rh = 8.106/1.5          # solved so big arc start/end heights match (R*u(120).y = A.y - R*u(30).y)
    C3 = J2 + Rh*u(120.0); A = C3 + Rh*u(30.0)
    # piece 3: big CW arc from E1 (heading 60, phi=150) to A (heading 300, phi=30)
    Rb = (E1[0]-A[0]) / (u(150.0)[0]-u(30.0)[0])
    C2 = E1 - Rb*u(150.0)
    cl = np.vstack([
        J1[None,:], S1[None,:],
        arc_pts(C1, 5.0, 255, 150, 50),
        arc_pts(C2, Rb, 150, 30, 120),
        arc_pts(C3, Rh, 30, -60, 45),
        J2[None,:]])
    return cl, (C1, 5.0, C2, Rb, C3, Rh)

cl_up, geo = loop_centerline()
C1,R1,C2,Rb,C3,Rh = geo
print("piece radii: R1=%.3f  Rbig=%.3f  Rhook=%.3f  (W/2=%.1f)" % (R1,Rb,Rh,W/2))
print("big-arc join height check: E1.y=%.4f  A.y=%.4f" % ((C1+5*u(150))[1], (C3+Rh*u(30))[1]))

d = np.gradient(cl_up, axis=0)
ang = np.degrees(np.arctan2(d[:,1], d[:,0]))
print("tangent leaving J1: %.2f deg ; arriving J2: %.2f deg" % (ang[2], (ang[-3]+180)%360-180 if False else ang[-3]))
d2 = np.gradient(d, axis=0)
curv = np.abs(d[:,0]*d2[:,1]-d[:,1]*d2[:,0])/(np.hypot(d[:,0],d[:,1])**3+1e-12)
print("min bend radius from samples: %.3f mm" % (1/curv.max()))
print("loop extent x [%.2f, %.2f], y [%.2f, %.2f]" % (cl_up[:,0].min(), cl_up[:,0].max(), cl_up[:,1].min(), cl_up[:,1].max()))

fig, ax = plt.subplots(figsize=(14,5))
ax.add_patch(plt.Rectangle((0,-W/2), L, W, fill=False, ec='k', lw=1.2))
for i in range(4):
    sgn = 1 if i%2==0 else -1
    cl = cl_up.copy(); cl[:,1] *= sgn; cl[:,0] += X0 + i*P
    a, b = offsets(cl, W)
    ax.plot(cl[:,0], cl[:,1], 'r-', lw=0.7)
    ax.plot(a[:,0], a[:,1], 'b-', lw=0.9); ax.plot(b[:,0], b[:,1], 'b-', lw=0.9)
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title('Tesla valve v2 (mm)')
plt.tight_layout(); plt.show()

# -- cell 4 -------------------------------------------------------------------------
# Leftover junk line — rerunning cleanly without it.
def arc_pts(c, r, a0, a1, n=60):
    ang = np.radians(np.linspace(a0, a1, n, endpoint=False))
    return c[None,:] + r*np.stack([np.cos(ang), np.sin(ang)], axis=1)

def loop_centerline():
    J1 = np.array([0.0, 0.0]); J2 = np.array([P, 0.0])
    S1 = J1 + 3.0*u(165.0)
    C1 = S1 + 5.0*u(75.0)
    E1 = C1 + 5.0*u(150.0)
    Rh = E1[1]/(np.sin(np.radians(120.0))+np.sin(np.radians(30.0)))
    C3 = J2 + Rh*u(120.0); A = C3 + Rh*u(30.0)
    Rb = (E1[0]-A[0]) / (u(150.0)[0]-u(30.0)[0])
    C2 = E1 - Rb*u(150.0)
    cl = np.vstack([arc_pts(C1, 5.0, 255, 150, 60),
                    arc_pts(C2, Rb, 150, 30, 140),
                    arc_pts(C3, Rh, 30, -60, 50),
                    J2[None,:]])
    return cl, (C1, 5.0, C2, Rb, C3, Rh)

cl_up, geo = loop_centerline()
C1,R1,C2,Rb,C3,Rh = geo
print("R1=%.3f Rbig=%.3f Rhook=%.3f  join dy = %.2e" % (R1, Rb, Rh,
      (C1+5*u(150))[1]-(C3+Rh*u(30))[1]))
d = np.gradient(cl_up, axis=0); d2 = np.gradient(d, axis=0)
curv = np.abs(d[:,0]*d2[:,1]-d[:,1]*d2[:,0])/(np.hypot(d[:,0],d[:,1])**3+1e-12)
ang = np.degrees(np.arctan2(d[:,1], d[:,0]))
print("tangent leaving J1: %.2f deg, arriving J2: %.2f deg, min bend R: %.3f mm"
      % (ang[1], ang[-2], 1/curv.max()))

fig, ax = plt.subplots(figsize=(14,5))
ax.add_patch(plt.Rectangle((0,-W/2), L, W, fill=False, ec='k', lw=1.2))
loops = []
for i in range(4):
    sgn = 1 if i%2==0 else -1
    cl = cl_up.copy(); cl[:,1] *= sgn; cl[:,0] += X0 + i*P
    loops.append(cl)
    a, b = offsets(cl, W)
    ax.plot(a[:,0], a[:,1], 'b-', lw=0.9); ax.plot(b[:,0], b[:,1], 'b-', lw=0.9)
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title('Tesla valve v3 (mm)')
plt.tight_layout(); plt.show()

for i in range(3):
    a1,b1 = offsets(loops[i],W); a2,b2 = offsets(loops[i+1],W)
    w1 = np.vstack([a1,b1]); w2 = np.vstack([a2,b2])
    dd = np.sqrt(((w1[:,None,:]-w2[None,:,:])**2).sum(-1)).min()
    print("min wall distance loop %d <-> loop %d: %.2f mm" % (i, i+1, dd))

# -- cell 5 -------------------------------------------------------------------------
import build123d as bd

def loop_face(cl, w, n_sub=4):
    a, b = offsets(cl, w)
    a = a[::n_sub]; b = b[::n_sub]
    e_out = bd.Spline([tuple(p) for p in a])
    e_in  = bd.Spline([tuple(p) for p in b[::-1]])
    cap1 = bd.Line(e_out @ 0, e_in @ 1)      # at J1 end
    cap2 = bd.Line(e_in @ 0, e_out @ 1)      # at J2 end
    return bd.Face(bd.Wire([e_out, cap2, e_in, cap1]))

main_face = bd.Pos(L/2, 0, 0) * bd.Rectangle(L, W)
faces = [main_face] + [loop_face(cl, W) for cl in loops]
profile = faces[0]
for f in faces[1:]:
    profile = profile + f
print("profile faces after union:", len(profile.faces()), " area: %.1f mm^2" % profile.area)

solid = bd.extrude(profile, amount=THICK)
print("solid bbox (mm):", solid.bounding_box())
bd.export_step(solid, "tesla_valve.step")
bd.export_stl(solid, "tesla_valve.stl")
print("exported tesla_valve.step / .stl")

# -- cell 6 -------------------------------------------------------------------------
# The union didn't merge — 5 separate faces means the loop faces aren't fusing with the rectangle. Let
for i, f in enumerate(faces):
    bb = f.bounding_box()
    print(i, "center:", tuple(round(v,3) for v in f.center()), 
          " z-range: %.3f..%.3f" % (bb.min.Z, bb.max.Z),
          " y-range: %.3f..%.3f" % (bb.min.Y, bb.max.Y),
          " area: %.1f" % f.area)
print(type(profile), len(profile.faces()))
# test pairwise: does loop 0 face intersect main face?
inter = faces[0] & faces[1]
print("intersection main & loop0:", type(inter), getattr(inter, 'area', None))

# -- cell 7 -------------------------------------------------------------------------
main_solid = bd.Pos(L/2, 0, 0) * bd.Box(L, W, THICK)
loop_solids = [bd.extrude(loop_face(cl, W), amount=THICK/2, both=True) for cl in loops]
solid = main_solid
for s in loop_solids:
    solid = solid + s
print("type:", type(solid).__name__, " solids:", len(solid.solids()), " faces:", len(solid.faces()))
print("volume: %.1f mm^3 (2D area %.1f)" % (solid.volume, solid.volume/THICK))
bb = solid.bounding_box()
print("bbox x [%.2f, %.2f]  y [%.2f, %.2f]  z [%.2f, %.2f]" % (bb.min.X, bb.max.X, bb.min.Y, bb.max.Y, bb.min.Z, bb.max.Z))
bd.export_step(solid, "tesla_valve.step")
bd.export_stl(solid, "tesla_valve.stl")

import pyvista as pv
m = pv.read("tesla_valve.stl")
pl = pv.Plotter(off_screen=True)
pl.add_mesh(m, color="lightsteelblue", show_edges=False)
pl.view_xy(); pl.camera.zoom(1.1)
pl.show()

# -- cell 8 -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14,5))
for ax, (xc, ttl) in zip(axs, [(36.0,'junction x=36 (up-tail + down-head)'), (9.0,'first branch J1 x=9')]):
    ax.add_patch(plt.Rectangle((xc-12,-W/2), 24, W, fill=False, ec='k', lw=1.2))
    for cl in loops:
        a, b = offsets(cl, W)
        ax.plot(a[:,0], a[:,1], 'b-', lw=1.0); ax.plot(b[:,0], b[:,1], 'b-', lw=1.0)
    ax.set_xlim(xc-12, xc+12); ax.set_ylim(-12, 12)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title(ttl)
plt.tight_layout(); plt.show()

# -- cell 9 -------------------------------------------------------------------------
# Final geometry constants (mm). X0 raised 9->10 so the first loop clears the x=0 inlet plane.
X0 = 10.0; L = 124.0
loops = []
for i in range(4):
    sgn = 1 if i % 2 == 0 else -1
    cl = cl_up.copy(); cl[:,1] *= sgn; cl[:,0] += X0 + i*P
    loops.append(cl)

main_solid = bd.Pos(L/2, 0, 0) * bd.Box(L, W, 6.0)          # z in [-3,3]: proud of future mesh slab
loop_solids = [bd.extrude(loop_face(cl, W), amount=3.0, both=True) for cl in loops]
solid = main_solid
for s in loop_solids:
    solid = solid + s
print("solids:", len(solid.solids()), " volume: %.1f mm^3" % solid.volume)
bb = solid.bounding_box()
print("bbox x [%.2f, %.2f]  y [%.2f, %.2f]  z [%.2f, %.2f]" % (bb.min.X,bb.max.X,bb.min.Y,bb.max.Y,bb.min.Z,bb.max.Z))
bd.export_step(solid, "tesla_valve.step")
bd.export_stl(solid, "tesla_valve.stl")

# quick measured numbers vs the request
print("channel width (built): %.1f mm | overall length (channel): %.0f mm | loops: 4" % (W, L))
