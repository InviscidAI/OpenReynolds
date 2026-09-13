"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run that first cell properly.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters (metres) ----
C      = 0.022          # chord
TH     = 0.06           # max thickness / chord
CAMB   = 0.06           # max camber / chord (design choice, NACA-4 style)
CAMB_P = 0.40
R_HUB  = 0.045
R_TIP  = 0.090
NB     = 28
PITCH  = 2*np.pi/NB
STAG_R = np.radians(30.0)
TWIST  = np.radians(35.0)

def stagger(r):
    return STAG_R + TWIST*(r-R_HUB)/(R_TIP-R_HUB)

def airfoil(n=81):
    b = (1-np.cos(np.linspace(0, np.pi, n)))/2
    yt = 5*TH*(0.2969*np.sqrt(b)-0.1260*b-0.3516*b**2+0.2843*b**3-0.1036*b**4)
    yc = np.where(b < CAMB_P, CAMB/CAMB_P**2*(2*CAMB_P*b-b**2),
                              CAMB/(1-CAMB_P)**2*((1-2*CAMB_P)+2*CAMB_P*b-b**2))
    dy = np.where(b < CAMB_P, 2*CAMB/CAMB_P**2*(CAMB_P-b),
                              2*CAMB/(1-CAMB_P)**2*(CAMB_P-b))
    ph = np.arctan(dy)
    xs, ys = b-yt*np.sin(ph), yc+yt*np.cos(ph)
    xp, yp = b+yt*np.sin(ph), yc-yt*np.cos(ph)
    return np.c_[xs,ys]*C, np.c_[xp,yp]*C, np.c_[b,yc]*C

S_AF, P_AF, CAM_AF = airfoil()
X0, Y0 = C/2, np.interp(C/2, CAM_AF[:,0], CAM_AF[:,1])

def to_ztheta(pts2d, r):
    l = stagger(r)
    u, v = pts2d[:,0]-X0, pts2d[:,1]-Y0
    return u*np.cos(l)-v*np.sin(l), (u*np.sin(l)+v*np.cos(l))/r

for r in (R_HUB, (R_HUB+R_TIP)/2, R_TIP):
    zs,ts = to_ztheta(S_AF,r); zp,tp = to_ztheta(P_AF,r)
    zmax=max(zs.max(),zp.max()); zmin=min(zs.min(),zp.min())
    print(f"r={r*1000:5.1f} stag={np.degrees(stagger(r)):4.1f}deg z-ext={(zmax-zmin)*1000:5.2f}mm "
          f"th-ext={(max(ts.max(),tp.max())-min(tp.min(),ts.min()))/PITCH:5.2f}pitch "
          f"chord={np.hypot(zs[-1]-zs[0],(ts[-1]-ts[0])*r)*1000:5.2f}mm "
          f"tang_thick={(ts-tp).max()*r/ (PITCH*r):.3f}pitch")
print("pitch hub/tip mm:", PITCH*R_HUB*1000, PITCH*R_TIP*1000)
print("max t/c:", np.max(np.hypot(S_AF[:,0]-P_AF[:,0], S_AF[:,1]-P_AF[:,1]))/C)

# -- cell 2 -------------------------------------------------------------------------
# Chord, thickness and stagger check out. Now the blade-to-blade passage outline (lower bound = suctio
z_LE = lambda r: to_ztheta(S_AF, r)[0][0]
z_TE = lambda r: to_ztheta(S_AF, r)[0][-1]
rr = np.linspace(R_HUB, R_TIP, 21)
Z_IN  = min(z_LE(r) for r in rr) - C          # one chord upstream of most-upstream LE
Z_OUT = max(z_TE(r) for r in rr) + C          # one chord downstream of most-downstream TE
print("Z_IN, Z_OUT (mm):", Z_IN*1000, Z_OUT*1000, " axial length:", (Z_OUT-Z_IN)*1000)

def section_loop(r, n_ax=15):
    """closed passage outline at radius r, as list of (z,theta) polyline segments in order:
       A(lower,upstream), suction(blade0), B(lower,downstream), outlet, B'(upper), pressure(blade1), A'(upper), inlet"""
    zs, ts = to_ztheta(S_AF, r)
    zp, tp = to_ztheta(P_AF, r)
    tLE, tTE = ts[0], ts[-1]
    A  = np.c_[np.linspace(Z_IN, zs[0], n_ax),  np.full(n_ax, tLE)]
    Sc = np.c_[zs, ts]
    B  = np.c_[np.linspace(zs[-1], Z_OUT, n_ax), np.full(n_ax, tTE)]
    out= np.c_[[Z_OUT, Z_OUT], [tTE, tTE+PITCH]]
    Bu = B[::-1].copy(); Bu[:,1] += PITCH
    Pc = np.c_[zp, tp][::-1].copy(); Pc[:,1] += PITCH        # pressure side of blade+1, TE->LE
    Au = A[::-1].copy(); Au[:,1] += PITCH
    inl= np.c_[[Z_IN, Z_IN], [tLE+PITCH, tLE]]
    return [A, Sc, B, out, Bu, Pc, Au, inl]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, r in zip(axes, (R_HUB, (R_HUB+R_TIP)/2, R_TIP)):
    segs = section_loop(r)
    names = ["per1_up","suction","per1_dn","outlet","per2_dn","pressure","per2_up","inlet"]
    for sgm, nm in zip(segs, names):
        ax.plot(sgm[:,0]*1000, sgm[:,1]*r*1000, lw=1.8, label=nm)
    ax.set_title(f"r={r*1000:.0f} mm, stagger {np.degrees(stagger(r)):.0f}°")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("r·θ [mm]"); ax.set_aspect("equal")
axes[0].legend(fontsize=7)
# validity: min gap between lower and upper boundary along z
for r in rr[::5]:
    zs,ts = to_ztheta(S_AF,r); zp,tp = to_ztheta(P_AF,r)
    gap = (tp+PITCH-ts).min()*r*1000
    print(f"r={r*1000:.0f}mm  min blade-to-blade gap = {gap:.2f} mm, z monotone S/P:",
          bool(np.all(np.diff(zs)>0)), bool(np.all(np.diff(zp)>0)))
fig

# -- cell 3 -------------------------------------------------------------------------
# Passage outline is valid at all radii (min gap 9–20 mm). Now lift each section onto its cylinder and
from build123d import *
import numpy as np

def seg_pts(r, n_ax=9, n_bl=41):
    """the 8 boundary segments at radius r as (z,theta) arrays with exactly shared endpoints"""
    zs, ts = to_ztheta(S_AF, r); zp, tp = to_ztheta(P_AF, r)
    k = np.linspace(0, len(zs)-1, n_bl).astype(int)
    zs, ts, zp, tp = zs[k], ts[k], zp[k], tp[k]
    tLE, tTE = ts[0], ts[-1]
    A  = np.c_[np.linspace(Z_IN, zs[0], n_ax),  np.full(n_ax, tLE)]
    Sc = np.c_[zs, ts]
    B  = np.c_[np.linspace(zs[-1], Z_OUT, n_ax), np.full(n_ax, tTE)]
    O  = np.c_[np.full(5, Z_OUT), np.linspace(tTE, tTE+PITCH, 5)]
    Bu = B[::-1].copy(); Bu[:,1] = Bu[:,1] + PITCH
    Pc = np.c_[zp, tp][::-1].copy(); Pc[:,1] = Pc[:,1] + PITCH
    Au = A[::-1].copy(); Au[:,1] = Au[:,1] + PITCH
    I  = np.c_[np.full(5, Z_IN), np.linspace(tLE+PITCH, tLE, 5)]
    return [A, Sc, B, O, Bu, Pc, Au, I]

def sec_wire(r):
    edges = []
    for sg in seg_pts(r):
        pts = [(r*np.cos(t), r*np.sin(t), z) for z, t in sg]
        edges.append(Spline(*[Vector(*p) for p in pts]))
    return Wire(edges)

R_LO, R_HI = R_HUB-0.003, R_TIP+0.003          # overshoot, trimmed by hub/shroud cylinders later
radii = np.linspace(R_LO, R_HI, 9)
wires = [sec_wire(r) for r in radii]
print("wires:", len(wires), "edges each:", [len(w.edges()) for w in wires][:3],
      "closed:", all(w.is_closed for w in wires))

# -- cell 4 -------------------------------------------------------------------------
# The loft cell errored before binding, so I re-emit it as one self-contained cell.
blank = Solid.make_loft(wires, ruled=True)
bb = blank.bounding_box()
print("vol(mm^3)=%.1f valid=%s faces=%d" % (blank.volume*1e9, blank.is_valid, len(blank.faces())))
print("bbox mm:", [round(v*1000,2) for v in tuple(bb.min)+tuple(bb.max)])

# -- cell 5 -------------------------------------------------------------------------
L  = Z_OUT - Z_IN
ZC = 0.5*(Z_IN + Z_OUT)
annulus = Pos(0,0,ZC) * (Cylinder(R_TIP, L) - Cylinder(R_HUB, 1.2*L))
passage = blank & annulus
print("passage vol(mm^3)=%.1f valid=%s faces=%d solids=%d" %
      (passage.volume*1e9, passage.is_valid, len(passage.faces()), len(passage.solids())))
bb = passage.bounding_box(); print("bbox mm:", [round(v*1000,2) for v in tuple(bb.min)+tuple(bb.max)])

# -- cell 6 -------------------------------------------------------------------------
# Volume matches the analytic estimate (sector 42980 mm³ − one blade ≈ 890 mm³ ⇒ 42091 vs 42107). Now 
SEG_GROUP = ["periodic1","suction","periodic1","outlet","periodic2","pressure","periodic2","inlet"]

def _seg_dist(pt_zt, seg, r):
    """min distance (metres, in z / r*theta plane) from point to polyline segment"""
    p = np.array([pt_zt[0], pt_zt[1]*r])
    q = np.c_[seg[:,0], seg[:,1]*r]
    d = np.hypot(*(p - q).T).min()
    for a, b in zip(q[:-1], q[1:]):
        ab = b - a; L2 = ab @ ab
        if L2 > 0:
            t = np.clip((p - a) @ ab / L2, 0, 1)
            d = min(d, np.hypot(*(p - (a + t*ab))))
    return d

def classify(x, y, z, tol=2e-4):
    r = np.hypot(x, y); th = np.arctan2(y, x)
    if abs(r - R_HUB) < tol:  return "hub", 0.0
    if abs(r - R_TIP) < tol:  return "shroud", 0.0
    if abs(z - Z_IN)  < tol:  return "inlet", 0.0
    if abs(z - Z_OUT) < tol:  return "outlet", 0.0
    segs = seg_pts(r)
    # theta is periodic: test the point and its +/- one-pitch images
    best = (1e9, None)
    for dth in (-PITCH, 0.0, PITCH):
        for i, sg in enumerate(segs):
            d = _seg_dist((z, th + dth), sg, r)
            if d < best[0]: best = (d, i)
    return SEG_GROUP[best[1]], best[0]

from collections import Counter
cnt = Counter(); worst = 0
for f in passage.faces():
    c = f.center()
    g, d = classify(c.X, c.Y, c.Z); cnt[g] += 1; worst = max(worst, d)
print(cnt, " worst curve distance (mm):", worst*1000)

# -- cell 7 -------------------------------------------------------------------------
# The ±pitch image loop was the bug — it maps periodic1 onto periodic2. The domain never wraps past ±π
def classify(x, y, z, tol=2e-4):
    r = np.hypot(x, y); th = np.arctan2(y, x)
    if abs(r - R_HUB) < tol:  return "hub", 0.0
    if abs(r - R_TIP) < tol:  return "shroud", 0.0
    if abs(z - Z_IN)  < tol:  return "inlet", 0.0
    if abs(z - Z_OUT) < tol:  return "outlet", 0.0
    segs = seg_pts(r)
    d = [_seg_dist((z, th), sg, r) for sg in segs]
    i = int(np.argmin(d))
    return SEG_GROUP[i], d[i]

cnt = Counter(); worst = 0
for f in passage.faces():
    c = f.center(); g, d = classify(c.X, c.Y, c.Z); cnt[g] += 1; worst = max(worst, d)
print(cnt, "total", sum(cnt.values()), " worst dist (mm):", worst*1000)
export_step(passage, "passage.step")
import os; print("step written", os.path.getsize("passage.step"))

# -- cell 8 -------------------------------------------------------------------------
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("passage")
gmsh.model.occ.importShapes("passage.step")
gmsh.model.occ.synchronize()
print("volumes:", gmsh.model.getEntities(3), "surfaces:", len(gmsh.model.getEntities(2)))
print("bbox:", [round(v,5) for v in gmsh.model.getBoundingBox(-1,-1)])

# -- cell 9 -------------------------------------------------------------------------
# Units are metres. I'll mesh coarse first, then classify each surface by majority vote over its own t
LC = 0.004   # coarse first
gmsh.option.setNumber("Mesh.MeshSizeMin", LC*0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", LC)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
gmsh.model.mesh.generate(3)
ne = gmsh.model.mesh.getElements(3)[1][0].size
print("tets:", ne)

# -- cell 10 ------------------------------------------------------------------------
# Mesh volume 4.2097e-5 m³ matches the CAD. Now classify each surface by its own triangle centroids an
nt, nc, _ = gmsh.model.mesh.getNodes()
XYZ = {int(t): nc[3*i:3*i+3] for i, t in enumerate(nt)}
groups = {}
report = []
for dim, tag in gmsh.model.getEntities(2):
    et, ei, en = gmsh.model.mesh.getElements(2, tag)
    tri = en[0].reshape(-1, 3)
    cen = np.array([(XYZ[a]+XYZ[b]+XYZ[c])/3 for a, b, c in tri])
    votes = Counter(classify(*p)[0] for p in cen)
    g, n = votes.most_common(1)[0]
    groups.setdefault(g, []).append(tag)
    report.append((tag, g, n/len(cen)))
print({k: len(v) for k, v in groups.items()})
print("min vote purity:", min(r[2] for r in report), " impure faces:", [r for r in report if r[2] < 0.999])

# -- cell 11 ------------------------------------------------------------------------
# Unanimous on all 52 faces. Tag physicals, write the mesh, and set up a minimal case for `gmshToFoam`
import os, textwrap, subprocess
NAME = {"pressure":"bladePressure","suction":"bladeSuction","hub":"hub","shroud":"shroud",
        "inlet":"inlet","outlet":"outlet","periodic1":"periodic1","periodic2":"periodic2"}
gmsh.model.removePhysicalGroups()
for g, tags in groups.items():
    p = gmsh.model.addPhysicalGroup(2, tags); gmsh.model.setPhysicalName(2, p, NAME[g])
pv = gmsh.model.addPhysicalGroup(3, [1]); gmsh.model.setPhysicalName(3, pv, "internal")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("passage.msh")

os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = lambda cls, obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
  "application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
  "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+
  "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
  "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
  "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} SIMPLE{}\n")
r = subprocess.run(["gmshToFoam","passage.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1200:], r.stderr[-500:])

# -- cell 12 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
i = out.find("Checking patch topology")
print(out[i:i+1500])
print(out[out.find("Checking geometry"):][:2500])

# -- cell 13 ------------------------------------------------------------------------
# Coarse mesh is on disk and `checkMesh` says OK. Now look at it, coloured by patch.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,"start_xvfb") else None
pv.global_theme.background = "white"
open("case.foam","w").write("")
mesh = pv.OpenFOAMReader("case.foam")
mesh.set_active_time_value(0.0)
d = mesh.read()
bnd = d["boundary"]
print(bnd.keys())
pl = pv.Plotter(off_screen=True, window_size=(1100, 800))
cols = dict(hub="tan", shroud="lightsteelblue", inlet="green", outlet="red",
            bladePressure="orange", bladeSuction="purple", periodic1="pink", periodic2="lightgray")
for k in bnd.keys():
    op = 0.35 if k in ("periodic1","shroud") else 1.0
    pl.add_mesh(bnd[k], color=cols.get(k,"black"), opacity=op, show_edges=True, line_width=0.3, label=k)
pl.add_legend(size=(0.18,0.25))
pl.camera_position = [(0.15,0.12,0.12),(0.065,0.01,0.0),(0,0,1)]
pl.show()

# -- cell 14 ------------------------------------------------------------------------
# Patch colours are hard to read from that angle. Let me hide both periodic faces and look down the ax
pl = pv.Plotter(off_screen=True, window_size=(1300, 600), shape=(1,2))
for j, cam in enumerate([[(0.02,0.03,0.16),(0.067,0.01,0.0),(0,1,0)],
                         [(0.30,0.05,0.05),(0.067,0.01,0.0),(0,0,1)]]):
    pl.subplot(0,j)
    for k in bnd.keys():
        if k.startswith("periodic"): continue
        pl.add_mesh(bnd[k].copy(), color=cols[k], show_edges=True, line_width=0.3)
    pl.camera_position = cam
pl.add_legend([(k,cols[k]) for k in cols if not k.startswith("per")], size=(0.25,0.3))
pl.show()

# -- cell 15 ------------------------------------------------------------------------
# Now measure the delivered numbers from the mesh itself, not from my parameters: patch-point clouds i
def unwrap(pts):
    r = np.hypot(pts[:,0], pts[:,1]); th = np.arctan2(pts[:,1], pts[:,0])
    return r, th, pts[:,2]

bl = np.vstack([bnd["bladePressure"].points, bnd["bladeSuction"].points])
rb, tb, zb = unwrap(bl)
print("blade radial range (mm): %.2f .. %.2f  span %.2f" % (rb.min()*1e3, rb.max()*1e3, (rb.max()-rb.min())*1e3))
for lo, hi, lbl in [(0.045,0.0465,"root"), (0.0885,0.0900,"tip")]:
    m = (rb>=lo)&(rb<=hi); rm = rb[m].mean()
    P = np.c_[zb[m], tb[m]*rm]; P = P - P.mean(0)
    u = np.linalg.svd(P, full_matrices=False)[2][0]
    ang = np.degrees(np.arctan2(abs(u[1]), abs(u[0])))
    # chord = extent along principal axis
    ch = (P@u).max() - (P@u).min()
    print(f"{lbl}: r={rm*1e3:.1f}mm  stagger(principal axis)={ang:.1f}deg  chord={ch*1e3:.2f}mm")
ri, ti, zi = unwrap(bnd["inlet"].points)
print("inlet: z=%.4f..%.4f  theta span=%.3f deg (pitch 360/28=%.3f)  r=%.1f..%.1f mm"
      % (zi.min(), zi.max(), np.degrees(ti.max()-ti.min()), 360/28, ri.min()*1e3, ri.max()*1e3))
ro,to,zo = unwrap(bnd["outlet"].points)
print("axial: inlet z=%.2f mm, blade LE z=%.2f, TE z=%.2f, outlet z=%.2f mm; up/down gaps %.2f/%.2f mm (chord 22)"
      % (zi.mean()*1e3, zb.min()*1e3, zb.max()*1e3, zo.mean()*1e3, (zb.min()-zi.mean())*1e3, (zo.mean()-zb.max())*1e3))
rh,_,_ = unwrap(bnd["hub"].points); rs,_,_ = unwrap(bnd["shroud"].points)
print("hub r: %.3f..%.3f mm   shroud r: %.3f..%.3f mm" % (rh.min()*1e3, rh.max()*1e3, rs.min()*1e3, rs.max()*1e3))
