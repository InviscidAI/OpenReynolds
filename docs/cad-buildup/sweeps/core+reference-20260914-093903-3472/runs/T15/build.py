"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, build123d as bd, math, os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

# ---- given by the request ----
m      = 3.0        # module, mm
Np, Nw = 20, 30     # teeth
alpha  = math.radians(20.0)
B      = 0.3        # backlash along pitch line, mm (total)
LX, LY, LZ = 160.0, 140.0, 120.0   # cavity, mm
SIGMA  = 90.0       # shaft angle, deg

# ---- derived bevel geometry ----
gp = math.atan2(Np, Nw); gw = math.atan2(Nw, Np)     # pitch cone half angles
rp, rw = m*Np/2, m*Nw/2                              # outer pitch radii
A0 = rp/math.sin(gp)
print(f"pitch dia  pinion {2*rp:.3f} mm   wheel {2*rw:.3f} mm")
print(f"cone angles {math.degrees(gp):.3f} + {math.degrees(gw):.3f} = {math.degrees(gp+gw):.3f} deg (target {SIGMA})")
print(f"outer cone distance A0 = {A0:.4f} mm  (check {rw/math.sin(gw):.4f})")
print(f"circular pitch p = {math.pi*m:.4f} mm, tooth thickness each = {math.pi*m/2 - B/2:.4f} mm")
print("build123d", bd.__version__)

# -- cell 2 -------------------------------------------------------------------------
w = bd.Polyline((0,0,0),(1,0,0),(1,1,0), close=True)
print(type(w), type(w.wrapped))
print(type(bd.Wire(w.edges())))

# -- cell 3 -------------------------------------------------------------------------
print(subprocess.run(["sed","-n","270,320p","/home/qiuzi/miniforge3/lib/python3.13/site-packages/build123d/operations_part.py"],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# Re-emitting self-contained (the earlier definitions came from an errored cell). Also verifying the f
FW = 18.0                   # face width, mm   -- CHOSEN (A0/3)
ha, hf = 1.0*m, 1.25*m      # addendum / dedendum, perpendicular to pitch cone
SHR, BORER, BOREL = 10.0, 15.0, 10.0   # CHOSEN shaft radius, wall bore radius, bore length

def gear_profile(N, gamma, nfl=9):
    """planar closed involute profile (mm) in the outer plane z=zo of a straight bevel gear"""
    r0 = m*N/2.0
    zo = A0*math.cos(gamma)
    rb = r0*math.cos(alpha)
    rt = (r0 + ha*math.cos(gamma))*zo/(zo - ha*math.sin(gamma))   # face cone at z=zo
    rr = (r0 - hf*math.cos(gamma))*zo/(zo + hf*math.sin(gamma))   # root cone at z=zo
    phi = lambda r: math.tan(math.acos(rb/r)) - math.acos(rb/r)
    ht = (math.pi*m/2 - B/2)/2.0/r0        # half tooth-thickness angle at pitch circle
    th = lambda r: ht + phi(r0) - phi(r)
    fl = [(r, th(r)) for r in np.linspace(max(rb, rr), rt, nfl)]
    if rr < rb: fl = [(rr, th(rb))] + fl
    pts = []
    for k in range(N):
        c = 2*math.pi*k/N
        seq = [(r, c - t) for r, t in reversed(fl)] + [(r, c + t) for r, t in fl]
        pts += [(r*math.cos(a), r*math.sin(a)) for r, a in seq]
    return np.array(pts), dict(r0=r0, rb=rb, rt=rt, rr=rr, zo=zo, ht=ht)

def mkwire(pts3):
    return bd.Wire(bd.Polyline(*pts3, close=True).edges())

def gear_solid(N, gamma, hubL, shaftEnd):
    pts, g = gear_profile(N, gamma)
    zo, zi = g['zo'], (A0-FW)*math.cos(gamma)
    s = zi/zo
    wo = mkwire([(x, y, zo) for x, y in pts])
    wi = mkwire([(x*s, y*s, zi) for x, y in pts])
    teeth = bd.Part(bd.Solid.make_loft([wo, wi], ruled=True).wrapped)
    hub   = bd.Pos(0, 0, zo)*bd.Cylinder(g['rr'], hubL, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
    shaft = bd.Pos(0, 0, zi)*bd.Cylinder(SHR, shaftEnd-zi, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
    return (teeth + hub + shaft), g

pin_local, gpin = gear_solid(Np, gp, 12.0, LX/2 + BOREL + 15)
whl_local, gwhl = gear_solid(Nw, gw, 12.0, LZ/2 + BOREL + 15)
for nm, g, gam in (("pinion", gpin, gp), ("wheel", gwhl, gw)):
    ga = math.degrees(math.atan(g['rt']/g['zo'])); gr = math.degrees(math.atan(g['rr']/g['zo']))
    print(f"{nm}: gamma {math.degrees(gam):.3f}  face cone {ga:.3f} (want {math.degrees(gam)+math.degrees(math.atan(ha/A0)):.3f})"
          f"  root cone {gr:.3f} (want {math.degrees(gam)-math.degrees(math.atan(hf/A0)):.3f})  rt {g['rt']:.3f} rr {g['rr']:.3f}")
print("volumes", round(pin_local.volume,1), round(whl_local.volume,1))
print("wheel bbox", whl_local.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
pinion = bd.Rotation(0, 90, 0) * pin_local          # local +z axis -> global +x
wheel  = bd.Rotation(0, 0, 6.0) * whl_local        # half-pitch turn: tooth space centred on +x
print("pinion bbox", pinion.bounding_box())
print("wheel  bbox", wheel.bounding_box())
inter = pinion & wheel
print("intersection volume:", inter.volume if inter is not None else None)
print("min distance gear-to-gear:", pinion.distance_to(wheel))

# -- cell 6 -------------------------------------------------------------------------
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
d = BRepExtrema_DistShapeShape(pinion.wrapped, wheel.wrapped); d.Perform()
print("nsol", d.NbSolution(), "dist", d.Value())
for i in range(1, min(d.NbSolution(),6)+1):
    p1, p2 = d.PointOnShape1(i), d.PointOnShape2(i)
    print(i, [round(v,4) for v in (p1.X(),p1.Y(),p1.Z())], [round(v,4) for v in (p2.X(),p2.Y(),p2.Z())])

# -- cell 7 -------------------------------------------------------------------------
# Boolean with the 30-tooth loft returns None, so first a quick robustness probe, and switching the tr
ptsw, gwd = gear_profile(Nw, gw)
zow = gwd['zo']; Aiw = A0-FW; ziw = Aiw*math.cos(gw); sw = ziw/zow
bw = bd.Solid.make_loft([mkwire([(x,y,zow) for x,y in ptsw]),
                         mkwire([(x*sw,y*sw,ziw) for x,y in ptsw])], ruled=True)
from OCP.BRepCheck import BRepCheck_Analyzer
print("bw valid:", BRepCheck_Analyzer(bw.wrapped).IsValid(), "vol", bw.volume)
cyl = bd.Pos(0,0,10)*bd.Cylinder(80, 30, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
r1 = bw & cyl; print("bw & cyl ->", None if r1 is None else r1.volume)
ball = bd.Sphere(A0)
r2 = bw & ball; print("bw & ball(A0) ->", None if r2 is None else r2.volume)

# -- cell 8 -------------------------------------------------------------------------
# The loft was self-intersecting: the axis-normal-plane profile gives pointed teeth on the wheel. The 
def bevel_profile(N, gamma, nfl=10):
    """planar profile in the outer plane z=zo: involute drawn on the back-cone
    development (Tredgold) and projected from the common apex."""
    r0 = m*N/2.0
    zo = A0*math.cos(gamma)
    zb = zo + r0*math.tan(gamma)            # back cone apex on the axis
    rv = r0/math.cos(gamma)                 # equivalent spur pitch radius (development)
    rbv = rv*math.cos(alpha)
    Rt, Rr = rv + ha, rv - hf
    phi = lambda R: math.tan(math.acos(rbv/R)) - math.acos(rbv/R)
    ht = (math.pi*m/2 - B/2)/2.0/rv         # half tooth-thickness angle at pitch, development
    th = lambda R: ht + phi(rv) - phi(R)
    fl = [(R, th(R)) for R in np.linspace(max(rbv, Rr), Rt, nfl)]
    if Rr < rbv: fl = [(Rr, th(rbv))] + fl
    def to_plane(R, thdev):
        rc, z = R*math.cos(gamma), zb - R*math.sin(gamma)   # on the back cone
        return rc*zo/z, thdev/math.cos(gamma)               # apex projection, azimuth
    pts = []
    for k in range(N):
        c = 2*math.pi*k/N
        seq = [(R, -t) for R, t in reversed(fl)] + [(R, t) for R, t in fl]
        for R, t in seq:
            rp_, dphi = to_plane(R, t)
            pts.append((rp_*math.cos(c + dphi), rp_*math.sin(c + dphi)))
    rt_, _ = to_plane(Rt, 0.0); rr_, _ = to_plane(Rr, 0.0)
    return np.array(pts), dict(r0=r0, zo=zo, zb=zb, rv=rv, rt=rt_, rr=rr_)

for N, gam, nm in ((Np, gp, "pinion"), (Nw, gw, "wheel")):
    p, gg = bevel_profile(N, gam)
    th_tip = None
    print(nm, {k: round(v,4) for k,v in gg.items()}, "npts", len(p))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(11,5.5))
for a,(N,gam,nm) in zip(ax, ((Np,gp,"pinion"),(Nw,gw,"wheel"))):
    p,_ = bevel_profile(N,gam); a.plot(*np.vstack([p,p[:1]]).T, lw=.8); a.set_aspect(1); a.set_title(nm)
plt.savefig("prof.png", dpi=90)

# -- cell 9 -------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(11,5.5))
for a,(N,gam,nm) in zip(ax, ((Np,gp,"pinion"),(Nw,gw,"wheel"))):
    p,_ = bevel_profile(N,gam); a.plot(*np.vstack([p,p[:1]]).T, lw=.8); a.set_aspect(1); a.set_title(nm)
fig

# -- cell 10 ------------------------------------------------------------------------
# The polyline ordering was wrong — teeth were being joined tip-to-tip instead of across the root land
def bevel_profile(N, gamma, nfl=10):
    """planar profile in the outer plane z=zo: involute drawn on the back-cone
    development (Tredgold), azimuth compressed by cos(gamma), projected from the apex."""
    r0 = m*N/2.0
    zo = A0*math.cos(gamma)
    zb = zo + r0*math.tan(gamma)
    rv = r0/math.cos(gamma)
    rbv = rv*math.cos(alpha)
    Rt, Rr = rv + ha, rv - hf
    phi = lambda R: math.tan(math.acos(rbv/R)) - math.acos(rbv/R)
    ht = (math.pi*m/2 - B/2)/2.0/rv
    th = lambda R: ht + phi(rv) - phi(R)
    fl = [(R, th(R)) for R in np.linspace(max(rbv, Rr), Rt, nfl)]   # root -> tip
    if Rr < rbv: fl = [(Rr, th(rbv))] + fl
    def to_plane(R, thdev):
        rc, z = R*math.cos(gamma), zb - R*math.sin(gamma)
        return rc*zo/z, thdev/math.cos(gamma)
    pts = []
    for k in range(N):
        c = 2*math.pi*k/N
        seq = [(R, -t) for R, t in fl] + [(R, t) for R, t in reversed(fl)]  # root- -> tip- -> tip+ -> root+
        for R, t in seq:
            rp_, dphi = to_plane(R, t)
            pts.append((rp_*math.cos(c + dphi), rp_*math.sin(c + dphi)))
    rt_, _ = to_plane(Rt, 0.0); rr_, _ = to_plane(Rr, 0.0)
    return np.array(pts), dict(r0=r0, zo=zo, zb=zb, rv=rv, rt=rt_, rr=rr_)

fig, ax = plt.subplots(1,2, figsize=(11,5.5))
for a,(N,gam,nm) in zip(ax, ((Np,gp,"pinion"),(Nw,gw,"wheel"))):
    p,_ = bevel_profile(N,gam); a.plot(*np.vstack([p,p[:1]]).T, lw=.8); a.set_aspect(1); a.set_title(nm)
fig

# -- cell 11 ------------------------------------------------------------------------
# Now the gear solids: apex-projection loft trimmed by spheres of radius A0 and Ai (constant cone dist
def bevel_gear(N, gamma, hubL, shaftEnd, coreOver=2.0):
    pts, g = bevel_profile(N, gamma)
    rr, zo = g['rr'], g['zo']
    Ai = A0 - FW
    zi = Ai*math.cos(gamma)
    s = zi/zo
    band = bd.Solid.make_loft([mkwire([(x, y, zo) for x, y in pts]),
                               mkwire([(x*s, y*s, zi) for x, y in pts])], ruled=True)
    print("  loft valid:", BRepCheck_Analyzer(band.wrapped).IsValid())
    band = (band & bd.Sphere(A0)) - bd.Sphere(Ai)
    zc = zi - coreOver
    core = bd.Pos(0,0,zc)*bd.Cone(rr*zc/zo, rr, zo-zc, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
    hub  = bd.Pos(0,0,zo)*bd.Cylinder(rr, hubL, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
    shaft= bd.Pos(0,0,zc)*bd.Cylinder(SHR, shaftEnd-zc, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
    return (band + core + hub + shaft), dict(g, zi=zi, Ai=Ai)

pinL, gpin = bevel_gear(Np, gp, 12.0, LX/2 + BOREL + 15)
whlL, gwhl = bevel_gear(Nw, gw, 12.0, LZ/2 + BOREL + 15)
for nm, s_ in (("pinion", pinL), ("wheel", whlL)):
    print(nm, "vol", round(s_.volume,1), "solids", len(s_.solids()),
          "valid", BRepCheck_Analyzer(s_.wrapped).IsValid(), s_.bounding_box())
