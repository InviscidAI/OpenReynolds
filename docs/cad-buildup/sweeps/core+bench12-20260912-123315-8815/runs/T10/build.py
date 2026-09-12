"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters (metres) ----
R_HUB, R_TIP = 0.045, 0.090
NB      = 28
PITCH   = 2*np.pi/NB                 # rad
CHORD   = 0.022
TMAX    = 0.06                       # 6% thickness
CAMBER  = np.radians(20.0)           # total camber (turning) angle, design choice
STAG_R  = np.radians(30.0)           # stagger at root
TWIST   = np.radians(35.0)           # further twist to tip

def stagger(r):
    f = (r - R_HUB)/(R_TIP - R_HUB)
    return STAG_R + TWIST*f

def section(r, n=81):
    """aerofoil section in local (s = r*theta, z) plane, stacked on mid-chord."""
    c = CHORD
    beta = np.linspace(0, np.pi, n)
    x = 0.5*c*(1-np.cos(beta))                      # cosine spacing 0..c
    R = c/(2*np.sin(CAMBER/2))
    yc = np.sqrt(R**2 - (x-c/2)**2) - R*np.cos(CAMBER/2)
    dyc = -(x-c/2)/np.sqrt(R**2-(x-c/2)**2)
    th = np.arctan(dyc)
    xi = x/c
    yt = 5*TMAX*c*(0.2969*np.sqrt(xi)-0.1260*xi-0.3516*xi**2+0.2843*xi**3-0.1036*xi**4)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    g = stagger(r)
    def to_sz(xx, yy):
        xx = xx - c/2
        z =  xx*np.cos(g) - yy*np.sin(g)
        s =  xx*np.sin(g) + yy*np.cos(g)
        return s, z
    su, zu = to_sz(xu, yu)
    sl, zl = to_sz(xl, yl)
    return (su, zu), (sl, zl)

fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, r, nm in zip(axs, [R_HUB, 0.5*(R_HUB+R_TIP), R_TIP], ["hub", "mid", "tip"]):
    p = r*PITCH
    for k in (-1, 0, 1):
        (su, zu), (sl, zl) = section(r)
        ax.plot(1000*(su+k*p), 1000*zu, 'b-', lw=1)
        ax.plot(1000*(sl+k*p), 1000*zl, 'r-', lw=1)
    ax.set_aspect('equal'); ax.set_title(f"{nm} r={1000*r:.0f}mm pitch={1000*p:.1f}mm stag={np.degrees(stagger(r)):.0f}d")
    ax.set_xlabel("s [mm]"); ax.set_ylabel("z [mm]"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("sections.png", dpi=110)

for r in (R_HUB, R_TIP):
    (su,zu),(sl,zl) = section(r)
    le = np.array([su[0], zu[0]]); te = np.array([su[-1], zu[-1]])
    ch = np.hypot(*(te-le))
    t = np.max(np.hypot(su-sl, zu-zl))
    print(f"r={1000*r:.0f}mm chord={1000*ch:.2f}mm tmax/c={t/ch*100:.2f}% stag_meas={np.degrees(np.arctan2(te[0]-le[0], te[1]-le[1])):.2f}d tang.ext={1000*(su.max()-su.min()):.2f}mm pitch={1000*r*PITCH:.2f}mm")
print("z extent overall mm:", 1000*min(section(R_HUB)[0][1].min(), section(R_TIP)[0][1].min()),
      1000*max(section(R_HUB)[0][1].max(), section(R_TIP)[0][1].max()))

# -- cell 2 -------------------------------------------------------------------------
# Numbers check out: chord 22.00 mm, t/c 6.00%, stagger 30.00°/65.00°. Let me look at the sections.
from IPython.display import Image, display
display(Image("sections.png"))

# -- cell 3 -------------------------------------------------------------------------
# Loft succeeded; `is_valid` is a property, not a call. Re-emitting the same cell with the print fixed
from build123d import *

TE_OPEN = 0.010            # TE closure added to half-thickness (fraction of chord), renormalised
R_LO, R_HI = 0.040, 0.095  # blade extended past hub/shroud for clean trimming
NSEC, NPT = 9, 41

def sec_sz(r, n=NPT):
    """aerofoil section in unwrapped (s, z) coords, stacked on mid-chord of the chord line"""
    c = CHORD
    beta = np.linspace(0, np.pi, n)
    x = 0.5*c*(1-np.cos(beta))
    R = c/(2*np.sin(CAMBER/2))
    yc = np.sqrt(R**2-(x-c/2)**2) - R*np.cos(CAMBER/2)
    th = np.arctan(-(x-c/2)/np.sqrt(R**2-(x-c/2)**2))
    xi = x/c
    yt = 5*TMAX*c*(0.2969*np.sqrt(xi)-0.1260*xi-0.3516*xi**2+0.2843*xi**3-0.1036*xi**4)
    yt = yt + TE_OPEN*c*xi
    yt = yt*(TMAX*c/2)/yt.max()                      # renormalise: max thickness = 6% chord
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    g = STAG_R + TWIST*(r-R_HUB)/(R_TIP-R_HUB)
    def to_sz(xx, yy):
        xx = xx - c/2
        return xx*np.sin(g)+yy*np.cos(g), xx*np.cos(g)-yy*np.sin(g)
    return to_sz(xu, yu), to_sz(xl, yl)

def wrap(r, s, z):
    t = s/r
    return [Vector(r*np.cos(ti), r*np.sin(ti), zi) for ti, zi in zip(t, z)]

def blade_wires():
    ws = []
    for r in np.linspace(R_LO, R_HI, NSEC):
        (su, zu), (sl, zl) = sec_sz(r)
        pu, pl = wrap(r, su, zu), wrap(r, sl, zl)
        ws.append(Wire([Spline(*pu), Spline(*pl[::-1])]))   # LE->TE suction, TE->LE pressure
    return ws

blade0 = Solid.make_loft(blade_wires(), ruled=False)
bb = blade0.bounding_box()
print(blade0.is_valid, "vol mm^3 =", round(blade0.volume*1e9,1), "faces:", len(blade0.faces()))
print("bbox mm:", [round(v*1000,2) for v in (bb.min.X,bb.min.Y,bb.min.Z,bb.max.X,bb.max.Y,bb.max.Z)])
for i,f in enumerate(blade0.faces()):
    c = f.center(); print(i, f.geom_type, "area mm2", round(f.area*1e6,2), "ctr mm", [round(v*1000,2) for v in (c.X,c.Y,c.Z)])

# -- cell 4 -------------------------------------------------------------------------
# Capping non-planar wires fails. Instead make the two sacrificial end sections planar (lying in the t
R_LO, R_HI = 0.038, 0.097          # sacrificial planar end sections (outside the trimmed annulus)
RADII = [0.043, 0.045, 0.050, 0.056, 0.062, 0.068, 0.074, 0.080, 0.086, 0.090, 0.092]

def flat(r, s, z):                 # planar approximation of the section, in plane X = r
    return [Vector(r, si, zi) for si, zi in zip(s, z)]

def blade_wires2():
    ws = []
    for r, planar in [(R_LO, True)] + [(x, False) for x in RADII] + [(R_HI, True)]:
        (su, zu), (sl, zl) = sec_sz(r)
        f = flat if planar else wrap
        pu, pl = f(r, su, zu), f(r, sl, zl)
        ws.append(Wire([Spline(*pu), Spline(*pl[::-1])]))
    return ws

blade_solid = Solid.make_loft(blade_wires2(), ruled=False)
bb = blade_solid.bounding_box()
print("valid:", blade_solid.is_valid, "vol mm^3:", round(blade_solid.volume*1e9,1), "faces:", len(blade_solid.faces()))
print("bbox mm:", [round(v*1000,2) for v in (bb.min.X,bb.min.Y,bb.min.Z,bb.max.X,bb.max.Y,bb.max.Z)])
for i,f in enumerate(blade_solid.faces()):
    c=f.center(); print(i, f.geom_type, round(f.area*1e6,2), [round(v*1000,2) for v in (c.X,c.Y,c.Z)])

# -- cell 5 -------------------------------------------------------------------------
# Caps exist but the shell orientation is off (negative face area, invalid). Run OCC's ShapeFix on it.
from OCP.ShapeFix import ShapeFix_Shape
from OCP.BRepCheck import BRepCheck_Analyzer
sf = ShapeFix_Shape(blade_solid.wrapped); sf.Perform()
blade = Solid(sf.Shape())
print("valid:", BRepCheck_Analyzer(blade.wrapped).IsValid(), "vol mm^3:", round(blade.volume*1e9,2), "faces:", len(blade.faces()))
print("areas mm2:", [round(f.area*1e6,2) for f in blade.faces()])

# -- cell 6 -------------------------------------------------------------------------
# Re-emitting self-contained, and going straight to the fluid domain: sector minus three blade copies.
from build123d.topology import Shape
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_ShapeEnum

def fix_solid(sol):
    sf = ShapeFix_Shape(sol.wrapped); sf.Perform()
    exp = TopExp_Explorer(sf.Shape(), TopAbs_ShapeEnum.TopAbs_SHELL); exp.More()
    return Solid(ShapeFix_Solid().SolidFromShell(TopoDS.Shell_s(exp.Current())))

blade = fix_solid(blade_solid)
print("blade vol mm^3:", round(blade.volume*1e9,2))

# axial extent of the blade inside the real annulus -> inlet/outlet one chord beyond
zs = [sec_sz(r)[i][1] for r in (R_HUB, 0.5*(R_HUB+R_TIP), R_TIP) for i in (0,1)]
Z_BL_MIN, Z_BL_MAX = min(a.min() for a in zs), max(a.max() for a in zs)
Z_IN, Z_OUT = Z_BL_MIN - CHORD, Z_BL_MAX + CHORD
L = Z_OUT - Z_IN
print("blade z mm:", round(1000*Z_BL_MIN,2), round(1000*Z_BL_MAX,2), " domain z mm:", round(1000*Z_IN,2), round(1000*Z_OUT,2))

prof = Plane.XZ * Pos(0.5*(R_HUB+R_TIP), 0.5*(Z_IN+Z_OUT)) * Rectangle(R_TIP-R_HUB, L)
sector = Rot(Z=-np.degrees(PITCH)/2) * revolve(prof, Axis.Z, revolution_arc=np.degrees(PITCH))
print("sector vol mm^3:", round(sector.volume*1e9,1), " analytic:", round(0.5*PITCH*(R_TIP**2-R_HUB**2)*L*1e9,1))

# -- cell 7 -------------------------------------------------------------------------
# Now subtract the blade and its ±1-pitch neighbours from the sector (needed because the blade's tange
blades = {k: Rot(Z=k*np.degrees(PITCH)) * blade for k in (-1, 0, 1)}
fluid = sector - blades[-1] - blades[0] - blades[1]
print("valid:", BRepCheck_Analyzer(fluid.wrapped).IsValid(), "solids:", len(fluid.solids()),
      "vol mm^3:", round(fluid.volume*1e9,1), "faces:", len(fluid.faces()))
for i,f in enumerate(fluid.faces()):
    c=f.center(); print(i, f.geom_type, round(f.area*1e6,1), "ctr mm", [round(v*1000,2) for v in (c.X,c.Y,c.Z)], "r=",round(1000*np.hypot(c.X,c.Y),2))

# -- cell 8 -------------------------------------------------------------------------
# Fluid volume dropped by only 9.8 mm³ — far less than the blade's share of the annulus. Checking the 
inter0 = sector & blades[0]
inter_m = sector & blades[-1]
inter_p = sector & blades[1]
print([round(x.volume*1e9,2) for x in (inter0, inter_m, inter_p)])
print("sum:", round((inter0.volume+inter_m.volume+inter_p.volume)*1e9,2), "sector-fluid:", round((sector.volume-fluid.volume)*1e9,2))

# -- cell 9 -------------------------------------------------------------------------
# Intersection with the central blade is zero, yet its faces appear in the cut — the blade solid is li
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_State
def where(sol, p):
    c = BRepClass3d_SolidClassifier(sol.wrapped, gp_Pnt(*p), 1e-9); return c.State()
pin = (0.068, 0.0, 0.0)           # mid-chord point at mid span: inside the blade
pfar = (0.0680, 0.0, 0.030)       # downstream of blade, inside sector only
for nm, s in [("blade", blade), ("blade_solid(raw loft)", blade_solid), ("sector", sector), ("fluid", fluid)]:
    print(nm, "pin:", where(s, pin), " pfar:", where(s, pfar))

# -- cell 10 ------------------------------------------------------------------------
# `fluid` is a compound (classifier needs a solid), and my "inside the blade" point may be wrong. Reco
r = 0.068
(su,zu),(sl,zl) = sec_sz(r)
i = len(su)//2
smid, zmid = 0.5*(su[i]+sl[i]), 0.5*(zu[i]+zl[i])
th = smid/r
pin = (r*np.cos(th), r*np.sin(th), zmid)
print("pin mm:", [round(1000*v,2) for v in pin])
fs = fluid.solids()[0]
pfar = (r*np.cos(th), r*np.sin(th), 0.025)
for nm, s in [("blade", blade), ("sector", sector.solids()[0]), ("fluid", fs)]:
    print(nm, "pin:", where(s, pin), "pfar:", where(s, pfar))
print("fluid solid vol mm^3:", round(fs.volume*1e9,1))

# -- cell 11 ------------------------------------------------------------------------
sec_solid = sector.solids()[0]
blades = {k: Rot(Z=k*np.degrees(PITCH)) * blade for k in (-1, 0, 1)}
fluid = (sector - blades[-1] - blades[0] - blades[1]).solids()[0]
print("valid:", BRepCheck_Analyzer(fluid.wrapped).IsValid(), "fluid vol mm^3:", round(fluid.volume*1e9,1),
      "removed:", round((sector.volume-fluid.volume)*1e9,1), "faces:", len(fluid.faces()))
print("blade-in-annulus vol mm^3:", round(sum(s.volume for s in sec_solid.intersect(blade).solids())*1e9,1))
print("fluid at blade interior point:", where(fluid, pin), " downstream point:", where(fluid, pfar))

# -- cell 12 ------------------------------------------------------------------------
# Fluid is valid: 42032 mm³, 978 mm³ of blade removed. Now list the 15 faces with surface type, radius
for i,f in enumerate(fluid.faces()):
    c = f.center(); n = f.normal_at(c)
    print(i, str(f.geom_type).split('.')[-1], "A=%8.1f"%(f.area*1e6), "ctr mm",
          [round(v*1000,2) for v in (c.X,c.Y,c.Z)], "r=%6.2f"%(1000*np.hypot(c.X,c.Y)),
          "th=%7.2f"%np.degrees(np.arctan2(c.Y,c.X)), "n", [round(v,2) for v in (n.X,n.Y,n.Z)])
print("pitch deg", np.degrees(PITCH))

# -- cell 13 ------------------------------------------------------------------------
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex

def dist_pt_face(p, face):
    v = BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex()
    d = BRepExtrema_DistShapeShape(v, face.wrapped); d.Perform(); return d.Value()

rr = 0.068; (su,zu),(sl,zl) = sec_sz(rr); j = len(su)//2
p_suc = wrap(rr, np.array([su[j]]), [zu[j]])[0]
p_prs = wrap(rr, np.array([sl[j]]), [zl[j]])[0]
p_te  = wrap(rr, np.array([0.5*(su[-1]+sl[-1])]), [0.5*(zu[-1]+zl[-1])])[0]
bf = blade.faces()
side_of = {}
for nm, p in [("suction", p_suc), ("pressure", p_prs), ("te", p_te)]:
    dl = [dist_pt_face((p.X,p.Y,p.Z), f) for f in bf]
    k = int(np.argmin(dl)); side_of[k] = nm
    print(nm, "-> blade face", k, "d=%.2e"%dl[k], "area", round(bf[k].area*1e6,1))

# -- cell 14 ------------------------------------------------------------------------
# Now map each fluid spline face back to a blade copy and side, printing the two best distances so I c
def rot_back(p, k):
    a = -k*PITCH
    return (p.X*np.cos(a)-p.Y*np.sin(a), p.X*np.sin(a)+p.Y*np.cos(a), p.Z)

names = {0:"suction", 1:"te", 2:"pressure"}
for i,f in enumerate(fluid.faces()):
    if f.geom_type != GeomType.BSPLINE: continue
    c = f.center()
    scores = []
    for k in (-1,0,1):
        p = rot_back(c, k)
        for idx in (0,1,2):
            scores.append((dist_pt_face(p, bf[idx]), k, names[idx]))
    scores.sort()
    print(i, "A=%7.1f"%(f.area*1e6), "best", scores[0][1], scores[0][2], "d=%.2e"%scores[0][0],
          "| 2nd", scores[1][1], scores[1][2], "d=%.2e"%scores[1][0])

# -- cell 15 ------------------------------------------------------------------------
# Face 11 is ambiguous from its centroid; rescoring every spline face by mean distance over its tessel
def face_side(f):
    verts, tris = f.tessellate(0.0002)
    pts = [verts[i] for i in range(0, len(verts), max(1, len(verts)//25))]
    best = None
    for k in (-1,0,1):
        for idx in (0,1,2):
            d = np.mean([dist_pt_face(rot_back(p, k), bf[idx]) for p in pts])
            if best is None or d < best[0]: best = (d, k, names[idx])
    return best

patch_of = {}
for i,f in enumerate(fluid.faces()):
    if f.geom_type != GeomType.BSPLINE: continue
    d,k,nm = face_side(f)
    patch_of[i] = ("blade_suction" if nm in ("suction","te") else "blade_pressure")
    print(i, "A=%7.1f"%(f.area*1e6), "copy", k, nm, "mean d=%.2e"%d, "->", patch_of[i])

# -- cell 16 ------------------------------------------------------------------------
# Classification agrees with the centroid test. Building the complete face→patch map intrinsically (cy
from OCP.BRepAdaptor import BRepAdaptor_Surface

def patch_name(f):
    c = f.center(); n = f.normal_at(c)
    if f.geom_type == GeomType.CYLINDER:
        R = BRepAdaptor_Surface(f.wrapped).Cylinder().Radius()
        return "hub" if abs(R-R_HUB) < abs(R-R_TIP) else "shroud"
    if f.geom_type == GeomType.PLANE:
        if abs(n.Z) > 0.9:
            return "inlet" if n.Z < 0 else "outlet"
        er = Vector(c.X, c.Y, 0); er = er/er.length
        et = Vector(-er.Y, er.X, 0)                      # +theta direction
        return "periodic_pos" if n.dot(et) > 0 else "periodic_neg"
    d,k,nm = face_side(f)
    return "blade_suction" if nm in ("suction","te") else "blade_pressure"

faces = fluid.faces()
PATCH = [patch_name(f) for f in faces]
areas = {}
for f,p in zip(faces, PATCH): areas[p] = areas.get(p,0)+f.area*1e6
for p,a in sorted(areas.items()): print(f"{p:15s} {a:9.1f} mm2  nfaces={PATCH.count(p)}")
export_step(fluid, "fluid.step")
print("wrote fluid.step")

# -- cell 17 ------------------------------------------------------------------------
# STEP written; periodic face areas match exactly (2776.4 mm² each). Now import into gmsh and match it
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("passage")
gmsh.model.occ.importShapes("fluid.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
print("gmsh surfaces:", len(surfs), " b123d faces:", len(faces))
ref = [(np.array([f.center().X,f.center().Y,f.center().Z]), f.area) for f in faces]
mapping = {}
for _, s in surfs:
    com = np.array(gmsh.model.occ.getCenterOfMass(2, s))
    ar = gmsh.model.occ.getMass(2, s)
    d = [np.linalg.norm(com-c) + abs(ar-a)/max(a,1e-12)*1e-3 for c,a in ref]
    i = int(np.argmin(d)); mapping[s] = i
    print(f"surf {s:3d} -> face {i:3d} {PATCH[i]:15s} err={d[i]:.2e}")
print("unique faces matched:", len(set(mapping.values())))
