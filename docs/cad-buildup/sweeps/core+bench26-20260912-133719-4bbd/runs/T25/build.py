"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll send that as an actual cell.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- flowpath definition, metres, axis = X, flow +X -------------------------
X_IN      = -0.15          # inlet plane
X_SPLIT   =  0.60          # splitter leading edge
X_BP_EXIT =  1.70          # bypass nozzle exit plane
X_CR_EXIT =  2.10          # core nozzle exit plane
T_SPLIT   =  0.02          # half-thickness of blunt splitter lip

NAC   = [(-0.15,0.95),(0.00,0.94),(0.15,0.915),(0.30,0.90),(0.60,0.90),
         (1.00,0.87),(1.40,0.82),(1.70,0.78)]                 # nacelle inner wall
HUB_F = [(0.00,0.00),(0.05,0.10),(0.12,0.18),(0.22,0.23),(0.30,0.25),
         (0.45,0.27),(0.60,0.30)]                             # spinner + front hub
COWL  = [(0.60,0.52),(0.80,0.58),(1.10,0.60),(1.40,0.58),(1.70,0.55)]  # core cowl outer
CASE  = [(0.60,0.48),(0.90,0.44),(1.20,0.40),(1.50,0.44),(1.90,0.48),(2.10,0.44)]
CHUB  = [(0.60,0.30),(0.90,0.33),(1.20,0.35),(1.50,0.33),(1.90,0.28),
         (2.00,0.22),(2.10,0.14)]                             # core hub + exhaust plug

fig,ax = plt.subplots(figsize=(11,4))
for pts,lab in [(NAC,"nacelle inner"),(HUB_F,"spinner/hub"),(COWL,"core cowl"),
                (CASE,"core casing"),(CHUB,"core hub + plug")]:
    a=np.array(pts); ax.plot(a[:,0],a[:,1],"-o",ms=3,label=lab)
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
plt.savefig("meridional.png",dpi=110,bbox_inches="tight")
print("fan tip dia %.3f m, fan hub/tip = %.3f" % (2*0.90, 0.25/0.90))
print("bypass exit annulus r %.2f..%.2f, core exit annulus r %.2f..%.2f" % (0.55,0.78,0.14,0.44))

# -- cell 2 -------------------------------------------------------------------------
# Profile numbers check out; let me actually look at it.
from IPython.display import display
display(fig)

# -- cell 3 -------------------------------------------------------------------------
# Now revolve the three meridional regions (inlet duct, bypass duct, core duct) into the fluid solid a
from build123d import *

def rev(pts):
    """closed (x,r) polygon in the XZ half-plane, revolved 360 deg about the X axis"""
    f = Plane.XZ * Polygon(*pts, align=None)
    return revolve(f, axis=Axis.X, revolution_arc=360)

reg_in   = rev([(X_IN,0.0)] + HUB_F + [(X_SPLIT,0.90)] + [p for p in reversed(NAC) if p[0] <= X_SPLIT])
reg_byp  = rev(COWL + [(X_BP_EXIT,0.78)] + [p for p in reversed(NAC) if p[0] >= X_SPLIT])
reg_core = rev(CHUB + [(X_CR_EXIT,0.44)] + list(reversed(CASE))[1:])

fluid0 = reg_in + reg_byp + reg_core
print("volumes: inlet %.4f  bypass %.4f  core %.4f  union %.4f m^3" %
      (reg_in.volume, reg_byp.volume, reg_core.volume, fluid0.volume))
print("bbox", fluid0.bounding_box())
print("solids in union:", len(fluid0.solids()), " faces:", len(fluid0.faces()))

# -- cell 4 -------------------------------------------------------------------------
# Blades next: cambered NACA-ish sections lofted radially with twist, then polar-patterned. Check one 
import math

def naca(chord, m=0.06, p=0.4, t=0.12, n=41):
    """closed 4-digit-style section, chord centred on origin, chord along local x"""
    xs = [0.5*(1-math.cos(math.pi*i/(n-1))) for i in range(n)]
    up, lo = [], []
    for xc_ in xs:
        yt = 5*t*(0.2969*math.sqrt(xc_)-0.1260*xc_-0.3516*xc_**2+0.2843*xc_**3-0.1036*xc_**4)
        if xc_ < p: yc = m/p**2*(2*p*xc_-xc_**2);         dy = 2*m/p**2*(p-xc_)
        else:       yc = m/(1-p)**2*((1-2*p)+2*p*xc_-xc_**2); dy = 2*m/(1-p)**2*(p-xc_)
        th = math.atan(dy)
        up.append(((xc_-yt*math.sin(th)-0.5)*chord, (yc+yt*math.cos(th))*chord))
        lo.append(((xc_+yt*math.sin(th)-0.5)*chord, (yc-yt*math.cos(th))*chord))
    return up + lo[-2:0:-1]

def blade(xc, r0, r1, chord, stag0, stag1, m=0.06):
    """one blade at azimuth 0, spanning radius r0..r1 (z direction), stagger twisting stag0->stag1 [deg]"""
    secs = []
    for r, g, ch in ((r0, stag0, chord), (r1, stag1, chord)):
        pl = Plane(origin=(xc, 0, r), x_dir=(math.cos(math.radians(g)), math.sin(math.radians(g)), 0),
                   z_dir=(0, 0, 1))
        secs.append(pl * Polygon(*naca(ch, m=m), align=None))
    return loft(secs)

def row(xc, r0, r1, n, chord, stag0, stag1, m=0.06):
    b = blade(xc, r0, r1, chord, stag0, stag1, m)
    return Compound([b.rotate(Axis.X, 360*i/n) for i in range(n)])

b1 = blade(0.30, 0.20, 0.95, 0.22, 25, 55)
print("single fan blade volume %.5f m^3, bbox %s" % (b1.volume, b1.bounding_box()))
print("max thickness ~ %.4f m" % (0.12*0.22))

# -- cell 5 -------------------------------------------------------------------------
# All six blade rows, subtracted from the fluid. Volume removed per row tells me each row actually int
ROWS = {                     # name: (xc, r0, r1, nblades, chord, stag0, stag1, camber)
 "fan"         : (0.30, 0.20, 0.95, 12, 0.22,  25,  55, 0.06),
 "ogv"         : (0.85, 0.50, 0.95, 12, 0.16,  18,  18, 0.04),
 "compressor1" : (0.80, 0.25, 0.52, 12, 0.11,  30,  42, 0.05),
 "compressor2" : (1.00, 0.25, 0.50, 12, 0.10, -25, -32, 0.05),
 "turbine1"    : (1.55, 0.24, 0.55, 12, 0.13,  40,  40, 0.09),
 "turbine2"    : (1.78, 0.22, 0.55, 12, 0.13, -38, -38, 0.09),
}
rows = {k: row(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) for k, p in ROWS.items()}

fluid = fluid0
for k, rc in rows.items():
    v0 = fluid.volume
    fluid = fluid - rc
    print("%-12s cut %.4f m^3 (%d blades), fluid now %.4f" % (k, v0-fluid.volume, ROWS[k][3], fluid.volume))
print("solids:", len(fluid.solids()), " faces:", len(fluid.faces()))

# -- cell 6 -------------------------------------------------------------------------
# Geometry is one solid. Save it (the boolean was expensive) and look at it, half-cut so the interior 
import os, subprocess
export_step(fluid, "fluid.step")
export_stl(fluid, "fluid_view.stl", tolerance=0.004, angular_tolerance=0.3)
import pyvista as pv
pv.start_xvfb() if False else None
m = pv.read("fluid_view.stl")
print(m.n_cells, m.bounds)
p = pv.Plotter(off_screen=True, window_size=(1400,700))
p.add_mesh(m.clip(normal="y", origin=(0,0,0)), color="lightsteelblue", show_edges=False)
p.camera_position = [(1.0, -4.5, 2.2), (0.95, 0, 0), (0, 0, 1)]
p.show(screenshot="view1.png")
from IPython.display import Image
Image("view1.png")

# -- cell 7 -------------------------------------------------------------------------
# The clip worked but I'm looking at its back. View the cut plane from +y.
p = pv.Plotter(off_screen=True, window_size=(1500,700))
p.add_mesh(m.clip(normal="y", origin=(0,0,0)), color="lightsteelblue", show_edges=True, edge_color="gray", line_width=1)
p.camera_position = [(0.95, 5.0, 1.2), (0.95, 0, 0), (0, 0, 1)]
p.show(screenshot="view2.png")
Image("view2.png")

# -- cell 8 -------------------------------------------------------------------------
# Let me render it more legibly: front quarter view (looking into the fan) and a shaded meridional cut
p = pv.Plotter(off_screen=True, window_size=(1500,750), shape=(1,2))
p.subplot(0,0)
p.add_mesh(m, color="steelblue")
p.camera_position = [(-2.6, -1.6, 1.1), (0.6, 0, 0), (0, 0, 1)]
p.subplot(0,1)
p.add_mesh(m.clip(normal="y", origin=(0,0,0), invert=False), color="lightsteelblue")
p.camera_position = [(0.95, 4.2, 0.05), (0.95, 0, 0.05), (0, 0, 1)]
p.show(screenshot="view3.png")
Image("view3.png")

# -- cell 9 -------------------------------------------------------------------------
# Wrong side again — for the half that keeps y>0 I must look from −y. Also zooming out.
p = pv.Plotter(off_screen=True, window_size=(1500,750), shape=(1,2))
p.subplot(0,0)
p.add_mesh(m, color="steelblue"); p.camera_position=[(-4.5,-3.0,2.0),(0.9,0,0),(0,0,1)]
p.subplot(0,1)
p.add_mesh(m.clip(normal="y", origin=(0,0,0), invert=False), color="lightsteelblue")
p.camera_position=[(0.95,-6.0,0.6),(0.95,0,0.1),(0,0,1)]
p.show(screenshot="view4.png")
Image("view4.png")

# -- cell 10 ------------------------------------------------------------------------
# Geometry reads correctly: spinner, fan, splitter, core stages, plug. Now patch identity — I'll test 
f0 = fluid.faces()[0]
print([a for a in dir(f0) if "dist" in a.lower()])
import numpy as np
b = rows["fan"].solids()[0]
c = b.faces()[3].center()
print("centroid of a fan-blade face:", c, " dist to that blade solid:", b.distance_to(c))

# -- cell 11 ------------------------------------------------------------------------
# Classifying every fluid face by the construction that made it: axial cut planes, the five revolved w
import numpy as np
CURVES = {"nacelle_inner":NAC, "spinner_hub":HUB_F, "core_cowl":COWL,
          "core_casing":CASE, "core_hub_plug":CHUB}

def r_of(curve, x):
    a = np.array(curve)
    if x < a[0,0]-1e-9 or x > a[-1,0]+1e-9: return None
    return float(np.interp(x, a[:,0], a[:,1]))

groups = {k: [] for k in list(CURVES) + ["inlet","outlet_bypass","outlet_core","splitter_nose"] + list(ROWS)}
unknown = []
for i, f in enumerate(fluid.faces()):
    c = f.center(); x, r = c.X, math.hypot(c.Y, c.Z)
    n = f.normal_at(c)
    if abs(n.X) > 0.99 and f.geom_type == GeomType.PLANE:
        for nm, xp in (("inlet",X_IN),("outlet_bypass",X_BP_EXIT),("outlet_core",X_CR_EXIT),("splitter_nose",X_SPLIT)):
            if abs(x-xp) < 1e-7: groups[nm].append(i); break
        else: unknown.append(i)
        continue
    for nm, cv in CURVES.items():
        rc = r_of(cv, x)
        if rc is not None and abs(r-rc) < 2e-4: groups[nm].append(i); break
    else: unknown.append(i)
print("after walls/planes: unknown =", len(unknown))
print({k: len(v) for k, v in groups.items() if v})

# -- cell 12 ------------------------------------------------------------------------
# 5760 leftovers should be exactly the blade surfaces. Assign each to a row by actually lying on that 
import time
rowbb = {k: rows[k].bounding_box() for k in rows}
faces = fluid.faces()
t0=time.time()
test = unknown[:50]
for i in test:
    c = faces[i].center()
    cands = [k for k in rows if rowbb[k].min.X-1e-6 <= c.X <= rowbb[k].max.X+1e-6]
    d = {k: rows[k].distance_to(c) for k in cands}
print("sample time %.2f s for 50 faces -> est %.0f s total" % (time.time()-t0, (time.time()-t0)/50*5760))
print("last face cands/dists:", d)

# -- cell 13 ------------------------------------------------------------------------
# Right — classify and get straight into gmsh. This cell finishes the blade assignment and stores face
for i in unknown:
    c = faces[i].center()
    cands = [k for k in rows if rowbb[k].min.X-1e-6 <= c.X <= rowbb[k].max.X+1e-6]
    k = min(cands, key=lambda k: rows[k].distance_to(c))
    assert rows[k].distance_to(c) < 1e-6, (i, k)
    groups[k].append(i)
patch_centers = {k: np.array([[faces[i].center().X, faces[i].center().Y, faces[i].center().Z] for i in v])
                 for k, v in groups.items() if v}
np.save("patch_centers.npy", patch_centers, allow_pickle=True)
print({k: (len(v), round(sum(faces[i].area for i in v), 4)) for k, v in groups.items()})

# -- cell 14 ------------------------------------------------------------------------
# Into the mesher. gmsh OCC on the STEP, surfaces matched to my named face groups by centroid, coarse 
import gmsh
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.model.add("turbofan")
gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
com = np.array([gmsh.model.occ.getCenterOfMass(2,t) for _,t in surfs])
allc = np.vstack([patch_centers[k] for k in patch_centers])
lab  = sum([[k]*len(patch_centers[k]) for k in patch_centers], [])
d = np.linalg.norm(com[:,None,:]-allc[None,:,:], axis=2)
idx = d.argmin(1)
print("surfaces:", len(surfs), " max centroid mismatch: %.2e" % d[np.arange(len(surfs)), idx].max())
assign = {}
for s,(dim,tag) in enumerate(surfs): assign.setdefault(lab[idx[s]], []).append(tag)
for k,v in assign.items(): gmsh.model.addPhysicalGroup(2, v, name=k)
gmsh.model.addPhysicalGroup(3, [t for _,t in gmsh.model.getEntities(3)], name="internal")
print({k:len(v) for k,v in assign.items()})

# -- cell 15 ------------------------------------------------------------------------
# Centroid convention mismatch: build123d's default face center is geometric, gmsh's is mass. Recomput
patch_centers = {k: np.array([list(faces[i].center(CenterOf.MASS).to_tuple()) for i in v])
                 for k, v in groups.items() if v}
allc = np.vstack([patch_centers[k] for k in patch_centers])
lab  = sum([[k]*len(patch_centers[k]) for k in patch_centers], [])
d = np.linalg.norm(com[:,None,:]-allc[None,:,:], axis=2); idx = d.argmin(1)
print("max centroid mismatch: %.2e" % d[np.arange(len(surfs)), idx].max())
gmsh.model.removePhysicalGroups()
assign = {}
for s,(dim,tag) in enumerate(surfs): assign.setdefault(lab[idx[s]], []).append(tag)
for k,v in assign.items(): gmsh.model.addPhysicalGroup(2, v, name=k)
gmsh.model.addPhysicalGroup(3, [t for _,t in gmsh.model.getEntities(3)], name="internal")
print({k:len(v) for k,v in assign.items()})
