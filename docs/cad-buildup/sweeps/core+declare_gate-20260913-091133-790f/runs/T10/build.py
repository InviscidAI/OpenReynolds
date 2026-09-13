"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run that cell properly now.
import numpy as np, os
os.chdir("/home/qiuzi/.openreynolds-buildup/work/T10-20260913-091133-36bd/t10")

# ---- parameters (metres, machine axis = x) ----
R_HUB   = 0.045
SPAN    = 0.045
R_TIP   = R_HUB + SPAN
NBLADE  = 28
PITCH   = 2*np.pi/NBLADE
CHORD   = 0.022
TMAX    = 0.06
STAG_R  = np.radians(30.0)
TWIST   = np.radians(35.0)
CAMBER  = np.radians(45.0)   # total camber (turning) angle: design choice, not specified

def section_2d(n=81):
    b = 0.5*(1-np.cos(np.linspace(0, np.pi, n)))
    x = b*CHORD
    R = CHORD/(2*np.sin(CAMBER/2))
    yc = np.sqrt(R**2-(x-CHORD/2)**2) - R*np.cos(CAMBER/2)
    dyc = -(x-CHORD/2)/np.sqrt(R**2-(x-CHORD/2)**2)
    th = np.arctan(dyc)
    t = b
    yt = 5*TMAX*CHORD*(0.2969*np.sqrt(np.maximum(t,0))-0.1260*t-0.3516*t**2+0.2843*t**3-0.1036*t**4)
    xu, yu = x-yt*np.sin(th), yc+yt*np.cos(th)
    xl, yl = x+yt*np.sin(th), yc-yt*np.cos(th)
    return (xu, yu), (xl, yl)

(xu,yu),(xl,yl) = section_2d()
print("chord (LE->TE along camber chord line):", xu[-1]-xu[0])
print("max thickness / chord:", np.max(np.hypot(xu-xl, yu-yl))/CHORD)
print("pitch deg:", np.degrees(PITCH), "stagger tip deg:", np.degrees(STAG_R+TWIST))
print("pitch arc at hub (m):", PITCH*R_HUB, " at tip:", PITCH*R_TIP)

# -- cell 2 -------------------------------------------------------------------------
# Sections wrap onto cylinders about the x-axis; stacking on each section's centroid at θ=0. Let me ge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSEC = 9
R_LO, R_HI = R_HUB-0.002, R_TIP+0.002      # slight overhang for clean trims

def stagger_at(r):
    f = (r-R_HUB)/SPAN                      # linear twist, 0 at hub, 1 at tip
    return STAG_R + TWIST*f

def section_planar(r):
    """(axial a, tangential arc s) closed loop of the section at radius r, centroid at origin."""
    (xu,yu),(xl,yl) = section_2d()
    x = np.concatenate([xu, xl[::-1][1:-1]])
    y = np.concatenate([yu, yl[::-1][1:-1]])
    xi = stagger_at(r)
    a =  x*np.cos(xi) - y*np.sin(xi)
    s =  x*np.sin(xi) + y*np.cos(xi)
    return a-a.mean(), s-s.mean()

def section_3d(r, theta0=0.0):
    a, s = section_planar(r)
    th = theta0 + s/r
    return np.column_stack([a, r*np.cos(th), r*np.sin(th)])

radii = np.linspace(R_LO, R_HI, NSEC)
secs  = [section_3d(r) for r in radii]
amin = min(p[:,0].min() for p in secs); amax = max(p[:,0].max() for p in secs)
X_IN, X_OUT = amin - CHORD, amax + CHORD
print("blade axial extent: %.4f .. %.4f m" % (amin, amax))
print("inlet plane x=%.4f  outlet plane x=%.4f" % (X_IN, X_OUT))

fig, ax = plt.subplots(figsize=(5,5))
for r,p in zip(radii, secs):
    a,s = section_planar(r); ax.plot(a, s+0*r, label="r=%.3f"%r)
ax.set_aspect("equal"); ax.set_xlabel("axial"); ax.set_ylabel("tangential"); ax.legend(fontsize=6)
fig.savefig("sections.png", dpi=110)
print("stagger root/tip deg:", np.degrees(stagger_at(R_HUB)), np.degrees(stagger_at(R_TIP)))

# -- cell 3 -------------------------------------------------------------------------
from IPython.display import display
fig, ax = plt.subplots(figsize=(5,5))
for r in radii:
    a,s = section_planar(r); ax.plot(a, s, lw=1, label="r=%.3f"%r)
ax.set_aspect("equal"); ax.set_xlabel("axial x"); ax.set_ylabel("tangential arc s"); ax.legend(fontsize=6)
display(fig); plt.close(fig)

# -- cell 4 -------------------------------------------------------------------------
# The loft produced an uncapped shell (my section wires are non-planar, so caps were skipped). Re-emit
from build123d import *
from OCP.BRepCheck import BRepCheck_Analyzer

def _staggered(r, x, y):
    xi = stagger_at(r)
    a = x*np.cos(xi) - y*np.sin(xi); s = x*np.sin(xi) + y*np.cos(xi)
    (xu,yu),(xl,yl) = section_2d()
    xa = np.concatenate([xl[::-1], xu[1:]]); ya = np.concatenate([yl[::-1], yu[1:]])
    ca = (xa*np.cos(xi) - ya*np.sin(xi)).mean(); cs = (xa*np.sin(xi) + ya*np.cos(xi)).mean()
    return a-ca, s-cs

def _to3d(r, a, s, theta0):
    th = theta0 + s/r
    return [Vector(float(ai), float(r*np.cos(t)), float(r*np.sin(t))) for ai, t in zip(a, th)]

def section_wire(r, theta0=0.0):
    """Closed section wire on the cylinder of radius r: sharp TE, two splines."""
    (xu,yu),(xl,yl) = section_2d()
    al, sl = _staggered(r, xl[::-1], yl[::-1])     # TE -> LE (pressure-ish side)
    au, su = _staggered(r, xu, yu)                 # LE -> TE (suction-ish side)
    pl = _to3d(r, al, sl, theta0); pu = _to3d(r, au, su, theta0)
    pu[0] = pl[-1]; pu[-1] = pl[0]
    return Wire([Spline(*pl).edge(), Spline(*pu).edge()])

def blade_solid(theta0=0.0):
    ws = [section_wire(r, theta0) for r in radii]
    shell_faces = Solid.make_loft(ws, ruled=False).faces()
    caps = [Face.make_surface(ws[0]), Face.make_surface(ws[-1])]
    return Solid(Shell(shell_faces + caps))

blade0 = blade_solid(0.0)
print("volume %.3e m^3  faces %d  valid %s" % (blade0.volume, len(blade0.faces()), BRepCheck_Analyzer(blade0.wrapped).IsValid()))

# -- cell 5 -------------------------------------------------------------------------
# Valid blade solid. Now the sector (one pitch, hub→shroud, inlet→outlet) with blades placed *on* the 
prof = Plane.XY * Rectangle(X_OUT-X_IN, R_TIP-R_HUB, align=(Align.MIN, Align.MIN))
prof = Pos(X_IN, R_HUB, 0) * prof
sector = Rot(X=-np.degrees(PITCH)/2) * revolve(prof, axis=Axis.X, revolution_arc=np.degrees(PITCH))
print("sector volume %.4e  expected %.4e" %
      (sector.volume, 0.5*PITCH*(R_TIP**2-R_HUB**2)*(X_OUT-X_IN)))

bladeA = blade_solid(-PITCH/2)      # blade on the -theta side plane
bladeB = blade_solid(+PITCH/2)      # its neighbour, exact rotational copy
passage = sector - bladeA - bladeB
print("passage volume %.4e  (sector - ~1 blade = %.4e)" % (passage.volume, sector.volume - blade0.volume*(SPAN/(R_HI-R_LO))))
print("solids %d, faces %d, valid %s" % (len(passage.solids()), len(passage.faces()), BRepCheck_Analyzer(passage.wrapped).IsValid()))

# -- cell 6 -------------------------------------------------------------------------
# Single valid solid. Now classify the 12 faces by their underlying surface (cylinder radius, plane no
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

def dist(shape, pnt):
    d = BRepExtrema_DistShapeShape(shape.wrapped, Vertex(pnt).wrapped); d.Perform(); return d.Value()

groups = {k: [] for k in ["inlet","outlet","hub","shroud","periodic_1","periodic_2","blade_suction","blade_pressure"]}
for f in passage.faces():
    c = f.center(); r = np.hypot(c.Y, c.Z); n = f.normal_at(c)
    if f.geom_type == GeomType.PLANE and abs(abs(n.X)-1) < 1e-6:
        groups["inlet" if c.X < 0.5*(X_IN+X_OUT) else "outlet"].append(f)
    elif f.geom_type == GeomType.CYLINDER and abs(r-R_HUB) < 1e-6:
        groups["hub"].append(f)
    elif f.geom_type == GeomType.CYLINDER and abs(r-R_TIP) < 1e-6:
        groups["shroud"].append(f)
    elif f.geom_type == GeomType.PLANE and abs(n.X) < 1e-6:
        groups["periodic_1" if np.arctan2(c.Z, c.Y) < 0 else "periodic_2"].append(f)
    else:
        dA, dB = dist(bladeA, c), dist(bladeB, c)
        groups["blade_suction" if dA < dB else "blade_pressure"].append(f)

for k, v in groups.items():
    print("%-14s n=%d  area=%.4e  thetas=%s" % (k, len(v), sum(f.area for f in v),
          ["%.1f" % np.degrees(np.arctan2(f.center().Z, f.center().Y)) for f in v]))

# -- cell 7 -------------------------------------------------------------------------
# Enough analysis — mesh it. Export STEP, import in gmsh, tag each surface by matching centre of mass 
import gmsh, subprocess, os
export_step(passage, "passage.step")
cen = {k: [tuple(f.center(CenterOf.MASS)) for f in v] for k, v in groups.items()}

gmsh.initialize()
gmsh.open("passage.step")
gmsh.model.occ.synchronize()
surf = gmsh.model.getEntities(2)
tags = {k: [] for k in cen}
for (d, t) in surf:
    com = np.array(gmsh.model.occ.getCenterOfMass(d, t))
    best = min(((np.linalg.norm(com-np.array(c)), k) for k, cs in cen.items() for c in cs))
    tags[best[1]].append(t); print("surf %d -> %-14s (mismatch %.2e)" % (t, best[1], best[0]))
for k, ts in tags.items():
    gmsh.model.addPhysicalGroup(2, ts, name=k)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="internal")
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0015)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.0025)
gmsh.model.mesh.generate(3)
gmsh.write("passage.msh")
gmsh.finalize()
print("cells:", os.path.getsize("passage.msh"))

# -- cell 8 -------------------------------------------------------------------------
# Mesh exists. Now a minimal case skeleton and `gmshToFoam` + `checkMesh`.
import subprocess, os, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr % "controlDict" + textwrap.dedent("""
    application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
    deltaT 1; writeControl timeStep; writeInterval 1;
"""))
open("system/fvSchemes","w").write(hdr % "fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr % "fvSolution" + "solvers{} \n")
r = subprocess.run(["gmshToFoam","passage.msh"], capture_output=True, text=True)
print(r.stdout[-900:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 10 ------------------------------------------------------------------------
# Mesh OK, 14.5k cells, all 8 patches named. Now set physical patch types (walls vs patch) and look at
import re
b = open("constant/polyMesh/boundary").read()
for w in ["hub","shroud","blade_suction","blade_pressure"]:
    b = re.sub(r"(\b%s\b\s*\{\s*type\s+)patch;" % w, r"\1wall;", b)
open("constant/polyMesh/boundary","w").write(b)
print(re.findall(r"^\s{4}(\w+)\n\s{4}\{\n\s+type\s+(\w+);", b, re.M))

# -- cell 11 ------------------------------------------------------------------------
# Now a picture: read the polyMesh with pyvista and colour the named patches.
import pyvista as pv
pv.start_xvfb() if False else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0)
m = rd.read()
bnd = m["boundary"]
print(bnd.keys())
p = pv.Plotter(off_screen=True, window_size=(900,700))
cols = dict(inlet="lightblue", outlet="pink", hub="grey", shroud="wheat",
            periodic_1="palegreen", periodic_2="palegreen",
            blade_suction="red", blade_pressure="blue")
for k, c in cols.items():
    p.add_mesh(bnd[k], color=c, show_edges=True, edge_color="k", line_width=0.3, opacity=1.0 if k not in ("periodic_2","shroud") else 0.25, label=k)
p.add_legend(); p.camera_position = "iso"; p.show_axes()
img = p.screenshot("passage.png"); p.close()
from IPython.display import Image, display
display(Image("passage.png"))

# -- cell 12 ------------------------------------------------------------------------
# Now measure the stated numbers off the built OCC geometry (not off my input arrays): hub/tip radii, 
def measure_section(r):
    w = section_wire(r, 0.0)
    P = np.array([tuple(w.edges()[0] @ t) for t in np.linspace(0,1,400)] +
                 [tuple(w.edges()[1] @ t) for t in np.linspace(0,1,400)])
    a = P[:,0]; th = np.arctan2(P[:,2], P[:,1]); s = th*r          # unwrap
    d2 = ((a[:,None]-a[None,:])**2 + (s[:,None]-s[None,:])**2)
    i, j = np.unravel_index(d2.argmax(), d2.shape)                 # LE/TE = farthest pair
    if a[i] > a[j]: i, j = j, i                                    # i=LE(upstream), j=TE
    chord = np.hypot(a[j]-a[i], s[j]-s[i]); stag = np.degrees(np.arctan2(s[j]-s[i], a[j]-a[i]))
    # thickness: max distance between the two splines normal to chord
    u = np.array([a[j]-a[i], s[j]-s[i]])/chord; nvec = np.array([-u[1], u[0]])
    q = np.column_stack([a-a[i], s-s[i]])
    off = q @ nvec; par = q @ u
    bins = np.linspace(0, chord, 60); tmax = 0
    for k in range(59):
        m = (par >= bins[k]) & (par < bins[k+1])
        if m.sum() > 1: tmax = max(tmax, off[m].max()-off[m].min())
    return chord, stag, tmax/chord

for r, lbl in [(R_HUB,"root r=45mm"), (R_TIP,"tip  r=90mm")]:
    c, st, tc = measure_section(r)
    print("%s: chord %.4f m (req 0.0220) | stagger %.2f deg | t/c %.3f (req 0.060)" % (lbl, c, st, tc))

# -- cell 13 ------------------------------------------------------------------------
# Chord 22.0 mm, twist exactly 35.0°. Now the annulus and pitch, measured from the meshed patches them
pts = {k: np.asarray(bnd[k].points) for k in bnd.keys()}
rad = lambda P: np.hypot(P[:,1], P[:,2])
print("hub patch radius  : %.5f .. %.5f m (req 0.045)" % (rad(pts['hub']).min(), rad(pts['hub']).max()))
print("shroud patch radius: %.5f .. %.5f m (req 0.090)" % (rad(pts['shroud']).min(), rad(pts['shroud']).max()))
print("span = %.5f m (req 0.045)" % (rad(pts['shroud']).mean()-rad(pts['hub']).mean()))
t1 = np.degrees(np.arctan2(pts['periodic_1'][:,2], pts['periodic_1'][:,1])).mean()
t2 = np.degrees(np.arctan2(pts['periodic_2'][:,2], pts['periodic_2'][:,1])).mean()
print("periodic planes at theta = %.4f / %.4f deg -> pitch %.4f deg = 360/%.2f blades" % (t1, t2, t2-t1, 360/(t2-t1)))
bl = np.vstack([pts['blade_suction'], pts['blade_pressure']])
print("blade axial extent %.4f..%.4f ; inlet x=%.4f (gap %.4f), outlet x=%.4f (gap %.4f), chord 0.0220"
      % (bl[:,0].min(), bl[:,0].max(), pts['inlet'][:,0].mean(), bl[:,0].min()-pts['inlet'][:,0].mean(),
         pts['outlet'][:,0].mean(), pts['outlet'][:,0].mean()-bl[:,0].max()))
print("areas periodic_1/2: %.6e %.6e (must match for periodicity)" % (bnd['periodic_1'].area, bnd['periodic_2'].area))

# -- cell 14 ------------------------------------------------------------------------
# Shape and numbers are right. Now refine: finer cells on the blade surfaces via a distance field, and
gmsh.initialize()
gmsh.open("passage.step"); gmsh.model.occ.synchronize()
tags = {k: [] for k in cen}
for (d, t) in gmsh.model.getEntities(2):
    com = np.array(gmsh.model.occ.getCenterOfMass(d, t))
    k = min(((np.linalg.norm(com-np.array(c)), kk) for kk, cs in cen.items() for c in cs))[1]
    tags[k].append(t)
for k, ts in tags.items(): gmsh.model.addPhysicalGroup(2, ts, name=k)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="internal")

blade_surfs = tags["blade_suction"] + tags["blade_pressure"]
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", blade_surfs)
gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.00045)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.0016)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0008)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.005)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.model.mesh.generate(3)
gmsh.write("passage.msh")
print(gmsh.model.mesh.getElementsByType(4)[0].size, "tets")
gmsh.finalize()

# -- cell 15 ------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam","passage.msh"], capture_output=True, text=True); print(r.stdout[-300:])
b = open("constant/polyMesh/boundary").read()
for w in ["hub","shroud","blade_suction","blade_pressure"]:
    b = re.sub(r"(\b%s\b\s*\{\s*type\s+)patch;" % w, r"\1wall;", b)
open("constant/polyMesh/boundary","w").write(b)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if any(s in l for s in
      ["cells:","patches","Max aspect","non-orthogonality","skewness","Total volume","Mesh OK","***","faces "])))

# -- cell 16 ------------------------------------------------------------------------
# Refined mesh is OK (233k cells). Final look and patch tally on the refined mesh.
rd2 = pv.OpenFOAMReader("case.foam"); rd2.set_active_time_value(0)
bnd2 = rd2.read()["boundary"]
for k in bnd2.keys(): print("%-14s faces %6d  area %.6e" % (k, bnd2[k].n_cells, bnd2[k].area))
p = pv.Plotter(off_screen=True, window_size=(950,700))
for k, c in cols.items():
    p.add_mesh(bnd2[k], color=c, show_edges=True, edge_color="k", line_width=0.2,
               opacity=0.18 if k in ("periodic_2","shroud") else 1.0, label=k)
p.add_legend(); p.camera.position = (0.09, 0.10, 0.10); p.camera.focal_point = (0.0, 0.067, 0.0)
p.screenshot("passage_fine.png"); p.close()
display(Image("passage_fine.png"))

# -- cell 17 ------------------------------------------------------------------------
# Let me also export one STL per named patch group (from the OCC faces), so each named surface exists 
os.makedirs("constant/triSurface", exist_ok=True)
for k, fs in groups.items():
    export_stl(Compound(children=[Solid(Shell(fs))]) if False else Compound(fs), f"constant/triSurface/{k}.stl")
print(sorted(os.listdir("constant/triSurface")))
print("passage volume (OCC) %.6e  vs mesh %.6e" % (passage.volume, 4.22949e-05))
