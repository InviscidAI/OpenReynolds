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

# ---- passage / blade parameters (metres, radians) ----
R_HUB, R_TIP = 0.045, 0.090          # hub 45 mm, tip 90 mm -> span 45 mm
N_BLADE      = 28
PITCH        = 2*np.pi/N_BLADE       # angular pitch
CHORD        = 0.022                 # 22 mm
TMAX         = 0.06                  # 6 % max thickness
CAM_M, CAM_P = 0.04, 0.40            # camber 4 % at 40 % chord (not specified -> chosen)
STAG_ROOT    = np.radians(30.0)
TWIST        = np.radians(35.0)      # further 35 deg root -> tip

def stagger(r):                      # linear twist in radius
    return STAG_ROOT + TWIST*(r-R_HUB)/(R_TIP-R_HUB)

def naca(n=81):
    b = np.linspace(0, np.pi, n); x = 0.5*(1-np.cos(b))       # cosine spacing
    m, p, t = CAM_M, CAM_P, TMAX
    yc = np.where(x < p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
    dy = np.where(x < p, 2*m/p**2*(p-x),      2*m/(1-p)**2*(p-x))
    th = np.arctan(dy)
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    up = (x-yt*np.sin(th), yc+yt*np.cos(th))
    lo = (x+yt*np.sin(th), yc-yt*np.cos(th))
    return x, yc, up, lo

XC, YC, UP, LO = naca()

def to_zt(r, xs, ys):
    """chord-frame (x/c, y/c) -> (axial z, tangential arc t) at radius r"""
    xi = stagger(r); X = (xs-0.5)*CHORD; Y = ys*CHORD
    return X*np.cos(xi)-Y*np.sin(xi), X*np.sin(xi)+Y*np.cos(xi)

for r in (R_HUB, R_TIP):
    zu, tu = to_zt(r, *UP); zl, tl = to_zt(r, *LO)
    plt.plot(zu*1e3, tu*1e3, '-', lw=1); plt.plot(zl*1e3, tl*1e3, '-', lw=1)

for r, nm in ((R_HUB, 'root'), (R_TIP, 'tip ')):
    z2, t2 = to_zt(r, np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    print("%s chord [mm] %.3f  stagger [deg] %.2f" % (nm, np.hypot(z2[1]-z2[0], t2[1]-t2[0])*1e3,
          np.degrees(np.arctan2(t2[1]-t2[0], z2[1]-z2[0]))))
yt_max = 2*max(5*TMAX*(0.2969*np.sqrt(XC)-0.1260*XC-0.3516*XC**2+0.2843*XC**3-0.1036*XC**4))
print("max thickness / chord %.4f" % yt_max)
print("pitch [deg] %.4f, pitch arc at hub [mm] %.3f, blade tangential extent at root [mm] %.3f"
      % (np.degrees(PITCH), R_HUB*PITCH*1e3, CHORD*np.sin(STAG_ROOT)*1e3))
plt.gca().set_aspect('equal'); plt.xlabel('z [mm]'); plt.ylabel('t [mm]'); plt.title('root & tip sections')
plt.savefig('sections.png', dpi=110)

# -- cell 2 -------------------------------------------------------------------------
# Numbers match the spec exactly. Let me see the sections and then build the blade solid.
from IPython.display import Image, display
display(Image('sections.png'))
from build123d import *
print([n for n in ('Spline','Wire','loft','Solid','Rot','Pos') if n in dir()])

# -- cell 3 -------------------------------------------------------------------------
# The previous cell errored, so its `sections` binding is not in the script. Re-emitting the blade bui
from build123d import *

def sect_pts(r, xs, ys):
    """chord-frame arrays -> 3D points lying on the cylinder of radius r"""
    z, t = to_zt(r, xs, ys); th = t/r
    return [(r*np.cos(a), r*np.sin(a), zz) for a, zz in zip(th, z)]

R_BL0, R_BL1 = 0.0435, 0.0915        # blade loft overshoots hub & shroud for clean cuts
blade_sections = []
for r in np.linspace(R_BL0, R_BL1, 7):
    up = Spline(*sect_pts(r, *UP))                        # suction (upper) LE -> TE
    lo = Spline(*sect_pts(r, LO[0][::-1], LO[1][::-1]))   # pressure (lower) TE -> LE
    blade_sections.append(Wire([up, lo]))
blade = Solid.make_loft(blade_sections, False)
print("blade volume [mm^3] %.2f  faces %d" % (blade.volume*1e9, len(blade.faces())))
bb = blade.bounding_box()
print("bbox min [mm]", [round(v*1e3,2) for v in bb.min], " max", [round(v*1e3,2) for v in bb.max])

# -- cell 4 -------------------------------------------------------------------------
# Only the `is_valid` print was wrong — the slab built. Re-emitting with the fix so the script stays v
zh_u, _ = to_zt(R_HUB, *UP); zh_l, _ = to_zt(R_HUB, *LO)
Z_IN  = float(min(zh_u.min(), zh_l.min())) - CHORD     # one chord upstream of blade
Z_OUT = float(max(zh_u.max(), zh_l.max())) + CHORD     # one chord downstream

def periodic_curve(r, sign):
    """edges of the mid-pitch periodic curve at radius r, sign=+1 upper / -1 lower"""
    zc, tc = to_zt(r, XC, YC)                     # camber line, z increasing
    th = tc/r + sign*PITCH/2
    P = lambda a, z: (r*np.cos(a), r*np.sin(a), z)
    pts = [P(a, z) for a, z in zip(th, zc)]
    return ([Line(P(th[0], Z_IN), pts[0]), Spline(*pts), Line(pts[-1], P(th[-1], Z_OUT))],
            P(th[0], Z_IN), P(th[-1], Z_OUT))

slab_sections = []
for r in np.linspace(0.040, 0.095, 5):
    eu, u_in, u_out = periodic_curve(r, +1)
    el, l_in, l_out = periodic_curve(r, -1)
    slab_sections.append(Wire(eu + [Line(u_out, l_out)] +
                              [e.reversed() for e in el[::-1]] + [Line(l_in, u_in)]))
slab = Solid.make_loft(slab_sections, False)
print("blade axial extent at hub [mm] %.2f .. %.2f ; inlet %.2f outlet %.2f"
      % (zh_u.min()*1e3, zh_u.max()*1e3, Z_IN*1e3, Z_OUT*1e3))
print("slab volume [mm^3] %.1f faces %d valid %s" % (slab.volume*1e9, len(slab.faces()), slab.is_valid))

# -- cell 5 -------------------------------------------------------------------------
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS
print("shells", len(slab.shells()), "faces", len(slab.faces()))
for i,f in enumerate(slab.faces()):
    print(i, type(f.geom_adaptor()).__name__, f.geom_type, "area[mm2] %.1f"%(f.area*1e6),
          "ok" if BRepCheck_Analyzer(f.wrapped).IsValid() else "BAD")

# -- cell 6 -------------------------------------------------------------------------
caps = [Face.make_surface(w) for w in (slab_sections[0], slab_sections[-1])]
shell = Shell(slab.faces() + caps)
slab_solid = Solid(shell)
print("cap areas [mm2]", [round(c.area*1e6,1) for c in caps])
print("slab solid volume [mm^3] %.1f  valid %s  faces %d"
      % (slab_solid.volume*1e9, slab_solid.is_valid, len(slab_solid.faces())))

# -- cell 7 -------------------------------------------------------------------------
zc = 0.5*(Z_IN+Z_OUT); hz = (Z_OUT-Z_IN) + 0.02
ring = Pos(0, 0, zc)*(Cylinder(R_TIP, hz) - Cylinder(R_HUB, hz))   # exact hub & shroud surfaces
domain = slab_solid & ring
passage = domain - blade
print("domain vol [mm^3] %.1f  passage vol [mm^3] %.1f  faces %d  valid %s"
      % (domain.volume*1e9, passage.volume*1e9, len(passage.faces()), passage.is_valid))
for i, f in enumerate(passage.faces()):
    c = f.center()
    print(i, f.geom_type, "A=%.1fmm2 c=(r %.1f, th %.2fdeg, z %.1f)mm" %
          (f.area*1e6, np.hypot(c.X, c.Y)*1e3, np.degrees(np.arctan2(c.Y, c.X)), c.Z*1e3))

# -- cell 8 -------------------------------------------------------------------------
# Enough analysis — label with the simple API and get to a mesh.
def theta_cam(r, z):
    zc, tc = to_zt(r, XC, YC)
    return float(np.interp(z, zc, tc))/r

labels = {}
for i, f in enumerate(passage.faces()):
    c = f.center(); n = f.normal_at()
    r, th, z = np.hypot(c.X, c.Y), np.arctan2(c.Y, c.X), c.Z
    d = np.degrees(th - theta_cam(r, z))
    et = np.array([-np.sin(th), np.cos(th), 0.0])
    ndott = float(np.array([n.X, n.Y, n.Z]) @ et)
    if   f.geom_type == GeomType.CYLINDER and abs(r-R_HUB) < 1e-5: lab = "hub"
    elif f.geom_type == GeomType.CYLINDER and abs(r-R_TIP) < 1e-5: lab = "shroud"
    elif abs(abs(n.Z)-1) < 1e-6: lab = "inlet" if z < 0 else "outlet"
    elif d >  3.0: lab = "periodic_upper"
    elif d < -3.0: lab = "periodic_lower"
    else: lab = "blade_suction" if ndott < 0 else "blade_pressure"
    labels.setdefault(lab, []).append(i)
    print("%2d %-15s dtheta=%7.2f n.e_t=%+6.3f r=%.1f z=%6.1f" % (i, lab, d, ndott, r*1e3, z*1e3))
print({k: len(v) for k, v in labels.items()})

# -- cell 9 -------------------------------------------------------------------------
import gmsh, subprocess, os
export_brep(passage, "passage.brep")

gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("passage")
gmsh.model.occ.importShapes("passage.brep"); gmsh.model.occ.synchronize()

bd_centers = [np.array([f.center().X, f.center().Y, f.center().Z]) for f in passage.faces()]
lab_of = {i: L for L, idx in labels.items() for i in idx}
gm = {}
for dim, tag in gmsh.model.getEntities(2):
    com = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
    j = int(np.argmin([np.linalg.norm(com-c) for c in bd_centers]))
    gm.setdefault(lab_of[j], []).append(tag)
print(gm)
for L, tags in gm.items():
    gmsh.model.addPhysicalGroup(2, tags, name=L)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="internal")
print("volumes", gmsh.model.getEntities(3))

# -- cell 10 ------------------------------------------------------------------------
# Periodic point-matching fails on the mid surfaces (trim curves aren't exact rotational copies). Drop
gmsh.clear(); gmsh.model.add("passage2")
gmsh.model.occ.importShapes("passage.brep"); gmsh.model.occ.synchronize()
gm = {}
for dim, tag in gmsh.model.getEntities(2):
    c = np.array(gmsh.model.occ.getCenterOfMass(2, tag))
    j = int(np.argmin([np.linalg.norm(c-b) for b in bd_centers]))
    gm.setdefault(lab_of[j], []).append(tag)
for L, tags in gm.items():
    gmsh.model.addPhysicalGroup(2, tags, name=L)
gmsh.model.addPhysicalGroup(3, [1], name="internal")
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.0030)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0010)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("passage.msh")
print(gm, "nodes", gmsh.model.mesh.getNodes()[0].size)

# -- cell 11 ------------------------------------------------------------------------
# Mesh volume 4.213e-5 m³ matches the CAD passage (4.211e-5). Convert to OpenFOAM and check.
import textwrap, pathlib
for d in ("system", "constant"): pathlib.Path(d).mkdir(exist_ok=True)
pathlib.Path("system/controlDict").write_text(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
    application     simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
    endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
    """))
for f in ("fvSchemes", "fvSolution"):
    pathlib.Path("system/"+f).write_text("FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n" % f)
print(subprocess.run(["gmshToFoam", "passage.msh"], capture_output=True, text=True).stdout[-700:])

# -- cell 12 ------------------------------------------------------------------------
print(pathlib.Path("constant/polyMesh/boundary").read_text())
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 13 ------------------------------------------------------------------------
print(r.stdout[-6000:-2000] if len(r.stdout)>2500 else r.stdout)
print("STDERR", r.stderr[-800:])

# -- cell 14 ------------------------------------------------------------------------
pathlib.Path("system/fvSchemes").write_text(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
    ddtSchemes { default steadyState; }
    gradSchemes { default Gauss linear; }
    divSchemes { default none; }
    laplacianSchemes { default Gauss linear corrected; }
    interpolationSchemes { default linear; }
    snGradSchemes { default corrected; }
    """))
pathlib.Path("system/fvSolution").write_text(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
    solvers {}
    """))
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):])

# -- cell 15 ------------------------------------------------------------------------
gmsh.model.mesh.clear()
f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", gm['blade_suction']+gm['blade_pressure'])
gmsh.model.mesh.field.setNumber(f1, "Sampling", 100)
f2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f2, "InField", f1)
gmsh.model.mesh.field.setNumber(f2, "SizeMin", 0.0005)   # 0.5 mm on the blade
gmsh.model.mesh.field.setNumber(f2, "SizeMax", 0.0018)   # 1.8 mm in the passage
gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.0005)
gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.006)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)
for o in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
    gmsh.option.setNumber("Mesh."+o, 0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.0018)
gmsh.model.mesh.generate(3)
gmsh.write("passage.msh")
print("nodes", gmsh.model.mesh.getNodes()[0].size//3, "tets", len(gmsh.model.mesh.getElementsByType(4)[0]))

# -- cell 16 ------------------------------------------------------------------------
subprocess.run(["gmshToFoam", "passage.msh"], capture_output=True, text=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
s = r.stdout[r.stdout.find("Mesh stats"):]
print(s[:900]); print(s[s.find("Checking geometry"):])

# -- cell 17 ------------------------------------------------------------------------
# Mesh OK at 71 k cells. Now export the per-patch STLs from the CAD faces and render them.
faces = passage.faces()
for L, idx in labels.items():
    export_stl(Compound(children=[faces[i] for i in idx]), L+".stl", tolerance=1e-5, angular_tolerance=0.1)
print(sorted(os.listdir('.')))

import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
p = pv.Plotter(off_screen=True, window_size=(900, 700))
cols = dict(hub='tan', shroud='lightsteelblue', inlet='green', outlet='red',
            blade_pressure='orange', blade_suction='purple',
            periodic_upper='lightgray', periodic_lower='lightgray')
for L, c in cols.items():
    p.add_mesh(pv.read(L+".stl"), color=c, opacity=0.45 if 'periodic' in L or L=='shroud' else 1.0,
               show_edges=False, label=L)
p.add_legend(); p.camera_position = [(0.20, 0.12, 0.14), (0.065, 0, 0), (0, 0, 1)]
p.show(screenshot="passage.png")
display(Image("passage.png"))

# -- cell 18 ------------------------------------------------------------------------
# Final verification: that the two side patches really are one pitch apart (rotational copies), plus t
lo = pv.read("periodic_lower.stl"); up = pv.read("periodic_upper.stl")
P = lo.points.copy()
Rz = np.array([[np.cos(PITCH), -np.sin(PITCH), 0], [np.sin(PITCH), np.cos(PITCH), 0], [0, 0, 1]])
d = np.array([np.linalg.norm(up.points[up.find_closest_point(q)]-q) for q in (P @ Rz.T)])
print("periodic_lower rotated by one pitch -> max dev from periodic_upper: %.4f mm (mean %.4f)"
      % (d.max()*1e3, d.mean()*1e3))
A = {L: sum(faces[i].area for i in idx)*1e6 for L, idx in labels.items()}
print("patch areas [mm2]", {k: round(v,1) for k, v in A.items()})
print("hub radius %.1f mm, shroud radius %.1f mm, span %.1f mm"
      % (R_HUB*1e3, R_TIP*1e3, (R_TIP-R_HUB)*1e3))
print("blades %d -> pitch %.3f deg ; axial domain %.2f -> %.2f mm (= blade +- 1 chord)"
      % (N_BLADE, np.degrees(PITCH), Z_IN*1e3, Z_OUT*1e3))
print("stagger root %.1f deg, tip %.1f deg (twist %.1f deg)"
      % (np.degrees(stagger(R_HUB)), np.degrees(stagger(R_TIP)), np.degrees(TWIST)))

# -- cell 19 ------------------------------------------------------------------------
# Areas of the two periodic patches are identical to 0.1 mm²; the 2.2 mm was nearest-*vertex*, not nea
Q = lo.points @ Rz.T
_, cp = up.find_closest_cell(Q, return_closest_point=True)
d = np.linalg.norm(cp-Q, axis=1)
print("periodic_lower rotated one pitch -> distance to periodic_upper surface: max %.4f mm, mean %.5f mm"
      % (d.max()*1e3, d.mean()*1e3))
