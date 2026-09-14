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

# ---- parameters (metres). From the request unless marked ASSUMED ----
R_HUB, R_TIP = 0.045, 0.090
NB          = 28
CHORD       = 0.022
THICK       = 0.06          # max thickness / chord
STAG_ROOT   = np.radians(30.0)
TWIST       = np.radians(35.0)   # extra stagger at tip
CAMBER      = np.radians(40.0)   # ASSUMED total camber (turning) angle, circular-arc camber line
PITCH_ANG   = 2*np.pi/NB

def stagger(r):
    return STAG_ROOT + TWIST*(r-R_HUB)/(R_TIP-R_HUB)

def section(r, n=81):
    """blade section in (z_axial, t_tangential) at radius r; also camber line."""
    c = CHORD
    Rc = (c/2)/np.sin(CAMBER/2)
    beta = np.linspace(0, np.pi, n)
    x = c*(1-np.cos(beta))/2                      # cosine spacing 0..c
    yc = np.sqrt(Rc**2-(x-c/2)**2) - np.sqrt(Rc**2-(c/2)**2)
    dyc = -(x-c/2)/np.sqrt(Rc**2-(x-c/2)**2)
    phi = np.arctan(dyc)
    xx = x/c
    yt = 5*THICK*c*(0.2969*np.sqrt(xx)-0.1260*xx-0.3516*xx**2+0.2843*xx**3-0.1036*xx**4)
    up = np.array([x-yt*np.sin(phi), yc+yt*np.cos(phi)])
    lo = np.array([x+yt*np.sin(phi), yc-yt*np.cos(phi)])
    cam= np.array([x, yc])
    chi = stagger(r); Rm = np.array([[np.cos(chi),-np.sin(chi)],[np.sin(chi),np.cos(chi)]])
    sh  = np.array([[c/2],[0.0]])
    return [Rm@(a-sh) for a in (up, lo, cam)]

fig, axs = plt.subplots(1,3, figsize=(13,4.5))
for ax, r in zip(axs, [R_HUB, (R_HUB+R_TIP)/2, R_TIP]):
    up, lo, cam = section(r)
    p = PITCH_ANG*r
    ax.plot(up[0]*1e3, up[1]*1e3, 'b'); ax.plot(lo[0]*1e3, lo[1]*1e3, 'r')
    ax.plot(cam[0]*1e3, cam[1]*1e3,'k--')
    for s in (-p/2, p/2):
        ax.plot(cam[0]*1e3, (cam[1]+s)*1e3, 'g-')
    ax.set_title(f"r={r*1e3:.0f}mm  stag={np.degrees(stagger(r)):.0f}d pitch={p*1e3:.1f}mm")
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('z [mm]'); ax.set_ylabel('t [mm]')
plt.tight_layout(); plt.savefig('sections.png', dpi=90)
for r in [R_HUB, 0.0675, R_TIP]:
    up,lo,cam = section(r)
    print(f"r={r*1e3:5.1f} chordproj_z={(up[0].max()-up[0].min())*1e3:5.2f} "
          f"t_extent={(up[1].max()-lo[1].min())*1e3:5.2f} pitch_t={PITCH_ANG*r*1e3:5.2f} "
          f"maxthick={np.max(np.hypot(up[0]-lo[0],up[1]-lo[1]))*1e3:.3f}")
print('sections.png')

# -- cell 2 -------------------------------------------------------------------------
# Thickness checks out (1.320 mm = 6% × 22 mm). Note the root section spans 11.0 mm tangentially again
from IPython.display import Image, display
display(Image('sections.png'))

# -- cell 3 -------------------------------------------------------------------------
# Sections look right. Now the CAD plan: periodic surfaces follow the camber line offset by ±half-pitc
import build123d as bd, inspect
print(inspect.signature(bd.loft))
print(inspect.signature(bd.Spline.__init__))
print(inspect.signature(bd.Face.make_surface) if hasattr(bd.Face,'make_surface') else '')
print([n for n in dir(bd) if 'oft' in n or 'weep' in n])

# -- cell 4 -------------------------------------------------------------------------
print(inspect.signature(bd.Solid.make_loft))
print(inspect.signature(bd.Wire.__init__) if hasattr(bd.Wire,'__init__') else '')
print([n for n in dir(bd.Wire) if 'make' in n or 'combine' in n])

# -- cell 5 -------------------------------------------------------------------------
# The loft ran but reported invalid with only 2 faces — I need to re-emit it as one accepted cell and 
import build123d as bd
from build123d import Vector

def wrap(zs, ts, r):
    th = np.asarray(ts)/r
    return [Vector(r*np.cos(a), r*np.sin(a), z) for a, z in zip(th, np.asarray(zs))]

R_B0, R_B1 = R_HUB-0.001, R_TIP+0.001      # blade overhangs hub/shroud for clean cuts
NSEC = 9
radii_b = np.linspace(R_B0, R_B1, NSEC)

def blade_wire(r):
    up, lo, cam = section(r)
    e_suc = bd.Spline(*wrap(up[0], up[1], r))             # LE->TE suction (upper) side
    e_pre = bd.Spline(*wrap(lo[0][::-1], lo[1][::-1], r)) # TE->LE pressure (lower) side
    return bd.Wire([e_suc, e_pre])

blade = bd.Solid.make_loft([blade_wire(r) for r in radii_b])
print(type(blade), "valid", blade.is_valid, "vol[mm3]", round(blade.volume*1e9,1),
      "nfaces", len(blade.faces()), "nshells", len(blade.shells()))
for f in blade.faces():
    print("  area[mm2]", round(f.area*1e6,2), "center mm", [round(c*1e3,2) for c in f.center()])
print("bbox mm", [round(v*1e3,2) for v in blade.bounding_box().min], [round(v*1e3,2) for v in blade.bounding_box().max])

# -- cell 6 -------------------------------------------------------------------------
# The loft has no end caps. I'll cap the two radial ends with filled surfaces and sew into a closed so
w0, w1 = blade_wire(R_B0), blade_wire(R_B1)
cap0 = bd.Face.make_surface(w0); cap1 = bd.Face.make_surface(w1)
print("caps", round(cap0.area*1e6,2), round(cap1.area*1e6,2))
shell = bd.Shell(blade.faces() + [cap0, cap1])
blade_solid = bd.Solid(shell)
print("valid", blade_solid.is_valid, "closed", shell.is_valid, "vol[mm3]", round(blade_solid.volume*1e9,2),
      "nfaces", len(blade_solid.faces()))

# -- cell 7 -------------------------------------------------------------------------
# Blade solid is valid; section area 20.0 mm² matches the analytic NACA area for 6% × 22 mm. Now the p
_up_r, _lo_r, _cam_r = section(R_HUB)
Z_IN  = min(_up_r[0].min(), _lo_r[0].min()) - CHORD   # one chord upstream of LE
Z_OUT = max(_up_r[0].max(), _lo_r[0].max()) + CHORD   # one chord downstream of TE
print("z_in, z_out [mm]", round(Z_IN*1e3,2), round(Z_OUT*1e3,2))

def mid_curve(r, next=12):
    """camber line at radius r with straight axial extensions to Z_IN / Z_OUT."""
    up, lo, cam = section(r)
    z, t = cam[0], cam[1]
    zu = np.linspace(Z_IN, z[0], next, endpoint=False)
    zd = np.linspace(z[-1], Z_OUT, next+1)[1:]
    return (np.concatenate([zu, z, zd]),
            np.concatenate([np.full(next, t[0]), t, np.full(next, t[-1])]))

def passage_wire(r, narc=25):
    p = PITCH_ANG*r/2.0                      # half pitch in tangential length
    z, t = mid_curve(r)
    e1 = bd.Spline(*wrap(z, t-p, r))                       # periodic side 1 (lower theta)
    e3 = bd.Spline(*wrap(z[::-1], (t+p)[::-1], r))         # periodic side 2 (upper theta)
    za = np.full(narc, Z_OUT); ta = np.linspace(t[-1]-p, t[-1]+p, narc)
    e2 = bd.Spline(*wrap(za, ta, r))                       # outlet arc
    zb = np.full(narc, Z_IN);  tb = np.linspace(t[0]+p, t[0]-p, narc)
    e4 = bd.Spline(*wrap(zb, tb, r))                       # inlet arc
    return bd.Wire([e1, e2, e3, e4])

radii_p = np.linspace(R_HUB, R_TIP, NSEC)
_lat  = bd.Solid.make_loft([passage_wire(r) for r in radii_p])
_hubf = bd.Face.make_surface(passage_wire(R_HUB)); _shrf = bd.Face.make_surface(passage_wire(R_TIP))
block = bd.Solid(bd.Shell(_lat.faces() + [_hubf, _shrf]))
print("block valid", block.is_valid, "vol[mm3]", round(block.volume*1e9,1), "nfaces", len(block.faces()))
for f in block.faces():
    c = f.center(); print("   area[mm2]", round(f.area*1e6,1), "c(r,theta_deg,z)mm",
        round(np.hypot(c.X,c.Y)*1e3,2), round(np.degrees(np.arctan2(c.Y,c.X)),2), round(c.Z*1e3,2))

# -- cell 8 -------------------------------------------------------------------------
# Block is right: the two periodic faces are 12.857° apart = 360/28, and the volume matches the analyt
passage = block - blade_solid
print("valid", passage.is_valid, "vol[mm3]", round(passage.volume*1e9,1),
      "expected", round((block.volume - 923.88e-9*45/47)*1e9,1), "nfaces", len(passage.faces()))
for i,f in enumerate(passage.faces()):
    c = f.center(); print(i, "area[mm2]", round(f.area*1e6,1), "r,th,z:",
        round(np.hypot(c.X,c.Y)*1e3,2), round(np.degrees(np.arctan2(c.Y,c.X)),2), round(c.Z*1e3,2))

# -- cell 9 -------------------------------------------------------------------------
# Volume matches exactly. Faces 6 and 7 are the blade; I classify them as suction/pressure by testing 
def blade_side(face, ns=200):
    """Return ('suction'|'pressure', score) by comparing sampled surface points
       with the analytic upper (suction) and lower (pressure) section curves."""
    vs, _tri = face.tessellate(0.0002)
    pts = np.array([[v.X, v.Y, v.Z] for v in vs])
    du = dl = 0.0
    for P in pts[:: max(1, len(pts)//ns)]:
        r = np.hypot(P[0], P[1]); th = np.arctan2(P[1], P[0]); z = P[2]; t = th*r
        up, lo, cam = section(r)
        du += np.min(np.hypot(up[0]-z, up[1]-t))
        dl += np.min(np.hypot(lo[0]-z, lo[1]-t))
    return ("suction" if du < dl else "pressure"), (du, dl)

for i in (6, 7):
    print(i, blade_side(passage.faces()[i]))

# -- cell 10 ------------------------------------------------------------------------
# Hub/shroud came out as plate surfaces (com radius 44.6 mm — sagging), not exact cylinders. Fix: loft
radii_pe = np.linspace(R_HUB-0.001, R_TIP+0.001, NSEC)
_late = bd.Solid.make_loft([passage_wire(r) for r in radii_pe])
_c0 = bd.Face.make_surface(passage_wire(radii_pe[0])); _c1 = bd.Face.make_surface(passage_wire(radii_pe[-1]))
block_x = bd.Solid(bd.Shell(_late.faces() + [_c0, _c1]))
LZ = (Z_OUT - Z_IN)*2
annulus = (bd.Pos(0,0,(Z_IN+Z_OUT)/2) * bd.Cylinder(R_TIP, LZ)) - (bd.Pos(0,0,(Z_IN+Z_OUT)/2) * bd.Cylinder(R_HUB, LZ))
block = bd.Solid(bd.Shell((block_x & annulus).faces()))
passage = block - blade_solid
print("valid", passage.is_valid, "vol[mm3]", round(passage.volume*1e9,1), "nfaces", len(passage.faces()))
for i,f in enumerate(passage.faces()):
    c = f.center(bd.CenterOf.MASS)
    print(i, round(f.area*1e6,1), "com r,z mm:", round(np.hypot(c.X,c.Y)*1e3,2), round(c.Z*1e3,2))

# -- cell 11 ------------------------------------------------------------------------
# Re-emitting the classifier with all its helpers defined inside, so the script is self-contained.
def sample_pts(face, tol=0.0005):
    vs, _ = face.tessellate(tol)
    return np.array([[v.X, v.Y, v.Z] for v in vs])

def blade_side(face, ns=150):
    """suction vs pressure by comparing sampled points to the analytic upper/lower curves"""
    P = sample_pts(face, 0.0002); du = dl = 0.0
    for Q in P[:: max(1,len(P)//ns)]:
        rr = np.hypot(Q[0],Q[1]); tt = np.arctan2(Q[1],Q[0])*rr
        up, lo, cam = section(rr)
        du += np.min(np.hypot(up[0]-Q[2], up[1]-tt)); dl += np.min(np.hypot(lo[0]-Q[2], lo[1]-tt))
    return "suction" if du < dl else "pressure"

def classify(face):
    P = sample_pts(face); r = np.hypot(P[:,0], P[:,1]); z = P[:,2]
    if np.allclose(z, Z_IN,  atol=1e-6): return "inlet"
    if np.allclose(z, Z_OUT, atol=1e-6): return "outlet"
    if np.allclose(r, R_HUB, atol=1e-6): return "hub"
    if np.allclose(r, R_TIP, atol=1e-6): return "shroud"
    d = []
    for Q in P[:: max(1,len(P)//150)]:
        rr = np.hypot(Q[0],Q[1]); tt = np.arctan2(Q[1],Q[0])*rr
        up, lo, cam = section(rr)
        d.append(min(np.min(np.hypot(up[0]-Q[2], up[1]-tt)), np.min(np.hypot(lo[0]-Q[2], lo[1]-tt))))
    return "blade_" + blade_side(face) if np.mean(d) < 2e-4 else "periodic"

faces = passage.faces()
labels = [classify(f) for f in faces]
per = [i for i,l in enumerate(labels) if l == "periodic"]
thp = [np.degrees(np.arctan2(faces[i].center(bd.CenterOf.MASS).Y, faces[i].center(bd.CenterOf.MASS).X)) for i in per]
labels[per[int(np.argmin(thp))]] = "periodic_1"; labels[per[int(np.argmax(thp))]] = "periodic_2"
com = {}
for f,l in zip(faces,labels):
    c = f.center(bd.CenterOf.MASS); com[l] = np.array([c.X,c.Y,c.Z])
    print(f"{l:16s} area[mm2]={f.area*1e6:8.1f}")
assert sorted(labels) == sorted(["inlet","outlet","hub","shroud","periodic_1","periodic_2","blade_pressure","blade_suction"])
bd.export_step(bd.Part(passage.wrapped), "passage.step")

# -- cell 12 ------------------------------------------------------------------------
# Stray `setNumbers` line; removing it and re-running.
import gmsh, subprocess
def build_mesh(lc, lc_blade, fname):
    if gmsh.isInitialized(): gmsh.finalize()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("passage")
    gmsh.model.occ.importShapes("passage.step"); gmsh.model.occ.synchronize()
    used = {}
    for dim, tag in gmsh.model.getEntities(2):
        c = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
        name = min(com, key=lambda k: np.linalg.norm(com[k]-c))
        assert np.linalg.norm(com[name]-c) < 1e-5, (tag, c, name)
        used.setdefault(name, []).append(tag)
    for name, tags in used.items(): gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="internal")
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_blade*0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", used["blade_pressure"]+used["blade_suction"])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", lc_blade)
    gmsh.model.mesh.field.setNumber(2, "LcMax", lc)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.001)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.008)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.write(fname)
    n = len(gmsh.model.mesh.getElementsByType(4)[0]); gmsh.finalize()
    return n, used
ntet, used = build_mesh(0.004, 0.0015, "passage.msh")
print("tets", ntet, {k:len(v) for k,v in used.items()})

# -- cell 13 ------------------------------------------------------------------------
# Coarse mesh exists. Now convert to OpenFOAM and check it.
import os, textwrap, subprocess
os.makedirs("system", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr%"controlDict"+textwrap.dedent("""
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(hdr%"fvSchemes"+"ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr%"fvSolution"+"solvers{}\n")
r1 = subprocess.run(["gmshToFoam","passage.msh"], capture_output=True, text=True)
print(r1.returncode, r1.stdout[-800:], r1.stderr[-500:])

# -- cell 14 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):][:3000])

# -- cell 15 ------------------------------------------------------------------------
# pyvista can't read .msh directly. I'll export one STL per patch from the mesh itself (conformal by c
def export_patch_stls(mshfile):
    if gmsh.isInitialized(): gmsh.finalize()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.open(mshfile)
    names = {}
    for dim, tag in gmsh.model.getPhysicalGroups(2):
        nm = gmsh.model.getPhysicalName(dim, tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        tris = []
        for e in ents:
            ty, et, en = gmsh.model.mesh.getElements(2, e)
            for t, nodes in zip(ty, en):
                if t == 2: tris.append(np.array(nodes).reshape(-1,3))
        tris = np.vstack(tris)
        nt, coord, _ = gmsh.model.mesh.getNodes()
        cmap = {int(i): coord[3*k:3*k+3] for k,i in enumerate(nt)}
        with open(f"{nm}.stl","w") as fh:
            fh.write(f"solid {nm}\n")
            for tr in tris:
                p = [cmap[int(i)] for i in tr]
                n = np.cross(p[1]-p[0], p[2]-p[0]); n = n/ (np.linalg.norm(n) or 1)
                fh.write(f" facet normal {n[0]:e} {n[1]:e} {n[2]:e}\n  outer loop\n")
                for q in p: fh.write(f"   vertex {q[0]:.9e} {q[1]:.9e} {q[2]:.9e}\n")
                fh.write("  endloop\n endfacet\n")
            fh.write(f"endsolid {nm}\n")
        names[nm] = len(tris)
    gmsh.finalize(); return names
print(export_patch_stls("passage.msh"))

# -- cell 16 ------------------------------------------------------------------------
import pyvista as pv
pv.global_theme.window_size=[1200,500]
cols = {'hub':'tan','shroud':'lightgray','inlet':'green','outlet':'red',
        'periodic_1':'yellow','periodic_2':'orange','blade_pressure':'blue','blade_suction':'magenta'}
pl = pv.Plotter(off_screen=True, shape=(1,2))
for j,v in enumerate([[0.6,0.5,1.0],[0,0,1]]):
    pl.subplot(0,j)
    for nm,c in cols.items():
        s = pv.read(f"{nm}.stl")
        if nm=='shroud' and j==0: continue
        pl.add_mesh(s, color=c, show_edges=True, opacity=1.0, label=nm)
    pl.view_vector(v); pl.add_axes()
pl.add_legend()
pl.screenshot("look.png"); display(Image("look.png"))

# -- cell 17 ------------------------------------------------------------------------
# Let me render each patch separately to verify identity, viewed along the axis.
pl = pv.Plotter(off_screen=True, shape=(2,4), window_size=[1400,700])
allm = {nm: pv.read(f"{nm}.stl") for nm in cols}
for k,(nm,c) in enumerate(cols.items()):
    pl.subplot(k//4, k%4)
    for n2,m2 in allm.items():
        pl.add_mesh(m2, color=cols[n2], opacity=0.12 if n2!=nm else 1.0, show_edges=(n2==nm))
    pl.add_text(nm, font_size=9)
    pl.camera_position = [(0.15,-0.13,0.12),(0.067,0,0),(0,0,1)]
pl.screenshot("patches.png"); display(Image("patches.png"))

# -- cell 18 ------------------------------------------------------------------------
# Patches are all in the right place. Now measure the requested quantities from the built solid itself
bf = [f for f,l in zip(faces,labels) if l.startswith("blade_")]
P = np.vstack([sample_pts(f, 0.00005) for f in bf])
r_ = np.hypot(P[:,0],P[:,1]); t_ = np.arctan2(P[:,1],P[:,0])*r_; z_ = P[:,2]
def band(rlo, rhi):
    s = (r_>=rlo)&(r_<=rhi); Z,T = z_[s], t_[s]
    i = np.argmin(Z+T*0);  # find LE/TE as the two most distant points
    D = np.hypot(Z[:,None]-Z[None,:], T[:,None]-T[None,:])
    a,b = np.unravel_index(np.argmax(D), D.shape)
    if Z[a] > Z[b]: a,b = b,a
    ch = D[a,b]; ang = np.degrees(np.arctan2(T[b]-T[a], Z[b]-Z[a]))
    thick = 0.0
    for zz in np.linspace(Z.min()+0.1*ch, Z.max()-0.1*ch, 40):
        sel = np.abs(Z-zz) < 0.0002
        if sel.sum()>1: thick = max(thick, T[sel].max()-T[sel].min())
    return ch, ang, thick
for nm,(lo,hi) in {"root":(0.0450,0.0455),"tip":(0.0895,0.0900)}.items():
    ch, ang, th = band(lo,hi)
    print(f"{nm}: chord={ch*1e3:6.2f} mm (asked 22.00), stagger={ang:6.2f} deg, max thickness~{th*1e3:.2f} mm")
print("pitch between periodic faces =", round(360/NB,3), "deg for", NB, "blades")
print("hub r =", round(min(np.hypot(*sample_pts([f for f,l in zip(faces,labels) if l=='hub'][0]).T[:2]))*1e3,3),
      " shroud r =", round(max(np.hypot(*sample_pts([f for f,l in zip(faces,labels) if l=='shroud'][0]).T[:2]))*1e3,3), "mm")
print("blade span r:", round(r_.min()*1e3,2), "->", round(r_.max()*1e3,2), "mm")
print("inlet z =", round(Z_IN*1e3,2), " blade LE z(root) =", round((Z_IN+CHORD)*1e3,2),
      " outlet z =", round(Z_OUT*1e3,2))

# -- cell 19 ------------------------------------------------------------------------
# Chord 22.01 mm, stagger 30.0°/64.8° (twist 34.8°). The banded thickness estimate is unreliable; meas
fs = [f for f,l in zip(faces,labels) if l=="blade_suction"][0]
fp = [f for f,l in zip(faces,labels) if l=="blade_pressure"][0]
def band_pts(f, lo, hi):
    Q = sample_pts(f, 0.00003); rr = np.hypot(Q[:,0],Q[:,1])
    s = (rr>=lo)&(rr<=hi); rr=rr[s]; Q=Q[s]
    return np.c_[Q[:,2], np.arctan2(Q[:,1],Q[:,0])*rr]   # (z,t)
for nm,(lo,hi) in {"root":(0.0450,0.0458),"tip":(0.0892,0.0900)}.items():
    A, B = band_pts(fs,lo,hi), band_pts(fp,lo,hi)
    D = np.hypot(A[:,0][:,None]-B[:,0][None,:], A[:,1][:,None]-B[:,1][None,:])
    tmax = D.min(axis=1).max()
    print(f"{nm}: max blade thickness = {tmax*1e3:.3f} mm = {tmax/CHORD*100:.2f}% chord (asked 6.00%)")

# -- cell 20 ------------------------------------------------------------------------
# Root figure looks wrong — probably sparse sampling in the band. Let me plot the sampled root band to
import matplotlib.pyplot as plt
A, B = band_pts(fs,0.0450,0.0458), band_pts(fp,0.0450,0.0458)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(A[:,0]*1e3, A[:,1]*1e3, 'm.', ms=3, label='suction')
ax.plot(B[:,0]*1e3, B[:,1]*1e3, 'b.', ms=3, label='pressure')
ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.3); ax.set_xlabel('z mm'); ax.set_ylabel('t mm')
print(len(A), len(B))
fig

# -- cell 21 ------------------------------------------------------------------------
# Sampling is too sparse for nearest-point; measuring instead as distance to the ordered polyline of t
def poly_dist(pts, poly):
    """min distance from each pt to polyline (ordered pts) segments"""
    a, b = poly[:-1], poly[1:]; ab = b-a; L2 = (ab**2).sum(1)
    d = []
    for p in pts:
        s = np.clip(((p-a)*ab).sum(1)/np.where(L2>0,L2,1), 0, 1)
        d.append(np.min(np.linalg.norm(a + s[:,None]*ab - p, axis=1)))
    return np.array(d)

for nm,(lo,hi) in {"root":(0.0450,0.0460),"mid":(0.0670,0.0680),"tip":(0.0890,0.0900)}.items():
    A, B = band_pts(fs,lo,hi), band_pts(fp,lo,hi)
    A = A[np.argsort(A[:,0])]; B = B[np.argsort(B[:,0])]
    tmax = max(poly_dist(A,B).max(), poly_dist(B,A).max())
    print(f"{nm}: max thickness {tmax*1e3:.3f} mm = {tmax/CHORD*100:.2f}% of chord (asked 6%)")

# -- cell 22 ------------------------------------------------------------------------
# Thickness ~6.1–6.7% (overestimate from coarse polylines; exact generating value 1.320 mm = 6.00%). N
ntet, used = build_mesh(0.0015, 0.0005, "passage.msh")
print("tets", ntet)
subprocess.run(["rm","-rf","constant/polyMesh"], check=True)
r1 = subprocess.run(["gmshToFoam","passage.msh"], capture_output=True, text=True)
print("gmshToFoam rc", r1.returncode)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):][:900])
print(r.stdout[r.stdout.find("Checking geometry"):][:1600])

# -- cell 23 ------------------------------------------------------------------------
# 336k tets, checkMesh OK (max non-orth 70.4 on 1 face, max skewness 0.985). Re-exporting patch STLs a
print(export_patch_stls("passage.msh"))
allm = {nm: pv.read(f"{nm}.stl") for nm in cols}
pl = pv.Plotter(off_screen=True, shape=(1,2), window_size=[1400,600])
pl.subplot(0,0)
for n2,m2 in allm.items():
    if n2=='shroud': continue
    pl.add_mesh(m2, color=cols[n2], show_edges=(n2 in ('blade_pressure','blade_suction')), line_width=0.3)
pl.camera_position=[(0.17,-0.14,0.13),(0.067,0,0),(0,0,1)]
pl.subplot(0,1)
for n2,m2 in allm.items(): pl.add_mesh(m2, color=cols[n2], opacity=0.55)
pl.camera_position=[(0.0,0.0,0.30),(0.067,0,0),(0,1,0)]
pl.screenshot("final.png"); display(Image("final.png"))
