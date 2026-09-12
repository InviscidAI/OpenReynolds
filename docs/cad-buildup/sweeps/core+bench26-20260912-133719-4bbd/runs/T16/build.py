"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

L = 0.900                      # duct length, m
Z = np.array([0.0, L/2, L])
R_OUT = np.array([0.450, 0.430, 0.410])   # outer cowl inner surface
R_IN  = np.array([0.280, 0.310, 0.250])   # core cowl outer surface

c_out = np.polyfit(Z, R_OUT, 2)   # quadratic through the 3 radii
c_in  = np.polyfit(Z, R_IN , 2)
r_out = lambda z: np.polyval(c_out, z)
r_in  = lambda z: np.polyval(c_in , z)

print("outer coeffs", np.round(c_out,6), "->", np.round(r_out(Z),6), "target", R_OUT)
print("inner coeffs", np.round(c_in ,6), "->", np.round(r_in (Z),6), "target", R_IN)
zz = np.linspace(0,L,901)
print("gap at z=0,0.45,0.9:", np.round(r_out(Z)-r_in(Z),4), "min gap:", (r_out(zz)-r_in(zz)).min())

# -- cell 2 -------------------------------------------------------------------------
# Solid built; only the numpy call was wrong. Re-running the whole cell so it replays cleanly.
from build123d import *
NS = 41
zs = np.linspace(0, L, NS)
pts_out = [(float(r_out(z)), 0.0, float(z)) for z in zs]
pts_in  = [(float(r_in(z)),  0.0, float(z)) for z in zs]

e_out = Spline(*pts_out)                      # outer cowl generatrix
e_in  = Spline(*pts_in)                       # core cowl generatrix
e_in_cap  = Line(pts_in[0],  pts_out[0])      # fan-exit inlet generatrix
e_out_cap = Line(pts_in[-1], pts_out[-1])     # nozzle outlet generatrix
prof = Face(Wire([e_in_cap, e_out, e_out_cap, e_in]))
duct = revolve(prof, Axis.Z, 360)
V_analytic = np.pi*np.trapezoid(r_out(zz)**2 - r_in(zz)**2, zz)
print("solid volume", duct.volume, " analytic", V_analytic, " rel err", abs(duct.volume-V_analytic)/V_analytic)
print("faces:", len(duct.faces()), "bbox", duct.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol

def rev_face(edge):                       # surface of revolution from one generatrix
    return Face(BRepPrimAPI_MakeRevol(edge.wrapped, Axis.Z.wrapped).Shape())

patches = {"outerCowl": rev_face(e_out), "coreCowl": rev_face(e_in),
           "inlet": rev_face(e_in_cap), "outlet": rev_face(e_out_cap)}

def lateral(c):
    dr = np.polyval(np.polyder(c), zz)
    return float(np.trapezoid(2*np.pi*np.polyval(c, zz)*np.sqrt(1+dr**2), zz))
ref = {"outerCowl": lateral(c_out), "coreCowl": lateral(c_in),
       "inlet": np.pi*(R_OUT[0]**2-R_IN[0]**2), "outlet": np.pi*(R_OUT[2]**2-R_IN[2]**2)}
for k, s in patches.items():
    print(f"{k:10s} area={s.area:.6f}  analytic={ref[k]:.6f}")

# -- cell 4 -------------------------------------------------------------------------
# All four patch surfaces match analytic areas to 1e-6. Export one STL per patch.
for k, s in patches.items():
    export_stl(s, f"constant/triSurface/{k}.stl", tolerance=2e-4, angular_tolerance=0.1)
print(subprocess.run(["ls","-l","constant/triSurface"],capture_output=True,text=True).stdout)
for k in patches:
    print(subprocess.run(["surfaceCheck",f"constant/triSurface/{k}.stl"],capture_output=True,text=True).stdout.splitlines()[-6:])

# -- cell 5 -------------------------------------------------------------------------
# Now the mesh: a fully structured O-grid annulus in blockMesh — 4 sectors of 90°, circumferential arc
import textwrap
os.makedirs("system", exist_ok=True)
open("system/controlDict","w").write(textwrap.dedent("""\
    FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
    application blockMesh; startFrom startTime; startTime 0; stopAt endTime;
    endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
    """))

NR, NT, NZ = 4, 8, 12            # per-sector cell counts (coarse pass)
NPOLY = 21                       # points defining each axial polyLine
ang = np.array([0.0, 90.0, 180.0, 270.0])

def P(a_deg, r, z): return (r*np.cos(np.radians(a_deg)), r*np.sin(np.radians(a_deg)), z)
verts, vid = [], {}
for i, a in enumerate(ang):
    for ring, rf in ((0, r_in), (1, r_out)):
        for k, z in ((0, 0.0), (1, L)):
            vid[(i, ring, k)] = len(verts); verts.append(P(a, float(rf(z)), z))

blocks, edges, fo, fc, fi, fu = [], [], [], [], [], []
for i in range(4):
    j = (i+1) % 4
    am = 0.5*(ang[i] + ang[i]+90.0)
    v = lambda ii, ring, k: vid[(ii, ring, k)]
    blocks.append(f"hex ({v(i,0,0)} {v(i,1,0)} {v(j,1,0)} {v(j,0,0)} "
                  f"{v(i,0,1)} {v(i,1,1)} {v(j,1,1)} {v(j,0,1)}) ({NR} {NT} {NZ}) simpleGrading (1 1 1)")
    for ring, rf in ((0, r_in), (1, r_out)):
        for k, z in ((0, 0.0), (1, L)):
            p = P(am, float(rf(z)), z)
            edges.append(f"arc {v(i,ring,k)} {v(j,ring,k)} ({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})")
        pl = " ".join("(%.9g %.9g %.9g)" % P(ang[i], float(rf(z)), z) for z in np.linspace(0, L, NPOLY)[1:-1])
        edges.append(f"polyLine {v(i,ring,0)} {v(i,ring,1)} ({pl})")
    fo.append(f"({v(i,1,0)} {v(j,1,0)} {v(j,1,1)} {v(i,1,1)})")
    fc.append(f"({v(i,0,0)} {v(i,0,1)} {v(j,0,1)} {v(j,0,0)})")
    fi.append(f"({v(i,0,0)} {v(j,0,0)} {v(j,1,0)} {v(i,1,0)})")
    fu.append(f"({v(i,0,1)} {v(i,1,1)} {v(j,1,1)} {v(j,0,1)})")

def blk(name, typ, faces): return f"{name} {{ type {typ}; faces ( {' '.join(faces)} ); }}"
dct = ("FoamFile{version 2.0;format ascii;class dictionary;object blockMeshDict;}\n"
       "scale 1;\nvertices (\n" + "\n".join("(%.9g %.9g %.9g)" % p for p in verts) + "\n);\n"
       "blocks (\n" + "\n".join(blocks) + "\n);\nedges (\n" + "\n".join(edges) + "\n);\n"
       "boundary (\n" + "\n".join([blk("outerCowl","wall",fo), blk("coreCowl","wall",fc),
                                   blk("inlet","patch",fi), blk("outlet","patch",fu)]) + "\n);\n")
open("system/blockMeshDict","w").write(dct)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:]); print(r.stderr[-800:])

# -- cell 6 -------------------------------------------------------------------------
# Coarse mesh exists. Check quality and measure mesh patch areas/volume against the analytic values.
r = subprocess.run(["checkMesh","-allTopology"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 7 -------------------------------------------------------------------------
print(r.stdout[r.stdout.find("Mesh stats"):][:3000])

# -- cell 8 -------------------------------------------------------------------------
r2 = subprocess.run(["checkMesh"], capture_output=True, text=True)
i = r2.stdout.find("Mesh stats")
print(r2.stdout[i:] if i>0 else r2.stdout[-2500:]); print("STDERR", r2.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
# checkMesh wants fvSchemes/fvSolution; write minimal ones.
open("system/fvSchemes","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\n"
    "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
    "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
    "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}\n")
r2 = subprocess.run(["checkMesh"], capture_output=True, text=True)
i = r2.stdout.find("Mesh stats"); print(r2.stdout[i:])

# -- cell 10 ------------------------------------------------------------------------
# Volume is 2.9% above analytic: with 90° sectors, blockMesh's Cartesian transfinite blend under-shoot
def write_bmd(nsec=8, nax=6, nr=4, nt=6, nz=3, npoly=9):
    A = np.arange(nsec)*360.0/nsec
    ZL = np.linspace(0, L, nax+1)
    def P(a, r, z): return (r*np.cos(np.radians(a)), r*np.sin(np.radians(a)), z)
    verts, vid = [], {}
    for i, a in enumerate(A):
        for ring, rf in ((0, r_in), (1, r_out)):
            for k, z in enumerate(ZL):
                vid[(i, ring, k)] = len(verts); verts.append(P(a, float(rf(z)), z))
    blocks, edges, fo, fc, fi, fu = [], [], [], [], [], []
    v = lambda ii, ring, k: vid[(ii, ring, k)]
    for i in range(nsec):
        j = (i+1) % nsec
        am = A[i] + 180.0/nsec
        for k in range(nax):
            blocks.append(f"hex ({v(i,0,k)} {v(i,1,k)} {v(j,1,k)} {v(j,0,k)} "
                          f"{v(i,0,k+1)} {v(i,1,k+1)} {v(j,1,k+1)} {v(j,0,k+1)}) "
                          f"({nr} {nt} {nz}) simpleGrading (1 1 1)")
            for ring, rf in ((0, r_in), (1, r_out)):
                zs_ = np.linspace(ZL[k], ZL[k+1], npoly)
                pl = " ".join("(%.9g %.9g %.9g)" % P(A[i], float(rf(z)), z) for z in zs_[1:-1])
                edges.append(f"polyLine {v(i,ring,k)} {v(i,ring,k+1)} ({pl})")
            fo.append(f"({v(i,1,k)} {v(j,1,k)} {v(j,1,k+1)} {v(i,1,k+1)})")
            fc.append(f"({v(i,0,k)} {v(i,0,k+1)} {v(j,0,k+1)} {v(j,0,k)})")
        for k, z in enumerate(ZL):
            for ring, rf in ((0, r_in), (1, r_out)):
                p = P(am, float(rf(z)), z)
                edges.append(f"arc {v(i,ring,k)} {v(j,ring,k)} ({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})")
        fi.append(f"({v(i,0,0)} {v(j,0,0)} {v(j,1,0)} {v(i,1,0)})")
        fu.append(f"({v(i,0,nax)} {v(i,1,nax)} {v(j,1,nax)} {v(j,0,nax)})")
    blk = lambda n,t,f: f"{n} {{ type {t}; faces ( {' '.join(f)} ); }}"
    open("system/blockMeshDict","w").write(
        "FoamFile{version 2.0;format ascii;class dictionary;object blockMeshDict;}\nscale 1;\n"
        "vertices (\n" + "\n".join("(%.9g %.9g %.9g)" % p for p in verts) + "\n);\n"
        "blocks (\n" + "\n".join(blocks) + "\n);\nedges (\n" + "\n".join(edges) + "\n);\n"
        "boundary (\n" + "\n".join([blk("outerCowl","wall",fo), blk("coreCowl","wall",fc),
                                    blk("inlet","patch",fi), blk("outlet","patch",fu)]) + "\n);\n")

write_bmd()
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-400:])

# -- cell 11 ------------------------------------------------------------------------
out = subprocess.run(["checkMesh"], capture_output=True, text=True).stdout
print(out[out.find("Checking geometry"):])
print("analytic volume", V_analytic)

# -- cell 12 ------------------------------------------------------------------------
# Volume now 0.2758 vs 0.2764 analytic (−0.2%, pure circumferential faceting). Look at it, and measure
import pyvista as pv
pv.start_xvfb() if False else None
pv.global_theme.background = "white"
mesh = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
grid = rd.read()
internal = grid["internalMesh"]
bnd = grid["boundary"]
print(internal.n_cells, list(bnd.keys()))
p = pv.Plotter(off_screen=True, window_size=(1000,450), shape=(1,2))
p.subplot(0,0)
cols = {"outerCowl":"steelblue","coreCowl":"indianred","inlet":"green","outlet":"orange"}
for k,c in cols.items(): p.add_mesh(bnd[k], color=c, show_edges=True, line_width=0.5)
p.camera_position = "iso"
p.subplot(0,1)
p.add_mesh(internal.clip(normal="y", origin=(0,0,0)), show_edges=True, color="lightgray")
p.camera_position = "xz"
p.screenshot("view0.png")

# -- cell 13 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view0.png"))

# -- cell 14 ------------------------------------------------------------------------
# The 3D shape is right. Now the meridional truth check: slice at y=0 and overlay mesh boundary radii 
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sl = internal.slice(normal="y", origin=(0,0,0))
pts = sl.points; keep = pts[:,0] > 0
fig, ax = plt.subplots(figsize=(9,3.2))
ax.plot(pts[keep,2], pts[keep,0], '.', ms=2, color='0.6', label='mesh slice y=0')
ax.plot(zz, r_out(zz), 'b-', lw=1.5, label='outer law')
ax.plot(zz, r_in(zz),  'r-', lw=1.5, label='inner law')
ax.plot(Z, R_OUT, 'bo'); ax.plot(Z, R_IN, 'ro')
ax.set_xlabel("z [m]"); ax.set_ylabel("r [m]"); ax.legend(); ax.grid(alpha=.3)
# measure from patch points at the four rays nearest the meridian
for name, law in (("outerCowl", r_out), ("coreCowl", r_in)):
    P_ = bnd[name].points; r_ = np.hypot(P_[:,0], P_[:,1])
    err = r_ - law(P_[:,2])
    print(f"{name}: max |r_mesh - r_law| over patch nodes = {np.abs(err).max():.2e} m")
for name, zq, tgt in (("outerCowl",Z,R_OUT), ("coreCowl",Z,R_IN)):
    P_ = bnd[name].points; r_ = np.hypot(P_[:,0], P_[:,1])
    got = [r_[np.abs(P_[:,2]-z)<1e-6].mean() for z in zq]
    print(f"{name} mesh radii at z=0,0.45,0.9: {np.round(got,5)} vs requested {tgt}")
plt.tight_layout(); plt.savefig("meridian.png", dpi=110); plt.close()
display(Image("meridian.png"))

# -- cell 15 ------------------------------------------------------------------------
# Walls match the laws (outer 3e-8 m, inner 1e-4 m max node deviation) and all six requested radii are
write_bmd(nsec=16, nax=6, nr=16, nt=8, nz=8, npoly=13)   # 128 circumferential, 48 axial, 16 radial
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-350:])

# -- cell 16 ------------------------------------------------------------------------
out = subprocess.run(["checkMesh"], capture_output=True, text=True).stdout
print(out[out.find("Checking geometry"):])
print("analytic volume", V_analytic, " rel err", )

# -- cell 17 ------------------------------------------------------------------------
# Volume now within 0.024% of analytic. Final verification on the refined mesh: patch areas, wall radi
rd2 = pv.OpenFOAMReader("case.foam"); g2 = rd2.read(); b2 = g2["boundary"]; im = g2["internalMesh"]
for k in ["outerCowl","coreCowl","inlet","outlet"]:
    print(f"{k:10s} mesh area={b2[k].area:.6f}  CAD/analytic={ref[k]:.6f}")
for name, law, tgt in (("outerCowl", r_out, R_OUT), ("coreCowl", r_in, R_IN)):
    P_ = b2[name].points; r_ = np.hypot(P_[:,0], P_[:,1])
    print(name, "max node dev from law %.2e m" % np.abs(r_-law(P_[:,2])).max(),
          "radii@z=0,.45,.9", np.round([r_[np.abs(P_[:,2]-z)<1e-6].mean() for z in Z],5), "req", tgt)
p = pv.Plotter(off_screen=True, window_size=(1000,460))
p.add_mesh(im.clip(normal="y", origin=(0,0,0)).clip(normal="x", origin=(0,0,0)), show_edges=True, color="lightgray", line_width=0.3)
for k,c in (("outerCowl","steelblue"),("coreCowl","indianred"),("inlet","green"),("outlet","orange")):
    p.add_mesh(b2[k].clip(normal="y", origin=(0,0,0)), color=c, opacity=0.55)
p.camera_position = [(2.0,-2.2,2.0),(0,0,0.45),(0,0,1)]
p.screenshot("final.png"); display(Image("final.png"))
