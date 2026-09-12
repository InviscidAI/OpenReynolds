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

# --- thread parameters (mm; converted to metres only at export) ---
PITCH      = 2.0      # mm
DEPTH      = 1.0      # mm radial thread depth
R_CREST    = 20.0     # mm  -> 40 mm OD
R_ROOT     = R_CREST - DEPTH
FLANK_ANG  = 15.0     # deg, flank from the radial direction (trapezoidal, 30 deg included)
CREST_W    = 0.7      # mm axial crest width
GAP        = 0.2      # mm lid cut shallower on every flank
H_THREAD   = 12.0     # mm threaded neck height
run   = DEPTH*np.tan(np.radians(FLANK_ANG))          # axial run of one flank
ROOT_W = PITCH - CREST_W - 2*run

# inner (neck) contour over one pitch in (z, r), starting mid-root
z0 = ROOT_W/2
inner = np.array([[0.0,R_ROOT],[z0,R_ROOT],[z0+run,R_CREST],
                  [z0+run+CREST_W,R_CREST],[z0+2*run+CREST_W,R_ROOT],[PITCH,R_ROOT]])

def outward_normal(a,b):
    d = b-a; d = d/np.linalg.norm(d)
    return np.array([-d[1], d[0]])            # (-dr, dz): points away from the solid

def line_isect(p1,d1,p2,d2):
    A = np.array([d1,-d2]).T
    t = np.linalg.solve(A, p2-p1)
    return p1 + t[0]*d1

offs = [(inner[i]+GAP*outward_normal(inner[i],inner[i+1]),
         (inner[i+1]-inner[i])/np.linalg.norm(inner[i+1]-inner[i])) for i in range(5)]
outer = [np.array([0.0, R_ROOT+GAP])]
for i in range(4):
    outer.append(line_isect(*offs[i], *offs[i+1]))
outer.append(np.array([PITCH, R_ROOT+GAP]))
outer = np.array(outer)

print("run=%.4f ROOT_W=%.4f  crest half-width in/out: %.4f / %.4f" %
      (run, ROOT_W, CREST_W/2, (outer[3,0]-outer[2,0])/2))
for i in range(5):
    n = outward_normal(inner[i],inner[i+1])
    d = np.dot(outer[i]-inner[i], n), np.dot(outer[i+1]-inner[i+1], n)
    print("  segment %d perpendicular gap: %.6f / %.6f mm" % (i,d[0],d[1]))

fig,ax = plt.subplots(figsize=(9,3))
for q in range(2):
    ax.plot(inner[:,0]+q*PITCH, inner[:,1],'-o',c='C0',ms=3)
    ax.plot(outer[:,0]+q*PITCH, outer[:,1],'-o',c='C3',ms=3)
ax.set_aspect('equal'); ax.set_xlabel('z [mm]'); ax.set_ylabel('r [mm]')
ax.legend(['neck (inner)','lid (outer)']); fig.tight_layout(); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# Offset is exact: 0.200 mm perpendicular on all five segments, crest widened 0.35→0.5035 (convex corn
from IPython.display import display
display(fig)

# -- cell 3 -------------------------------------------------------------------------
# Profile is right. Now the mesh generator: a structured (helix × profile × thickness) hex grid, with 
TURNS = 5            # z span = (TURNS+1)*PITCH = 12 mm of threaded neck

def profile_nodes(nr2, nf, nc):
    """node counts per segment: half-root, flank, crest, flank, half-root"""
    counts = [nr2, nf, nc, nf, nr2]
    pin, pout = [], []
    for i,n in enumerate(counts):
        for k in range(n):
            f = k/n
            pin.append(inner[i] + f*(inner[i+1]-inner[i]))
            pout.append(outer[i] + f*(outer[i+1]-outer[i]))
    pin.append(inner[5]); pout.append(outer[5])
    return np.array(pin), np.array(pout)

def build_grid(nr2, nf, nc, Nk, T):
    pin, pout = profile_nodes(nr2, nf, nc)
    m  = len(pin)-1
    Nt = T*TURNS
    ID = -np.ones((Nt+1, m+1, Nk+1), dtype=int)
    n = 0
    for a in range(Nt+1):
        for b in range(m):
            for c in range(Nk+1):
                ID[a,b,c] = n; n += 1
    for a in range(Nt+1):
        for c in range(Nk+1):
            ID[a,m,c] = ID[a+T,0,c] if a+T <= Nt else -1
    for a in range(Nt+1):
        for c in range(Nk+1):
            if ID[a,m,c] < 0:
                ID[a,m,c] = n; n += 1
    pts = np.zeros((n,3))
    for a in range(Nt+1):
        t = a/T; th = 2*np.pi*t
        for b in range(m+1):
            for c in range(Nk+1):
                f = c/Nk
                zl, r = pin[b] + f*(pout[b]-pin[b])
                pts[ID[a,b,c]] = (r*np.cos(th), r*np.sin(th), PITCH*t + zl)
    return pin, pout, m, Nt, ID, pts

pin, pout, m, Nt, ID, pts = build_grid(1,2,1,2,40)   # coarse
print("profile nodes per pitch m =", m, " helix stations =", Nt+1, " nodes =", len(pts))
print("cells =", Nt*m*2)
print("z range [mm]: %.4f .. %.4f   r range: %.4f .. %.4f" %
      (pts[:,2].min(), pts[:,2].max(), np.hypot(pts[:,0],pts[:,1]).min(), np.hypot(pts[:,0],pts[:,1]).max()))

# -- cell 4 -------------------------------------------------------------------------
# Grid spans z 0–12 mm and r 19–20.2 mm as designed. Now connectivity, orientation check, and a look a
def build_cells(m, Nt, Nk, ID, T):
    hexes = []
    for a in range(Nt):
        for b in range(m):
            for c in range(Nk):
                hexes.append([ID[a,b,c],ID[a+1,b,c],ID[a+1,b+1,c],ID[a,b+1,c],
                              ID[a,b,c+1],ID[a+1,b,c+1],ID[a+1,b+1,c+1],ID[a,b+1,c+1]])
    P = {"neckFlank":[], "lidFlank":[], "innerEnd":[], "outerEnd":[]}
    for a in range(Nt):
        for b in range(m):
            P["neckFlank"].append([ID[a,b,0],ID[a+1,b,0],ID[a+1,b+1,0],ID[a,b+1,0]])
            P["lidFlank" ].append([ID[a,b,Nk],ID[a+1,b,Nk],ID[a+1,b+1,Nk],ID[a,b+1,Nk]])
    for b in range(m):
        for c in range(Nk):
            P["innerEnd"].append([ID[0,b,c],ID[0,b+1,c],ID[0,b+1,c+1],ID[0,b,c+1]])
            P["outerEnd"].append([ID[Nt,b,c],ID[Nt,b+1,c],ID[Nt,b+1,c+1],ID[Nt,b,c+1]])
    for a in range(Nt):
        for c in range(Nk):
            if a < T:            # free bottom edge of the first turn -> opens to jar interior
                P["innerEnd"].append([ID[a,0,c],ID[a+1,0,c],ID[a+1,0,c+1],ID[a,0,c+1]])
            if a >= Nt-T:        # free top edge of the last turn -> opens to the air
                P["outerEnd"].append([ID[a,m,c],ID[a+1,m,c],ID[a+1,m,c+1],ID[a,m,c+1]])
    return np.array(hexes), {k:np.array(v) for k,v in P.items()}

hexes, patches = build_cells(m, Nt, 2, ID, 40)

def hex_volumes(pts, h):
    c = pts[h]                      # (n,8,3)
    ctr = c.mean(axis=1)
    quads = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
    V = np.zeros(len(h))
    for q in quads:
        fc = c[:,q,:].mean(axis=1)
        for i in range(4):
            a,b = c[:,q[i],:]-ctr, c[:,q[(i+1)%4],:]-ctr
            V += np.einsum('ij,ij->i', np.cross(a,b), fc-ctr)/6.0
    return V

V = hex_volumes(pts, hexes)
print("cells %d  faces: %s" % (len(hexes), {k:len(v) for k,v in patches.items()}))
print("volume min/max [mm^3]: %.3e / %.3e   total %.4f mm^3" % (V.min(), V.max(), V.sum()))

# -- cell 5 -------------------------------------------------------------------------
# Volumes came out uniformly negative — just node ordering. Flip it and render the coarse shell.
def build_cells(m, Nt, Nk, ID, T):
    hexes = []
    for a in range(Nt):
        for b in range(m):
            for c in range(Nk):
                hexes.append([ID[a,b,c],ID[a,b+1,c],ID[a+1,b+1,c],ID[a+1,b,c],
                              ID[a,b,c+1],ID[a,b+1,c+1],ID[a+1,b+1,c+1],ID[a+1,b,c+1]])
    P = {"neckFlank":[], "lidFlank":[], "innerEnd":[], "outerEnd":[]}
    for a in range(Nt):
        for b in range(m):
            P["neckFlank"].append([ID[a,b,0],ID[a+1,b,0],ID[a+1,b+1,0],ID[a,b+1,0]])
            P["lidFlank" ].append([ID[a,b,Nk],ID[a,b+1,Nk],ID[a+1,b+1,Nk],ID[a+1,b,Nk]])
    for b in range(m):
        for c in range(Nk):
            P["innerEnd"].append([ID[0,b,c],ID[0,b,c+1],ID[0,b+1,c+1],ID[0,b+1,c]])
            P["outerEnd"].append([ID[Nt,b,c],ID[Nt,b+1,c],ID[Nt,b+1,c+1],ID[Nt,b,c+1]])
    for a in range(Nt):
        for c in range(Nk):
            if a < T:            # free bottom edge of the first turn -> opens to the jar interior
                P["innerEnd"].append([ID[a,0,c],ID[a+1,0,c],ID[a+1,0,c+1],ID[a,0,c+1]])
            if a >= Nt-T:        # free top edge of the last turn -> opens to the air
                P["outerEnd"].append([ID[a,m,c],ID[a,m,c+1],ID[a+1,m,c+1],ID[a+1,m,c]])
    return np.array(hexes), {k:np.array(v) for k,v in P.items()}

hexes, patches = build_cells(m, Nt, 2, ID, 40)
V = hex_volumes(pts, hexes)
print("volume min/max [mm^3]: %.3e / %.3e   total %.4f mm^3" % (V.min(), V.max(), V.sum()))
print("shell volume / (mid-area*0.2) sanity, area = %.1f mm^2" % (V.sum()/GAP))

# -- cell 6 -------------------------------------------------------------------------
# Volumes positive, total 434 mm³ ≈ 2171 mm² of flank × 0.2 mm. Now look at it, coloured by patch.
import pyvista as pv
pv.global_theme.allow_empty_mesh = True

def patch_poly(pts, quads):
    f = np.hstack([np.full((len(quads),1),4), quads]).astype(np.int64)
    return pv.PolyData(pts, faces=f)

cols = {"neckFlank":"steelblue","lidFlank":"indianred","innerEnd":"lime","outerEnd":"gold"}
pl = pv.Plotter(off_screen=True, window_size=(1000,800))
for k,c in cols.items():
    pl.add_mesh(patch_poly(pts, patches[k]), color=c, show_edges=True, line_width=1, label=k)
pl.add_legend(); pl.camera_position = [(70,-70,40),(0,0,6),(0,0,1)]
pl.screenshot("shell_coarse.png")
from IPython.display import Image
Image("shell_coarse.png")

# -- cell 7 -------------------------------------------------------------------------
# The band is there but I can't see the thread profile. Clip the volume at y=0 to see the corrugated c
cells_vtk = np.hstack([np.full((len(hexes),1),8), hexes]).astype(np.int64).ravel()
grid = pv.UnstructuredGrid(cells_vtk, np.full(len(hexes), pv.CellType.HEXAHEDRON), pts)
clip = grid.clip(normal=(0,1,0), origin=(0,0,0), crinkle=True)
pl = pv.Plotter(off_screen=True, window_size=(1100,700))
pl.add_mesh(clip, color="lightblue", show_edges=True, line_width=1)
pl.camera_position = [(0,-90,6),(0,0,6),(0,0,1)]; pl.camera.zoom(2.6)
pl.screenshot("clip_coarse.png")
Image("clip_coarse.png")

# -- cell 8 -------------------------------------------------------------------------
# Too zoomed and crinkle kept whole cells. A true slice, plotted in (x,z) so I can read the profile di
sl = grid.slice(normal=(0,1,0), origin=(0,0,6))
sp = sl.points
fig2, ax = plt.subplots(figsize=(7,7))
for c in range(sl.n_cells):
    q = sp[sl.get_cell(c).point_ids]
    ax.fill(q[:,0], q[:,2], facecolor="lightblue", edgecolor="k", lw=0.4)
ax.set_xlim(18.5,20.5); ax.set_aspect('equal'); ax.set_xlabel("x (=r) [mm]"); ax.set_ylabel("z [mm]")
ax.set_title("gap shell, plane y=0, x>0 side"); fig2.tight_layout()
display(fig2)

# -- cell 9 -------------------------------------------------------------------------
# Cross-section is exactly the 0.2 mm corrugated layer, continuous through root–flank–crest. Now write
import os, subprocess, textwrap
MM = 1e-3    # build is in mm; OpenFOAM gets metres

def write_stl(path, pts_mm, quads, name):
    P = pts_mm*MM
    with open(path,"w") as f:
        f.write("solid %s\n" % name)
        for q in quads:
            for tri in ((q[0],q[1],q[2]), (q[0],q[2],q[3])):
                a,b,c = P[list(tri)]
                n = np.cross(b-a, c-a); L = np.linalg.norm(n); n = n/L if L>0 else n
                f.write("facet normal %.6e %.6e %.6e\n outer loop\n" % tuple(n))
                for v in (a,b,c): f.write("  vertex %.9e %.9e %.9e\n" % tuple(v))
                f.write(" endloop\nendfacet\n")
        f.write("endsolid %s\n" % name)

def write_msh(path, pts_mm, hexes, patches):
    P = pts_mm*MM
    names = list(patches.keys())
    with open(path,"w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n%d\n" % (len(names)+1))
        for i,n in enumerate(names): f.write('2 %d "%s"\n' % (i+1, n))
        f.write('3 %d "internal"\n$EndPhysicalNames\n' % (len(names)+1))
        f.write("$Nodes\n%d\n" % len(P))
        for i,p in enumerate(P): f.write("%d %.10e %.10e %.10e\n" % (i+1,p[0],p[1],p[2]))
        f.write("$EndNodes\n$Elements\n%d\n" % (sum(len(v) for v in patches.values())+len(hexes)))
        e = 0
        for i,n in enumerate(names):
            for q in patches[n]:
                e += 1; f.write("%d 3 2 %d %d %s\n" % (e, i+1, i+1, " ".join(str(x+1) for x in q)))
        for h in hexes:
            e += 1; f.write("%d 5 2 %d %d %s\n" % (e, len(names)+1, len(names)+1, " ".join(str(x+1) for x in h)))
        f.write("$EndElements\n")

os.makedirs("constant/triSurface", exist_ok=True); os.makedirs("system", exist_ok=True)
head = "FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}\n"
open("system/controlDict","w").write(head + textwrap.dedent("""
    application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
    deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
    writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
    """))
for k,v in patches.items(): write_stl("constant/triSurface/%s.stl" % k, pts, v, k)
write_msh("gap.msh", pts, hexes, patches)
print(sorted(os.listdir("constant/triSurface")), os.path.getsize("gap.msh"))

# -- cell 10 ------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam","gap.msh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:]); print(r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
# Two red flags: hexes reported inverted, and 2988 faces left undefined. Let me audit my own connectiv
from collections import Counter
qf = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
cnt = Counter()
for h in hexes:
    for q in qf: cnt[tuple(sorted(h[list(q)]))] += 1
bnd = {k for k,v in cnt.items() if v==1}
print("total distinct faces", len(cnt), " boundary", len(bnd), " >2 shared:", sum(1 for v in cnt.values() if v>2))
mine = set()
for k,v in patches.items():
    for q in v: mine.add(tuple(sorted(q)))
print("my patch faces", len(mine), " matching mesh boundary:", len(mine & bnd), " missing:", len(bnd-mine))
# sign convention test on a unit cube in standard (gmsh/VTK) order
cube = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
print("unit-cube signed volume by my routine:", hex_volumes(cube, np.array([[0,1,2,3,4,5,6,7]]))[0])

# -- cell 12 ------------------------------------------------------------------------
# Connectivity is exactly right (merge works: no face shared by >2 cells, boundary = my 2988 patch fac
def build_cells(m, Nt, Nk, ID, T):
    hexes = []
    for a in range(Nt):
        for b in range(m):
            for c in range(Nk):
                hexes.append([ID[a,b,c],ID[a+1,b,c],ID[a+1,b+1,c],ID[a,b+1,c],
                              ID[a,b,c+1],ID[a+1,b,c+1],ID[a+1,b+1,c+1],ID[a,b+1,c+1]])
    P = {"neckFlank":[], "lidFlank":[], "innerEnd":[], "outerEnd":[]}
    for a in range(Nt):
        for b in range(m):
            P["neckFlank"].append([ID[a,b,0],ID[a+1,b,0],ID[a+1,b+1,0],ID[a,b+1,0]])
            P["lidFlank" ].append([ID[a,b,Nk],ID[a,b+1,Nk],ID[a+1,b+1,Nk],ID[a+1,b,Nk]])
    for b in range(m):
        for c in range(Nk):
            P["innerEnd"].append([ID[0,b,c],ID[0,b,c+1],ID[0,b+1,c+1],ID[0,b+1,c]])
            P["outerEnd"].append([ID[Nt,b,c],ID[Nt,b+1,c],ID[Nt,b+1,c+1],ID[Nt,b,c+1]])
    for a in range(Nt):
        for c in range(Nk):
            if a < T:            # free bottom edge of the first turn -> opens to the jar interior
                P["innerEnd"].append([ID[a,0,c],ID[a+1,0,c],ID[a+1,0,c+1],ID[a,0,c+1]])
            if a >= Nt-T:        # free top edge of the last turn -> opens to the air
                P["outerEnd"].append([ID[a,m,c],ID[a,m,c+1],ID[a+1,m,c+1],ID[a+1,m,c]])
    return np.array(hexes), {k:np.array(v) for k,v in P.items()}

hexes, patches = build_cells(m, Nt, 2, ID, 40)
print("signed volume (OpenFOAM convention) min/max:", -hex_volumes(pts,hexes).max(), -hex_volumes(pts,hexes).min())
for k,v in patches.items(): write_stl("constant/triSurface/%s.stl" % k, pts, v, k)
write_msh("gap.msh", pts, hexes, patches)
subprocess.run(["rm","-rf","constant/polyMesh"])
r = subprocess.run(["gmshToFoam","gap.msh"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "undefined" in l or "Inverting" in l or "gets name" in l][:12])

# -- cell 13 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read()[-1200:])
print([l for l in r.stdout.splitlines() if "Patch" in l or "faces" in l][:20])

# -- cell 14 ------------------------------------------------------------------------
# All four patches got their faces (1400/1400/94/94) and the total face count matches my own 9894 — th
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 15 ------------------------------------------------------------------------
print(len(r.stdout)); print(r.stdout[3000:]); print("STDERR", r.stderr[-500:])

# -- cell 16 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSchemes;}\n"
    "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
    "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
    "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSolution;}\n"
    "solvers{} relaxationFactors{}\n")
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):])

# -- cell 17 ------------------------------------------------------------------------
# Coarse mesh passes (volume 4.3427e-7 m³, bbox 20.2 mm radius × 12 mm). Now refine: ~0.2 mm along the
NR2, NF, NC, NK, T = 3, 4, 3, 4, 250          # final resolution
pin, pout, m, Nt, ID, pts = build_grid(NR2, NF, NC, NK, T)
hexes, patches = build_cells(m, Nt, NK, ID, T)
V = -hex_volumes(pts, hexes)
print("m=%d Nt=%d cells=%d nodes=%d" % (m, Nt, len(hexes), len(pts)))
print("vol min %.3e max %.3e total %.4f mm^3 (coarse gave 434.2660)" % (V.min(), V.max(), V.sum()))

# -- cell 18 ------------------------------------------------------------------------
# Profile index only spans one pitch; successive crests at θ=0 are at helix stations q·T. Fixing the m
rr = np.hypot(pts[:,0], pts[:,1])
print("OD at neck crest      : %.4f mm   (asked 40)"     % (2*rr[ID[:,:,0]].max()))
print("thread depth          : %.4f mm   (asked 1)"      % (rr[ID[:,:,0]].max()-rr[ID[:,:,0]].min()))
print("threaded neck height  : %.4f mm   (asked 12)"     % (pts[:,2].max()-pts[:,2].min()))
bc = NR2+NF+NC//2
zc = [pts[ID[q*T, bc, 0], 2] for q in range(6)]
print("crest z at theta=0    :", np.round(zc,4), "-> pitch %.4f mm (asked 2)" % np.mean(np.diff(zc)))
print("turns of helix        : %.3f   (single start)" % (Nt/T))
d = []
for a in range(0, Nt+1, 97):
    A = pts[ID[a,:,NK]]; B = pts[ID[a,:,0]]
    d.append(np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1)).min(axis=1))
print("gap neck->lid min/max : %.4f / %.4f mm   (asked 0.2)" % (np.min(d), np.max(d)))
fl = np.degrees(np.arctan2(inner[2,0]-inner[1,0], inner[2,1]-inner[1,1]))
print("flank angle from radial: %.3f deg (%.1f deg included, trapezoidal)" % (fl, 2*fl))
mid = pts[ID[:,bc,NK//2]]
print("channel length along crest helix: %.1f mm" % np.linalg.norm(np.diff(mid,axis=0),axis=1).sum())
print("lid crest OD          : %.4f mm" % (2*rr[ID[:,:,NK]].max()))

# -- cell 19 ------------------------------------------------------------------------
# All measured numbers match. Export the four STLs and the final mesh.
import time
t0=time.time()
for k,v in patches.items(): write_stl("constant/triSurface/%s.stl" % k, pts, v, k)
write_msh("gap.msh", pts, hexes, patches)
print("written in %.1fs" % (time.time()-t0), {k:len(v) for k,v in patches.items()})
subprocess.run(["rm","-rf","constant/polyMesh"])
r = subprocess.run(["gmshToFoam","gap.msh"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "gets name" in l or "Inverting" in l][:8])
print(subprocess.run(["ls","-la","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 20 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):])

# -- cell 21 ------------------------------------------------------------------------
# Mesh OK: 85 000 hexes, all checks pass. A final look at the four named patches.
pl = pv.Plotter(off_screen=True, window_size=(1200,600), shape=(1,2))
for j,cam in enumerate([[(80,-80,45),(0,0,6),(0,0,1)], [(0,-40,2),(0,-19,1),(0,0,1)]]):
    pl.subplot(0,j)
    for k,c in cols.items():
        pl.add_mesh(patch_poly(pts, patches[k]).extract_surface(), color=c,
                    show_edges=(j==1), line_width=0.4, label=k)
    pl.camera_position = cam
    if j==1: pl.camera.zoom(3.0)
pl.add_legend()
pl.screenshot("final.png"); Image("final.png")
