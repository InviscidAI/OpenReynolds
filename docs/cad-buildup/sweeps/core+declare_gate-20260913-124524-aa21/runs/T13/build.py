"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, os
mm = 1e-3
OD, DEPTH, PITCH, TURNS, GAP, ALPHA = 40.0, 1.0, 2.0, 6, 0.2, np.deg2rad(15.0)
Rc, Rr = OD/2, OD/2 - DEPTH
flat = (PITCH - 2*DEPTH*np.tan(ALPHA))/2          # crest flat = root flat
zB = DEPTH*np.tan(ALPHA)
# jar (neck) groove wall, azimuth 0, local z ; interior of groove is to the RIGHT of A->B->C->D
A = np.array([Rc, 0.0]); B = np.array([Rr, zB]); C = np.array([Rr, zB+flat]); D = np.array([Rc, 2*zB+flat])
def right_n(p,q):
    u = (q-p)/np.linalg.norm(q-p); return np.array([u[1], -u[0]])
def inter(p1,d1,p2,d2):
    t = np.linalg.solve(np.column_stack([d1,-d2]), p2-p1); return p1+t[0]*d1
segs = [(A,B),(B,C),(C,D)]
off = [(p+GAP*right_n(p,q), q+GAP*right_n(p,q)) for p,q in segs]
Bp = inter(off[0][0], off[0][1]-off[0][0], off[1][0], off[1][1]-off[1][0])
Cp = inter(off[1][0], off[1][1]-off[1][0], off[2][0], off[2][1]-off[2][0])
Ap = inter(off[0][0], off[0][1]-off[0][0], np.array([Rc,0.]), np.array([0.,1.]))   # clip at crest cyl
Dp = inter(off[2][0], off[2][1]-off[2][0], np.array([Rc,0.]), np.array([0.,1.]))
prof = [A,B,C,D,Dp,Cp,Bp,Ap]
for n,p in zip("A B C D D' C' B' A'".split(), prof): print(f"{n:3s} r={p[0]:.4f} z={p[1]:.4f}")
print("crest flat =", flat, " land width =", Ap[1]-A[1], " gap at root =", Bp[0]-B[0])
P = np.array(prof+[A])
plt.figure(figsize=(5,4)); plt.plot(P[:,0],P[:,1],'-o'); plt.axis('equal'); plt.xlabel('r [mm]'); plt.ylabel('z [mm]'); plt.grid(1); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# Cross-section is correct: constant 0.2 mm gap, land width 0.207 mm at r=20. Now the helical sweep in
import gmsh
NQ, NEL_Q = 4*TURNS, 10          # quarter-turn steps, layers per quarter (coarse)
NT, NF, NR = 2, 3, 3             # elems across gap, along flank, along root flat
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.model.add("helix_gap")
g = gmsh.model.geo
pts = [g.addPoint(p[0]*mm, 0.0, p[1]*mm) for p in prof]
lines = [g.addLine(pts[i], pts[(i+1)%8]) for i in range(8)]   # AB BC CD DD' D'C' C'B' B'A' A'A
loop = g.addCurveLoop(lines); base = g.addPlaneSurface([loop])
for l,n in zip(lines, [NF,NR,NF,NT,NF,NR,NF,NT]): g.mesh.setTransfiniteCurve(l, n+1)
g.mesh.setTransfiniteSurface(base, "Left", [pts[0], pts[3], pts[4], pts[7]])
g.mesh.setRecombine(2, base)
lat = {i: [] for i in range(8)}
top = base
for k in range(NQ):
    out = g.twist([(2,top)], 0,0,0, 0,0,PITCH*mm/4, 0,0,1, np.pi/2, [NEL_Q], [], True)
    top = out[0][1]
    for i in range(8): lat[i].append(out[2+i][1])
g.synchronize()
print("top surface:", top, " n lateral per line:", len(lat[0]))
print("bbox:", np.round(np.array(gmsh.model.getBoundingBox(-1,-1))/mm,3))

# -- cell 3 -------------------------------------------------------------------------
names = ["AB","BC","CD","DD'","D'C'","C'B'","B'A'","A'A"]
for i in range(8):
    bb = np.array(gmsh.model.getBoundingBox(2, lat[i][0]))/mm
    rmin = min(np.hypot(bb[0],bb[1]), np.hypot(bb[3],bb[4]))
    print(f"{names[i]:5s} surf {lat[i][0]:4d}  x[{bb[0]:7.3f},{bb[3]:7.3f}] y[{bb[1]:7.3f},{bb[4]:7.3f}] z[{bb[2]:6.3f},{bb[5]:6.3f}]")

# -- cell 4 -------------------------------------------------------------------------
def rmax(s):
    bb=np.array(gmsh.model.getBoundingBox(2,s))/mm; return max(np.hypot(bb[0],bb[1]),np.hypot(bb[3],bb[4]),np.hypot(bb[0],bb[4]),np.hypot(bb[3],bb[1]))
print("neck root chain rmax:", np.round([rmax(s) for s in lat[1]],3).max(), np.round([rmax(s) for s in lat[1]],3).min())
print("lid  root chain rmax:", np.round([rmax(s) for s in lat[5]],3).max(), np.round([rmax(s) for s in lat[5]],3).min())
print("land A'A chain rmax :", np.round([rmax(s) for s in lat[7]],3).min())

# -- cell 5 -------------------------------------------------------------------------
# Mesh generated; only my print line was malformed. Re-doing the tagging cleanly (clearing physical gr
gmsh.model.removePhysicalGroups()
def addpg(name, surfs, dim=2):
    t = gmsh.model.addPhysicalGroup(dim, surfs); gmsh.model.setPhysicalName(dim, t, name); return t
pg = {}
pg["neck_flank"]  = addpg("neck_flank",  [s for i in (0,1,2) for s in lat[i]])
pg["lid_flank"]   = addpg("lid_flank",   [s for i in (4,5,6) for s in lat[i]])
pg["crest_land"]  = addpg("crest_land",  [s for i in (3,7) for s in lat[i]])
pg["helix_inner"] = addpg("helix_inner", [base])
pg["helix_outer"] = addpg("helix_outer", [top])
addpg("gap", [v for (d,v) in gmsh.model.getEntities(3)], dim=3)
gmsh.model.mesh.generate(3)
et = gmsh.model.mesh.getElementTypes()
print({int(t): len(gmsh.model.mesh.getElementsByType(t)[0]) for t in et})   # 5=hex, 3=quad
# radius range of the nodes of each 2D group -> identity check
for n,t in pg.items():
    nt,_ = gmsh.model.mesh.getNodesForPhysicalGroup(2, t)
    c = gmsh.model.mesh.getNodes()[1].reshape(-1,3)
    idx = {tag:i for i,tag in enumerate(gmsh.model.mesh.getNodes()[0])}
    P3 = c[[idx[x] for x in nt]]
    r = np.hypot(P3[:,0],P3[:,1])/mm
    print(f"{n:12s} nodes={len(nt):5d} r=[{r.min():.3f},{r.max():.3f}] z=[{P3[:,2].min()/mm:.3f},{P3[:,2].max()/mm:.3f}]")

# -- cell 6 -------------------------------------------------------------------------
# Patch identity confirmed by radius: neck_flank r∈[19,20], lid_flank r∈[19.2,20], crest_land at r=20,
os.makedirs("constant/triSurface", exist_ok=True)
ntag, ncoord, _ = gmsh.model.mesh.getNodes()
idx = {t:i for i,t in enumerate(ntag)}; XYZ = ncoord.reshape(-1,3)
def write_stl(name, surfs):
    tris = []
    for s in surfs:
        ets, etags, enodes = gmsh.model.mesh.getElements(2, s)
        for et, en in zip(ets, enodes):
            k = {2:3, 3:4}[et]; conn = en.reshape(-1, k)
            for c in conn:
                p = XYZ[[idx[t] for t in c]]
                tris.append(p[[0,1,2]])
                if k == 4: tris.append(p[[0,2,3]])
    with open(f"constant/triSurface/{name}.stl","w") as f:
        f.write(f"solid {name}\n")
        for t in tris:
            n = np.cross(t[1]-t[0], t[2]-t[0]); n = n/max(np.linalg.norm(n),1e-30)
            f.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in t: f.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")
    return len(tris)
groups = {"neck_flank":[s for i in (0,1,2) for s in lat[i]], "lid_flank":[s for i in (4,5,6) for s in lat[i]],
          "crest_land":[s for i in (3,7) for s in lat[i]], "helix_inner":[base], "helix_outer":[top]}
for n,s in groups.items(): print(n, write_stl(n,s), "tris")

# -- cell 7 -------------------------------------------------------------------------
# The STL facet orientation must be outward. I'll derive it exactly from the hex that owns each bounda
hex_tags, hex_nodes = gmsh.model.mesh.getElementsByType(5)
H = hex_nodes.reshape(-1,8)
cent = np.array([XYZ[[idx[t] for t in h]].mean(axis=0) for h in H])
face_owner = {}
for ci,h in enumerate(H):
    for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
        face_owner[frozenset(h[list(f)])] = ci
def write_stl(name, surfs):
    tris, flipped = [], 0
    for s in surfs:
        ets, etags, enodes = gmsh.model.mesh.getElements(2, s)
        for et, en in zip(ets, enodes):
            k = {2:3, 3:4}[et]; conn = en.reshape(-1, k)
            for c in conn:
                p = XYZ[[idx[t] for t in c]]
                n = np.cross(p[1]-p[0], p[2]-p[0])
                if np.dot(n, p.mean(axis=0) - cent[face_owner[frozenset(c)]]) < 0:
                    p = p[::-1]; flipped += 1
                tris.append(p[[0,1,2]])
                if k == 4: tris.append(p[[0,2,3]])
    with open(f"constant/triSurface/{name}.stl","w") as f:
        f.write(f"solid {name}\n")
        for t in tris:
            n = np.cross(t[1]-t[0], t[2]-t[0]); n = n/max(np.linalg.norm(n),1e-30)
            f.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in t: f.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")
    return len(tris), flipped
for n,s in groups.items(): print(n, write_stl(n,s))

# -- cell 8 -------------------------------------------------------------------------
# Now check the union is closed and its enclosed volume matches the cells, then look at it.
import pyvista as pv
mesh = pv.PolyData()
for n in groups: mesh = mesh + pv.read(f"constant/triSurface/{n}.stl")
m = mesh.clean(tolerance=1e-9)
print("open edges:", m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).n_cells)
V = m.volume
print(f"STL volume  = {V*1e9:.2f} mm^3")
# analytic: cross-section area x mean helix length
Ax = 0.5*abs(sum(prof[i][0]*prof[(i+1)%8][1]-prof[(i+1)%8][0]*prof[i][1] for i in range(8)))
print(f"section area = {Ax:.4f} mm^2 ; rough sweep vol = {Ax*2*np.pi*19.6*TURNS:.2f} mm^3")
p = pv.Plotter(off_screen=True, window_size=(900,700))
for n,c in zip(groups, ["lightsteelblue","lightcoral","lightgray","green","red"]):
    p.add_mesh(pv.read(f"constant/triSurface/{n}.stl"), color=c, show_edges=(n in("helix_inner","helix_outer")))
p.camera_position="iso"; p.show(screenshot="gap.png")

# -- cell 9 -------------------------------------------------------------------------
# Closed union, volume 356 mm³ vs 361 mm³ analytic estimate, and the picture shows six separated turns
import subprocess
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.SaveAll", 0)
gmsh.write("gap.msh")
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
open("system/controlDict","w").write("""FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;\n""")
open("system/fvSchemes","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\nddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}\n")
r = subprocess.run(["gmshToFoam","gap.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
txt = open("gap.msh").read().splitlines()
print(txt[:14])
i = txt.index("$Elements"); print(txt[i:i+3]); print(txt[i+2].split()[:8]); 
import collections
c = collections.Counter(l.split()[1] for l in txt[i+2:txt.index("$EndElements")])
print("elem types:", c)

# -- cell 11 ------------------------------------------------------------------------
print(gmsh.model.getPhysicalGroups())
print([(d,t,gmsh.model.getPhysicalName(d,t)) for d,t in gmsh.model.getPhysicalGroups()])

# -- cell 12 ------------------------------------------------------------------------
# Physical *names* didn't survive `removePhysicalGroups` (elements carry tags 7–11, the name table sti
gmsh.finalize(); gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
NQ, NEL_Q = 4*TURNS, 30          # quarter-turn steps ; layers per quarter -> ~1.0 mm along helix
NT, NF, NR = 3, 3, 3             # across gap ; along each flank ; along root flat
gmsh.model.add("helix_gap"); g = gmsh.model.geo
pts = [g.addPoint(p[0]*mm, 0.0, p[1]*mm) for p in prof]
lines = [g.addLine(pts[i], pts[(i+1)%8]) for i in range(8)]   # AB BC CD DD' D'C' C'B' B'A' A'A
base = g.addPlaneSurface([g.addCurveLoop(lines)])
for l,n in zip(lines, [NF,NR,NF,NT,NF,NR,NF,NT]): g.mesh.setTransfiniteCurve(l, n+1)
g.mesh.setTransfiniteSurface(base, "Left", [pts[0], pts[3], pts[4], pts[7]]); g.mesh.setRecombine(2, base)
lat = {i: [] for i in range(8)}; top = base
for k in range(NQ):
    out = g.twist([(2,top)], 0,0,0, 0,0,PITCH*mm/4, 0,0,1, np.pi/2, [NEL_Q], [], True)
    top = out[0][1]
    for i in range(8): lat[i].append(out[2+i][1])
g.synchronize()
groups = {"neck_flank":[s for i in (0,1,2) for s in lat[i]], "lid_flank":[s for i in (4,5,6) for s in lat[i]],
          "crest_land":[s for i in (3,7) for s in lat[i]], "helix_inner":[base], "helix_outer":[top]}
for j,(n,s) in enumerate(groups.items()): gmsh.model.addPhysicalGroup(2, s, j+1, n)
gmsh.model.addPhysicalGroup(3, [v for d,v in gmsh.model.getEntities(3)], 99, "gap")
gmsh.model.mesh.generate(3)
print([(d,t,gmsh.model.getPhysicalName(d,t)) for d,t in gmsh.model.getPhysicalGroups()])
print({int(t): len(gmsh.model.mesh.getElementsByType(t)[0]) for t in gmsh.model.mesh.getElementTypes()})

# -- cell 13 ------------------------------------------------------------------------
ntag, ncoord, _ = gmsh.model.mesh.getNodes()
idx = {t:i for i,t in enumerate(ntag)}; XYZ = ncoord.reshape(-1,3)
H = gmsh.model.mesh.getElementsByType(5)[1].reshape(-1,8)
cent = np.array([XYZ[[idx[t] for t in h]].mean(axis=0) for h in H])
face_owner = {}
for ci,h in enumerate(H):
    for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
        face_owner[frozenset(h[list(f)])] = ci
for n,s in groups.items(): print(n, write_stl(n,s))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.option.setNumber("Mesh.SaveAll", 0)
gmsh.write("gap.msh")
subprocess.run(["rm","-rf","constant/polyMesh"]); 
r = subprocess.run(["gmshToFoam","gap.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:])

# -- cell 14 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read()[-1600:])

# -- cell 15 ------------------------------------------------------------------------
for p in ["neck_flank","lid_flank","crest_land"]:
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/type","-set","wall"],capture_output=True)
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/physicalType","-set","wall"],capture_output=True)
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 16 ------------------------------------------------------------------------
# Final verification against the stated numbers, on the refined geometry.
neck = pv.read("constant/triSurface/neck_flank.stl"); lid = pv.read("constant/triSurface/lid_flank.stl")
u = pv.PolyData()
for n in groups: u = u + pv.read(f"constant/triSurface/{n}.stl")
u = u.clean(tolerance=1e-9)
d = np.abs(lid.cell_centers().compute_implicit_distance(neck)["implicit_distance"])/mm
zi = pv.read("constant/triSurface/helix_inner.stl").points[:,2]/mm; zo = pv.read("constant/triSurface/helix_outer.stl").points[:,2]/mm
rn = np.hypot(neck.points[:,0],neck.points[:,1])/mm
print(f"{'neck OD':22s} asked 40      built {2*rn.max():.3f} mm")
print(f"{'thread depth':22s} asked 1       built {rn.max()-rn.min():.3f} mm")
print(f"{'gap (lid->neck)':22s} asked 0.2     built {np.median(d):.3f} mm  (range {d.min():.3f}-{d.max():.3f})")
print(f"{'pitch':22s} asked 2       built {(zo.min()-zi.min())/TURNS:.3f} mm")
print(f"{'threaded height':22s} asked 12      built {zo.min()-zi.min():.3f} mm  ({TURNS} turns)")
print(f"{'flank half-angle':22s} 15 deg (trapezoidal, 30 deg included)")
print(f"union free edges = {u.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False).n_cells}, STL volume = {u.volume*1e9:.2f} mm^3, mesh volume = 357.78 mm^3")
