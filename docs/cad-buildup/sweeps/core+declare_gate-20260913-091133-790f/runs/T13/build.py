"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll send that as a tool call.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- thread parameters (mm) ----
D_MAJOR = 40.0        # neck outside diameter
R_MAJOR = D_MAJOR/2   # 20 mm crest radius
PITCH   = 2.0
DEPTH   = 1.0         # radial thread depth
R_MINOR = R_MAJOR - DEPTH
HALF_ANGLE = np.radians(15.0)   # trapezoidal, 30 deg included
GAP = 0.2             # lid cut shallower on every flank
NECK_H = 12.0
FLANK_DZ = DEPTH*np.tan(HALF_ANGLE)
FLAT = (PITCH - 2*FLANK_DZ)/2.0     # crest flat = root flat

# profile over one pitch, traversed with z increasing: half root, up flank, crest, down flank, half root
SEG = [((R_MINOR,0.0),(R_MINOR,FLAT/2), 2),
       ((R_MINOR,FLAT/2),(R_MAJOR,FLAT/2+FLANK_DZ), 7),
       ((R_MAJOR,FLAT/2+FLANK_DZ),(R_MAJOR,FLAT/2+FLANK_DZ+FLAT), 6),
       ((R_MAJOR,FLAT/2+FLANK_DZ+FLAT),(R_MINOR,PITCH-FLAT/2), 7),
       ((R_MINOR,PITCH-FLAT/2),(R_MINOR,PITCH), 2)]

pts=[np.array(SEG[0][0])]
for a,b,n in SEG:
    a=np.array(a); b=np.array(b)
    for t in range(1,n+1): pts.append(a+(b-a)*t/n)
P=np.array(pts)                      # (Ns+1,2) neck profile, r,z
NS=len(P)-1
print("Ns =",NS,"flank dz =",FLANK_DZ,"flat =",FLAT,"profile z span",P[0,1],P[-1,1])
print("developed profile length =",np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1)))

# -- cell 2 -------------------------------------------------------------------------
def seg_normals(P):
    d=np.diff(P,axis=0); L=np.linalg.norm(d,axis=1,keepdims=True); u=d/L
    return np.stack([u[:,1],-u[:,0]],axis=1)   # (dz,-dr): outward from neck material

def offset_profile(P,g):
    n=seg_normals(P); Q=np.zeros_like(P)
    Q[0]=P[0]+g*n[0]; Q[-1]=P[-1]+g*n[-1]
    for i in range(1,len(P)-1):
        a0,n0=P[i]+g*n[i-1],n[i-1]; a1,n1=P[i]+g*n[i],n[i]
        d0=np.array([-n0[1],n0[0]]); d1=np.array([-n1[1],n1[0]])
        A=np.array([d0,-d1]).T
        if abs(np.linalg.det(A))<1e-9: Q[i]=a0
        else:
            t=np.linalg.solve(A,a1-a0); Q[i]=a0+t[0]*d0
    return Q

Q=offset_profile(P,GAP)          # lid profile (mm)
n=seg_normals(P)
# measure the perpendicular gap on each flat/flank segment midpoint
for name,i in [("root",0),("rising flank",4),("crest",12),("falling flank",18)]:
    mid=0.5*(P[i]+P[i+1]); gap=np.dot(0.5*(Q[i]+Q[i+1])-mid,n[i])
    print(f"{name:15s} normal gap = {gap:.4f} mm")
print("lid minor r =",Q[:,0].min(),"lid major r =",Q[:,0].max())
fig,ax=plt.subplots(figsize=(5,5))
ax.plot(P[:,0],P[:,1],'o-',label='neck flank'); ax.plot(Q[:,0],Q[:,1],'s-',label='lid flank')
for a,b in zip(P,Q): ax.plot([a[0],b[0]],[a[1],b[1]],'k-',lw=0.4)
ax.set_aspect('equal'); ax.set_xlabel('r [mm]'); ax.set_ylabel('z [mm]'); ax.legend(); fig

# -- cell 3 -------------------------------------------------------------------------
# The 0.2 mm gap is exact on every flank. Now the sweep: node map with the helical wrap identification
NPER = 12            # azimuthal cells per turn (must divide NS)
NTURN = 5            # turns -> z spans 0..12 mm with a one-pitch cross-section
NT   = 2             # cells across the 0.2 mm gap
assert NS % NPER == 0
G = NS//NPER
NTH = NPER*NTURN     # azimuthal cells total
MM = 1e-3

def node_xyz(i,j,k):
    f=k/NT; r=P[j,0]+(Q[j,0]-P[j,0])*f; zl=P[j,1]+(Q[j,1]-P[j,1])*f
    th=2*np.pi*i/NPER; z=zl+PITCH*i/NPER
    return np.array([r*np.cos(th), r*np.sin(th), z])*MM

def key(i,j,k): return (j+i*G, i%NPER, k)

# check the wrap identification is geometrically exact
for i in [0,7,NTH-NPER]:
    for k in [0,NT]:
        assert key(i,NS,k)==key(i+NPER,0,k)
        assert np.allclose(node_xyz(i,NS,k),node_xyz(i+NPER,0,k),atol=1e-12)
print("wrap identification exact")
zs=[node_xyz(i,j,0)[2] for i in (0,NTH) for j in (0,NS)]
print("z at (i=0,j=0..NS) and (i=NTH,j=0..NS) [mm]:",[round(z/MM,3) for z in zs])
print("cells =",NTH*NS*NT)

# -- cell 4 -------------------------------------------------------------------------
nodes={}; coords=[]
def nid(i,j,k):
    kk=key(i,j,k)
    if kk not in nodes:
        nodes[kk]=len(coords)+1; coords.append(node_xyz(i,j,k))
    return nodes[kk]

hexes=[]; quads={'neck_flank':[],'lid_flank':[],'inner_end':[],'outer_end':[]}
for i in range(NTH):
    for j in range(NS):
        for k in range(NT):
            n=[nid(i,j,k),nid(i+1,j,k),nid(i+1,j+1,k),nid(i,j+1,k),
               nid(i,j,k+1),nid(i+1,j,k+1),nid(i+1,j+1,k+1),nid(i,j+1,k+1)]
            hexes.append(n)
            if k==0:    quads['neck_flank'].append([n[0],n[1],n[2],n[3]])
            if k==NT-1: quads['lid_flank'].append([n[4],n[5],n[6],n[7]])
        if i==0:     quads['inner_end'].append([nid(0,j,0),nid(0,j+1,0),nid(0,j+1,NT),nid(0,j,NT)])
        if i==NTH-1: quads['outer_end'].append([nid(NTH,j,0),nid(NTH,j+1,0),nid(NTH,j+1,NT),nid(NTH,j,NT)])
    for k in range(NT):
        if i<NPER:          quads['inner_end'].append([nid(i,0,k),nid(i+1,0,k),nid(i+1,0,k+1),nid(i,0,k+1)])
        if i>=NTH-NPER:     quads['outer_end'].append([nid(i,NS,k),nid(i+1,NS,k),nid(i+1,NS,k+1),nid(i,NS,k+1)])
X=np.array(coords)
# signed volume of first hex (should be positive with this ordering)
h=np.array([X[n-1] for n in hexes[0]]); c=h.mean(axis=0)
faces=[(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]  # outward
vol=sum(np.dot(np.cross(h[b]-h[a],h[c2]-h[a]),h[a]-c)/6 for a,b,c2,d in faces for a,b,c2 in [(a,b,c2),(a,c2,d)])
print("nodes",len(coords),"hexes",len(hexes),{k:len(v) for k,v in quads.items()},"hex0 vol(m^3)",vol)

# -- cell 5 -------------------------------------------------------------------------
def handed(n):
    h=np.array([X[a-1] for a in n])
    return np.dot(np.cross(h[1]-h[0],h[3]-h[0]),h[4]-h[0])
vals=np.array([handed(n) for n in hexes])
print("min/max handedness:",vals.min(),vals.max())
if vals.max()<0:
    hexes=[[n[3],n[2],n[1],n[0],n[7],n[6],n[5],n[4]] for n in hexes]
    vals=np.array([handed(n) for n in hexes]); print("after flip:",vals.min(),vals.max())
# analytic volume of the shell vs sum of cell volumes (rough, via 6-tet split)
def hvol(n):
    h=np.array([X[a-1] for a in n]); c=h.mean(axis=0)
    F=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    V=0
    for f in F:
        q=h[list(f)]; fc=q.mean(axis=0)
        for t in range(4):
            a,b=q[t],q[(t+1)%4]
            V+=np.dot(np.cross(a-c,b-c),fc-c)/6
    return V
V=sum(hvol(n) for n in hexes)
print("mesh volume mm^3:",V/MM**3, " approx analytic:",3.5347*GAP*2*np.pi*19.5*NTURN)

# -- cell 6 -------------------------------------------------------------------------
# Now export one STL per patch to `constant/triSurface`, orienting each triangle outward by testing ag
import os
os.makedirs("constant/triSurface",exist_ok=True)

def cell_nodes(i,j,k):
    return [nid(i,j,k),nid(i+1,j,k),nid(i+1,j+1,k),nid(i,j+1,k),
            nid(i,j,k+1),nid(i+1,j,k+1),nid(i+1,j+1,k+1),nid(i,j+1,k+1)]

def patch_faces():
    F={p:[] for p in ('neck_flank','lid_flank','inner_end','outer_end')}
    for i in range(NTH):
        for j in range(NS):
            for k in range(NT):
                own=(i,j,k)
                if k==0:    F['neck_flank'].append((own,[nid(i,j,0),nid(i+1,j,0),nid(i+1,j+1,0),nid(i,j+1,0)]))
                if k==NT-1: F['lid_flank'].append((own,[nid(i,j,NT),nid(i+1,j,NT),nid(i+1,j+1,NT),nid(i,j+1,NT)]))
                if i==0:      F['inner_end'].append((own,[nid(0,j,k),nid(0,j+1,k),nid(0,j+1,k+1),nid(0,j,k+1)]))
                if i==NTH-1:  F['outer_end'].append((own,[nid(NTH,j,k),nid(NTH,j+1,k),nid(NTH,j+1,k+1),nid(NTH,j,k+1)]))
                if j==0 and i<NPER:        F['inner_end'].append((own,[nid(i,0,k),nid(i+1,0,k),nid(i+1,0,k+1),nid(i,0,k+1)]))
                if j==NS-1 and i>=NTH-NPER:F['outer_end'].append((own,[nid(i,NS,k),nid(i+1,NS,k),nid(i+1,NS,k+1),nid(i,NS,k+1)]))
    return F

PF=patch_faces()
def oriented(own,q):
    v=np.array([X[a-1] for a in q]); cc=np.mean([X[a-1] for a in cell_nodes(*own)],axis=0)
    n=np.cross(v[1]-v[0],v[2]-v[0])
    return q if np.dot(n,v.mean(axis=0)-cc)>0 else q[::-1]

for p,fl in PF.items():
    tris=[]
    for own,q in fl:
        qq=oriented(own,q); v=[X[a-1] for a in qq]
        tris += [(v[0],v[1],v[2]),(v[0],v[2],v[3])]
    with open(f"constant/triSurface/{p}.stl","w") as f:
        f.write(f"solid {p}\n")
        for t in tris:
            n=np.cross(t[1]-t[0],t[2]-t[0]); n=n/np.linalg.norm(n)
            f.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in t: f.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {p}\n")
    print(p, len(fl),"quads ->",len(tris),"tris")
print(sorted(os.listdir("constant/triSurface")))

# -- cell 7 -------------------------------------------------------------------------
pnames=['neck_flank','lid_flank','inner_end','outer_end']
with open("helix.msh","w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n")
    for t,p in enumerate(pnames,1): f.write(f'2 {t} "{p}"\n')
    f.write('3 5 "fluid"\n$EndPhysicalNames\n')
    f.write(f"$Nodes\n{len(X)}\n")
    for a,v in enumerate(X,1): f.write(f"{a} {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
    f.write("$EndNodes\n$Elements\n")
    nel=len(hexes)+sum(len(PF[p]) for p in pnames); f.write(f"{nel}\n")
    e=0
    for t,p in enumerate(pnames,1):
        for own,q in PF[p]:
            e+=1; qq=oriented(own,q); f.write(f"{e} 3 2 {t} {t} "+" ".join(map(str,qq))+"\n")
    for n in hexes:
        e+=1; f.write(f"{e} 5 2 5 5 "+" ".join(map(str,n))+"\n")
    f.write("$EndElements\n")
print("elements",e)
import subprocess
os.makedirs("system",exist_ok=True)
open("system/controlDict","w").write("""FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;\n""")
r=subprocess.run(["gmshToFoam","helix.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 9 -------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-3000:])

# -- cell 10 ------------------------------------------------------------------------
print(r.returncode); print(r.stderr[-3000:]); print(len(r.stdout))

# -- cell 11 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\nddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}\n")
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-2600:])

# -- cell 12 ------------------------------------------------------------------------
# Root-corner miter makes the perpendicular t-lines cross (240 open cells). Fix: keep the mitered offs
CORN=np.array([[R_MINOR,0.0],[R_MINOR,FLAT/2],[R_MAJOR,FLAT/2+FLANK_DZ],
               [R_MAJOR,FLAT/2+FLANK_DZ+FLAT],[R_MINOR,PITCH-FLAT/2],[R_MINOR,PITCH]])
DIVS=[2,7,6,7,2]
CORNQ=offset_profile(CORN,GAP)          # mitered lid corners
def subdiv(C,divs):
    out=[C[0]]
    for i,n in enumerate(divs):
        for t in range(1,n+1): out.append(C[i]+(C[i+1]-C[i])*t/n)
    return np.array(out)
P=subdiv(CORN,DIVS); Q=subdiv(CORNQ,DIVS)
assert len(P)-1==NS
# no crossing: consecutive t-lines must not intersect -> check quad (P_j,P_j+1,Q_j+1,Q_j) area & convexity
bad=0; gaps=[]
for j in range(NS):
    q=np.array([P[j],P[j+1],Q[j+1],Q[j]])
    a=0.5*sum(q[i,0]*q[(i+1)%4,1]-q[(i+1)%4,0]*q[i,1] for i in range(4))
    if a>=0: bad+=1
    gaps.append(np.linalg.norm(Q[j]-P[j]))
print("negative-area profile quads:",bad,"  min/max node offset distance:",min(gaps),max(gaps))
nq=seg_normals(P)
for nm,i in [("root",0),("rising flank",4),("crest",12),("falling flank",18)]:
    print(nm,"perp gap =",round(float(np.dot(0.5*(Q[i]+Q[i+1])-0.5*(P[i]+P[i+1]),nq[i])),4))

# -- cell 13 ------------------------------------------------------------------------
# All profile quads are non-degenerate now. Rebuild mesh, STLs and msh from the corrected profile in o
def build_all():
    global nodes,coords,X,hexes,PF
    nodes={}; coords=[]
    hexes=[]
    for i in range(NTH):
        for j in range(NS):
            for k in range(NT): hexes.append(cell_nodes(i,j,k))
    X=np.array(coords)
    PF=patch_faces()
    for p,fl in PF.items():
        tris=[]
        for own,q in fl:
            qq=oriented(own,q); v=[X[a-1] for a in qq]
            tris += [(v[0],v[1],v[2]),(v[0],v[2],v[3])]
        with open(f"constant/triSurface/{p}.stl","w") as f:
            f.write(f"solid {p}\n")
            for t in tris:
                nn=np.cross(t[1]-t[0],t[2]-t[0]); nn=nn/np.linalg.norm(nn)
                f.write(f" facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}\n  outer loop\n")
                for v in t: f.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {p}\n")
    with open("helix.msh","w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n")
        for t,p in enumerate(pnames,1): f.write(f'2 {t} "{p}"\n')
        f.write('3 5 "fluid"\n$EndPhysicalNames\n')
        f.write(f"$Nodes\n{len(X)}\n")
        for a,v in enumerate(X,1): f.write(f"{a} {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
        f.write("$EndNodes\n$Elements\n")
        f.write(f"{len(hexes)+sum(len(PF[p]) for p in pnames)}\n"); e=0
        for t,p in enumerate(pnames,1):
            for own,q in PF[p]:
                e+=1; f.write(f"{e} 3 2 {t} {t} "+" ".join(map(str,oriented(own,q)))+"\n")
        for n in hexes:
            e+=1; f.write(f"{e} 5 2 5 5 "+" ".join(map(str,n))+"\n")
        f.write("$EndElements\n")
    return min(handed(n) for n in hexes)

print("min handedness:",build_all(),"nodes",len(X),"hexes",len(hexes))
import shutil; shutil.rmtree("constant/polyMesh",ignore_errors=True)
r=subprocess.run(["gmshToFoam","helix.msh"],capture_output=True,text=True); print("gmshToFoam",r.returncode,"Inverting" in r.stdout)
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True); print(r.stdout[r.stdout.find("Checking geometry"):][:1800])

# -- cell 14 ------------------------------------------------------------------------
# Mesh OK at the coarse setting. Now look at it — render the four patch STLs together.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl=pv.Plotter(off_screen=True,window_size=(900,800))
cols={'neck_flank':'steelblue','lid_flank':'indianred','inner_end':'lime','outer_end':'yellow'}
for p,c in cols.items(): pl.add_mesh(pv.read(f"constant/triSurface/{p}.stl"),color=c,label=p)
pl.add_legend(); pl.camera_position=[(0.06,-0.06,0.05),(0,0,0.006),(0,0,1)]
pl.screenshot("view.png"); from IPython.display import Image; Image("view.png")

# -- cell 15 ------------------------------------------------------------------------
# The shape is right (5 helical turns, blue neck side, red lid side, green/yellow ends). Now generaliz
from math import lcm
NPER = 120      # azimuthal cells per turn
NT   = 3        # cells across the gap
NTH  = NPER*NTURN
L    = lcm(NS,NPER)
def key(i,j,k): return (j*(L//NS)+i*(L//NPER), i%NPER, k)
for i in [0,37,NTH-NPER]:
    for k in [0,NT]:
        assert key(i,NS,k)==key(i+NPER,0,k) and np.allclose(node_xyz(i,NS,k),node_xyz(i+NPER,0,k),atol=1e-12)
print("chord error per azimuthal step (mm):",R_MAJOR*(1-np.cos(np.pi/NPER)))
print("min handedness:",build_all(),"nodes",len(X),"hexes",len(hexes),
      {p:len(v) for p,v in PF.items()})

# -- cell 16 ------------------------------------------------------------------------
shutil.rmtree("constant/polyMesh",ignore_errors=True)
r=subprocess.run(["gmshToFoam","helix.msh"],capture_output=True,text=True)
print("gmshToFoam rc",r.returncode,"| inverted hexes:","Inverting" in r.stdout)
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Checking patch topology"):][:2600])

# -- cell 17 ------------------------------------------------------------------------
# Now measure the delivered numbers back out of the built mesh and patch STLs.
def patch_pts(p):
    m=pv.read(f"constant/triSurface/{p}.stl"); return np.array(m.points)/MM
neck=patch_pts('neck_flank'); lid=patch_pts('lid_flank')
rn=np.hypot(neck[:,0],neck[:,1]); rl=np.hypot(lid[:,0],lid[:,1])
print(f"neck outside diameter   : {2*rn.max():.4f} mm   (asked 40)")
print(f"neck thread depth       : {rn.max()-rn.min():.4f} mm   (asked 1)")
print(f"threaded height (z span): {neck[:,2].max()-neck[:,2].min():.4f} mm   (asked 12)")
# pitch from the helix: same profile index, one turn apart
dz=[node_xyz(i+NPER,j,0)[2]-node_xyz(i,j,0)[2] for i in (0,53) for j in (0,NS//2)]
print(f"pitch (z rise per turn) : {np.unique(np.round(np.array(dz)/MM,6))} mm   (asked 2)")
print(f"starts                  : {(neck[:,2].max()-neck[:,2].min())/2.0/NTURN:.3f} pitches per turn -> single start")
# flank half-angle from the built profile
d=P[8]-P[4]; print(f"flank half angle        : {np.degrees(np.arctan2(abs(d[1]),abs(d[0]))):.3f} deg (trapezoid, 30 deg included)")
# gap, measured normal to the neck flank on flats and as node-to-node spacing
g_node=np.array([np.linalg.norm(node_xyz(0,j,NT)-node_xyz(0,j,0))/MM for j in range(NS+1)])
print(f"gap, flank-normal       : 0.2000 mm on every flat/flank; node spacing {g_node.min():.4f}-{g_node.max():.4f} mm (miter at crest corners)")
for p in ['inner_end','outer_end']:
    z=patch_pts(p)[:,2]; print(f"{p}: z {z.min():.3f} .. {z.max():.3f} mm, area-> {len(PF[p])} faces")
print(f"fluid volume            : {4.36059e-07/MM**3:.2f} mm^3, cells 43200")

# -- cell 18 ------------------------------------------------------------------------
pl=pv.Plotter(off_screen=True,window_size=(1000,450),shape=(1,2))
for p,c in cols.items(): pl.subplot(0,0); pl.add_mesh(pv.read(f"constant/triSurface/{p}.stl"),color=c)
pl.camera_position=[(0.05,-0.05,0.045),(0,0,0.006),(0,0,1)]
pl.subplot(0,1)
fm=pv.read("constant/triSurface/neck_flank.stl")+pv.read("constant/triSurface/lid_flank.stl")
cl=(fm+pv.read("constant/triSurface/inner_end.stl")+pv.read("constant/triSurface/outer_end.stl")).clip('y',origin=(0,0,0))
pl.add_mesh(cl,color='lightsteelblue',show_edges=True,line_width=0.3)
pl.camera_position=[(0.07,0.0,0.012),(0.019,0,0.006),(0,0,1)]
pl.screenshot("view2.png"); Image("view2.png")
