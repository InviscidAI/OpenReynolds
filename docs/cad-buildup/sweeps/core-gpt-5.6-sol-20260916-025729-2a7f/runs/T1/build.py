"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid passage itself: two 120 mm horizontal centerline legs at \(y=0\) and \(y=30\) m
import build123d as bd
import matplotlib.pyplot as plt
# All dimensions in metres.
W = 0.010
L = 0.120
R = 0.015
T = 0.001  # assumed numerical extrusion thickness for the 2-D mesh
lower = bd.Pos(L/2, 0, 0) * bd.Rectangle(L, W)
upper = bd.Pos(L/2, 2*R, 0) * bd.Rectangle(L, W)
outer_disk = bd.Pos(L, R, 0) * bd.Circle(R + W/2)
inner_disk = bd.Pos(L, R, 0) * bd.Circle(R - W/2)
right_clip = bd.Pos(L + (R + W/2)/2, R, 0) * bd.Rectangle(R + W/2, 2*(R + W/2))
bend = (outer_disk - inner_disk) & right_clip
footprint = lower + upper + bend
fluid = bd.extrude(footprint, T)
print(f"requested width={W:.6f} m; measured straight width={lower.bounding_box().size.Y:.6f} m")
print(f"requested leg length={L:.6f} m; measured={lower.bounding_box().size.X:.6f} m")
print(f"requested centreline radius={R:.6f} m; constructed mean radius={((R-W/2)+(R+W/2))/2:.6f} m")
print(f"fluid volume={fluid.volume:.9e} m^3; expected area*thickness={(2*L*W + 3.141592653589793*R*W)*T:.9e} m^3")
# Plot exact footprint boundaries in the x-y plane.
fig, ax = plt.subplots(figsize=(8,3))
for e in footprint.edges():
    pts = [e.position_at(i/100) for i in range(101)]
    ax.plot([p.X*1000 for p in pts], [p.Y*1000 for p in pts], 'k-', lw=2)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('U-bend fluid footprint (coarse CAD inspection)'); ax.grid(True)
plt.tight_layout(); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The inspected footprint has the intended constant width and tangent joins. I’ll now make the coarse 
from pathlib import Path
import numpy as np, math
# Structured 2-D blocks; each quad is counter-clockwise and boundary labels are created here.
nL, nW, nQ = 48, 4, 10
raw_pts, quads, edge_labels = [], [], {}
def node(x,y):
    raw_pts.append((float(x),float(y))); return len(raw_pts)-1
def add_block(X, Y, labels):
    ids=np.empty(X.shape,dtype=int)
    for ij in np.ndindex(X.shape): ids[ij]=node(X[ij],Y[ij])
    ni,nj=X.shape[0]-1,X.shape[1]-1
    for i in range(ni):
        for j in range(nj): quads.append([ids[i,j],ids[i+1,j],ids[i+1,j+1],ids[i,j+1]])
    for side,label in labels.items():
        seq = ids[0,:] if side=='i0' else ids[-1,:] if side=='i1' else ids[:,0] if side=='j0' else ids[:,-1]
        for a,b in zip(seq[:-1],seq[1:]): edge_labels[frozenset((a,b))]=label
# lower: x increases, y increases
x=np.linspace(0,L,nL+1); y=np.linspace(-W/2,W/2,nW+1); X,Y=np.meshgrid(x,y,indexing='ij')
add_block(X,Y,{'i0':'inlet','j0':'walls','j1':'walls'})
# bend: theta increases; radius decreases, preserving CCW cell order
theta=np.linspace(-math.pi/2,math.pi/2,2*nQ+1); rad=np.linspace(R+W/2,R-W/2,nW+1)
TH,RR=np.meshgrid(theta,rad,indexing='ij'); X=L+RR*np.cos(TH); Y=R+RR*np.sin(TH)
add_block(X,Y,{'j0':'walls','j1':'walls'})
# upper: both x and y decrease, preserving CCW cell order
x=np.linspace(L,0,nL+1); y=np.linspace(2*R+W/2,2*R-W/2,nW+1); X,Y=np.meshgrid(x,y,indexing='ij')
add_block(X,Y,{'i1':'outlet','j0':'walls','j1':'walls'})
# Merge coincident nodes across block interfaces and transfer boundary labels.
merged=[]; key_to_new={}; old_to_new={}
for old,p in enumerate(raw_pts):
    key=(round(p[0],12),round(p[1],12))
    if key not in key_to_new: key_to_new[key]=len(merged); merged.append(p)
    old_to_new[old]=key_to_new[key]
quads=[[old_to_new[v] for v in q] for q in quads]
labels2={frozenset(old_to_new[v] for v in e):lab for e,lab in edge_labels.items()}
N=len(merged); points=[(x,y,0.0) for x,y in merged]+[(x,y,T) for x,y in merged]
# Assemble outward-oriented hex faces, retaining first-use orientation for owner.
face_map={}; faces=[]; owners=[]; neighbours=[]; patch_for_face={}
for ci,q in enumerate(quads):
    b=q; t=[v+N for v in q]
    candidates=[([b[0],b[3],b[2],b[1]],'frontAndBack'),([t[0],t[1],t[2],t[3]],'frontAndBack')]
    for k in range(4):
        a,c=b[k],b[(k+1)%4]; candidates.append(([a,a+N,c+N,c],labels2.get(frozenset((a,c)))))
    for f,lab in candidates:
        key=tuple(sorted(f))
        if key in face_map: neighbours.append((face_map[key],ci)); patch_for_face.pop(face_map[key],None)
        else:
            fi=len(faces); face_map[key]=fi; faces.append(f); owners.append(ci)
            if lab: patch_for_face[fi]=lab
# OpenFOAM requires internal faces first.
internal_ids=[fi for fi,_ in neighbours]; neighbour_by_face={fi:nb for fi,nb in neighbours}
patch_order=['inlet','outlet','walls','frontAndBack']
boundary_ids=[fi for p in patch_order for fi in range(len(faces)) if patch_for_face.get(fi)==p]
order=internal_ids+boundary_ids; remap={old:new for new,old in enumerate(order)}
faces_o=[faces[i] for i in order]; owners_o=[owners[i] for i in order]; neigh_o=[neighbour_by_face[i] for i in internal_ids]
root=Path('.'); pm=root/'constant/polyMesh'; pm.mkdir(parents=True,exist_ok=True); (root/'system').mkdir(exist_ok=True); (root/'triSurface').mkdir(exist_ok=True)
head=lambda cls,obj: f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n"
(pm/'points').write_text(head('vectorField','points')+f"{len(points)}\n(\n"+''.join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x,y,z in points)+")\n")
(pm/'faces').write_text(head('faceList','faces')+f"{len(faces_o)}\n(\n"+''.join(f"4({' '.join(map(str,f))})\n" for f in faces_o)+")\n")
(pm/'owner').write_text(head('labelList','owner')+f"{len(owners_o)}\n(\n"+'\n'.join(map(str,owners_o))+"\n)\n")
(pm/'neighbour').write_text(head('labelList','neighbour')+f"{len(neigh_o)}\n(\n"+'\n'.join(map(str,neigh_o))+"\n)\n")
start=len(internal_ids); entries=[]
for p in patch_order:
    count=sum(patch_for_face.get(fi)==p for fi in boundary_ids)
    typ='empty' if p=='frontAndBack' else 'patch' if p in ('inlet','outlet') else 'wall'
    entries.append(f"    {p}\n    {{\n        type {typ};\n        nFaces {count};\n        startFace {start};\n    }}\n"); start+=count
(pm/'boundary').write_text(head('polyBoundaryMesh','boundary')+f"{len(entries)}\n(\n"+''.join(entries)+")\n")
(root/'system/controlDict').write_text(head('dictionary','controlDict')+'application checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n')
# Export each named patch as its own triangulated STL in metres.
def stl_for_patch(name):
    out=[f"solid {name}\n"]
    for fi,lab in patch_for_face.items():
        if lab!=name: continue
        f=faces[fi]
        for tri in ((f[0],f[1],f[2]),(f[0],f[2],f[3])):
            a,b,c=[np.array(points[v]) for v in tri]; n=np.cross(b-a,c-a); n=n/(np.linalg.norm(n) or 1)
            out.append(f" facet normal {' '.join(f'{v:.12g}' for v in n)}\n  outer loop\n"+''.join(f"   vertex {' '.join(f'{v:.12g}' for v in points[k])}\n" for k in tri)+"  endloop\n endfacet\n")
    out.append(f"endsolid {name}\n"); (root/'triSurface'/f'{name}.stl').write_text(''.join(out))
for p in patch_order: stl_for_patch(p)
bd.export_step(fluid, root/'fluid_volume.step')
print(f"coarse mesh: {len(quads)} hex cells, {len(points)} points, {len(internal_ids)} internal faces")
print(f"resolution: {nW} cells across 10 mm, {nL} along each 120 mm leg, {2*nQ} around 180 deg")
print({p:sum(v==p for v in patch_for_face.values()) for p in patch_order})

# -- cell 3 -------------------------------------------------------------------------
# The coarse mesh has 464 hexahedra and the expected patch counts. I’ll now run OpenFOAM’s topology/qu
import subprocess
cm = subprocess.run(['checkMesh','-constant','-allTopology','-allGeometry'],text=True,capture_output=True)
print(cm.stdout[-6000:])
print('checkMesh return code:',cm.returncode)

# -- cell 4 -------------------------------------------------------------------------
# `checkMesh` stopped during mesh loading rather than reporting quality, so I’ll inspect its complete 
print('STDOUT:\n',cm.stdout)
print('STDERR:\n',cm.stderr)

# -- cell 5 -------------------------------------------------------------------------
# The mesh files loaded far enough to reveal only missing mandatory case dictionaries. I’ll add minima
for name in ('fvSchemes','fvSolution'):
    Path('system',name).write_text(head('dictionary',name))
print('wrote minimal system/fvSchemes and system/fvSolution')

# -- cell 6 -------------------------------------------------------------------------
# I’ll rerun the full topology and geometry check now that the minimal case metadata is complete.
cm2=subprocess.run(['checkMesh','-constant','-allTopology','-allGeometry'],text=True,capture_output=True)
print(cm2.stdout[-7000:]); print(cm2.stderr); print('return code:',cm2.returncode)

# -- cell 7 -------------------------------------------------------------------------
# OpenFOAM 2512 additionally requires the standard scheme subdictionaries even for mesh checking. I’ll
Path('system/fvSchemes').write_text(head('dictionary','fvSchemes')+'''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text(head('dictionary','fvSolution')+'solvers {}\n')
print('completed required checkMesh dictionaries')

# -- cell 8 -------------------------------------------------------------------------
# I’ll run the quality check again.
cm3=subprocess.run(['checkMesh','-constant','-allTopology','-allGeometry'],text=True,capture_output=True)
print(cm3.stdout[-7000:]); print(cm3.stderr); print('return code:',cm3.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The topology is connected, but every cell has negative signed volume and every internal-face angle i
import shutil
shutil.rmtree(pm/'sets',ignore_errors=True)
face_map={}; faces=[]; owners=[]; neighbours=[]; patch_for_face={}
for ci,q in enumerate(quads):
    b=q; t=[v+N for v in q]
    candidates=[([b[0],b[3],b[2],b[1]],'frontAndBack'),([t[0],t[1],t[2],t[3]],'frontAndBack')]
    for k in range(4):
        a,c=b[k],b[(k+1)%4]; candidates.append(([a,c,c+N,a+N],labels2.get(frozenset((a,c)))))
    for f,lab in candidates:
        key=tuple(sorted(f))
        if key in face_map: neighbours.append((face_map[key],ci)); patch_for_face.pop(face_map[key],None)
        else:
            fi=len(faces); face_map[key]=fi; faces.append(f); owners.append(ci)
            if lab: patch_for_face[fi]=lab
internal_ids=[fi for fi,_ in neighbours]; neighbour_by_face={fi:nb for fi,nb in neighbours}
boundary_ids=[fi for p in patch_order for fi in range(len(faces)) if patch_for_face.get(fi)==p]
order=internal_ids+boundary_ids
faces_o=[faces[i] for i in order]; owners_o=[owners[i] for i in order]; neigh_o=[neighbour_by_face[i] for i in internal_ids]
(pm/'faces').write_text(head('faceList','faces')+f"{len(faces_o)}\n(\n"+''.join(f"4({' '.join(map(str,f))})\n" for f in faces_o)+")\n")
(pm/'owner').write_text(head('labelList','owner')+f"{len(owners_o)}\n(\n"+'\n'.join(map(str,owners_o))+"\n)\n")
(pm/'neighbour').write_text(head('labelList','neighbour')+f"{len(neigh_o)}\n(\n"+'\n'.join(map(str,neigh_o))+"\n)\n")
for p in patch_order: stl_for_patch(p)
print('reversed lateral face winding and regenerated mesh/STLs')

# -- cell 10 ------------------------------------------------------------------------
# I’ll verify the corrected winding with the same exhaustive mesh check.
cm4=subprocess.run(['checkMesh','-constant','-allTopology','-allGeometry'],text=True,capture_output=True)
print(cm4.stdout[-6500:]); print(cm4.stderr); print('return code:',cm4.returncode)

# -- cell 11 ------------------------------------------------------------------------
# The corrected coarse mesh passes all checks, with maximum non-orthogonality 1.85° and maximum skewne
fig,ax=plt.subplots(figsize=(10,3.2))
# draw every unique planar mesh edge
seen=set()
for q in quads:
    for a,b in zip(q,q[1:]+q[:1]):
        e=tuple(sorted((a,b)))
        if e not in seen:
            seen.add(e); xa,ya=merged[a]; xb,yb=merged[b]; ax.plot([xa*1000,xb*1000],[ya*1000,yb*1000],color='0.45',lw=.45)
colors={'inlet':'tab:blue','outlet':'tab:orange','walls':'tab:red'}
for e,lab in labels2.items():
    a,b=tuple(e); xa,ya=merged[a]; xb,yb=merged[b]
    ax.plot([xa*1000,xb*1000],[ya*1000,yb*1000],color=colors[lab],lw=2)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.grid(True,alpha=.25)
ax.set_title('Checked 2-D OpenFOAM mesh: inlet blue, outlet orange, walls red')
plt.tight_layout(); plt.show()

# -- cell 12 ------------------------------------------------------------------------
# The visual inspection shows a conformal, smoothly fanned bend mesh with correctly located inlet/outl
nL,nW,nQ=120,10,24
raw_pts=[]; quads=[]; edge_labels={}
x=np.linspace(0,L,nL+1); y=np.linspace(-W/2,W/2,nW+1); X,Y=np.meshgrid(x,y,indexing='ij'); add_block(X,Y,{'i0':'inlet','j0':'walls','j1':'walls'})
theta=np.linspace(-math.pi/2,math.pi/2,2*nQ+1); rad=np.linspace(R+W/2,R-W/2,nW+1); TH,RR=np.meshgrid(theta,rad,indexing='ij'); add_block(L+RR*np.cos(TH),R+RR*np.sin(TH),{'j0':'walls','j1':'walls'})
x=np.linspace(L,0,nL+1); y=np.linspace(2*R+W/2,2*R-W/2,nW+1); X,Y=np.meshgrid(x,y,indexing='ij'); add_block(X,Y,{'i1':'outlet','j0':'walls','j1':'walls'})
merged=[]; key_to_new={}; old_to_new={}
for old,p in enumerate(raw_pts):
    key=(round(p[0],12),round(p[1],12))
    if key not in key_to_new: key_to_new[key]=len(merged); merged.append(p)
    old_to_new[old]=key_to_new[key]
quads=[[old_to_new[v] for v in q] for q in quads]; labels2={frozenset(old_to_new[v] for v in e):lab for e,lab in edge_labels.items()}
N=len(merged); points=[(x,y,0.0) for x,y in merged]+[(x,y,T) for x,y in merged]
face_map={}; faces=[]; owners=[]; neighbours=[]; patch_for_face={}
for ci,q in enumerate(quads):
    b=q; t=[v+N for v in q]; candidates=[([b[0],b[3],b[2],b[1]],'frontAndBack'),([t[0],t[1],t[2],t[3]],'frontAndBack')]
    candidates += [([b[k],b[(k+1)%4],b[(k+1)%4]+N,b[k]+N],labels2.get(frozenset((b[k],b[(k+1)%4])))) for k in range(4)]
    for f,lab in candidates:
        key=tuple(sorted(f))
        if key in face_map: neighbours.append((face_map[key],ci)); patch_for_face.pop(face_map[key],None)
        else:
            fi=len(faces); face_map[key]=fi; faces.append(f); owners.append(ci)
            if lab: patch_for_face[fi]=lab
internal_ids=[fi for fi,_ in neighbours]; neighbour_by_face={fi:nb for fi,nb in neighbours}; boundary_ids=[fi for p in patch_order for fi in range(len(faces)) if patch_for_face.get(fi)==p]; order=internal_ids+boundary_ids
faces_o=[faces[i] for i in order]; owners_o=[owners[i] for i in order]; neigh_o=[neighbour_by_face[i] for i in internal_ids]
(pm/'points').write_text(head('vectorField','points')+f"{len(points)}\n(\n"+''.join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x,y,z in points)+")\n")
(pm/'faces').write_text(head('faceList','faces')+f"{len(faces_o)}\n(\n"+''.join(f"4({' '.join(map(str,f))})\n" for f in faces_o)+")\n")
(pm/'owner').write_text(head('labelList','owner')+f"{len(owners_o)}\n(\n"+'\n'.join(map(str,owners_o))+"\n)\n"); (pm/'neighbour').write_text(head('labelList','neighbour')+f"{len(neigh_o)}\n(\n"+'\n'.join(map(str,neigh_o))+"\n)\n")
start=len(internal_ids); entries=[]
for p in patch_order:
    count=sum(patch_for_face.get(fi)==p for fi in boundary_ids); typ='empty' if p=='frontAndBack' else 'patch' if p in ('inlet','outlet') else 'wall'; entries.append(f"    {p}\n    {{\n        type {typ};\n        nFaces {count};\n        startFace {start};\n    }}\n"); start+=count
(pm/'boundary').write_text(head('polyBoundaryMesh','boundary')+f"{len(entries)}\n(\n"+''.join(entries)+")\n")
for p in patch_order: stl_for_patch(p)
shutil.rmtree(pm/'sets',ignore_errors=True)
print(f'production mesh: {len(quads)} cells; nominal straight spacing {L/nL*1000:.3f} mm; {nW} cells across width')

# -- cell 13 ------------------------------------------------------------------------
# The refined mesh contains 2,880 structured hexahedra. I’ll run the exhaustive OpenFOAM check on this
final_check=subprocess.run(['checkMesh','-constant','-allTopology','-allGeometry'],text=True,capture_output=True)
print(final_check.stdout[-5000:]); print(final_check.stderr); print('return code:',final_check.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The final mesh passes every topology and geometry check. I’ll make one final rendering of the delive
fig,ax=plt.subplots(figsize=(10,3.2)); seen=set()
for q in quads:
    for a,b in zip(q,q[1:]+q[:1]):
        e=tuple(sorted((a,b)))
        if e not in seen:
            seen.add(e); xa,ya=merged[a]; xb,yb=merged[b]; ax.plot([xa*1000,xb*1000],[ya*1000,yb*1000],color='0.55',lw=.28)
for e,lab in labels2.items():
    a,b=tuple(e); xa,ya=merged[a]; xb,yb=merged[b]; ax.plot([xa*1000,xb*1000],[ya*1000,yb*1000],color=colors[lab],lw=1.8)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title('Final 2-D U-bend mesh (2,880 cells)'); ax.grid(True,alpha=.2)
plt.tight_layout(); plt.savefig('final_mesh.png',dpi=180); plt.show()
