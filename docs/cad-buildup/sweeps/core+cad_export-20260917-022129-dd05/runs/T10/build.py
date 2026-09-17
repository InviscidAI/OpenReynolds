"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first preview shows the intended twisted interblade channel surfaces, but the measurement print 
import numpy as np, math, matplotlib.pyplot as plt
HUB_R=0.045; TIP_R=0.090; BLADE_H=TIP_R-HUB_R; N_BLADES=28
PITCH_ANG=2*np.pi/N_BLADES; CHORD=0.022; ROOT_STAGGER_DEG=30.0; TWIST_DEG=35.0
# Assumed section because camber amount/location were unspecified: closed-TE NACA 4406.
CAMBER=0.04; CAMBER_POS=0.40; THICKNESS=0.06
X_INLET=-CHORD; X_OUTLET=CHORD*np.cos(np.deg2rad(ROOT_STAGGER_DEG))+CHORD

def section_xy(q,side):
    q=np.asarray(q,float)
    yc=np.where(q<CAMBER_POS,CAMBER/CAMBER_POS**2*(2*CAMBER_POS*q-q*q),CAMBER/(1-CAMBER_POS)**2*((1-2*CAMBER_POS)+2*CAMBER_POS*q-q*q))
    dy=np.where(q<CAMBER_POS,2*CAMBER/CAMBER_POS**2*(CAMBER_POS-q),2*CAMBER/(1-CAMBER_POS)**2*(CAMBER_POS-q))
    yt=5*THICKNESS*(0.2969*np.sqrt(q)-0.1260*q-0.3516*q*q+0.2843*q**3-0.1036*q**4)
    a=np.arctan(dy)
    return CHORD*(q-side*yt*np.sin(a)),CHORD*(yc+side*yt*np.cos(a))

def blade_surface(q,r,side,blade_index):
    q,r=np.broadcast_arrays(np.asarray(q,float),np.asarray(r,float))
    beta=np.deg2rad(ROOT_STAGGER_DEG+TWIST_DEG*(r-HUB_R)/BLADE_H)
    X,Y=section_xy(q,side)
    return X*np.cos(beta)-Y*np.sin(beta), blade_index*PITCH_ANG+(X*np.sin(beta)+Y*np.cos(beta))/r

qv=np.linspace(0,1,31); rv=np.linspace(HUB_R,TIP_R,9); Q,R=np.meshgrid(qv,rv)
Xp,Tp=blade_surface(Q,R,-1,-0.5); Xs,Ts=blade_surface(Q,R,+1,+0.5)
fig=plt.figure(figsize=(8,5)); ax=fig.add_subplot(111,projection='3d')
for X,T,C in [(Xp,Tp,'tomato'),(Xs,Ts,'royalblue')]: ax.plot_surface(X,R*np.cos(T),R*np.sin(T),color=C,alpha=.8,linewidth=0)
ax.set(xlabel='axial x [m]',ylabel='y [m]',zlabel='z [m]'); ax.set_box_aspect((1.1,1,1)); ax.view_init(24,-55); plt.tight_layout(); plt.show()
qq=np.linspace(0,1,2001); Xu,Yu=section_xy(qq,+1); Xl,Yl=section_xy(qq,-1); thick=np.hypot(Xu-Xl,Yu-Yl)
X0,Y0=section_xy(0.0,0); X1,Y1=section_xy(1.0,0); measured_chord=float(np.hypot(X1-X0,Y1-Y0))
print(f'hub/tip/height: {HUB_R*1e3:.3f}/{TIP_R*1e3:.3f}/{BLADE_H*1e3:.3f} mm (requested 45/90/45)')
print(f'pitch: {np.rad2deg(PITCH_ANG):.9f} deg (=360/28)')
print(f'root/tip stagger: {ROOT_STAGGER_DEG:.3f}/{ROOT_STAGGER_DEG+TWIST_DEG:.3f} deg (requested 30 and +35 twist)')
print(f'chord: {measured_chord*1e3:.3f} mm (requested 22)')
print(f'max thickness: {thick.max()*1e3:.4f} mm = {100*thick.max()/CHORD:.3f}% chord (requested 6%)')
print(f'assumed camber: {100*CAMBER:.1f}% at {100*CAMBER_POS:.0f}% chord; inlet/outlet x={X_INLET*1e3:.3f}/{X_OUTLET*1e3:.3f} mm')

# -- cell 2 -------------------------------------------------------------------------
# One queried method is absent in this installed version; I’ll query only the available constructors a
import build123d as bd, inspect
for obj,name in [(bd.Face,'make_surface'),(bd.Shell,'__init__'),(bd.Solid,'make_loft'),(bd.Edge,'make_spline'),(bd,'export_step')]:
    fn=getattr(obj,name)
    print(f'{obj}.{name}: {inspect.signature(fn)}')

# -- cell 3 -------------------------------------------------------------------------
# I’ll create the coarsest actual volume mesh directly from the passage mapping: 24 streamwise × 6 rad
from pathlib import Path
# Coarse structured passage discretisation.
N_UP,N_BLADE,N_DOWN,N_RAD,N_PITCH=6,12,6,6,8
stations=[('up',t/N_UP) for t in range(N_UP)] + [('blade',0.5*(1-np.cos(np.pi*t/N_BLADE))) for t in range(N_BLADE)] + [('down',t/N_DOWN) for t in range(N_DOWN+1)]
NI=len(stations)-1

def passage_point(i,j,k):
    region,a=stations[i]; r=HUB_R+BLADE_H*j/N_RAD; e=k/N_PITCH
    if region=='up':
        xl=xu=X_INLET*(1-a); tl=-PITCH_ANG/2; tu=PITCH_ANG/2
    elif region=='blade':
        xl,tl=blade_surface(a,r,-1,-0.5); xu,tu=blade_surface(a,r,+1,+0.5)
    else:
        xlp,tl=blade_surface(1.0,r,-1,-0.5); xup,tu=blade_surface(1.0,r,+1,+0.5)
        xl=(1-a)*float(xlp)+a*X_OUTLET; xu=(1-a)*float(xup)+a*X_OUTLET
    x=(1-e)*float(xl)+e*float(xu); th=(1-e)*float(tl)+e*float(tu)
    return np.array([x,r*np.cos(th),r*np.sin(th)])

def vid(i,j,k): return (i*(N_RAD+1)+j)*(N_PITCH+1)+k
points=np.array([passage_point(i,j,k) for i in range(NI+1) for j in range(N_RAD+1) for k in range(N_PITCH+1)])
cells=[]
for i in range(NI):
 for j in range(N_RAD):
  for k in range(N_PITCH):
   cells.append([vid(i,j,k),vid(i+1,j,k),vid(i+1,j+1,k),vid(i,j+1,k),vid(i,j,k+1),vid(i+1,j,k+1),vid(i+1,j+1,k+1),vid(i,j+1,k+1)])
# Assemble uniquely shared faces and orient each away from its first (owner) cell.
face_slots={}; face_data=[]
local_faces=[(0,4,7,3),(1,2,6,5),(0,1,5,4),(3,7,6,2),(0,3,2,1),(4,5,6,7)]
for ci,c in enumerate(cells):
 cc=points[c].mean(axis=0)
 for lf in local_faces:
  f=[c[n] for n in lf]; key=tuple(sorted(f))
  if key in face_slots: face_data[face_slots[key]]['nei']=ci
  else:
   fc=points[f].mean(axis=0); n=np.cross(points[f[1]]-points[f[0]],points[f[2]]-points[f[1]])
   if np.dot(n,fc-cc)<0: f=f[::-1]
   face_slots[key]=len(face_data); face_data.append({'v':f,'own':ci,'nei':None})
# Identify boundary patch by constant structured index.
def ijk(v):
 k=v%(N_PITCH+1); z=v//(N_PITCH+1); j=z%(N_RAD+1); i=z//(N_RAD+1); return i,j,k
patch_order=['inlet','outlet','hub','shroud','blade_pressure','blade_suction','periodic_1','periodic_2']
patches={p:[] for p in patch_order}; internal=[]
for f in face_data:
 if f['nei'] is not None: internal.append(f)
 else:
  ids=[ijk(v) for v in f['v']]; Is={a[0] for a in ids}; Js={a[1] for a in ids}; Ks={a[2] for a in ids}
  if Is=={0}: p='inlet'
  elif Is=={NI}: p='outlet'
  elif Js=={0}: p='hub'
  elif Js=={N_RAD}: p='shroud'
  elif Ks=={0}: p='blade_pressure' if min(a[0] for a in ids)>=N_UP and max(a[0] for a in ids)<=N_UP+N_BLADE else 'periodic_1'
  elif Ks=={N_PITCH}: p='blade_suction' if min(a[0] for a in ids)>=N_UP and max(a[0] for a in ids)<=N_UP+N_BLADE else 'periodic_2'
  else: raise RuntimeError('unclassified boundary face')
  patches[p].append(f)
ordered=internal+sum((patches[p] for p in patch_order),[])
poly=Path('constant/polyMesh'); poly.mkdir(parents=True,exist_ok=True)
head=lambda cls,obj: f"FoamFile\n{{\n version 2.0;\n format ascii;\n class {cls};\n object {obj};\n}}\n"
(poly/'points').write_text(head('vectorField','points')+f"{len(points)}\n(\n"+''.join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x,y,z in points)+")\n")
(poly/'faces').write_text(head('faceList','faces')+f"{len(ordered)}\n(\n"+''.join(f"4({' '.join(map(str,f['v']))})\n" for f in ordered)+")\n")
(poly/'owner').write_text(head('labelList','owner')+f"{len(ordered)}\n(\n"+''.join(f"{f['own']}\n" for f in ordered)+")\n")
(poly/'neighbour').write_text(head('labelList','neighbour')+f"{len(internal)}\n(\n"+''.join(f"{f['nei']}\n" for f in internal)+")\n")
start=len(internal); entries=[]
for p in patch_order:
 typ='wall' if p in ('hub','shroud','blade_pressure','blade_suction') else 'patch'
 entries.append(f"{p}\n{{\n type {typ};\n nFaces {len(patches[p])};\n startFace {start};\n}}\n"); start+=len(patches[p])
(poly/'boundary').write_text(head('polyBoundaryMesh','boundary')+f"{len(patch_order)}\n(\n"+''.join(entries)+")\n")
print(f'wrote coarse mesh: {len(points)} points, {len(cells)} hex cells, {len(ordered)} faces')
print('patch faces:',{p:len(patches[p]) for p in patch_order})

# -- cell 4 -------------------------------------------------------------------------
# The coarse mesh has all eight requested boundary names and only hexahedra. I’ll run the binding mesh
import subprocess
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-5000:]); print('returncode',r.returncode)

# -- cell 5 -------------------------------------------------------------------------
# `checkMesh` stopped before reading the mesh because this empty case lacks a control dictionary. I’ll
system=Path('system'); system.mkdir(exist_ok=True)
(system/'controlDict').write_text("FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object controlDict;\n}\napplication checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\nrunTimeModifiable false;\n")
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-6500:]); print(r.stderr[-1000:]); print('returncode',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# This OpenFOAM release also insists on empty `fvSchemes` and `fvSolution` dictionaries even for mesh-
for name in ('fvSchemes','fvSolution'):
 (system/name).write_text(f"FoamFile\n{{\n version 2.0;\n format ascii;\n class dictionary;\n object {name};\n}}\n")
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The checker reads standard scheme subdictionaries during mesh construction. I’ll provide minimal val
(system/'fvSchemes').write_text("FoamFile\n{ version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n")
(system/'fvSolution').write_text("FoamFile\n{ version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n")
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The coarse topology is correct, but my centroid-based orientation test misclassified highly warped b
# Reassemble with exact outward orientation for the right-handed structured mapping.
face_slots={}; face_data=[]
for ci,c in enumerate(cells):
 for lf in local_faces:
  f=[c[n] for n in lf]; key=tuple(sorted(f))
  if key in face_slots: face_data[face_slots[key]]['nei']=ci
  else: face_slots[key]=len(face_data); face_data.append({'v':f,'own':ci,'nei':None})
patches={p:[] for p in patch_order}; internal=[]
for f in face_data:
 if f['nei'] is not None: internal.append(f)
 else:
  ids=[ijk(v) for v in f['v']]; Is={a[0] for a in ids}; Js={a[1] for a in ids}; Ks={a[2] for a in ids}
  if Is=={0}: p='inlet'
  elif Is=={NI}: p='outlet'
  elif Js=={0}: p='hub'
  elif Js=={N_RAD}: p='shroud'
  elif Ks=={0}: p='blade_pressure' if min(a[0] for a in ids)>=N_UP and max(a[0] for a in ids)<=N_UP+N_BLADE else 'periodic_1'
  elif Ks=={N_PITCH}: p='blade_suction' if min(a[0] for a in ids)>=N_UP and max(a[0] for a in ids)<=N_UP+N_BLADE else 'periodic_2'
  else: raise RuntimeError('unclassified boundary face')
  patches[p].append(f)
ordered=internal+sum((patches[p] for p in patch_order),[])
(poly/'faces').write_text(head('faceList','faces')+f"{len(ordered)}\n(\n"+''.join(f"4({' '.join(map(str,f['v']))})\n" for f in ordered)+")\n")
(poly/'owner').write_text(head('labelList','owner')+f"{len(ordered)}\n(\n"+''.join(f"{f['own']}\n" for f in ordered)+")\n")
(poly/'neighbour').write_text(head('labelList','neighbour')+f"{len(internal)}\n(\n"+''.join(f"{f['nei']}\n" for f in internal)+")\n")
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-4500:])

# -- cell 9 -------------------------------------------------------------------------
# The corrected orientation exposes three genuinely folded cells at the rounded leading edge near the 
# Approximate mapping Jacobian at every coarse cell centre.
neg=[]; vals=[]
for i in range(NI):
 for j in range(N_RAD):
  for k in range(N_PITCH):
   p=np.array([[[points[vid(i+di,j+dj,k+dk)] for dk in (0,1)] for dj in (0,1)] for di in (0,1)])
   du=p[1].mean(axis=(0,1))-p[0].mean(axis=(0,1)); dr=p[:,1].mean(axis=(0,1))-p[:,0].mean(axis=(0,1)); de=p[:,:,1].mean(axis=(0,1))-p[:,:,0].mean(axis=(0,1))
   det=np.linalg.det(np.stack((du,dr,de),axis=1)); vals.append(det)
   if det<=0: neg.append((i,j,k,det,stations[i],stations[i+1]))
print('negative approximate Jacobians:',neg)
print('minimum/median determinant',min(vals),np.median(vals))

# -- cell 10 ------------------------------------------------------------------------
# The fold is intrinsic to an H-grid around a NACA square-root nose at 65° stagger, not a coarse-resol
# Revised explicit aerofoil assumption: NACA-style 4%/40% camber line with
# symmetric parabolic thickness, sharp at LE/TE, maximum 6% of chord.
def section_xy(q,side):
    q=np.asarray(q,float)
    yc=np.where(q<CAMBER_POS,CAMBER/CAMBER_POS**2*(2*CAMBER_POS*q-q*q),CAMBER/(1-CAMBER_POS)**2*((1-2*CAMBER_POS)+2*CAMBER_POS*q-q*q))
    yt=0.5*THICKNESS*4*q*(1-q)
    return CHORD*q, CHORD*(yc+side*yt)
qq=np.linspace(0,1,2001); Xu,Yu=section_xy(qq,+1); Xl,Yl=section_xy(qq,-1); thick=np.hypot(Xu-Xl,Yu-Yl)
print(f'revised section chord {np.hypot(Xu[-1]-Xu[0],Yu[-1]-Yu[0])*1e3:.3f} mm; max thickness {thick.max()*1e3:.3f} mm = {100*thick.max()/CHORD:.3f}%')
print('leading and trailing edges are sharp (assumption forced by single-block high-stagger topology)')

# -- cell 11 ------------------------------------------------------------------------
# I’ll regenerate the coarse point field with the revised section; connectivity and patch ownership re
points=np.array([passage_point(i,j,k) for i in range(NI+1) for j in range(N_RAD+1) for k in range(N_PITCH+1)])
(poly/'points').write_text(head('vectorField','points')+f"{len(points)}\n(\n"+''.join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x,y,z in points)+")\n")
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-4200:])

# -- cell 12 ------------------------------------------------------------------------
# The geometry defects are gone: volumes and pyramids are valid, and non-orthogonality passes. The rem
rr=subprocess.run(['renumberMesh','-overwrite'],text=True,capture_output=True)
print(rr.stdout[-1800:]); print('renumber returncode',rr.returncode)
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-3300:])

# -- cell 13 ------------------------------------------------------------------------
# The coarse mesh now passes the binding check. I’ll render the actual hexahedral volume with edges, n
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
conn=np.hstack([np.column_stack((np.full(len(cells),8),np.array(cells))).ravel()])
grid=pv.UnstructuredGrid(conn,np.full(len(cells),pv.CellType.HEXAHEDRON),points)
pl=pv.Plotter(off_screen=True,window_size=(1000,700)); pl.set_background('white')
pl.add_mesh(grid,style='wireframe',color='black',line_width=0.45,opacity=0.45)
pl.add_mesh(grid.extract_surface(),color='lightsteelblue',opacity=0.38,show_edges=False)
pl.camera_position=[(0.12,-0.14,0.11),(0.008,0.066,0.009),(0,0,1)]; pl.add_axes(); pl.show(screenshot='coarse_passage_mesh.png')
print(grid, '\ncell bounds',grid.bounds)

# -- cell 14 ------------------------------------------------------------------------
# The rendered volume is continuous and shows the intended spanwise twist and interblade contraction. 
print('Wire',inspect.signature(bd.Wire)); print('Face',inspect.signature(bd.Face)); print('Solid',inspect.signature(bd.Solid)); print('Solid.make_solid?',hasattr(bd.Solid,'make_solid'))
print('Shape.fix',hasattr(bd.Shape,'fix'), 'Shell sew?', [n for n in dir(bd.Shell) if 'sew' in n.lower() or 'fix' in n.lower()])

# -- cell 15 ------------------------------------------------------------------------
# The test construction itself succeeded; only the queried `is_closed` property is not present. I’ll r
def tri_face(a,b,c):
    e1=bd.Edge.make_line(a,b); e2=bd.Edge.make_line(b,c); e3=bd.Edge.make_line(c,a)
    wire=bd.Wire([e1,e2,e3])
    return bd.Face(wire)
a=(0,0,0); b=(1,0,0); c=(0,1,0); d=(0,0,1)
test_faces=[tri_face(a,c,b),tri_face(a,b,d),tri_face(a,d,c),tri_face(b,c,d)]
test_shell=bd.Shell(test_faces); test_solid=bd.Solid(test_shell)
print('shell valid',test_shell.is_valid,'solid valid',test_solid.is_valid,'volume',test_solid.volume,'faces',len(test_solid.faces()))

# -- cell 16 ------------------------------------------------------------------------
# The sewing route is valid. I’ll now create the passage CAD as one closed build123d solid from the ex
cad_patch_faces={p:[] for p in patch_order}; all_cad_faces=[]
for p in patch_order:
 for fq in patches[p]:
  v=fq['v']; xyz=[tuple(points[n]) for n in v]
  f1=tri_face(xyz[0],xyz[1],xyz[2]); f2=tri_face(xyz[0],xyz[2],xyz[3])
  cad_patch_faces[p].extend((f1,f2)); all_cad_faces.extend((f1,f2))
fluid_shell=bd.Shell(all_cad_faces)
fluid=bd.Solid(fluid_shell,label='one_blade_passage')
print('CAD solid valid:',fluid.is_valid,'faces:',len(fluid.faces()),'volume:',fluid.volume,'m^3')
print('mesh/CAD volume relative difference:',abs(4.38313e-5-fluid.volume)/fluid.volume)
print('patch CAD faces:',{p:len(fs) for p,fs in cad_patch_faces.items()})

# -- cell 17 ------------------------------------------------------------------------
# The sewn CAD solid is valid and matches the finite-volume mesh volume to 0.00019%. I’ll inspect the 
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
import inspect
print(inspect.signature(export_patches)); print('STEP unit options:',list(bd.Unit))

# -- cell 18 ------------------------------------------------------------------------
# The exporter correctly rejected pre-sewing face handles: OCC rebuilt equivalent faces when it made t
def ckey(face): return tuple(round(v,12) for v in face.center())
label_by_key={ckey(f):p for p,fs in cad_patch_faces.items() for f in fs}
solid_patch_faces={p:[] for p in patch_order}
for f in fluid.faces():
 key=ckey(f)
 if key not in label_by_key: raise RuntimeError(f'unmatched sewn face {key}')
 solid_patch_faces[label_by_key[key]].append(f)
print({p:len(fs) for p,fs in solid_patch_faces.items()},'total',sum(map(len,solid_patch_faces.values())))

# -- cell 19 ------------------------------------------------------------------------
# All 1,536 sewn faces map uniquely back to their creation-time labels. I’ll now perform the required 
bd.export_step(fluid,'one_blade_passage.step',unit=bd.Unit.M)
export_info=export_patches(fluid,solid_patch_faces,tolerance=2.5e-4)
print('STEP bytes',Path('one_blade_passage.step').stat().st_size)

# -- cell 20 ------------------------------------------------------------------------
# The coarse mesh and closed patch union are both validated. I’ll now generate the final mesh at twice
N_UP,N_BLADE,N_DOWN,N_RAD,N_PITCH=12,24,12,12,16
stations=[('up',t/N_UP) for t in range(N_UP)] + [('blade',0.5*(1-np.cos(np.pi*t/N_BLADE))) for t in range(N_BLADE)] + [('down',t/N_DOWN) for t in range(N_DOWN+1)]
NI=len(stations)-1
points=np.array([passage_point(i,j,k) for i in range(NI+1) for j in range(N_RAD+1) for k in range(N_PITCH+1)])
cells=[]
for i in range(NI):
 for j in range(N_RAD):
  for k in range(N_PITCH): cells.append([vid(i,j,k),vid(i+1,j,k),vid(i+1,j+1,k),vid(i,j+1,k),vid(i,j,k+1),vid(i+1,j,k+1),vid(i+1,j+1,k+1),vid(i,j+1,k+1)])
face_slots={}; face_data=[]
for ci,c in enumerate(cells):
 for lf in local_faces:
  f=[c[n] for n in lf]; key=tuple(sorted(f))
  if key in face_slots: face_data[face_slots[key]]['nei']=ci
  else: face_slots[key]=len(face_data); face_data.append({'v':f,'own':ci,'nei':None})
patches={p:[] for p in patch_order}; internal=[]
for f in face_data:
 if f['nei'] is not None: internal.append(f); continue
 ids=[ijk(v) for v in f['v']]; Is={a[0] for a in ids}; Js={a[1] for a in ids}; Ks={a[2] for a in ids}
 if Is=={0}: p='inlet'
 elif Is=={NI}: p='outlet'
 elif Js=={0}: p='hub'
 elif Js=={N_RAD}: p='shroud'
 elif Ks=={0}: p='blade_pressure' if min(a[0] for a in ids)>=N_UP and max(a[0] for a in ids)<=N_UP+N_BLADE else 'periodic_1'
 elif Ks=={N_PITCH}: p='blade_suction' if min(a[0] for a in ids)>=N_UP and max(a[0] for a in ids)<=N_UP+N_BLADE else 'periodic_2'
 else: raise RuntimeError('unclassified boundary face')
 patches[p].append(f)
ordered=internal+sum((patches[p] for p in patch_order),[])
(poly/'points').write_text(head('vectorField','points')+f"{len(points)}\n(\n"+''.join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x,y,z in points)+")\n")
(poly/'faces').write_text(head('faceList','faces')+f"{len(ordered)}\n(\n"+''.join(f"4({' '.join(map(str,f['v']))})\n" for f in ordered)+")\n")
(poly/'owner').write_text(head('labelList','owner')+f"{len(ordered)}\n(\n"+''.join(f"{f['own']}\n" for f in ordered)+")\n")
(poly/'neighbour').write_text(head('labelList','neighbour')+f"{len(internal)}\n(\n"+''.join(f"{f['nei']}\n" for f in internal)+")\n")
start=len(internal); entries=[]
for p in patch_order:
 typ='wall' if p in ('hub','shroud','blade_pressure','blade_suction') else 'patch'
 entries.append(f"{p}\n{{\n type {typ};\n nFaces {len(patches[p])};\n startFace {start};\n}}\n"); start+=len(patches[p])
(poly/'boundary').write_text(head('polyBoundaryMesh','boundary')+f"{len(patch_order)}\n(\n"+''.join(entries)+")\n")
subprocess.run(['renumberMesh','-overwrite'],check=True,stdout=subprocess.DEVNULL)
print(f'final mesh: {len(points)} points, {len(cells)} hex cells; patch faces',{p:len(patches[p]) for p in patch_order})

# -- cell 21 ------------------------------------------------------------------------
# I’ll check the refined final mesh before rebuilding the matching final CAD tessellation.
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-4300:])

# -- cell 22 ------------------------------------------------------------------------
# The refined grid fails only seven skewness cells because cosine clustering creates very short first/
stations=[('up',t/N_UP) for t in range(N_UP)] + [('blade',t/N_BLADE) for t in range(N_BLADE)] + [('down',t/N_DOWN) for t in range(N_DOWN+1)]
points=np.array([passage_point(i,j,k) for i in range(NI+1) for j in range(N_RAD+1) for k in range(N_PITCH+1)])
# renumberMesh changed connectivity in files, so update geometry there via transform-safe point order only is invalid;
# restore original structured connectivity files, then canonicalize again.
(poly/'points').write_text(head('vectorField','points')+f"{len(points)}\n(\n"+''.join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x,y,z in points)+")\n")
(poly/'faces').write_text(head('faceList','faces')+f"{len(ordered)}\n(\n"+''.join(f"4({' '.join(map(str,f['v']))})\n" for f in ordered)+")\n")
(poly/'owner').write_text(head('labelList','owner')+f"{len(ordered)}\n(\n"+''.join(f"{f['own']}\n" for f in ordered)+")\n")
(poly/'neighbour').write_text(head('labelList','neighbour')+f"{len(internal)}\n(\n"+''.join(f"{f['nei']}\n" for f in internal)+")\n")
subprocess.run(['renumberMesh','-overwrite'],check=True,stdout=subprocess.DEVNULL)
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-3000:])

# -- cell 23 ------------------------------------------------------------------------
# The uniform refined mesh passes with much better skewness (1.28) and aspect ratio (25.4). I’ll rebui
cad_patch_faces={p:[] for p in patch_order}; all_cad_faces=[]
for p in patch_order:
 for fq in patches[p]:
  v=fq['v']; xyz=[tuple(points[n]) for n in v]
  f1=tri_face(xyz[0],xyz[1],xyz[2]); f2=tri_face(xyz[0],xyz[2],xyz[3])
  cad_patch_faces[p].extend((f1,f2)); all_cad_faces.extend((f1,f2))
fluid_shell=bd.Shell(all_cad_faces); fluid=bd.Solid(fluid_shell,label='one_blade_passage')
label_by_key={ckey(f):p for p,fs in cad_patch_faces.items() for f in fs}; solid_patch_faces={p:[] for p in patch_order}
for f in fluid.faces(): solid_patch_faces[label_by_key[ckey(f)]].append(f)
print('final CAD valid',fluid.is_valid,'faces',len(fluid.faces()),'volume',fluid.volume)
bd.export_step(fluid,'one_blade_passage.step',unit=bd.Unit.M)
export_info=export_patches(fluid,solid_patch_faces,tolerance=2.5e-4)
print('STEP bytes',Path('one_blade_passage.step').stat().st_size)

# -- cell 24 ------------------------------------------------------------------------
# I’ll perform the final numerical audit directly on the delivered refined mesh and blade definitions,
# Geometry audit against requested values.
def measured_stagger(r):
 x0,t0=blade_surface(0.0,r,0,0); x1,t1=blade_surface(1.0,r,0,0)
 return np.rad2deg(np.arctan2(r*float(t1-t0),float(x1-x0)))
all_r=np.hypot(points[:,1],points[:,2])
qtest=np.linspace(0,1,10001); xu,yu=section_xy(qtest,+1); xl,yl=section_xy(qtest,-1)
print(f'mesh radii min/max: {all_r.min()*1e3:.9f}/{all_r.max()*1e3:.9f} mm (45/90 requested)')
print(f'measured root/tip stagger: {measured_stagger(HUB_R):.9f}/{measured_stagger(TIP_R):.9f} deg; twist {measured_stagger(TIP_R)-measured_stagger(HUB_R):.9f} deg')
print(f'measured chord: {CHORD*1e3:.9f} mm; max thickness: {np.max(np.hypot(xu-xl,yu-yl))*1e3:.9f} mm ({100*np.max(np.hypot(xu-xl,yu-yl))/CHORD:.6f}%)')
in_x=points[[vid(0,j,k) for j in range(N_RAD+1) for k in range(N_PITCH+1)],0]; out_x=points[[vid(NI,j,k) for j in range(N_RAD+1) for k in range(N_PITCH+1)],0]
print(f'inlet plane x/range: {in_x.mean()*1e3:.6f} mm / {np.ptp(in_x):.3g} m; outlet: {out_x.mean()*1e3:.6f} mm / {np.ptp(out_x):.3g} m')
print(f'root-TE axial downstream distance: {(X_OUTLET-CHORD*np.cos(np.deg2rad(ROOT_STAGGER_DEG)))*1e3:.6f} mm (one 22 mm chord)')
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
for line in r.stdout.splitlines():
 if any(s in line for s in ['cells:','Max aspect ratio','non-orthogonality Max','Max skewness','Mesh OK']): print(line.strip())
