"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid region—not the solid—as a 120 mm × 6 mm straight channel plus four 6 mm-wide cu
import build123d as bd, inspect
print('Polygon', inspect.signature(bd.Polygon))
print('Circle', inspect.signature(bd.Circle))
print('extrude', inspect.signature(bd.extrude))
print('fuse', inspect.signature(bd.Shape.fuse))

# -- cell 2 -------------------------------------------------------------------------
# The installed API supports direct planar primitives and extrusion. I’ll now build the coarse fluid s
import numpy as np
# Requested dimensions (metres)
L = 0.120
W = 0.006
N_LOOPS = 4
# Assumptions needed to define the otherwise unspecified teardrops
T = 0.001              # 2D extrusion thickness
BRANCH_W = 0.006       # bypass width, assumed equal to main channel
CELL_PITCH = 0.026
LOOP_SPAN = 0.022
LOOP_RISE = 0.014
X0 = 0.008
SAMPLES = 25

def capsule_tube(points, radius):
    pieces = []
    for p in points:
        pieces.append(bd.Pos(float(p[0]), float(p[1])) * bd.Circle(radius))
    for a, b in zip(points[:-1], points[1:]):
        d = b-a; n = radius*np.array([-d[1], d[0]])/np.linalg.norm(d)
        pieces.append(bd.Polygon(tuple(a+n), tuple(b+n), tuple(b-n), tuple(a-n)))
    out = pieces[0]
    for piece in pieces[1:]: out = out.fuse(piece)
    return out

main_2d = bd.Pos(0, -W/2) * bd.Rectangle(L, W, align=(bd.Align.MIN, bd.Align.MIN))
branch_2d = []
branch_centerlines = []
for i in range(N_LOOPS):
    xa = X0 + i*CELL_PITCH; xb = xa + LOOP_SPAN; sgn = 1 if i%2 == 0 else -1
    p0=np.array([xa,0.0]); p1=np.array([xa+0.002,sgn*LOOP_RISE]); p2=np.array([xb-0.008,sgn*LOOP_RISE]); p3=np.array([xb,0.0])
    ts=np.linspace(0,1,SAMPLES)
    pts=np.array([(1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3 for t in ts])
    branch_centerlines.append(pts)
    branch_2d.append(capsule_tube(pts, BRANCH_W/2))
fluid_2d = main_2d
for branch in branch_2d: fluid_2d = fluid_2d.fuse(branch)
fluid = bd.extrude(fluid_2d, amount=T)
bb=fluid.bounding_box()
print('solids:', len(fluid.solids()), 'faces:', len(fluid.faces()))
print('extent [m]:', bb.min, bb.max)
print('overall length measured/requested [m]:', bb.size.X, L)
print('main width requested [m]:', W, 'extrusion assumed [m]:', T)
print('fluid volume [m3]:', fluid.volume)

# -- cell 3 -------------------------------------------------------------------------
# The first solid check exposed two construction issues before any meshing: the planar sketch operands
print(type(main_2d), len(main_2d.faces()), type(main_2d.faces()[0]))
print(type(branch_2d[0]), len(branch_2d[0].faces()))
print('first branch area sum', sum(f.area for f in branch_2d[0].faces()))

# -- cell 4 -------------------------------------------------------------------------
# The diagnostics confirm that primitive sketches each contain valid faces, but iterative sketch-level
tube_faces=[]
for pts in branch_centerlines:
    for p in pts:
        tube_faces.append((bd.Pos(float(p[0]), float(p[1])) * bd.Circle(BRANCH_W/2)).faces()[0])
    for a,b in zip(pts[:-1],pts[1:]):
        d=b-a; n=(BRANCH_W/2)*np.array([-d[1],d[0]])/np.linalg.norm(d)
        tube_faces.append(bd.Polygon(tuple(a+n),tuple(b+n),tuple(b-n),tuple(a-n)).faces()[0])
fluid_face = main_2d.faces()[0].fuse(*tube_faces, tol=1e-9)
print(type(fluid_face), 'faces', len(fluid_face.faces()), 'area', fluid_face.area)
print('wires per face:', [len(f.wires()) for f in fluid_face.faces()])
fluid = bd.extrude(fluid_face, amount=T, dir=(0,0,1))
bb=fluid.bounding_box()
print('solid count',len(fluid.solids()),'z extent',bb.min.Z,bb.max.Z,'volume',fluid.volume)

# -- cell 5 -------------------------------------------------------------------------
# The one-shot union created a main connected face with the expected five boundary wires (one exterior
areas=sorted([(f.area, len(f.wires()),j) for j,f in enumerate(fluid_face.faces())], reverse=True)
print('largest ten (area, wires, index):')
for row in areas[:10]: print(row)
largest_face=fluid_face.faces()[areas[0][2]]
clean_fluid=bd.extrude(largest_face, amount=T, dir=(0,0,1))
print('clean solids/faces/volume:',len(clean_fluid.solids()),len(clean_fluid.faces()),clean_fluid.volume)
print('clean planar area:',largest_face.area,'holes:',len(largest_face.wires())-1)

# -- cell 6 -------------------------------------------------------------------------
# The geometry itself is intact; only the plotting helper used the wrong `Location` accessor and produ
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(12,4))
for wire in largest_face.wires():
    for edge in wire.edges():
        ep=np.array([(loc.position.X,loc.position.Y) for loc in edge.distribute_locations(20)])
        ax.plot(ep[:,0]*1000,ep[:,1]*1000,'k-',lw=1.3)
for i,pts in enumerate(branch_centerlines):
    ax.plot(pts[:,0]*1000,pts[:,1]*1000,'--',lw=.8,label='bypass centerline' if i==0 else None)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('Coarse Tesla-valve fluid region (top view)'); ax.grid(True,alpha=.25); ax.legend(loc='upper right')
plt.tight_layout(); plt.show()

# -- cell 7 -------------------------------------------------------------------------
# The coarse view has the intended topology: a continuous 6 mm straight passage and four alternating b
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
import inspect
print(inspect.signature(export_patches))
for j,f in enumerate(clean_fluid.faces()):
    n=f.normal_at(); c=f.center()
    if abs(n.X)>.9 or abs(n.Z)>.9:
        print(j,'area',f.area,'normal',tuple(round(q,3) for q in (n.X,n.Y,n.Z)),'center',tuple(round(q,5) for q in (c.X,c.Y,c.Z)))

# -- cell 8 -------------------------------------------------------------------------
# The orientation audit finds exactly one −X face of area 6 mm² (inlet), one +X face of the same area 
fluid = clean_fluid
all_faces=list(fluid.faces())
inlet=[f for f in all_faces if f.area>5.9e-6 and f.normal_at().X < -0.999]
outlet=[f for f in all_faces if f.area>5.9e-6 and f.normal_at().X > 0.999]
front_back=[f for f in all_faces if abs(f.normal_at().Z)>0.999]
used_ids={id(f) for f in inlet+outlet+front_back}
walls=[f for f in all_faces if id(f) not in used_ids]
print('patch face counts:',len(inlet),len(outlet),len(front_back),len(walls))
patch_report=export_patches(fluid, {'inlet':inlet,'outlet':outlet,'frontAndBack':front_back,'walls':walls}, tolerance=2.5e-4)

# -- cell 9 -------------------------------------------------------------------------
# For the coarse mesh I’ll use Gmsh’s OCC kernel to reproduce the validated planar profile from the sa
import gmsh, os
gmsh.initialize(); gmsh.model.add('teslaValve')
occ=gmsh.model.occ
# Main rectangle and four polygonal tubes from the validated centerline offsets
surfs=[(2,occ.addRectangle(0,-W/2,0,L,W))]
for pts in branch_centerlines:
    left=[]; right=[]
    for j,p in enumerate(pts):
        if j==0: d=pts[1]-pts[0]
        elif j==len(pts)-1: d=pts[-1]-pts[-2]
        else: d=pts[j+1]-pts[j-1]
        n=(BRANCH_W/2)*np.array([-d[1],d[0]])/np.linalg.norm(d)
        left.append(p+n); right.append(p-n)
    poly=np.vstack([left,right[::-1]])
    pTags=[occ.addPoint(float(x),float(y),0) for x,y in poly]
    lTags=[occ.addLine(pTags[j],pTags[(j+1)%len(pTags)]) for j in range(len(pTags))]
    loop=occ.addCurveLoop(lTags)
    surfs.append((2,occ.addPlaneSurface([loop])))
occ.synchronize()
fused,_=occ.fuse([surfs[0]],surfs[1:],removeObject=True,removeTool=True)
occ.synchronize()
plane_surfs=[tag for dim,tag in fused if dim==2]
print('fused planar surfaces:',plane_surfs)
extr=occ.extrude([(2,plane_surfs[0])],0,0,T,numElements=[1],recombine=True)
occ.synchronize()
vols=[tag for dim,tag in extr if dim==3]
all_s=gmsh.model.getEntities(2)
in_s=[]; out_s=[]; fb_s=[]; wall_s=[]
for _,tag in all_s:
    area=occ.getMass(2,tag); com=occ.getCenterOfMass(2,tag)
    # Surface type and extents distinguish broad 2D planes from vertical sides
    box=gmsh.model.getBoundingBox(2,tag); dx=box[3]-box[0]; dy=box[4]-box[1]; dz=box[5]-box[2]
    if dz < 1e-8: fb_s.append(tag)
    elif area > 5.9e-6 and dx < 1e-8:
        (in_s if com[0] < L/2 else out_s).append(tag)
    else: wall_s.append(tag)
for dim,tags,name in [(3,vols,'fluid'),(2,in_s,'inlet'),(2,out_s,'outlet'),(2,fb_s,'frontAndBack'),(2,wall_s,'walls')]:
    pg=gmsh.model.addPhysicalGroup(dim,tags); gmsh.model.setPhysicalName(dim,pg,name)
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0015); gmsh.option.setNumber('Mesh.MeshSizeMax',0.0015)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3)
gmsh.write('teslaValve.msh')
print('surface groups:',{'inlet':in_s,'outlet':out_s,'frontAndBack':len(fb_s),'walls':len(wall_s)})
print('mesh nodes/elements:',len(gmsh.model.mesh.getNodes()[0]),sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 10 ------------------------------------------------------------------------
# Gmsh’s all-at-once OCC union retained one surface per loop despite the caps, so I’m narrowing this t
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.model.add('teslaTopology'); occ=gmsh.model.occ
main_tag=occ.addRectangle(0,-W/2,0,L,W)
branch_tags=[]
for pts in branch_centerlines:
    left=[]; right=[]
    for j,p in enumerate(pts):
        d=pts[1]-pts[0] if j==0 else (pts[-1]-pts[-2] if j==len(pts)-1 else pts[j+1]-pts[j-1])
        n=(BRANCH_W/2)*np.array([-d[1],d[0]])/np.linalg.norm(d); left.append(p+n); right.append(p-n)
    poly=np.vstack([left,right[::-1]])
    pp=[occ.addPoint(float(x),float(y),0) for x,y in poly]
    ll=[occ.addLine(pp[j],pp[(j+1)%len(pp)]) for j in range(len(pp))]
    tube=occ.addPlaneSurface([occ.addCurveLoop(ll)])
    d0=occ.addDisk(float(pts[0,0]),float(pts[0,1]),0,BRANCH_W/2,BRANCH_W/2)
    d1=occ.addDisk(float(pts[-1,0]),float(pts[-1,1]),0,BRANCH_W/2,BRANCH_W/2)
    br,_=occ.fuse([(2,tube)],[(2,d0),(2,d1)],removeObject=True,removeTool=True)
    branch_tags.extend([t for dim,t in br if dim==2])
occ.synchronize()
current=[(2,main_tag)]
for bt in branch_tags:
    current,_=occ.fuse(current,[(2,bt)],removeObject=True,removeTool=True); occ.synchronize()
plane_surfs=[t for d,t in current if d==2]
print('branch pieces after local unions:',len(branch_tags),'final surfaces:',plane_surfs)
print('surface masses:',[occ.getMass(2,t) for t in plane_surfs])
gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# I’m switching to the shortest reliable path: export the already validated single build123d face (one
bd.export_step(largest_face,'teslaProfile.step')
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.model.add('teslaValve'); occ=gmsh.model.occ
imp=occ.importShapes('teslaProfile.step'); occ.synchronize()
ps=[t for d,t in imp if d==2]
print('imported surfaces:',ps,'area:',[occ.getMass(2,t) for t in ps])
if len(ps)!=1: raise RuntimeError('STEP profile is not one surface')
extr=occ.extrude([(2,ps[0])],0,0,T,numElements=[1],recombine=True); occ.synchronize()
vols=[t for d,t in extr if d==3]; in_s=[]; out_s=[]; fb_s=[]; wall_s=[]
for _,tag in gmsh.model.getEntities(2):
    area=occ.getMass(2,tag); com=occ.getCenterOfMass(2,tag); box=gmsh.model.getBoundingBox(2,tag); dx=box[3]-box[0]
    if area>1e-4: fb_s.append(tag)
    elif area>5.5e-6 and dx<1e-6: (in_s if com[0]<L/2 else out_s).append(tag)
    else: wall_s.append(tag)
print('patch counts:',len(in_s),len(out_s),len(fb_s),len(wall_s))
for dim,tags,name in [(3,vols,'fluid'),(2,in_s,'inlet'),(2,out_s,'outlet'),(2,fb_s,'frontAndBack'),(2,wall_s,'walls')]:
    pg=gmsh.model.addPhysicalGroup(dim,tags); gmsh.model.setPhysicalName(dim,pg,name)
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0015); gmsh.option.setNumber('Mesh.MeshSizeMax',0.0015); gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3); gmsh.write('teslaValve.msh')
print('nodes/cells:',len(gmsh.model.mesh.getNodes()[0]),sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 12 ------------------------------------------------------------------------
# The exact STEP face meshed successfully: one connected volume, 5,518 one-layer cells, and all four p
import subprocess, pathlib, re, os
pathlib.Path('system').mkdir(exist_ok=True)
pathlib.Path('system/controlDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
r=subprocess.run(['gmshToFoam','teslaValve.msh'],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-1000:])
if r.returncode: raise RuntimeError('gmshToFoam failed')
bp=pathlib.Path('constant/polyMesh/boundary'); txt=bp.read_text()
# Physical group is already named; only set the 2D patch type.
pos=txt.find('frontAndBack'); start=txt.find('{',pos); end=txt.find('}',start)
block=txt[start:end+1]; block=re.sub(r'type\s+patch\s*;', 'type empty;', block)
txt=txt[:start]+block+txt[end+1:]; bp.write_text(txt)
cm=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(cm.stdout[-4000:]); print(cm.stderr)
print('checkMesh return code:',cm.returncode)

# -- cell 13 ------------------------------------------------------------------------
# Conversion produced the intended 5,518 triangular prisms and named patches. `checkMesh` did not exam
pathlib.Path('system/fvSchemes').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n''')
pathlib.Path('system/fvSolution').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
cm=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(cm.stdout[-5000:]); print(cm.stderr); print('return code',cm.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The binding check now passes: one region, 5,518 prisms, correct 2D geometric/solution directions, ma
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.open('teslaValve.msh')
node_tags,coords,_=gmsh.model.mesh.getNodes(); xyz=np.asarray(coords).reshape(-1,3); idx={int(t):i for i,t in enumerate(node_tags)}
types,_,conns=gmsh.model.mesh.getElements(3)
pr=np.asarray(conns[list(types).index(6)]).reshape(-1,6)
tri=np.array([[idx[int(a)],idx[int(b)],idx[int(c)]] for a,b,c in pr[:,:3]])
gmsh.finalize()
fig,ax=plt.subplots(figsize=(12,4))
ax.triplot(xyz[:,0]*1000,xyz[:,1]*1000,tri,lw=.22,color='#1f4e79')
ax.set_aspect('equal'); ax.set_xlim(-2,122); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('Coarse OpenFOAM mesh: 5,518 one-layer triangular prisms'); ax.grid(True,alpha=.15)
plt.tight_layout(); plt.show()
