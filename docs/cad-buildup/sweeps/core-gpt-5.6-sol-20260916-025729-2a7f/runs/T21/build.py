"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the internal air volume in metres as a 160 × 100 × 40 mm cavity, subtract the four 10 mm 
import inspect, build123d as bd
print('build123d', getattr(bd, '__version__', 'installed'))
print('Shape.tessellate', inspect.signature(bd.Shape.tessellate))
print('Box', inspect.signature(bd.Box), 'Cylinder', inspect.signature(bd.Cylinder))

# -- cell 2 -------------------------------------------------------------------------
# The installed API supports direct tessellation, so I’ll now create the coarse algebraic fluid solid 
from pathlib import Path
import numpy as np, pyvista as pv, matplotlib.pyplot as plt
L, W, H = 0.160, 0.100, 0.040
WALL = 0.003
STAND_D, STAND_H = 0.010, 0.025
HOLE_D, HOLE_DEPTH = 0.004, 0.012
INSET = 0.015
VENT_W, VENT_H, VENT_Z0 = 0.030, 0.020, 0.010
centers = [(INSET,INSET),(INSET,W-INSET),(L-INSET,INSET),(L-INSET,W-INSET)]
cavity = bd.Box(L,W,H, align=(bd.Align.MIN,bd.Align.MIN,bd.Align.MIN))
annuli=[]
for x,y in centers:
    outer = bd.Pos(x,y,0) * bd.Cylinder(STAND_D/2,STAND_H,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
    hole = bd.Pos(x,y,STAND_H-HOLE_DEPTH) * bd.Cylinder(HOLE_D/2,HOLE_DEPTH,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
    annuli.append(outer-hole)
standoffs = annuli[0].fuse(*annuli[1:])
fluid = cavity-standoffs
expected_volume=L*W*H-4*(np.pi*(STAND_D/2)**2*STAND_H-np.pi*(HOLE_D/2)**2*HOLE_DEPTH)
bb=fluid.bounding_box()
print(f'envelope measured = {bb.size.X:.6f} x {bb.size.Y:.6f} x {bb.size.Z:.6f} m; requested = {L:.6f} x {W:.6f} x {H:.6f} m')
print(f'fluid volume measured = {fluid.volume:.12g} m^3; analytic = {expected_volume:.12g} m^3; rel.err = {(fluid.volume/expected_volume-1):.3e}')
verts,tris=fluid.tessellate(0.001,0.15)
pts=np.array([[v.X,v.Y,v.Z] for v in verts]); faces=np.hstack([np.full((len(tris),1),3),np.array(tris)]).ravel()
poly=pv.PolyData(pts,faces)
pl=pv.Plotter(off_screen=True,window_size=(900,650)); pl.add_mesh(poly,color='lightblue',opacity=0.55,show_edges=True,edge_color='navy')
pl.camera_position=[(0.24,-0.19,0.15),(0.08,0.05,0.015),(0,0,1)]; pl.add_axes(); pl.show(screenshot='coarse_geometry.png')
print('coarse tessellation:',poly.n_cells,'triangles; screenshot coarse_geometry.png')

# -- cell 3 -------------------------------------------------------------------------
# The coarse view confirms all four annular standoffs and their blind cavities. For a conformal volume
import gmsh, math
gmsh.initialize()
gmsh.model.add('enclosure_air')
occ=gmsh.model.occ
vbox=occ.addBox(0,0,0,L,W,H)
ann_tags=[]
for x,y in centers:
    out=occ.addCylinder(x,y,0,0,0,STAND_H,STAND_D/2)
    inn=occ.addCylinder(x,y,STAND_H-HOLE_DEPTH,0,0,HOLE_DEPTH,HOLE_D/2)
    cut,_=occ.cut([(3,out)],[(3,inn)],removeObject=True,removeTool=True)
    ann_tags.extend(cut)
fluid_dimtags,_=occ.cut([(3,vbox)],ann_tags,removeObject=True,removeTool=True)
vent=occ.addRectangle(0,(W-VENT_W)/2,0,VENT_H,VENT_W)
occ.rotate([(2,vent)],0,0,0,0,1,0,-math.pi/2)
occ.translate([(2,vent)],0,0,VENT_Z0)
frag,_=occ.fragment(fluid_dimtags,[(2,vent)],removeObject=True,removeTool=True)
occ.synchronize()
vols=gmsh.model.getEntities(3); surfs=gmsh.model.getBoundary(vols,oriented=False,recursive=False)
print('volumes',vols,'boundary surface count',len(surfs))
for d,t in surfs:
    typ=gmsh.model.getType(d,t); com=occ.getCenterOfMass(d,t); area=occ.getMass(d,t)
    print(f's{t:2d} {typ:10s} area={area:.9g} com=({com[0]:.5f},{com[1]:.5f},{com[2]:.5f})')

# -- cell 4 -------------------------------------------------------------------------
# The imprinted CAD has exactly the expected 23 boundary faces: 1 top, 1 vent, 5 remaining enclosure w
surface_info={t:(gmsh.model.getType(2,t),occ.getMass(2,t),occ.getCenterOfMass(2,t)) for _,t in surfs}
def near(a,b,tol=1e-9): return abs(a-b)<tol
open_top=[t for t,(typ,a,c) in surface_info.items() if typ=='Plane' and near(a,L*W) and near(c[2],H)]
vent_faces=[t for t,(typ,a,c) in surface_info.items() if typ=='Plane' and near(a,VENT_W*VENT_H) and near(c[0],0)]
stand_cyl=[t for t,(typ,a,c) in surface_info.items() if typ=='Cylinder' and near(a,math.pi*STAND_D*STAND_H)]
stand_tops=[t for t,(typ,a,c) in surface_info.items() if typ=='Plane' and near(a,math.pi*((STAND_D/2)**2-(HOLE_D/2)**2))]
hole_cyl=[t for t,(typ,a,c) in surface_info.items() if typ=='Cylinder' and near(a,math.pi*HOLE_D*HOLE_DEPTH)]
hole_bottoms=[t for t,(typ,a,c) in surface_info.items() if typ=='Plane' and near(a,math.pi*(HOLE_D/2)**2)]
used=set(open_top+vent_faces+stand_cyl+stand_tops+hole_cyl+hole_bottoms)
walls_floor=[t for _,t in surfs if t not in used]
groups={'openTop':open_top,'vent':vent_faces,'enclosureWallsAndFloor':walls_floor,'standoffOuterSurfaces':stand_cyl+stand_tops,'blindHoleSurfaces':hole_cyl+hole_bottoms}
for name,tags in groups.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name); print(name,tags)
vg=gmsh.model.addPhysicalGroup(3,[t for _,t in vols]); gmsh.model.setPhysicalName(3,vg,'air')
# Curvature-aware refinement fields on semantic surfaces
fd=gmsh.model.mesh.field.add('Distance'); gmsh.model.mesh.field.setNumbers(fd,'FacesList',stand_cyl+stand_tops)
fs=gmsh.model.mesh.field.add('Threshold'); gmsh.model.mesh.field.setNumber(fs,'InField',fd); gmsh.model.mesh.field.setNumber(fs,'SizeMin',0.0015); gmsh.model.mesh.field.setNumber(fs,'SizeMax',0.004); gmsh.model.mesh.field.setNumber(fs,'DistMin',0.002); gmsh.model.mesh.field.setNumber(fs,'DistMax',0.012)
fh=gmsh.model.mesh.field.add('Distance'); gmsh.model.mesh.field.setNumbers(fh,'FacesList',hole_cyl+hole_bottoms)
fht=gmsh.model.mesh.field.add('Threshold'); gmsh.model.mesh.field.setNumber(fht,'InField',fh); gmsh.model.mesh.field.setNumber(fht,'SizeMin',0.0008); gmsh.model.mesh.field.setNumber(fht,'SizeMax',0.004); gmsh.model.mesh.field.setNumber(fht,'DistMin',0.001); gmsh.model.mesh.field.setNumber(fht,'DistMax',0.006)
fmin=gmsh.model.mesh.field.add('Min'); gmsh.model.mesh.field.setNumbers(fmin,'FieldsList',[fs,fht]); gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0007); gmsh.option.setNumber('Mesh.MeshSizeMax',0.004)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',24); gmsh.option.setNumber('Mesh.Algorithm3D',10); gmsh.option.setNumber('Mesh.Optimize',1); gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize('Netgen')
gmsh.write('enclosure_air.msh')
types,elemTags,nodeTags=gmsh.model.mesh.getElements(3)
print('3D elements by type:',[(gmsh.model.mesh.getElementProperties(t)[0],len(e)) for t,e in zip(types,elemTags)])

# -- cell 5 -------------------------------------------------------------------------
# The Gmsh volume mesh contains 102,401 tetrahedra after optimization. Before converting it into the O
tri_dir=Path('constant/triSurface'); tri_dir.mkdir(parents=True,exist_ok=True)
all_node_tags,all_xyz,_=gmsh.model.mesh.getNodes()
xyz_map={int(tag):all_xyz[3*i:3*i+3] for i,tag in enumerate(all_node_tags)}
signed_bnd=gmsh.model.getBoundary(vols,combined=False,oriented=True,recursive=False)
orient={abs(t):(1 if t>0 else -1) for d,t in signed_bnd}
def surface_triangles(tags):
    out=[]
    for st in tags:
        etypes,etags,enodes=gmsh.model.mesh.getElements(2,st)
        for et,nds in zip(etypes,enodes):
            name,dim,order,nper,_,_=gmsh.model.mesh.getElementProperties(et)
            if name.startswith('Triangle') and nper==3:
                arr=np.array(nds,dtype=np.int64).reshape(-1,3)
                if orient.get(st,1)<0: arr=arr[:,[0,2,1]]
                out.extend(arr.tolist())
    return out
def write_ascii_stl(path,name,triangles):
    area=0.0
    with open(path,'w') as f:
        f.write(f'solid {name}\n')
        for ids in triangles:
            p=np.array([xyz_map[i] for i in ids]); cr=np.cross(p[1]-p[0],p[2]-p[0]); mag=np.linalg.norm(cr); area+=mag/2
            n=cr/mag
            f.write(f' facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n')
            for q in p: f.write(f'   vertex {q[0]:.12g} {q[1]:.12g} {q[2]:.12g}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')
    return area
for name,tags in groups.items():
    ts=surface_triangles(tags); area=write_ascii_stl(tri_dir/f'{name}.stl',name,ts)
    print(f'{name}.stl: {len(ts)} triangles, triangulated area {area:.9g} m^2')
print('Exported',len(list(tri_dir.glob('*.stl'))),'patch STL files before OpenFOAM conversion.')

# -- cell 6 -------------------------------------------------------------------------
# All five required STL patch files now exist with areas matching the CAD faces. I’ll convert the phys
import subprocess, textwrap, os
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text(textwrap.dedent('''\
FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
'''))
r=subprocess.run(['gmshToFoam','enclosure_air.msh'],text=True,capture_output=True)
print(r.stdout[-5000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)
print('\nBoundary file patch lines:')
btxt=Path('constant/polyMesh/boundary').read_text()
for line in btxt.splitlines():
    if line.strip() in list(groups)+['air']: print(line.strip())

# -- cell 7 -------------------------------------------------------------------------
# The converter retained all five requested names. Its “undefined faces” message is a normal intermedi
print(Path('constant/polyMesh/boundary').read_text())
r=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('\n--- checkMesh tail ---\n',r.stdout[-7000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The mesh boundary file has exactly five patches and no `defaultFaces`. The extended check requested 
Path('system/fvSchemes').write_text(textwrap.dedent('''\
FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''))
Path('system/fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
r=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print(r.stdout[-9000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The full topology and geometry check passes: one connected region, 102,401 tetrahedra, maximum non-o
# Recover dimensions from the actual imprinted OCC surfaces
vbb=occ.getBoundingBox(2,vent_faces[0]); vdim=(vbb[4]-vbb[1],vbb[5]-vbb[2]); vz0=vbb[2]
sbb=occ.getBoundingBox(2,stand_cyl[0]); sdiam=max(sbb[3]-sbb[0],sbb[4]-sbb[1]); sheight=sbb[5]-sbb[2]
hbb=occ.getBoundingBox(2,hole_cyl[0]); hdiam=max(hbb[3]-hbb[0],hbb[4]-hbb[1]); hdepth=hbb[5]-hbb[2]
actual_centers=sorted([(surface_info[t][2][0],surface_info[t][2][1]) for t in stand_cyl])
print(f'internal envelope measured {bb.size.X:.6f} x {bb.size.Y:.6f} x {bb.size.Z:.6f} m; requested 0.160 x 0.100 x 0.040 m')
print(f'standoff measured diameter {sdiam:.6f} m, height {sheight:.6f} m; requested 0.010, 0.025 m')
print(f'blind hole measured diameter {hdiam:.6f} m, depth {hdepth:.6f} m; requested 0.004, 0.012 m')
print('standoff centres measured:',actual_centers,'m; requested 0.015 m inset from internal corners')
print(f'vent measured width(y) {vdim[0]:.6f} m, height(z) {vdim[1]:.6f} m, lower edge {vz0:.6f} m; requested 0.030, 0.020, 0.010 m')
colors={'openTop':'deepskyblue','vent':'limegreen','enclosureWallsAndFloor':'lightgray','standoffOuterSurfaces':'orange','blindHoleSurfaces':'magenta'}
pl2=pv.Plotter(off_screen=True,window_size=(950,680))
for name,tags in groups.items():
    ts=surface_triangles(tags); unique=sorted(set(i for tri in ts for i in tri)); imap={n:i for i,n in enumerate(unique)}
    p=np.array([xyz_map[i] for i in unique]); f=np.hstack([np.full((len(ts),1),3),np.array([[imap[i] for i in tri] for tri in ts])]).ravel()
    pl2.add_mesh(pv.PolyData(p,f),color=colors[name],opacity=0.28 if name=='openTop' else 0.82,show_edges=False,label=name)
pl2.add_legend(); pl2.add_axes(); pl2.camera_position=[(0.24,-0.19,0.14),(0.08,0.05,0.015),(0,0,1)]
pl2.show(screenshot='final_patch_mesh.png')
print('Rendered actual final surface mesh to final_patch_mesh.png')

# -- cell 10 ------------------------------------------------------------------------
# The final patch rendering shows the vent cutout, standoff exteriors, and blind-hole interiors as dis
def cylinder_measures(st):
    curves=gmsh.model.getBoundary([(2,st)],oriented=False,recursive=False)
    circumferences=[occ.getMass(1,t) for d,t in curves if d==1]
    circ=sum(circumferences)/len(circumferences)
    area=occ.getMass(2,st)
    return circ/math.pi, area/circ
sd,sh=cylinder_measures(stand_cyl[0]); hd,hdep=cylinder_measures(hole_cyl[0])
mesh_vol=0.000632807; cad_vol=fluid.volume
print(f'standoff diameter/height recovered = {sd:.9f}, {sh:.9f} m; requested = 0.010000000, 0.025000000 m')
print(f'blind-hole diameter/depth recovered = {hd:.9f}, {hdep:.9f} m; requested = 0.004000000, 0.012000000 m')
print(f'CAD fluid volume = {cad_vol:.12g} m^3; checkMesh volume = {mesh_vol:.12g} m^3; difference = {(mesh_vol/cad_vol-1)*100:.4f}%')
print('STLs:',sorted(str(p) for p in tri_dir.glob('*.stl')))
gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# The STL winding check found 322 same-direction shared edges. This is an export-orientation issue, no
def read_ascii_tris(path):
    vs=[]
    for line in Path(path).read_text().splitlines():
        s=line.strip().split()
        if s and s[0]=='vertex': vs.append(tuple(map(float,s[1:4])))
    return np.array(vs).reshape(-1,3,3)
def expected_outward(name,c):
    if name=='openTop': return np.array([0.,0.,1.])
    if name=='vent': return np.array([-1.,0.,0.])
    if name=='enclosureWallsAndFloor':
        distances=[(c[2],np.array([0.,0.,-1.])),(c[0],np.array([-1.,0.,0.])),(L-c[0],np.array([1.,0.,0.])),(c[1],np.array([0.,-1.,0.])),(W-c[1],np.array([0.,1.,0.]))]
        return min(distances,key=lambda q:q[0])[1]
    x,y=min(centers,key=lambda q:(c[0]-q[0])**2+(c[1]-q[1])**2)
    if name=='standoffOuterSurfaces':
        return np.array([0.,0.,-1.]) if abs(c[2]-STAND_H)<1e-7 else np.array([x-c[0],y-c[1],0.])
    return np.array([0.,0.,-1.]) if abs(c[2]-(STAND_H-HOLE_DEPTH))<1e-7 else np.array([c[0]-x,c[1]-y,0.])
for name in groups:
    path=tri_dir/f'{name}.stl'; arr=read_ascii_tris(path); flipped=0
    for i,p in enumerate(arr):
        cr=np.cross(p[1]-p[0],p[2]-p[0]); exp=expected_outward(name,p.mean(axis=0))
        if np.dot(cr,exp)<0: arr[i,[1,2]]=arr[i,[2,1]]; flipped+=1
    write_ascii_stl(path,name,[[j for j in range(3)]]) if False else None
    with open(path,'w') as f:
        f.write(f'solid {name}\n')
        for p in arr:
            cr=np.cross(p[1]-p[0],p[2]-p[0]); n=cr/np.linalg.norm(cr)
            f.write(f' facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n')
            for q in p: f.write(f'   vertex {q[0]:.12g} {q[1]:.12g} {q[2]:.12g}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')
    print(name,'triangles',len(arr),'flipped',flipped)
# Independent welded-edge audit across all patch STLs
from collections import defaultdict
edges=defaultdict(list); scaleq=10**11
for name in groups:
    for p in read_ascii_tris(tri_dir/f'{name}.stl'):
        q=[tuple(np.rint(v*scaleq).astype(np.int64)) for v in p]
        for a,b in [(q[0],q[1]),(q[1],q[2]),(q[2],q[0])]: edges[tuple(sorted((a,b)))].append((a,b))
free=sum(len(v)==1 for v in edges.values()); nonman=sum(len(v)!=2 for v in edges.values()); same=sum(len(v)==2 and v[0]==v[1] for v in edges.values())
print('union edge audit: free =',free,'non-manifold =',nonman,'same-direction =',same,'of',len(edges),'unique edges')
