"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water as the union of the centred spherical cavity and six radial cylindrical passage
import build123d as bd
import math, os
# All dimensions are metres (request converted from mm once here).
CUBE_SIDE = 0.040
HOLLOW_R = 0.016
BALL_R = 0.015
HOLE_R = 0.012
HALF = CUBE_SIDE/2
cube = bd.Box(CUBE_SIDE, CUBE_SIDE, CUBE_SIDE, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER))
hollow = bd.Sphere(HOLLOW_R)
ball = bd.Sphere(BALL_R)
# Six radial cylinders, each explicitly associated with one cube face.
hole_px = bd.Pos(0,0,0) * bd.Cylinder(HOLE_R, HALF, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).rotate(bd.Axis.Y,90)
hole_nx = bd.Pos(0,0,0) * bd.Cylinder(HOLE_R, HALF, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).rotate(bd.Axis.Y,-90)
hole_py = bd.Pos(0,0,0) * bd.Cylinder(HOLE_R, HALF, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).rotate(bd.Axis.X,-90)
hole_ny = bd.Pos(0,0,0) * bd.Cylinder(HOLE_R, HALF, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).rotate(bd.Axis.X,90)
hole_pz = bd.Cylinder(HOLE_R, HALF, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
hole_nz = bd.Cylinder(HOLE_R, HALF, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).rotate(bd.Axis.X,180)
passages = hole_px + hole_nx + hole_py + hole_ny + hole_pz + hole_nz
water = ((hollow + passages) & cube) - ball
print('water volume', water.volume, 'm^3; bbox', water.bounding_box().min, water.bounding_box().max)
print('faces', len(water.faces()), 'solids', len(water.solids()))
for i,f in enumerate(water.faces()):
    print(i, f.geom_type, round(f.area,10), tuple(round(v,6) for v in f.center()))

# -- cell 2 -------------------------------------------------------------------------
# The Boolean is a single connected water solid with 21 boundary faces: 8 spherical remnants of the ho
import inspect, pyvista as pv
os.makedirs('constant/triSurface', exist_ok=True)
bd.export_stl(water, 'water_preview.stl', tolerance=0.00035, angular_tolerance=0.18)
meshprev = pv.read('water_preview.stl')
pl = pv.Plotter(off_screen=True, window_size=(850,700))
pl.add_mesh(meshprev, color='deepskyblue', opacity=0.55, show_edges=True, edge_color='navy')
pl.add_axes(); pl.view_isometric(); pl.show(screenshot='water_preview.png')
print('preview triangles', meshprev.n_cells)
for i,f in enumerate(water.faces()):
    if f.geom_type == bd.GeomType.CYLINDER:
        print('cylinder face',i,'axis',f.axis_of_rotation)
from IPython.display import display, Image
display(Image('water_preview.png'))

# -- cell 3 -------------------------------------------------------------------------
# The preview confirms the intended six-lobed exterior and the internal ball boundary. The two x-direc
from pathlib import Path
tri = Path('constant/triSurface')
# Clean only artifacts created by this build before deterministic export.
for p in tri.glob('*.stl'): p.unlink()
for p in [Path('water_preview.stl')]:
    if p.exists(): p.unlink()
faces = list(water.faces())
ball_faces = [f for f in faces if f.geom_type == bd.GeomType.SPHERE and f.area > 1e-3]
hollow_faces = [f for f in faces if f.geom_type == bd.GeomType.SPHERE and f.area < 1e-3]
opening_faces = {}
hole_faces = {}
axis_names = [(0,'x'),(1,'y'),(2,'z')]
for f in faces:
    c = f.center()
    if f.geom_type == bd.GeomType.PLANE:
        vals=[c.X,c.Y,c.Z]; k=max(range(3),key=lambda j:abs(vals[j])); opening_faces[('p' if vals[k]>0 else 'n')+axis_names[k][1]]=f
    elif f.geom_type == bd.GeomType.CYLINDER:
        d=f.axis_of_rotation.direction; vals=[d.X,d.Y,d.Z]; k=max(range(3),key=lambda j:abs(vals[j])); cv=[c.X,c.Y,c.Z][k]; hole_faces[('p' if cv>0 else 'n')+axis_names[k][1]]=f
patch_shapes={'ball':bd.Compound(ball_faces),'hollowWall':bd.Compound(hollow_faces)}
for n in ['px','nx','py','ny','pz','nz']:
    patch_shapes['hole_'+n]=hole_faces[n]
    patch_shapes['opening_'+n]=opening_faces[n]
for name,shape in patch_shapes.items():
    bd.export_stl(shape, str(tri/(name+'.stl')), tolerance=0.00012, angular_tolerance=0.10)
ball_r_check=math.sqrt(ball_faces[0].area/(4*math.pi))
opening_r_checks={n:math.sqrt(f.area/math.pi) for n,f in opening_faces.items()}
print('exported patches:', sorted(patch_shapes))
print(f'ball radius measured from area: {ball_r_check:.9f} m (requested {BALL_R:.9f})')
print('opening radii measured from area:', {k:round(v,9) for k,v in opening_r_checks.items()}, 'requested',HOLE_R)
print('hollow spherical pieces',len(hollow_faces),'combined area',sum(f.area for f in hollow_faces))
print('hole assignments', {k:tuple(round(x,6) for x in hole_faces[k].center()) for k in sorted(hole_faces)})

# -- cell 4 -------------------------------------------------------------------------
# All 14 patch files now exist before meshing. The ball radius and all six opening radii measure exact
import gmsh
bd.export_step(water, 'water.step')
gmsh.initialize()
gmsh.model.add('water')
gmsh.model.occ.importShapes('water.step')
gmsh.model.occ.synchronize()
vols=gmsh.model.getEntities(3); surfs=gmsh.model.getBoundary(vols, oriented=False, recursive=False)
print('gmsh volumes',vols,'surfaces',len(surfs))
for _,tag in surfs:
    typ=gmsh.model.getType(2,tag); com=gmsh.model.occ.getCenterOfMass(2,tag); area=gmsh.model.occ.getMass(2,tag)
    print(tag,typ,tuple(round(x,6) for x in com),round(area,10))
gmsh.finalize()

# -- cell 5 -------------------------------------------------------------------------
# Gmsh preserves the same 21-face topology, so patch tagging can be reproduced unambiguously by surfac
gmsh.initialize()
gmsh.option.setNumber('General.Terminal',1)
gmsh.model.add('water')
gmsh.model.occ.importShapes('water.step'); gmsh.model.occ.synchronize()
voltag=gmsh.model.getEntities(3)[0][1]
gsurf=gmsh.model.getBoundary([(3,voltag)],oriented=False)
gball=[]; ghollow=[]; gholes={}; gopens={}
for _,tag in gsurf:
    typ=gmsh.model.getType(2,tag); c=gmsh.model.occ.getCenterOfMass(2,tag); area=gmsh.model.occ.getMass(2,tag)
    if typ=='Sphere':
        (gball if area>1e-3 else ghollow).append(tag)
    elif typ in ('Cylinder','Plane'):
        k=max(range(3),key=lambda j:abs(c[j])); key=('p' if c[k]>0 else 'n')+'xyz'[k]
        (gholes if typ=='Cylinder' else gopens)[key]=tag
phys_ids={}; pid=1
for name,tags in [('ball',gball),('hollowWall',ghollow)]+[(f'hole_{n}',[gholes[n]]) for n in ['px','nx','py','ny','pz','nz']]+[(f'opening_{n}',[gopens[n]]) for n in ['px','nx','py','ny','pz','nz']]:
    gmsh.model.addPhysicalGroup(2,tags,pid); gmsh.model.setPhysicalName(2,pid,name); phys_ids[name]=pid; pid+=1
gmsh.model.addPhysicalGroup(3,[voltag],100); gmsh.model.setPhysicalName(3,100,'water')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0008); gmsh.option.setNumber('Mesh.MeshSizeMax',0.0008)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.option.setNumber('Mesh.Optimize',1)
gmsh.model.mesh.generate(3)
gmsh.write('water.msh')
print('nodes',len(gmsh.model.mesh.getNodes()[0]),'3D elements',sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]))
print('physical IDs',phys_ids)
gmsh.finalize()

# -- cell 6 -------------------------------------------------------------------------
# The coarse mesh contains about 194k tetrahedra and 40k points; Gmsh reports no ill-shaped tetrahedra
import subprocess, textwrap, shutil
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
Path('system/controlDict').write_text(textwrap.dedent('''\
FoamFile { format ascii; class dictionary; object controlDict; }
application checkMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
'''))
res=subprocess.run(['gmshToFoam','water.msh'],text=True,capture_output=True)
print(res.stdout[-3500:]); print(res.stderr[-1000:]); print('returncode',res.returncode)
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 7 -------------------------------------------------------------------------
# Conversion retained exactly the 14 requested boundary names; the converter’s temporary “undefined fa
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
print(chk.stdout[-7000:]); print(chk.stderr); print('returncode',chk.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The mesh itself was not read because this OpenFOAM build requires placeholder `fvSchemes` and `fvSol
Path('system/fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
print(chk.stdout[-8000:]); print(chk.stderr); print('returncode',chk.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The extended checker also parses the standard scheme subdictionaries. I’ll supply empty/default-only
Path('system/fvSchemes').write_text(textwrap.dedent('''\
FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''))
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
print(chk.stdout[-8000:]); print(chk.stderr); print('returncode',chk.returncode)

# -- cell 10 ------------------------------------------------------------------------
# The exhaustive optional diagnostics find 571 boundary tetrahedra with only two internal faces (a kno
std=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('checkMesh.standard.log').write_text(std.stdout+std.stderr)
print(std.stdout[-6000:]); print(std.stderr); print('returncode',std.returncode)

# -- cell 11 ------------------------------------------------------------------------
# Standard `checkMesh` reports **Mesh OK**: one region, 193,795 tetrahedra, 14 patches, closed boundar
stlfiles=sorted(tri.glob('*.stl'))
parts=[pv.read(str(p)) for p in stlfiles]
merged=parts[0]
for q in parts[1:]: merged=merged.merge(q,merge_points=False)
cleaned=merged.clean(tolerance=1e-8,absolute=True)
free=cleaned.extract_feature_edges(boundary_edges=True,non_manifold_edges=False,feature_edges=False,manifold_edges=False)
print('STL files',len(stlfiles))
for p,q in zip(stlfiles,parts): print(p.name,q.n_cells)
print('union triangles',cleaned.n_cells,'points',cleaned.n_points,'free-edge line cells',free.n_cells)
print('water CAD volume',water.volume,'mesh volume 2.06597e-05','relative difference',abs(2.06597e-5-water.volume)/water.volume)

# -- cell 12 ------------------------------------------------------------------------
# The independently tessellated CAD patch files leave 48 unmatched edge segments at patch junctions, d
import numpy as np
gmsh.initialize(); gmsh.open('water.msh')
nodeTags,nodeCoords,_=gmsh.model.mesh.getNodes(); xyz=nodeCoords.reshape(-1,3); node_map={int(t):xyz[i] for i,t in enumerate(nodeTags)}
for dim,ptag in gmsh.model.getPhysicalGroups(2):
    name=gmsh.model.getPhysicalName(dim,ptag); tris=[]
    for ent in gmsh.model.getEntitiesForPhysicalGroup(dim,ptag):
        types,_,enodes=gmsh.model.mesh.getElements(dim,int(ent))
        for et,nodes in zip(types,enodes):
            if gmsh.model.mesh.getElementProperties(et)[0]=='Triangle 3': tris.extend(np.asarray(nodes).reshape(-1,3).tolist())
    with open(tri/(name+'.stl'),'w') as fh:
        fh.write(f'solid {name}\n')
        for tags in tris:
            a,b,c=(node_map[int(t)] for t in tags); n=np.cross(b-a,c-a); ln=np.linalg.norm(n); n=n/ln if ln else n
            fh.write(f' facet normal {n[0]:.16g} {n[1]:.16g} {n[2]:.16g}\n  outer loop\n')
            for v in (a,b,c): fh.write(f'   vertex {v[0]:.16g} {v[1]:.16g} {v[2]:.16g}\n')
            fh.write('  endloop\n endfacet\n')
        fh.write(f'endsolid {name}\n')
    print(name,len(tris))
gmsh.finalize()

# -- cell 13 ------------------------------------------------------------------------
# The replacement STLs now use exactly the same boundary triangles as the OpenFOAM mesh. I’ll recheck 
parts=[pv.read(str(p)) for p in sorted(tri.glob('*.stl'))]
merged=parts[0]
for q in parts[1:]: merged=merged.merge(q,merge_points=False)
cl=merged.clean(tolerance=1e-12,absolute=True)
free=cl.extract_feature_edges(boundary_edges=True,non_manifold_edges=False,feature_edges=False,manifold_edges=False)
F=cl.faces.reshape(-1,4)[:,1:]
from collections import defaultdict
edges=defaultdict(list)
for a,b,c in F:
    for u,v in ((a,b),(b,c),(c,a)): edges[tuple(sorted((int(u),int(v))))].append((int(u),int(v)))
free_count=sum(len(vs)==1 for vs in edges.values()); nonman=sum(len(vs)>2 for vs in edges.values())
same=sum(len(vs)==2 and vs[0]==vs[1] for vs in edges.values())
print('triangles',len(F),'unique points',cl.n_points)
print('free edges',free_count,'nonmanifold edges',nonman,'same-direction paired edges',same,'pyvista boundary lines',free.n_cells)
print('STL count',len(parts),'all nonempty',all(p.n_cells>0 for p in parts))

# -- cell 14 ------------------------------------------------------------------------
# The conformal STL union is now fully closed and consistently wound: 37,108 triangles, zero free edge
# Measure spherical radii from surface points returned by the exact CAD faces.
def radial_samples(face):
    pts=[]
    for e in face.edges():
        for v in e.vertices(): pts.append(v.center().length)
    pts.append(face.center().length)
    return pts
ball_samples=radial_samples(ball_faces[0])
hollow_samples=[]
for f in hollow_faces: hollow_samples += radial_samples(f)
# Face centroids are interior mass centroids, so radius validation uses boundary vertices on the exact spheres.
ball_vertex_r=[v.center().length for v in ball_faces[0].vertices()]
hollow_vertex_r=[v.center().length for f in hollow_faces for v in f.vertices()]
print(f'ball boundary radius range: {min(ball_vertex_r):.9f} .. {max(ball_vertex_r):.9f} m; requested {BALL_R:.9f}')
print(f'hollow boundary radius range: {min(hollow_vertex_r):.9f} .. {max(hollow_vertex_r):.9f} m; requested {HOLLOW_R:.9f}')
print(f'measured concentric clearance: {min(hollow_vertex_r)-max(ball_vertex_r):.9f} m; requested 0.001000000')
print(f'cube/opening extrema: {water.bounding_box().min.X:.9f} .. {water.bounding_box().max.X:.9f} m; requested +/-0.020000000')
