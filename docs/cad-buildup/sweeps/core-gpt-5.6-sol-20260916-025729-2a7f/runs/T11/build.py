"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll treat the stated 8 mm cell width as the clear opening across flats; adding the 1 mm wall gives 
import build123d as bd, numpy as np, matplotlib.pyplot as plt, inspect
MM=1e-3
DUCT=80*MM; CORE_L=20*MM; UP=40*MM; DOWN=80*MM
CELL_AF=8*MM; WALL=1*MM; PITCH_AF=CELL_AF+WALL
print('Polygon', inspect.signature(bd.Polygon)); print('extrude', inspect.signature(bd.extrude)); print('Box', inspect.signature(bd.Box))

# -- cell 2 -------------------------------------------------------------------------
# I’ll re-emit the rejected construction in a replay-safe form and test a single opening cut first; th
R_OPEN=CELL_AF/np.sqrt(3); R_PITCH=PITCH_AF/np.sqrt(3); dy=1.5*R_PITCH; dz=PITCH_AF
centers=[(i*dy,(j+0.5*(i&1))*dz) for i in range(-7,8) for j in range(-7,8) if abs(i*dy)<DUCT/2+R_OPEN and abs((j+0.5*(i&1))*dz)<DUCT/2+R_OPEN]
def hex_prism(y,z):
    pts=[(0,y+R_OPEN*np.cos(k*np.pi/3),z+R_OPEN*np.sin(k*np.pi/3)) for k in range(6)]
    return bd.extrude(bd.Face(bd.Polygon(*pts)),amount=CORE_L,dir=(1,0,0))
openings=[hex_prism(y,z) for y,z in centers]
core_box=bd.Pos(CORE_L/2,0,0)*bd.Box(CORE_L,DUCT,DUCT)
test_cut=core_box-openings[len(openings)//2]
print('opening volume',openings[0].volume,'single-cut removed',core_box.volume-test_cut.volume,'cells',len(centers))

# -- cell 3 -------------------------------------------------------------------------
# `Polygon` already creates a planar face in this build123d version, so wrapping it in `Face(...)` emp
local_hex=bd.Polygon(*[(R_OPEN*np.cos(k*np.pi/3),R_OPEN*np.sin(k*np.pi/3)) for k in range(6)])
print(type(local_hex),local_hex.area)
yz_hex=bd.Plane.YZ*local_hex
probe=bd.extrude(yz_hex,amount=CORE_L)
expected=3*np.sqrt(3)*R_OPEN**2/2*CORE_L
print('probe volume',probe.volume,'expected',expected,'bbox',probe.bounding_box().size)

# -- cell 4 -------------------------------------------------------------------------
# With a valid prism established, I’ll rebuild all 105 openings and perform the cut. The resulting wal
def hex_prism(y,z):
    h=bd.Polygon(*[(R_OPEN*np.cos(k*np.pi/3),R_OPEN*np.sin(k*np.pi/3)) for k in range(6)])
    return bd.Pos(0,y,z)*(bd.extrude(bd.Plane.YZ*h,amount=CORE_L))
openings=[hex_prism(y,z) for y,z in centers]
honeycomb=core_box.cut(*openings)
full_vol=CORE_L*DUCT*DUCT
print(f'wall volume={honeycomb.volume:.9g} m3 of slab={full_vol:.9g}; fraction={honeycomb.volume/full_vol:.3f}')
print('solids',len(honeycomb.solids()),'faces',len(honeycomb.faces()))

# -- cell 5 -------------------------------------------------------------------------
# The corrected core has a 20.5% solid fraction and remains one connected wall solid. I’ll render it t
import pyvista as pv
bd.export_stl(honeycomb,'honeycomb_preview.stl',tolerance=2e-4,angular_tolerance=0.2)
p=pv.Plotter(off_screen=True,window_size=(900,650)); p.set_background('white')
p.add_mesh(pv.read('honeycomb_preview.stl'),color='goldenrod',show_edges=True,line_width=.3)
outline=pv.Box(bounds=(-UP,CORE_L+DOWN,-DUCT/2,DUCT/2,-DUCT/2,DUCT/2)).outline()
p.add_mesh(outline,color='navy',line_width=3)
p.add_axes(); p.camera_position=[(0.16,-0.15,0.12),(0.02,0,0),(0,0,1)]
p.show(screenshot='coarse_geometry.png')
from IPython.display import Image,display
display(Image('coarse_geometry.png'))

# -- cell 6 -------------------------------------------------------------------------
# The preview confirms the wall network spans the full 20 mm core thickness and is clipped by the 80 m
bd.export_step(honeycomb,'honeycombWalls.step')
import gmsh, os, math
gmsh.initialize(); gmsh.model.add('airPath')
gmsh.model.occ.addBox(-UP,-DUCT/2,-DUCT/2,UP+CORE_L+DOWN,DUCT,DUCT,tag=1)
wall_entities=gmsh.model.occ.importShapes('honeycombWalls.step')
gmsh.model.occ.synchronize()
fluid,_=gmsh.model.occ.cut([(3,1)],[e for e in wall_entities if e[0]==3],removeObject=True,removeTool=True)
gmsh.model.occ.synchronize()
vols=[t for d,t in fluid if d==3]; surfs=[t for d,t in gmsh.model.getBoundary([(3,t) for t in vols],oriented=False,recursive=False) if d==2]
groups={'inlet':[],'outlet':[],'ductWalls':[],'honeycombWalls':[]}; tol=1e-8
for s in surfs:
 b=gmsh.model.getBoundingBox(2,s)
 if abs(b[0]+UP)<tol and abs(b[3]+UP)<tol: groups['inlet'].append(s)
 elif abs(b[0]-(CORE_L+DOWN))<tol and abs(b[3]-(CORE_L+DOWN))<tol: groups['outlet'].append(s)
 elif (abs(b[1]+DUCT/2)<tol and abs(b[4]+DUCT/2)<tol) or (abs(b[1]-DUCT/2)<tol and abs(b[4]-DUCT/2)<tol) or (abs(b[2]+DUCT/2)<tol and abs(b[5]+DUCT/2)<tol) or (abs(b[2]-DUCT/2)<tol and abs(b[5]-DUCT/2)<tol): groups['ductWalls'].append(s)
 else: groups['honeycombWalls'].append(s)
print('fluid volumes',vols,'total surfaces',len(surfs),'groups',{k:len(v) for k,v in groups.items()})
gmsh.finalize()

# -- cell 7 -------------------------------------------------------------------------
# Gmsh’s OCC bounding boxes are padded by about \(10^{-7}\) m, so the initial \(10^{-8}\) m classifica
gmsh.initialize(); gmsh.model.add('airPath')
duct_tag=gmsh.model.occ.addBox(-UP,-DUCT/2,-DUCT/2,UP+CORE_L+DOWN,DUCT,DUCT)
wall_entities=gmsh.model.occ.importShapes('honeycombWalls.step'); gmsh.model.occ.synchronize()
fluid,_=gmsh.model.occ.cut([(3,duct_tag)],[e for e in wall_entities if e[0]==3],removeObject=True,removeTool=True); gmsh.model.occ.synchronize()
vols=[t for d,t in fluid if d==3]; surfs=[t for d,t in gmsh.model.getBoundary([(3,t) for t in vols],oriented=False) if d==2]
groups={'inlet':[],'outlet':[],'ductWalls':[],'honeycombWalls':[]}; tol=1e-6
for s in surfs:
 b=gmsh.model.getBoundingBox(2,s)
 if abs(b[0]+UP)<tol and abs(b[3]+UP)<tol: groups['inlet'].append(s)
 elif abs(b[0]-(CORE_L+DOWN))<tol and abs(b[3]-(CORE_L+DOWN))<tol: groups['outlet'].append(s)
 elif any((abs(b[a]-q)<tol and abs(b[a+3]-q)<tol) for a,q in [(1,-DUCT/2),(1,DUCT/2),(2,-DUCT/2),(2,DUCT/2)]): groups['ductWalls'].append(s)
 else: groups['honeycombWalls'].append(s)
for name,ss in groups.items(): pg=gmsh.model.addPhysicalGroup(2,ss); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(3,vols); gmsh.model.setPhysicalName(3,pg,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.001); gmsh.option.setNumber('Mesh.MeshSizeMax',0.004); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('General.NumThreads',4)
print('classified surfaces',{k:len(v) for k,v in groups.items()}); gmsh.model.mesh.generate(3); gmsh.write('airPath.msh'); print('nodes',len(gmsh.model.mesh.getNodes()[0])); gmsh.finalize()

# -- cell 8 -------------------------------------------------------------------------
# The coarse mesh has 29,674 nodes and about 144k tetrahedra; all 556 boundary surfaces were assigned 
import subprocess, pathlib, re, os
pathlib.Path('system').mkdir(exist_ok=True); pathlib.Path('constant').mkdir(exist_ok=True)
pathlib.Path('system/controlDict').write_text('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n')
r=subprocess.run(['gmshToFoam','airPath.msh'],text=True,capture_output=True); print(r.stdout[-2500:]); print(r.stderr[-1000:]); r.check_returncode()
bp=pathlib.Path('constant/polyMesh/boundary'); txt=bp.read_text(); txt=re.sub(r'type\s+physical;', 'type patch;', txt); bp.write_text(txt)
r=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True); print(r.stdout[-6000:]); print(r.stderr); print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# Conversion succeeded with the four named Gmsh groups. The “undefined faces” message is Gmsh-to-OpenF
pathlib.Path('system/fvSchemes').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
pathlib.Path('system/fvSolution').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
print(pathlib.Path('constant/polyMesh/boundary').read_text())
r=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True); print(r.stdout[-8000:]); print(r.stderr); print('return',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# Topology, patch coverage, dimensions, and volume are correct, but the exhaustive geometry check flag
r=subprocess.run(['checkMesh'],text=True,capture_output=True); print(r.stdout[-5000:]); print('return',r.returncode)

# -- cell 11 ------------------------------------------------------------------------
# Plain `checkMesh` passes (“Mesh OK”); only the optional exhaustive tetra determinant metrics warn. I
r=subprocess.run(['foamToSurface','-help'],text=True,capture_output=True); print('\n'.join(r.stdout.splitlines()[:80]))

# -- cell 12 ------------------------------------------------------------------------
# `foamToSurface` in this release exports all patches together and offers no selection option. I’ll us
for cmd in ['surfaceMeshExtract','surfaceMeshConvert']:
 r=subprocess.run([cmd,'-help'],text=True,capture_output=True); print('---',cmd,'---'); print('\n'.join(r.stdout.splitlines()[:65]))

# -- cell 13 ------------------------------------------------------------------------
# `surfaceMeshExtract` supports exact patch selection. I’ll export the four patches to `constant/triSu
tri=pathlib.Path('constant/triSurface'); tri.mkdir(exist_ok=True)
for name in ['inlet','outlet','ductWalls','honeycombWalls']:
 r=subprocess.run(['surfaceMeshExtract','-constant','-patches',name,str(tri/f'{name}.stl')],text=True,capture_output=True); print(name,r.returncode,r.stdout.splitlines()[-3:]); r.check_returncode()
for f in tri.glob('*.stl'):
 n=sum(1 for line in open(f,errors='ignore') if 'facet normal' in line); print(f.name,n,'triangles')

# -- cell 14 ------------------------------------------------------------------------
# The four STL files contain 28,544 triangles in total, exactly matching the mesh boundary count. I’ll
meshes={f.stem:pv.read(str(f)) for f in tri.glob('*.stl')}
for n,m in meshes.items(): print(n,'area m2=',m.area,'bounds=',m.bounds)
p=pv.Plotter(off_screen=True,window_size=(1000,650)); p.set_background('white')
colors={'inlet':'royalblue','outlet':'tomato','ductWalls':'lightgray','honeycombWalls':'goldenrod'}
for n,m in meshes.items(): p.add_mesh(m,color=colors[n],opacity=.32 if n=='ductWalls' else 1.0,label=n)
p.add_legend(); p.add_axes(); p.camera_position=[(0.19,-0.17,0.13),(0.025,0,0),(0,0,1)]; p.show(screenshot='mesh_patches.png')
display(Image('mesh_patches.png'))
