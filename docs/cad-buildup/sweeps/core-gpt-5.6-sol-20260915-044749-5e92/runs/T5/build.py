"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll first inspect the STEP declaration and import it with OCCT, then report its topology, dimension
from pathlib import Path
import re, subprocess, json, os
import build123d as bd
STEP_PATH = Path('/home/qiuzi/.openreynolds-buildup/work/T5-20260915-045413-d2ba/geometry/assembly.step')
step_head = STEP_PATH.read_text(errors='ignore')
unit_lines = [ln.strip() for ln in step_head.splitlines() if ('SI_UNIT' in ln or 'CONVERSION_BASED_UNIT' in ln)][:20]
assembly = bd.import_step(STEP_PATH)
solids = list(assembly.solids())
print('STEP unit declarations:')
print('\n'.join(unit_lines) if unit_lines else 'NONE FOUND')
print(f'Imported solids: {len(solids)}, faces: {len(assembly.faces())}, edges: {len(assembly.edges())}')
print(f'OCCT valid: {assembly.is_valid}')
print(f'Bounding box raw: min={assembly.bounding_box().min}, max={assembly.bounding_box().max}, size={assembly.bounding_box().size}')
for i,s in enumerate(solids):
    print(f'solid {i}: valid={s.is_valid}, volume={s.volume:.9g}, faces={len(s.faces())}, bbox={s.bounding_box().size}')

# -- cell 2 -------------------------------------------------------------------------
# The file explicitly declares metres, contains three valid solids, and spans 53.95 × 46.29 × 31.30 m.
import pyvista as pv
colors = ['#d95f02','#1b9e77','#7570b3']
pl = pv.Plotter(off_screen=True, window_size=(900,700))
for i,s in enumerate(solids):
    fn = f'inspect_solid_{i}.stl'
    bd.export_stl(s, fn, tolerance=0.05, angular_tolerance=0.15)
    m = pv.read(fn)
    pl.add_mesh(m, color=colors[i], opacity=0.65, show_edges=True, label=f'solid {i}')
pl.add_axes(); pl.add_legend(); pl.view_isometric(); pl.camera.zoom(1.25)
pl.show(screenshot='assembly_overview.png')
print('Rendered assembly_overview.png')

# -- cell 3 -------------------------------------------------------------------------
# The solids form stacked, partly overlapping housing components rather than nested “fluid” and “wall”
pl = pv.Plotter(off_screen=True, shape=(2,3), window_size=(1200,750))
for i in range(3):
    m=pv.read(f'inspect_solid_{i}.stl')
    pl.subplot(0,i); pl.add_mesh(m,color=colors[i],show_edges=True); pl.add_text(f'Solid {i} isometric'); pl.view_isometric(); pl.camera.zoom(1.15)
    pl.subplot(1,i); pl.add_mesh(m,color=colors[i],show_edges=True); pl.add_text(f'Solid {i} front (+Y)'); pl.view_xz(); pl.camera.zoom(1.15)
pl.show(screenshot='solids_individual.png')
print('Rendered solids_individual.png')

# -- cell 4 -------------------------------------------------------------------------
# The parts resemble a lower cover, an upper circular insert, and a surrounding casing. To avoid guess
def show_tree(sh, depth=0):
    print('  '*depth + f'{type(sh).__name__}: label={getattr(sh,"label","")!r} children={len(getattr(sh,"children",[]))}')
    for c in getattr(sh,'children',[]): show_tree(c, depth+1)
show_tree(assembly)
print('\nSTEP product/entity names:')
for ln in step_head.splitlines():
    if any(k in ln for k in ('PRODUCT(','MANIFOLD_SOLID_BREP','NEXT_ASSEMBLY_USAGE_OCCURRENCE')):
        print(ln[:300])

# -- cell 5 -------------------------------------------------------------------------
# One slice contained no intersection and the plotting call rejected the empty result. I’ll repeat wit
meshes=[pv.read(f'inspect_solid_{i}.stl') for i in range(3)]
zlevels=[1.0,3.0,6.0,12.0,20.0,28.0]
pl=pv.Plotter(off_screen=True,shape=(2,3),window_size=(1200,800))
for k,z in enumerate(zlevels):
    pl.subplot(k//3,k%3)
    for i,m in enumerate(meshes):
        sl=m.slice(normal=(0,0,1),origin=(0,0,z))
        if sl.n_points: pl.add_mesh(sl,color=colors[i],line_width=4)
    pl.add_text(f'z = {z} m'); pl.view_xy(); pl.camera.parallel_projection=True
pl.show(screenshot='horizontal_sections.png')
print('Rendered horizontal_sections.png')

# -- cell 6 -------------------------------------------------------------------------
# The first pair has no Boolean common and OCCT returns `None` for that case. I’ll handle empty inters
print('Pairwise common volumes and distances:')
for i in range(3):
    for j in range(i+1,3):
        common=solids[i] & solids[j]
        cv=0.0 if common is None else sum(x.volume for x in common.solids())
        print(f'{i}-{j}: common volume={cv:.9g}, distance={solids[i].distance_to(solids[j])}')
fluid_raw=solids[0].fuse(solids[1],solids[2])
print(f'Fuse: solids={len(fluid_raw.solids())}, valid={fluid_raw.is_valid}, volume={sum(s.volume for s in fluid_raw.solids()):.9g}; input sum={sum(s.volume for s in solids):.9g}')

# -- cell 7 -------------------------------------------------------------------------
# The parts only touch at interfaces and do not enclose a separate void; the `BODY` itself is almost t
for si,s in enumerate(solids):
    print(f'\nSolid {si} {s.label}: planar faces (area, center, normal)')
    rows=[]
    for f in s.faces():
        try:
            n=f.normal_at(); c=f.center()
            # planar if normals at sample are effectively constant is exposed by geom_type
            gt=str(f.geom_type)
            if 'PLANE' in gt.upper(): rows.append((f.area,c,n,gt))
        except Exception: pass
    for a,c,n,gt in sorted(rows,reverse=True,key=lambda r:r[0])[:12]:
        print(f' area={a:.7g} center=({c.X:.5g},{c.Y:.5g},{c.Z:.5g}) normal=({n.X:.3g},{n.Y:.3g},{n.Z:.3g})')

# -- cell 8 -------------------------------------------------------------------------
# The interfaces match exactly: `BASE` top to `BODY` bottom, and `BODY` top to `EMITTER` bottom (the l
import gmsh
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 1)
gmsh.model.add('fluid')
dimtags=gmsh.model.occ.importShapes(str(STEP_PATH))
gmsh.model.occ.synchronize()
vols=[dt for dt in gmsh.model.getEntities(3)]
print('Imported gmsh volumes:',vols)
result,_=gmsh.model.occ.fuse([vols[0]],vols[1:],removeObject=True,removeTool=True)
gmsh.model.occ.removeAllDuplicates(); gmsh.model.occ.synchronize()
print('Fuse result:',result,'final volumes:',gmsh.model.getEntities(3))
for tag in [t for d,t in gmsh.model.getBoundary(gmsh.model.getEntities(3),combined=True,oriented=False) if d==2]:
    bb=gmsh.model.getBoundingBox(2,tag); mass=gmsh.model.occ.getMass(2,tag)
    if bb[2]<0.01 or bb[5]>31.29:
        print(f'candidate end surface {tag}: area={mass:.7g}, bbox={tuple(round(x,5) for x in bb)}')
gmsh.write('fluid_fused.brep')
gmsh.finalize()

# -- cell 9 -------------------------------------------------------------------------
# OCC preserves two touching volumes because one interface is purely coincident, but Boolean fragmenta
gmsh.initialize(); gmsh.model.add('inspect_fused'); gmsh.model.occ.importShapes('fluid_fused.brep'); gmsh.model.occ.synchronize()
gvols=gmsh.model.getEntities(3)
print('Volumes:',[(t,gmsh.model.occ.getMass(3,t)) for d,t in gvols])
bsets=[]
for d,v in gvols:
    bs={t for dd,t in gmsh.model.getBoundary([(3,v)],oriented=False) if dd==2}; bsets.append(bs)
print('Shared interface surfaces:',sorted(bsets[0]&bsets[1]))
extsurf=(bsets[0]|bsets[1])-(bsets[0]&bsets[1])
print('Top exterior candidates:')
for tag in sorted(extsurf):
    bb=gmsh.model.getBoundingBox(2,tag)
    if bb[5]>31.299:
        print(tag,'area',gmsh.model.occ.getMass(2,tag),'bbox',tuple(round(x,5) for x in bb))
gmsh.finalize()

# -- cell 10 ------------------------------------------------------------------------
# I’ll re-emit the successful coarse mesh cell with its missing `Counter` import included, making it v
from collections import Counter
MESH_SIZE=2.5
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.model.add('fluid_mesh')
inputs=gmsh.model.occ.importShapes(str(STEP_PATH)); gmsh.model.occ.synchronize()
gmsh.model.occ.fragment(inputs[:1],inputs[1:],removeObject=True,removeTool=True); gmsh.model.occ.removeAllDuplicates(); gmsh.model.occ.synchronize()
mesh_vols=gmsh.model.getEntities(3)
counts=Counter(t for v in mesh_vols for d,t in gmsh.model.getBoundary([v],oriented=False) if d==2)
exterior=sorted(t for t,n in counts.items() if n==1); tol=1e-5
inlet=[t for t in exterior if abs(gmsh.model.getBoundingBox(2,t)[2])<tol and abs(gmsh.model.getBoundingBox(2,t)[5])<tol]
outlet=[t for t in exterior if abs(gmsh.model.getBoundingBox(2,t)[2]-31.3)<tol and abs(gmsh.model.getBoundingBox(2,t)[5]-31.3)<tol]
walls=sorted(set(exterior)-set(inlet)-set(outlet))
for dim,tags,name in [(3,[t for d,t in mesh_vols],'fluid'),(2,inlet,'inlet'),(2,outlet,'outlet'),(2,walls,'walls')]:
    p=gmsh.model.addPhysicalGroup(dim,tags); gmsh.model.setPhysicalName(dim,p,name)
gmsh.option.setNumber('Mesh.MeshSizeMin',MESH_SIZE); gmsh.option.setNumber('Mesh.MeshSizeMax',MESH_SIZE)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0); gmsh.option.setNumber('Mesh.Algorithm3D',10)
gmsh.model.mesh.generate(3); gmsh.write('fluid.msh')
node_count=len(gmsh.model.mesh.getNodes()[0]); element_count=sum(len(a) for a in gmsh.model.mesh.getElements()[1])
areas={n:sum(gmsh.model.occ.getMass(2,t) for t in ts) for n,ts in [('inlet',inlet),('outlet',outlet),('walls',walls)]}
print(f'Wrote fluid.msh: {node_count} nodes, {element_count} elements, {len(mesh_vols)} conformal volumes')
print('Patch areas m^2:',areas)
gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# OpenFOAM requires a minimal case dictionary even for mesh conversion. I’ll create only that mesh-inf
Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile
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
''')
print('Wrote minimal system/controlDict for mesh utilities')

# -- cell 12 ------------------------------------------------------------------------
# I’ll rerun the converter now that the minimal case structure exists.
conv=subprocess.run(['gmshToFoam','fluid.msh'],text=True,capture_output=True)
print(conv.stdout[-5000:]); print(conv.stderr[-1000:])
if conv.returncode!=0: raise RuntimeError(f'gmshToFoam failed: {conv.returncode}')

# -- cell 13 ------------------------------------------------------------------------
# Conversion produced 15,173 tetrahedra and the requested named patches, but also 5,534 `defaultFaces`
chk=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
print((chk.stdout+chk.stderr)[-9000:])
print('checkMesh return code:',chk.returncode)

# -- cell 14 ------------------------------------------------------------------------
# This OpenFOAM build loads `fvSchemes` for the extended geometry checks. I’ll add empty mesh-utility 
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
''')
print('Wrote mesh-check utility dictionaries')

# -- cell 15 ------------------------------------------------------------------------
# I’ll rerun the full topology and geometry check.
chk=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
print((chk.stdout+chk.stderr)[-10000:]); print('checkMesh return code:',chk.returncode)

# -- cell 16 ------------------------------------------------------------------------
# The mesh is topologically one region and closed, but it fails five quality checks because sub-millim
HEAL_TOL=0.10
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.model.add('fluid_healed')
inputs=gmsh.model.occ.importShapes(str(STEP_PATH)); gmsh.model.occ.synchronize()
healed=gmsh.model.occ.healShapes(inputs,tolerance=HEAL_TOL,fixDegenerated=True,fixSmallEdges=True,fixSmallFaces=True,sewFaces=True,makeSolids=True)
gmsh.model.occ.synchronize(); hvols=gmsh.model.getEntities(3)
gmsh.model.occ.fragment(hvols[:1],hvols[1:],removeObject=True,removeTool=True); gmsh.model.occ.removeAllDuplicates(); gmsh.model.occ.synchronize()
mesh_vols=gmsh.model.getEntities(3); counts=Counter(t for v in mesh_vols for d,t in gmsh.model.getBoundary([v],oriented=False) if d==2)
exterior=sorted(t for t,n in counts.items() if n==1); tol=1e-5
inlet=[t for t in exterior if abs(gmsh.model.getBoundingBox(2,t)[2])<tol and abs(gmsh.model.getBoundingBox(2,t)[5])<tol]
outlet=[t for t in exterior if abs(gmsh.model.getBoundingBox(2,t)[2]-31.3)<tol and abs(gmsh.model.getBoundingBox(2,t)[5]-31.3)<tol]
walls=sorted(set(exterior)-set(inlet)-set(outlet))
for dim,tags,name in [(3,[t for d,t in mesh_vols],'fluid'),(2,inlet,'inlet'),(2,outlet,'outlet'),(2,walls,'walls')]:
    p=gmsh.model.addPhysicalGroup(dim,tags); gmsh.model.setPhysicalName(dim,p,name)
gmsh.option.setNumber('Mesh.MeshSizeMin',MESH_SIZE); gmsh.option.setNumber('Mesh.MeshSizeMax',MESH_SIZE)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0); gmsh.option.setNumber('Mesh.Algorithm3D',1)
gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize('Netgen'); gmsh.write('fluid.msh')
print(f'Healed at {HEAL_TOL} m: volumes={len(mesh_vols)}, surfaces={len(exterior)}, nodes={len(gmsh.model.mesh.getNodes()[0])}')
gmsh.finalize()

# -- cell 17 ------------------------------------------------------------------------
# The healed geometry retains three conformal volumes and has fewer exterior faces. I’ll reconvert it 
conv=subprocess.run(['gmshToFoam','fluid.msh'],text=True,capture_output=True,check=True)
chk=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
text=chk.stdout+chk.stderr
print(text[-6500:]); print('Failed line:',[x for x in text.splitlines() if 'Failed ' in x][-1:])

# -- cell 18 ------------------------------------------------------------------------
# OCC healing did not remove the microscopic edge defects and also dropped the BASE cells from the phy
import inspect
print(inspect.signature(solids[0].is_inside))
for p in [bd.Vector(0,0,1),bd.Vector(0,0,10),bd.Vector(0,0,25),bd.Vector(100,100,100)]:
    print(tuple(p),[s.is_inside(p) for s in solids])

# -- cell 19 ------------------------------------------------------------------------
# Point classification confirms the three solids occupy successive stacked portions of one flow volume
import numpy as np, shutil
VOXEL_SIZE=1.5
bb=assembly.bounding_box(); xmin,ymin,zmin=bb.min.X,bb.min.Y,bb.min.Z; xmax,ymax,zmax=bb.max.X,bb.max.Y,bb.max.Z
def breaks(a,b,h):
    q=list(np.arange(a,b,h));
    if not q or abs(q[-1]-b)>1e-10: q.append(b)
    return np.array(q)
xs,ys,zs=breaks(xmin,xmax,VOXEL_SIZE),breaks(ymin,ymax,VOXEL_SIZE),breaks(0.0,zmax,VOXEL_SIZE)
occupied=[]
for k in range(len(zs)-1):
 for j in range(len(ys)-1):
  for i in range(len(xs)-1):
   p=bd.Vector((xs[i]+xs[i+1])/2,(ys[j]+ys[j+1])/2,(zs[k]+zs[k+1])/2)
   if any(s.is_inside(p,1e-5) for s in solids): occupied.append((i,j,k))
cell_id={ijk:n for n,ijk in enumerate(occupied)}
face_defs=[((-1,0,0),lambda i,j,k:[(i,j,k),(i,j+1,k),(i,j+1,k+1),(i,j,k+1)]),
((1,0,0),lambda i,j,k:[(i+1,j,k),(i+1,j,k+1),(i+1,j+1,k+1),(i+1,j+1,k)]),
((0,-1,0),lambda i,j,k:[(i,j,k),(i,j,k+1),(i+1,j,k+1),(i+1,j,k)]),
((0,1,0),lambda i,j,k:[(i,j+1,k),(i+1,j+1,k),(i+1,j+1,k+1),(i,j+1,k+1)]),
((0,0,-1),lambda i,j,k:[(i,j,k),(i+1,j,k),(i+1,j+1,k),(i,j+1,k)]),
((0,0,1),lambda i,j,k:[(i,j,k+1),(i+1,j,k+1),(i+1,j+1,k+1),(i,j+1,k+1)])]
raw={}
for cid,(i,j,k) in enumerate(occupied):
 for (di,dj,dk),vf in face_defs:
  vv=vf(i,j,k); key=tuple(sorted(vv))
  if key in raw: raw[key]['neighbour']=cid
  else: raw[key]={'verts':vv,'owner':cid,'neighbour':None,'dir':(di,dj,dk)}
internal=[]; patches={'inlet':[],'outlet':[],'walls':[]}
for f in raw.values():
 if f['neighbour'] is not None: internal.append(f)
 else:
  kk=[v[2] for v in f['verts']]; dz=f['dir'][2]
  name='inlet' if dz==-1 and max(kk)==0 else ('outlet' if dz==1 and min(kk)==len(zs)-1 else 'walls')
  patches[name].append(f)
ordered=internal+patches['inlet']+patches['outlet']+patches['walls']
used=sorted({v for f in ordered for v in f['verts']}); pmap={v:n for n,v in enumerate(used)}
coords=[(xs[i],ys[j],zs[k]) for i,j,k in used]
pm=Path('constant/polyMesh'); shutil.rmtree(pm,ignore_errors=True); pm.mkdir(parents=True)
def header(cls,obj): return f'''FoamFile\n{{\n format ascii;\n class {cls};\n object {obj};\n}}\n'''
Path(pm/'points').write_text(header('vectorField','points')+f'{len(coords)}\n(\n'+'\n'.join(f'({x:.12g} {y:.12g} {z:.12g})' for x,y,z in coords)+'\n)\n')
Path(pm/'faces').write_text(header('faceList','faces')+f'{len(ordered)}\n(\n'+'\n'.join('4('+' '.join(str(pmap[v]) for v in f['verts'])+')' for f in ordered)+'\n)\n')
Path(pm/'owner').write_text(header('labelList','owner')+f'{len(ordered)}\n(\n'+'\n'.join(str(f['owner']) for f in ordered)+'\n)\n')
Path(pm/'neighbour').write_text(header('labelList','neighbour')+f'{len(internal)}\n(\n'+'\n'.join(str(f['neighbour']) for f in internal)+'\n)\n')
start=len(internal); blocks=[]
for name in ['inlet','outlet','walls']:
 n=len(patches[name]); blocks.append(f'{name}\n{{\n type patch;\n nFaces {n};\n startFace {start};\n}}'); start+=n
Path(pm/'boundary').write_text(header('polyBoundaryMesh','boundary')+'3\n(\n'+'\n'.join(blocks)+'\n)\n')
print(f'Cartesian CAD-classified mesh: cells={len(occupied)}, points={len(coords)}, faces={len(ordered)}')
print('Patch faces:',{k:len(v) for k,v in patches.items()},'bounds:',(xs[0],xs[-1],ys[0],ys[-1],zs[0],zs[-1]))

# -- cell 20 ------------------------------------------------------------------------
# The Cartesian mesh contains 12,545 cells and preserves the exact declared end planes. I’ll now run t
chk=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr)
text=chk.stdout+chk.stderr
print(text[-7000:]); print('return:',chk.returncode)

# -- cell 21 ------------------------------------------------------------------------
# The Cartesian topology is correct, but five of the six local face templates were wound inward; only 
for f in ordered:
    if f['dir'] != (0,0,1): f['verts']=list(reversed(f['verts']))
Path(pm/'faces').write_text(header('faceList','faces')+f'{len(ordered)}\n(\n'+'\n'.join('4('+' '.join(str(pmap[v]) for v in f['verts'])+')' for f in ordered)+'\n)\n')
print('Rewrote Cartesian face winding: -x,+x,-y,+y,-z reversed; +z retained')

# -- cell 22 ------------------------------------------------------------------------
# I’ll rerun the binding check after correcting face orientation.
chk=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr); text=chk.stdout+chk.stderr
print(text[-6500:])

# -- cell 23 ------------------------------------------------------------------------
# The corrected mesh passes all checks except 53 under-determined cells. These are one-cell remnants o
occset=set(occupied); removed_total=0
while True:
    bad=[]
    for c in occset:
        vecs=[]
        for d in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            if tuple(c[q]+d[q] for q in range(3)) in occset: vecs.append(d)
        if np.linalg.matrix_rank(np.array(vecs,dtype=float))<3: bad.append(c)
    if not bad: break
    occset.difference_update(bad); removed_total+=len(bad)
occupied=sorted(occset,key=lambda c:(c[2],c[1],c[0])); cell_id={ijk:n for n,ijk in enumerate(occupied)}
raw={}
for cid,(i,j,k) in enumerate(occupied):
 for (di,dj,dk),vf in face_defs:
  vv=vf(i,j,k); key=tuple(sorted(vv))
  if key in raw: raw[key]['neighbour']=cid
  else: raw[key]={'verts':vv,'owner':cid,'neighbour':None,'dir':(di,dj,dk)}
internal=[]; patches={'inlet':[],'outlet':[],'walls':[]}
for f in raw.values():
 if f['neighbour'] is not None: internal.append(f)
 else:
  kk=[v[2] for v in f['verts']]; dz=f['dir'][2]
  name='inlet' if dz==-1 and max(kk)==0 else ('outlet' if dz==1 and min(kk)==len(zs)-1 else 'walls'); patches[name].append(f)
ordered=internal+patches['inlet']+patches['outlet']+patches['walls']
for f in ordered:
    if f['dir'] != (0,0,1): f['verts']=list(reversed(f['verts']))
used=sorted({v for f in ordered for v in f['verts']}); pmap={v:n for n,v in enumerate(used)}; coords=[(xs[i],ys[j],zs[k]) for i,j,k in used]
Path(pm/'points').write_text(header('vectorField','points')+f'{len(coords)}\n(\n'+'\n'.join(f'({x:.12g} {y:.12g} {z:.12g})' for x,y,z in coords)+'\n)\n')
Path(pm/'faces').write_text(header('faceList','faces')+f'{len(ordered)}\n(\n'+'\n'.join('4('+' '.join(str(pmap[v]) for v in f['verts'])+')' for f in ordered)+'\n)\n')
Path(pm/'owner').write_text(header('labelList','owner')+f'{len(ordered)}\n(\n'+'\n'.join(str(f['owner']) for f in ordered)+'\n)\n')
Path(pm/'neighbour').write_text(header('labelList','neighbour')+f'{len(internal)}\n(\n'+'\n'.join(str(f['neighbour']) for f in internal)+'\n)\n')
start=len(internal); blocks=[]
for name in ['inlet','outlet','walls']:
 n=len(patches[name]); blocks.append(f'{name}\n{{\n type patch;\n nFaces {n};\n startFace {start};\n}}'); start+=n
Path(pm/'boundary').write_text(header('polyBoundaryMesh','boundary')+'3\n(\n'+'\n'.join(blocks)+'\n)\n')
print(f'Pruned {removed_total} unsupported sub-grid cells; retained {len(occupied)} cells')
print('Patch faces:',{k:len(v) for k,v in patches.items()})

# -- cell 24 ------------------------------------------------------------------------
# I’ll run the full mesh check on the pruned Cartesian mesh.
chk=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
Path('checkMesh.log').write_text(chk.stdout+chk.stderr); text=chk.stdout+chk.stderr
print(text[-6000:])
