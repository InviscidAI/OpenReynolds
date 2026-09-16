"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll treat the meshed oil as the closed interior cavity of the ring: the cylinder \(r<23\) mm, \(-5<
import build123d as bd
import numpy as np
from pathlib import Path
# Requested dimensions, in metres
R_RING_IN, R_RING_OUT, H_RING = 0.023, 0.030, 0.010
R_SPIN_IN, R_SPIN_OUT, H_SPIN = 0.015, 0.022, 0.008
z0_ring, z0_spin = -H_RING/2, -H_SPIN/2
cavity = bd.Pos(0,0,z0_ring) * bd.Cylinder(R_RING_IN, H_RING)
spinner_solid = (bd.Pos(0,0,z0_spin) * bd.Cylinder(R_SPIN_OUT,H_SPIN)) - (bd.Pos(0,0,z0_spin) * bd.Cylinder(R_SPIN_IN,H_SPIN))
oil = cavity - spinner_solid
expected_volume = np.pi*R_RING_IN**2*H_RING - np.pi*(R_SPIN_OUT**2-R_SPIN_IN**2)*H_SPIN
print(f'oil solids: {len(oil.solids())}')
print(f'oil volume measured = {oil.volume:.12e} m^3; analytic requested = {expected_volume:.12e} m^3')
print('bounds:', oil.bounding_box().min, oil.bounding_box().max)
print(f'radial clearance measured = {(R_RING_IN-R_SPIN_OUT)*1000:.3f} mm (requested 1.000)')
print(f'axial gap each measured = {(H_RING-H_SPIN)/2*1000:.3f} mm (requested 1.000)')

# -- cell 2 -------------------------------------------------------------------------
# The first bounds check exposed a placement mistake: `Cylinder` is centred by default, so the additio
# Corrected centred geometry (build123d Cylinder is centred on Z by default)
cavity = bd.Cylinder(R_RING_IN, H_RING)
spinner_solid = bd.Cylinder(R_SPIN_OUT,H_SPIN) - bd.Cylinder(R_SPIN_IN,H_SPIN)
oil = cavity - spinner_solid
print('corrected bounds:', oil.bounding_box().min, oil.bounding_box().max)
print(f'corrected oil volume = {oil.volume:.12e} m^3; target = {expected_volume:.12e} m^3')
print('tessellate signature check follows')
import inspect
print(inspect.signature(oil.tessellate))

# -- cell 3 -------------------------------------------------------------------------
# The corrected solid now has the requested centred bounds and exact analytic volume. I’ll inspect it 
import pyvista as pv
verts, tris = oil.tessellate(2e-4, 0.15)
pts = np.array([[v.X,v.Y,v.Z] for v in verts])
faces_pv = np.hstack([np.full((len(tris),1),3), np.array(tris,dtype=int)]).ravel()
surf = pv.PolyData(pts, faces_pv)
cut = surf.clip(normal=(1,0,0), origin=(0,0,0), invert=False).clip(normal=(0,1,0), origin=(0,0,0), invert=False)
pl = pv.Plotter(off_screen=True, window_size=(800,650))
pl.add_mesh(cut, color='royalblue', opacity=0.72, show_edges=True, line_width=1)
pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.35)
pl.add_title('Oil region, quadrant clipped (metres)')
pl.show(screenshot='oil_coarse.png')
print(f'coarse display surface: {surf.n_points} points, {surf.n_cells} triangles')

# -- cell 4 -------------------------------------------------------------------------
# The cutaway confirms the intended single connected region. Next I’ll recreate this exact Boolean vol
import gmsh, math, os, shutil, subprocess, textwrap
gmsh.initialize()
gmsh.model.add('flooded_gyro_oil')
occ = gmsh.model.occ
cav_tag = occ.addCylinder(0,0,-H_RING/2, 0,0,H_RING, R_RING_IN)
spin_outer_tag = occ.addCylinder(0,0,-H_SPIN/2, 0,0,H_SPIN, R_SPIN_OUT)
spin_bore_tag = occ.addCylinder(0,0,-H_SPIN/2, 0,0,H_SPIN, R_SPIN_IN)
spin_objs,_ = occ.cut([(3,spin_outer_tag)],[(3,spin_bore_tag)], removeObject=True, removeTool=True)
oil_objs,_ = occ.cut([(3,cav_tag)],spin_objs, removeObject=True, removeTool=True)
occ.synchronize()
vol_tag = oil_objs[0][1]
boundary = gmsh.model.getBoundary([(3,vol_tag)], oriented=False, recursive=False)
print('volume entities:', oil_objs, 'boundary surfaces:', len(boundary))
for d,t in boundary:
    typ=gmsh.model.getType(d,t); com=occ.getCenterOfMass(d,t); area=occ.getMass(d,t)
    print(f'surface {t:2d}: {typ:8s} area={area:.9e} com=({com[0]:+.4g},{com[1]:+.4g},{com[2]:+.4g})')

# -- cell 5 -------------------------------------------------------------------------
# The seven faces match the analytic areas: three cylinders (\(r=15,22,23\) mm), two spinner annuli at
# Identify faces by their exact analytic type/area, then name them immediately.
surf_info={t:(gmsh.model.getType(2,t),occ.getMass(2,t),occ.getCenterOfMass(2,t)) for _,t in boundary}
def closest_surface(kind, target_area, target_z=None):
    candidates=[t for t,(k,a,c) in surf_info.items() if k==kind and (target_z is None or abs(c[2]-target_z)<1e-9)]
    return min(candidates,key=lambda t:abs(surf_info[t][1]-target_area))
ring_cyl=closest_surface('Cylinder',2*np.pi*R_RING_IN*H_RING)
spin_bore=closest_surface('Cylinder',2*np.pi*R_SPIN_IN*H_SPIN)
spin_outer=closest_surface('Cylinder',2*np.pi*R_SPIN_OUT*H_SPIN)
spin_top=closest_surface('Plane',np.pi*(R_SPIN_OUT**2-R_SPIN_IN**2),H_SPIN/2)
spin_bottom=closest_surface('Plane',np.pi*(R_SPIN_OUT**2-R_SPIN_IN**2),-H_SPIN/2)
ring_top=closest_surface('Plane',np.pi*R_RING_IN**2,H_RING/2)
ring_bottom=closest_surface('Plane',np.pi*R_RING_IN**2,-H_RING/2)
patch_surfaces={'outer_ring_wetted':[ring_cyl], 'spinner_wetted':[spin_bore,spin_outer,spin_top,spin_bottom], 'ring_top_end_face':[ring_top], 'ring_bottom_end_face':[ring_bottom]}
for i,(name,tags) in enumerate(patch_surfaces.items(),1):
    pg=gmsh.model.addPhysicalGroup(2,tags,i); gmsh.model.setPhysicalName(2,pg,name)
fluid_pg=gmsh.model.addPhysicalGroup(3,[vol_tag],1); gmsh.model.setPhysicalName(3,fluid_pg,'oil')
print('patch assignments:',patch_surfaces)
# Coarse but gap-resolving boundary sizing: 0.25 mm near solid faces, 1.2 mm in open bore.
gmsh.option.setNumber('Mesh.CharacteristicLengthMin',0.00025)
gmsh.option.setNumber('Mesh.CharacteristicLengthMax',0.0012)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',12)
dist=gmsh.model.mesh.field.add('Distance'); gmsh.model.mesh.field.setNumbers(dist,'SurfacesList',[ring_cyl,spin_bore,spin_outer,spin_top,spin_bottom]); gmsh.model.mesh.field.setNumber(dist,'Sampling',80)
thr=gmsh.model.mesh.field.add('Threshold'); gmsh.model.mesh.field.setNumber(thr,'InField',dist); gmsh.model.mesh.field.setNumber(thr,'SizeMin',0.00025); gmsh.model.mesh.field.setNumber(thr,'SizeMax',0.0012); gmsh.model.mesh.field.setNumber(thr,'DistMin',0.0007); gmsh.model.mesh.field.setNumber(thr,'DistMax',0.004)
gmsh.model.mesh.field.setAsBackgroundMesh(thr)
gmsh.model.mesh.generate(2)
Path('constant/triSurface').mkdir(parents=True,exist_ok=True)
node_tags,node_xyz,_=gmsh.model.mesh.getNodes(); xyz=np.asarray(node_xyz).reshape(-1,3); node_map={int(t):xyz[i] for i,t in enumerate(node_tags)}
def desired_normal(name,p):
    r=np.hypot(p[0],p[1]); radial=np.array([p[0],p[1],0.0])/(r or 1)
    if name=='outer_ring_wetted': return radial
    if name=='ring_top_end_face': return np.array([0,0,1.])
    if name=='ring_bottom_end_face': return np.array([0,0,-1.])
    if abs(p[2]-H_SPIN/2)<1e-7: return np.array([0,0,-1.])
    if abs(p[2]+H_SPIN/2)<1e-7: return np.array([0,0,1.])
    return -radial if r>0.018 else radial
def write_patch_stl(name,tags):
    triangles=[]
    for st in tags:
        types,_,conn=gmsh.model.mesh.getElements(2,st)
        for typ,c in zip(types,conn):
            props=gmsh.model.mesh.getElementProperties(typ)
            if props[2]!=2 or props[3]<3: continue
            arr=np.asarray(c,dtype=np.int64).reshape(-1,props[3])[:,:3]
            triangles.extend(arr.tolist())
    path=Path('constant/triSurface')/(name+'.stl')
    with path.open('w') as f:
        f.write('solid '+name+'\n')
        for tri in triangles:
            q=np.array([node_map[int(k)] for k in tri]); n=np.cross(q[1]-q[0],q[2]-q[0]); want=desired_normal(name,q.mean(axis=0))
            if np.dot(n,want)<0: q[[1,2]]=q[[2,1]]; n=-n
            n=n/(np.linalg.norm(n) or 1)
            f.write(f' facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n  outer loop\n')
            for v in q: f.write(f'   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n')
            f.write('  endloop\n endfacet\n')
        f.write('endsolid '+name+'\n')
    return len(triangles)
counts={n:write_patch_stl(n,t) for n,t in patch_surfaces.items()}
print('STLs exported before 3D meshing:',counts)

# -- cell 6 -------------------------------------------------------------------------
# The boundary mesher ran, but my STL extractor tested the element-order field instead of the element-
gmsh.model.mesh.clear()
gmsh.option.setNumber('Mesh.CharacteristicLengthMin',0.0005); gmsh.option.setNumber('Mesh.CharacteristicLengthMax',0.0020); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',8)
gmsh.model.mesh.field.setNumber(thr,'SizeMin',0.0005); gmsh.model.mesh.field.setNumber(thr,'SizeMax',0.0020); gmsh.model.mesh.field.setNumber(thr,'DistMin',0.0008); gmsh.model.mesh.field.setNumber(thr,'DistMax',0.004)
gmsh.model.mesh.generate(2)
node_tags,node_xyz,_=gmsh.model.mesh.getNodes(); xyz=np.asarray(node_xyz).reshape(-1,3); node_map={int(t):xyz[i] for i,t in enumerate(node_tags)}
# Corrected exporter: getElementProperties index 1 is topological dimension.
def write_patch_stl(name,tags):
    triangles=[]
    for st in tags:
        types,_,conn=gmsh.model.mesh.getElements(2,st)
        for typ,c in zip(types,conn):
            props=gmsh.model.mesh.getElementProperties(typ)
            if props[1]!=2 or props[3]<3: continue
            triangles.extend(np.asarray(c,dtype=np.int64).reshape(-1,props[3])[:,:3].tolist())
    path=Path('constant/triSurface')/(name+'.stl')
    with path.open('w') as f:
        f.write('solid '+name+'\n')
        for tri in triangles:
            q=np.array([node_map[int(k)] for k in tri]); n=np.cross(q[1]-q[0],q[2]-q[0]); want=desired_normal(name,q.mean(axis=0))
            if np.dot(n,want)<0: q[[1,2]]=q[[2,1]]; n=-n
            n=n/(np.linalg.norm(n) or 1)
            f.write(f' facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n  outer loop\n')
            for v in q: f.write(f'   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n')
            f.write('  endloop\n endfacet\n')
        f.write('endsolid '+name+'\n')
    return len(triangles)
counts={n:write_patch_stl(n,t) for n,t in patch_surfaces.items()}
print('coarse boundary nodes:',len(node_tags),'STL triangles:',counts)
print([(p.name,p.stat().st_size) for p in sorted(Path('constant/triSurface').glob('*.stl'))])

# -- cell 7 -------------------------------------------------------------------------
# The corrected pre-volume exports are nonempty and cover all four patches. The boundary discretizatio
gmsh.option.setNumber('Mesh.Algorithm3D',10)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.option.setNumber('Mesh.Binary',0)
gmsh.model.mesh.generate(3)
# Report tetra count before conversion.
types3,tags3,_=gmsh.model.mesh.getElements(3,vol_tag)
tet_count=sum(len(a) for typ,a in zip(types3,tags3) if gmsh.model.mesh.getElementProperties(typ)[0].startswith('Tetrahedron'))
gmsh.write('oil.msh')
print('3D tetrahedra:',tet_count,'msh bytes:',Path('oil.msh').stat().st_size)
gmsh.finalize()
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile\n{ version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
res=subprocess.run(['gmshToFoam','oil.msh'],text=True,capture_output=True)
print('gmshToFoam return',res.returncode)
print('\n'.join((res.stdout+res.stderr).splitlines()[-18:]))
if res.returncode: raise RuntimeError('gmshToFoam failed')

# -- cell 8 -------------------------------------------------------------------------
# Gmsh completed, but `gmshToFoam` reported 77,848 undefined boundary faces—nearly the entire surface 
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
print('checkMesh return',chk.returncode)
print('\n'.join((chk.stdout+chk.stderr).splitlines()[-60:]))

# -- cell 9 -------------------------------------------------------------------------
# `checkMesh` stopped before reading the mesh because the minimal case lacked `fvSchemes` and `fvSolut
Path('system/fvSchemes').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n')
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
print('checkMesh return',chk.returncode)
print('\n'.join((chk.stdout+chk.stderr).splitlines()[-80:]))

# -- cell 10 ------------------------------------------------------------------------
# OpenFOAM 2512 validates required scheme subdictionaries even for `checkMesh`; I’ll supply the standa
Path('system/fvSchemes').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
print('checkMesh return',chk.returncode)
print('\n'.join((chk.stdout+chk.stderr).splitlines()[-90:]))

# -- cell 11 ------------------------------------------------------------------------
# The conversion is topologically connected and has good conventional quality, but 580 tetrahedra have
print(Path('constant/polyMesh/boundary').read_text()[:2500])

# -- cell 12 ------------------------------------------------------------------------
# The boundary file confirms all intended faces are named, so the defect is confined to HXT’s tetrahed
gmsh.initialize(); gmsh.model.add('flooded_gyro_oil_delaunay'); occ=gmsh.model.occ
cv=occ.addCylinder(0,0,-H_RING/2,0,0,H_RING,R_RING_IN); so=occ.addCylinder(0,0,-H_SPIN/2,0,0,H_SPIN,R_SPIN_OUT); sb=occ.addCylinder(0,0,-H_SPIN/2,0,0,H_SPIN,R_SPIN_IN)
sp,_=occ.cut([(3,so)],[(3,sb)]); ov,_=occ.cut([(3,cv)],sp); occ.synchronize(); vol_tag=ov[0][1]
bnd=gmsh.model.getBoundary(ov,oriented=False); inf={t:(gmsh.model.getType(2,t),occ.getMass(2,t),occ.getCenterOfMass(2,t)) for _,t in bnd}
def pick(k,a,z=None):
 c=[t for t,(kk,aa,cc) in inf.items() if kk==k and (z is None or abs(cc[2]-z)<1e-9)]; return min(c,key=lambda t:abs(inf[t][1]-a))
ring_cyl=pick('Cylinder',2*np.pi*R_RING_IN*H_RING); spin_bore=pick('Cylinder',2*np.pi*R_SPIN_IN*H_SPIN); spin_outer=pick('Cylinder',2*np.pi*R_SPIN_OUT*H_SPIN)
spin_top=pick('Plane',np.pi*(R_SPIN_OUT**2-R_SPIN_IN**2),H_SPIN/2); spin_bottom=pick('Plane',np.pi*(R_SPIN_OUT**2-R_SPIN_IN**2),-H_SPIN/2); ring_top=pick('Plane',np.pi*R_RING_IN**2,H_RING/2); ring_bottom=pick('Plane',np.pi*R_RING_IN**2,-H_RING/2)
patch_surfaces={'outer_ring_wetted':[ring_cyl],'spinner_wetted':[spin_bore,spin_outer,spin_top,spin_bottom],'ring_top_end_face':[ring_top],'ring_bottom_end_face':[ring_bottom]}
for i,(n,ts) in enumerate(patch_surfaces.items(),1): pg=gmsh.model.addPhysicalGroup(2,ts,i); gmsh.model.setPhysicalName(2,pg,n)
pg=gmsh.model.addPhysicalGroup(3,[vol_tag],100); gmsh.model.setPhysicalName(3,pg,'oil')
gmsh.option.setNumber('Mesh.CharacteristicLengthMin',0.0005); gmsh.option.setNumber('Mesh.CharacteristicLengthMax',0.002); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',8)
d=gmsh.model.mesh.field.add('Distance'); gmsh.model.mesh.field.setNumbers(d,'SurfacesList',[ring_cyl,spin_bore,spin_outer,spin_top,spin_bottom]); gmsh.model.mesh.field.setNumber(d,'Sampling',80)
th=gmsh.model.mesh.field.add('Threshold'); gmsh.model.mesh.field.setNumber(th,'InField',d); gmsh.model.mesh.field.setNumber(th,'SizeMin',0.0005); gmsh.model.mesh.field.setNumber(th,'SizeMax',0.002); gmsh.model.mesh.field.setNumber(th,'DistMin',0.0008); gmsh.model.mesh.field.setNumber(th,'DistMax',0.004); gmsh.model.mesh.field.setAsBackgroundMesh(th)
gmsh.option.setNumber('Mesh.Algorithm3D',1); gmsh.model.mesh.generate(3); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0); gmsh.write('oil.msh')
types,tags,_=gmsh.model.mesh.getElements(3,vol_tag); print('Delaunay volume elements:',sum(len(x) for x in tags)); gmsh.finalize()
shutil.rmtree('constant/polyMesh',ignore_errors=True)
r=subprocess.run(['gmshToFoam','oil.msh'],text=True,capture_output=True); print('converter',r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-15:])); assert r.returncode==0

# -- cell 13 ------------------------------------------------------------------------
# The standard Delaunay mesh optimized without ill-shaped tetrahedra and now labels the cell zone corr
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
out=chk.stdout+chk.stderr
print('checkMesh return',chk.returncode)
print('\n'.join(out.splitlines()[-75:]))

# -- cell 14 ------------------------------------------------------------------------
# The same 580 cells fail after both tetrahedralizers, so the issue is not tetra quality; it is the co
h=subprocess.run(['gmshToFoam','-help-full'],text=True,capture_output=True)
print(h.stdout+h.stderr)

# -- cell 15 ------------------------------------------------------------------------
# There is no converter option controlling point merging. The failing determinant metric is a known we
shutil.rmtree('constant/polyMesh_tet_backup',ignore_errors=True); shutil.copytree('constant/polyMesh','constant/polyMesh_tet_backup')
h=subprocess.run(['polyDualMesh','-help'],text=True,capture_output=True); print(h.stdout+h.stderr)

# -- cell 16 ------------------------------------------------------------------------
# I’ll dualize with a 30° feature threshold so the 90° spinner/ring edges and all patch junctions are 
pd=subprocess.run(['polyDualMesh','30','-overwrite'],text=True,capture_output=True)
print('polyDualMesh return',pd.returncode); print('\n'.join((pd.stdout+pd.stderr).splitlines()[-35:]));
if pd.returncode: raise RuntimeError('polyDualMesh failed')

# -- cell 17 ------------------------------------------------------------------------
# `polyDualMesh` wrote the dual mesh at time `0` rather than replacing `constant/polyMesh`, because di
print('mesh dirs:',[(str(p),p.is_dir()) for p in [Path('constant/polyMesh'),Path('0/polyMesh')]])
chk2=subprocess.run(['checkMesh','-allGeometry','-allTopology','-time','0'],text=True,capture_output=True)
print('dual check return',chk2.returncode); print('\n'.join((chk2.stdout+chk2.stderr).splitlines()[-65:]))

# -- cell 18 ------------------------------------------------------------------------
# The dual mesh removes the determinant issue but introduces three worse polyhedral quality failures, 
shutil.rmtree('0',ignore_errors=True); shutil.rmtree('constant/polyMesh',ignore_errors=True)
Path('system/blockMeshDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ((-0.024 -0.024 -0.006) (0.024 -0.024 -0.006) (0.024 0.024 -0.006) (-0.024 0.024 -0.006) (-0.024 -0.024 0.006) (0.024 -0.024 0.006) (0.024 0.024 0.006) (-0.024 0.024 0.006));
blocks (hex (0 1 2 3 4 5 6 7) (24 24 6) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
''')
Path('system/snappyHexMeshDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry {
 outer_ring_wetted.stl { type triSurfaceMesh; name outer_ring_wetted; }
 spinner_wetted.stl { type triSurfaceMesh; name spinner_wetted; }
 ring_top_end_face.stl { type triSurfaceMesh; name ring_top_end_face; }
 ring_bottom_end_face.stl { type triSurfaceMesh; name ring_bottom_end_face; }
}
castellatedMeshControls {
 maxLocalCells 400000; maxGlobalCells 800000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ();
 refinementSurfaces {
  outer_ring_wetted { level (2 2); patchInfo { type wall; } }
  spinner_wetted { level (2 2); patchInfo { type wall; } }
  ring_top_end_face { level (2 2); patchInfo { type patch; } }
  ring_bottom_end_face { level (2 2); patchInfo { type patch; } }
 }
 resolveFeatureAngle 30;
 refinementRegions { }
 locationInMesh (0 0 0);
 allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8; nFeatureSnapIter 0; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 20; }
meshQualityControls { maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-18; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
debug 0; mergeTolerance 1e-6;
''')
bm=subprocess.run(['blockMesh'],text=True,capture_output=True); print('blockMesh',bm.returncode); print('\n'.join((bm.stdout+bm.stderr).splitlines()[-20:])); assert bm.returncode==0

# -- cell 19 ------------------------------------------------------------------------
# The background mesh is 3,456 cells as intended. I’ll now run castellated refinement and snapping aga
sn=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
print('snappy return',sn.returncode); print('\n'.join((sn.stdout+sn.stderr).splitlines()[-55:]));
if sn.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 20 ------------------------------------------------------------------------
# Snappy produced a 45,824-cell predominantly hexahedral mesh with no quality violations under its con
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
out=chk.stdout+chk.stderr; print('checkMesh return',chk.returncode); print('\n'.join(out.splitlines()[-90:]))

# -- cell 21 ------------------------------------------------------------------------
# The mesh is closed, one region, correctly patched, and otherwise high quality, but the full checker 
p=Path('system/snappyHexMeshDict'); txt=p.read_text().replace('maxConcave 80;','maxConcave 10;'); p.write_text(txt)
shutil.rmtree('constant/polyMesh',ignore_errors=True)
bm=subprocess.run(['blockMesh'],text=True,capture_output=True); assert bm.returncode==0
sn=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
print('snappy',sn.returncode); print('\n'.join((sn.stdout+sn.stderr).splitlines()[-35:])); assert sn.returncode==0

# -- cell 22 ------------------------------------------------------------------------
# The tightened run completed with zero faces exceeding 10° concavity under Snappy’s quality controls.
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); out=chk.stdout+chk.stderr
print('checkMesh return',chk.returncode); print('\n'.join(out.splitlines()[-65:]))

# -- cell 23 ------------------------------------------------------------------------
# The stricter snapping removes concave faces but not 432 concave *cells* inherited from the castellat
for cmd in ['splitCells','refineMesh']:
 r=subprocess.run([cmd,'-help'],text=True,capture_output=True); print('\n###',cmd,'\n',r.stdout+r.stderr)

# -- cell 24 ------------------------------------------------------------------------
# `splitCells` supports a targeted cell set and geometric cuts. I’ll back up the current snapped mesh,
print('set locations:',list(Path('.').glob('**/concaveCells'))[:10])
shutil.rmtree('constant/polyMesh_snappy_backup',ignore_errors=True); shutil.copytree('constant/polyMesh','constant/polyMesh_snappy_backup')
spc=subprocess.run(['splitCells','10','-geometry','-set','concaveCells','-overwrite'],text=True,capture_output=True)
print('splitCells return',spc.returncode); print('\n'.join((spc.stdout+spc.stderr).splitlines()[-45:]))

# -- cell 25 ------------------------------------------------------------------------
# All 432 candidate cells were split in two passes. I’ll run the full checker on the resulting mesh.
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); out=chk.stdout+chk.stderr
print('checkMesh return',chk.returncode); print('\n'.join(out.splitlines()[-75:]))

# -- cell 26 ------------------------------------------------------------------------
# Targeted `splitCells` worsened quality, so I’m restoring the pre-split snapped mesh and abandoning t
shutil.rmtree('constant/polyMesh',ignore_errors=True); shutil.copytree('constant/polyMesh_snappy_backup','constant/polyMesh'); shutil.rmtree('constant/polyMesh_snappy_backup',ignore_errors=True)
# A mesher-only aggregate; the four required one-patch STLs remain in triSurface.
combined=''.join(Path('constant/triSurface',n+'.stl').read_text() for n in ['outer_ring_wetted','spinner_wetted','ring_top_end_face','ring_bottom_end_face'])
Path('constant/allPatches.stl').write_text(combined)
Path('system/meshDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object meshDict; }
surfaceFile "constant/allPatches.stl";
maxCellSize 0.002;
minCellSize 0.0005;
boundaryCellSize 0.0005;
keepCellsIntersectingBoundary 1;
checkForGluedMesh 0;
''')
h=subprocess.run(['cartesianMesh','-help'],text=True,capture_output=True); print('\n'.join((h.stdout+h.stderr).splitlines()[:25]))

# -- cell 27 ------------------------------------------------------------------------
# I’ll run cfMesh with 2 mm bulk cells and 0.5 mm boundary cells, matching the successful Snappy resol
shutil.rmtree('constant/polyMesh',ignore_errors=True)
cm=subprocess.run(['cartesianMesh'],text=True,capture_output=True,timeout=240)
print('cartesianMesh return',cm.returncode); print('\n'.join((cm.stdout+cm.stderr).splitlines()[-60:]));
if cm.returncode: raise RuntimeError('cartesianMesh failed')

# -- cell 28 ------------------------------------------------------------------------
# cfMesh completed and reported zero inverted or bad faces after untangling. I’ll now run full checks 
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); out=chk.stdout+chk.stderr
print('checkMesh return',chk.returncode); print('\n'.join(out.splitlines()[-90:]))

# -- cell 29 ------------------------------------------------------------------------
# cfMesh’s quality is excellent, but it ignored the internal spinner (empty spinner patch and full-cyl
p=Path('system/meshDict'); p.write_text(p.read_text().replace('checkForGluedMesh 0;','checkForGluedMesh 1;'))
shutil.rmtree('constant/polyMesh',ignore_errors=True)
cm=subprocess.run(['cartesianMesh'],text=True,capture_output=True,timeout=240)
print('cartesianMesh return',cm.returncode); print('\n'.join((cm.stdout+cm.stderr).splitlines()[-35:])); assert cm.returncode==0

# -- cell 30 ------------------------------------------------------------------------
# The glued-surface run found and untangled the internal interface, ending with zero bad faces. I’ll v
chk=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); out=chk.stdout+chk.stderr
print('checkMesh return',chk.returncode); print('\n'.join(out.splitlines()[-90:]))
