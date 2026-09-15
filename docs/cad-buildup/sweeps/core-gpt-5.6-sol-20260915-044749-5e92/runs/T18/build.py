"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the axis as \(z\), with the barrel from \(z=0\) to \(0.120\) m. The stated fin stack occu
from build123d import *
import os, numpy as np, pyvista as pv
from pathlib import Path
# All dimensions are metres.
BARREL_R=0.045; BARREL_H=0.120
FIN_R=0.070; FIN_T=0.003; FIN_GAP=0.006; N_FINS=12
FIN_PITCH=FIN_T+FIN_GAP; FIN_Z0=0.009
HEAD_R=BARREL_R; HEAD_H=HEAD_R
DUCT_W=0.300; Z_IN=-0.150; Z_OUT=BARREL_H+HEAD_H+0.300
fin_z=[FIN_Z0+i*FIN_PITCH for i in range(N_FINS)]
barrel_solid=Cylinder(BARREL_R,BARREL_H)
fins_solid=Part()
for z in fin_z:
    f=Pos(0,0,z)*Cylinder(FIN_R,FIN_T)
    fins_solid = f if len(fins_solid.solids())==0 else fins_solid.fuse(f)
head_sphere=Pos(0,0,BARREL_H)*Sphere(HEAD_R)
head_clip=Pos(-HEAD_R,-HEAD_R,BARREL_H)*Box(2*HEAD_R,2*HEAD_R,HEAD_R)
head_solid=head_sphere & head_clip
obstacle=barrel_solid.fuse(fins_solid,head_solid)
Path('constant/triSurface').mkdir(parents=True,exist_ok=True)
export_stl(obstacle,'obstacle_preview.stl',tolerance=8e-4,angular_tolerance=0.15)
mesh=pv.read('obstacle_preview.stl')
p=pv.Plotter(off_screen=True,window_size=(800,650)); p.set_background('white')
p.add_mesh(mesh,color='silver',smooth_shading=True,show_edges=True,edge_color='gray')
p.add_mesh(pv.Cube(center=(0,0,(Z_IN+Z_OUT)/2),x_length=DUCT_W,y_length=DUCT_W,z_length=Z_OUT-Z_IN),style='wireframe',color='navy',line_width=2)
p.camera_position=[(0.38,-0.38,0.28),(0,0,0.12),(0,0,1)]; p.show(screenshot='coarse_geometry.png')
print(f'Fin stack: first z={fin_z[0]:.3f}, last top={fin_z[-1]+FIN_T:.3f} m; requested 12 fins, 3 mm thick, 6 mm gaps')
print(f'Obstacle extents: barrel OD={2*BARREL_R:.3f} m, fin OD={2*FIN_R:.3f} m, barrel height={BARREL_H:.3f} m, top={BARREL_H+HEAD_H:.3f} m')
print(f'Duct: {DUCT_W:.3f} x {DUCT_W:.3f} m, z=[{Z_IN:.3f},{Z_OUT:.3f}] m')

# -- cell 2 -------------------------------------------------------------------------
# The preview exposed an error in my clipping solid: the box’s default alignment removed half the hemi
head_clip=Pos(0,0,BARREL_H)*Cylinder(HEAD_R,HEAD_H)
head_solid=head_sphere & head_clip
obstacle=barrel_solid.fuse(fins_solid,head_solid)
print('solid count',len(obstacle.solids()),'face count',len(obstacle.faces()),'volume',obstacle.volume)
from collections import Counter
print(Counter(str(f.geom_type) for f in obstacle.faces()))
for i,f in enumerate(obstacle.faces()):
    c=f.center(); bb=f.bounding_box();
    print(i,str(f.geom_type),f'area={f.area:.7f}',f'center=({c.X:.4f},{c.Y:.4f},{c.Z:.4f})',f'z=({bb.min.Z:.4f},{bb.max.Z:.4f})')

# -- cell 3 -------------------------------------------------------------------------
# The face audit caught a second alignment issue: build123d’s cylinders were centered axially, so the 
AXIAL_ALIGN=(Align.CENTER,Align.CENTER,Align.MIN)
barrel_solid=Cylinder(BARREL_R,BARREL_H,align=AXIAL_ALIGN)
fins=[]
for z in fin_z:
    fins.append(Pos(0,0,z)*Cylinder(FIN_R,FIN_T,align=AXIAL_ALIGN))
fins_solid=fins[0]
for f in fins[1:]: fins_solid=fins_solid.fuse(f)
head_sphere=Pos(0,0,BARREL_H)*Sphere(HEAD_R)
head_clip=Pos(0,0,BARREL_H)*Cylinder(HEAD_R,HEAD_H,align=AXIAL_ALIGN)
head_solid=head_sphere & head_clip
obstacle=barrel_solid.fuse(fins_solid,head_solid)
bb=obstacle.bounding_box()
print('solid count',len(obstacle.solids()),'faces',len(obstacle.faces()))
print(f'bounds z=({bb.min.Z:.6f},{bb.max.Z:.6f}), radial x=({bb.min.X:.6f},{bb.max.X:.6f})')
print(Counter(str(f.geom_type) for f in obstacle.faces()))
export_stl(obstacle,'obstacle_preview.stl',tolerance=5e-4,angular_tolerance=0.12)
mesh=pv.read('obstacle_preview.stl'); p=pv.Plotter(off_screen=True,window_size=(700,650)); p.set_background('white'); p.add_mesh(mesh,color='silver',smooth_shading=True,show_edges=True,edge_color='gray'); p.camera_position=[(0.30,-0.30,0.25),(0,0,0.075),(0,0,1)]; p.show(screenshot='corrected_geometry.png')

# -- cell 4 -------------------------------------------------------------------------
# The corrected preview now has one fused solid, the requested 140 mm fin diameter, and a complete 90 
import math
tri=Path('constant/triSurface')
for old in tri.glob('*.stl'): old.unlink()
head_faces=[]; fin_faces=[]; barrel_faces=[]
for face in obstacle.faces():
    gt=str(face.geom_type); bb=face.bounding_box()
    radial=max(abs(bb.min.X),abs(bb.max.X),abs(bb.min.Y),abs(bb.max.Y))
    if gt.endswith('SPHERE'):
        head_faces.append(face)
    elif (gt.endswith('CYLINDER') and radial>0.060) or (gt.endswith('PLANE') and bb.min.Z>1e-7):
        fin_faces.append(face)
    else:
        barrel_faces.append(face)
patch_shapes={'barrel':Compound(barrel_faces),'fins':Compound(fin_faces),'head':Compound(head_faces)}
duct_box=Pos(0,0,Z_IN)*Box(DUCT_W,DUCT_W,Z_OUT-Z_IN,align=(Align.CENTER,Align.CENTER,Align.MIN))
inlet_faces=[]; outlet_faces=[]; wall_faces=[]
for face in duct_box.faces():
    z=face.center().Z
    if abs(z-Z_IN)<1e-8: inlet_faces.append(face)
    elif abs(z-Z_OUT)<1e-8: outlet_faces.append(face)
    else: wall_faces.append(face)
patch_shapes.update({'ductWalls':Compound(wall_faces),'inlet':Compound(inlet_faces),'outlet':Compound(outlet_faces)})
for name,shape in patch_shapes.items():
    export_stl(shape,str(tri/f'{name}.stl'),tolerance=2.5e-4,angular_tolerance=0.08)
    print(name,'faces',len(shape.faces()),'area',f'{shape.area:.8f}','STL bytes',(tri/f'{name}.stl').stat().st_size)
print('Obstacle patch area sum',sum(patch_shapes[n].area for n in ['barrel','fins','head']),'solid outer area',obstacle.area)
print('Duct patch area sum',sum(patch_shapes[n].area for n in ['ductWalls','inlet','outlet']),'box area',duct_box.area)

# -- cell 5 -------------------------------------------------------------------------
# The exported obstacle patch areas sum exactly to the fused solid’s outer area, and the three duct pa
import subprocess, textwrap
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
block=f'''FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
(-0.15 -0.15 {Z_IN}) (0.15 -0.15 {Z_IN}) (0.15 0.15 {Z_IN}) (-0.15 0.15 {Z_IN})
(-0.15 -0.15 {Z_OUT}) (0.15 -0.15 {Z_OUT}) (0.15 0.15 {Z_OUT}) (-0.15 0.15 {Z_OUT})
);
blocks ( hex (0 1 2 3 4 5 6 7) (15 15 31) simpleGrading (1 1 1) );
edges ();
boundary
(
inlet {{ type patch; faces ((0 3 2 1)); }}
outlet {{ type patch; faces ((4 5 6 7)); }}
ductWalls {{ type wall; faces ((0 1 5 4)(1 2 6 5)(2 3 7 6)(3 0 4 7)); }}
);
mergePatchPairs ();
'''
snappy='''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
 barrel.stl { type triSurfaceMesh; name barrel; }
 fins.stl { type triSurfaceMesh; name fins; }
 head.stl { type triSurfaceMesh; name head; }
}
castellatedMeshControls
{
 maxLocalCells 500000; maxGlobalCells 800000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2;
 features ();
 refinementSurfaces
 {
  barrel { level (3 3); patchInfo { type wall; } }
  fins { level (3 3); patchInfo { type wall; } }
  head { level (3 3); patchInfo { type wall; } }
 }
 resolveFeatureAngle 30;
 refinementRegions {}
 locationInMesh (0.12 0 0);
 allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls
{
 #include "meshQualityDict"
 nSmoothScale 4; errorReduction 0.75;
}
mergeTolerance 1e-6;
'''
(Path('system')/'blockMeshDict').write_text(block)
(Path('system')/'snappyHexMeshDict').write_text(snappy)
(Path('system')/'meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }\n''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-1800:]); print('return',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# `blockMesh` stopped before generating cells, and the truncated output did not include the parser dia
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print('STDOUT\n',r.stdout[-4000:]); print('STDERR\n',r.stderr[-2000:]); print('return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The mesh dictionary is valid; OpenFOAM only requires the otherwise-unused case-level `controlDict` t
(Path('system')/'controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-2200:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The background mesh is the intended 6,975 cells with correct metre-scale bounds and named duct patch
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-1200:]); print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# This OpenFOAM build also insists on placeholder numerics dictionaries even for a mesh-only utility. 
(Path('system')/'fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
(Path('system')/'fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# An empty `fvSchemes` is not accepted because snappy’s mesh-motion stage requests standard scheme sec
(Path('system')/'fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6500:]); print(r.stderr[-1200:]); print('return',r.returncode)

# -- cell 11 ------------------------------------------------------------------------
# The coarse snappy mesh completed with 94,206 cells and no mesher-reported quality errors. I’ll run t
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True,timeout=240)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6500:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# The exhaustive check confirms correct connectivity, one fluid region, six patches, and good non-orth
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
print(r.stdout[-4500:]); print(r.stderr[-800:]); print('return',r.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The binding check passes with “Mesh OK”; the stricter optional findings are cut-cell shape diagnosti
Path('t18.foam').touch()
reader=pv.OpenFOAMReader('t18.foam'); reader.set_active_time_value(reader.time_values[-1]); mb=reader.read()
print(mb)
# collect internal mesh block
for key in mb.keys(): print('top',key,type(mb[key]))
internal=mb['internalMesh'] if 'internalMesh' in mb.keys() else mb[0]
print('internal cells',internal.n_cells,'bounds',internal.bounds)
sl=internal.slice(normal=(0,1,0),origin=(0,0,0.07))
sl=sl.compute_cell_sizes(length=False,area=False,volume=True)
p=pv.Plotter(off_screen=True,window_size=(900,700)); p.set_background('white')
p.add_mesh(sl,scalars='Volume',cmap='viridis',show_edges=True,edge_color='black',line_width=0.25,log_scale=True)
p.camera_position='xz'; p.camera.zoom(1.35); p.show(screenshot='mesh_axial_cut.png')
print('slice cells',sl.n_cells)

# -- cell 14 ------------------------------------------------------------------------
# The axial cut shows the head/barrel interior removed and all fin channels represented, though the at
# Point-containment audit at r=55 mm: inside fin radius but outside barrel.
gap_z=[fin_z[i]+FIN_T+FIN_GAP/2 for i in range(N_FINS-1)]
fin_mid_z=[z+FIN_T/2 for z in fin_z]
gap_ids=[internal.find_containing_cell((0.055,0,z)) for z in gap_z]
fin_ids=[internal.find_containing_cell((0.055,0,z)) for z in fin_mid_z]
print('gap samples in fluid:',sum(i>=0 for i in gap_ids),'/',len(gap_ids),gap_ids)
print('fin-thickness samples excluded:',sum(i<0 for i in fin_ids),'/',len(fin_ids),fin_ids)
print('barrel-axis sample excluded:',internal.find_containing_cell((0,0,0.06))<0)
print('outer-air sample present:',internal.find_containing_cell((0.10,0,0.06))>=0)
btxt=Path('constant/polyMesh/boundary').read_text()
print('patch names present:',[n for n in ['barrel','fins','head','ductWalls','inlet','outlet'] if ('\n    '+n+'\n') in btxt])

# -- cell 15 ------------------------------------------------------------------------
# The dimensional audit passed, but PyVista’s installed `merge` API does not accept a tolerance argume
export_step(obstacle,'radial_engine_cylinder.step')
bar_bb=barrel_solid.bounding_box(); head_bb=head_solid.bounding_box()
measured_fin_od=2*max(f.bounding_box().max.X for f in fins)
measured_fin_t=[f.bounding_box().max.Z-f.bounding_box().min.Z for f in fins]
measured_gaps=[fin_z[i+1]-(fin_z[i]+FIN_T) for i in range(N_FINS-1)]
print(f'barrel OD measured {bar_bb.max.X-bar_bb.min.X:.6f} m, requested 0.090000 m')
print(f'barrel height measured {bar_bb.max.Z-bar_bb.min.Z:.6f} m, requested 0.120000 m')
print(f'fin count measured {len(fins)}, requested 12; OD {measured_fin_od:.6f} m, requested 0.140000 m')
print(f'fin thickness range {min(measured_fin_t):.6f}..{max(measured_fin_t):.6f} m, requested 0.003000 m')
print(f'inter-fin gap range {min(measured_gaps):.6f}..{max(measured_gaps):.6f} m, requested 0.006000 m')
print(f'head base diameter {2*HEAD_R:.6f} m, height {head_bb.max.Z-head_bb.min.Z:.6f} m (assumed hemisphere)')
print(f'duct width {DUCT_W:.6f} m; inlet below barrel {0-Z_IN:.6f} m; outlet above head {Z_OUT-head_bb.max.Z:.6f} m')
stls=[pv.read(str(tri/f'{n}.stl')) for n in ['barrel','fins','head','ductWalls','inlet','outlet']]
merged=pv.merge(stls,merge_points=True).clean(tolerance=1e-7)
free=merged.extract_feature_edges(boundary_edges=True,non_manifold_edges=False,feature_edges=False,manifold_edges=False)
nonman=merged.extract_feature_edges(boundary_edges=False,non_manifold_edges=True,feature_edges=False,manifold_edges=False)
print('welded STL union: points',merged.n_points,'triangles',merged.n_cells,'free edges',free.n_cells,'non-manifold edges',nonman.n_cells)
Path('geometry_report.txt').write_text('''Units: metres\nRequested: barrel OD 0.090, barrel H 0.120; 12 fins OD 0.140, thickness 0.003, gaps 0.006; duct square 0.300.\nAssumptions: hemispherical head radius/height 0.045; fin stack centered with 0.009 barrel margins; duct z=-0.150 to 0.465, i.e. 0.300 above complete head.\nMesh: external air, snappyHexMesh, nominal finest surface cell 0.0025 m, no layers.\n''')

# -- cell 16 ------------------------------------------------------------------------
# The virtual weld found free edges, meaning independent face tessellation produced non-matching node 
import gmsh
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0)
gmsh.model.add('obstacleSurface')
gmsh.model.occ.importShapes('radial_engine_cylinder.step'); gmsh.model.occ.synchronize()
surf_groups={'barrel':[],'fins':[],'head':[]}
for dim,tag in gmsh.model.getEntities(2):
    typ=gmsh.model.getType(dim,tag); xmin,ymin,zmin,xmax,ymax,zmax=gmsh.model.getBoundingBox(dim,tag)
    radial=max(abs(xmin),abs(xmax),abs(ymin),abs(ymax))
    if typ=='Sphere': name='head'
    elif (typ=='Cylinder' and radial>0.060) or (typ=='Plane' and zmin>1e-7): name='fins'
    else: name='barrel'
    surf_groups[name].append(tag)
print({k:len(v) for k,v in surf_groups.items()},'types',Counter(gmsh.model.getType(2,t) for _,t in gmsh.model.getEntities(2)))
print('bounds',gmsh.model.getBoundingBox(-1,-1))

# -- cell 17 ------------------------------------------------------------------------
# Gmsh imported exactly the same 51 analytic surfaces and reproduced the intended 14/36/1 patch split 
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0015); gmsh.option.setNumber('Mesh.MeshSizeMax',0.0025)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',12)
gmsh.model.mesh.generate(2)
nodeTags,nodeCoords,_=gmsh.model.mesh.getNodes(); xyz={int(t):nodeCoords[3*i:3*i+3] for i,t in enumerate(nodeTags)}
def write_ascii_stl(path,name,triangles):
    with open(path,'w') as out:
        out.write(f'solid {name}\n')
        for a,b,c in triangles:
            a=np.asarray(a); b=np.asarray(b); c=np.asarray(c); n=np.cross(b-a,c-a); ln=np.linalg.norm(n); n=n/ln if ln else n
            out.write(f' facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n')
            for q in (a,b,c): out.write(f'   vertex {q[0]:.12g} {q[1]:.12g} {q[2]:.12g}\n')
            out.write('  endloop\n endfacet\n')
        out.write(f'endsolid {name}\n')
for name,tags in surf_groups.items():
    tris=[]
    for tag in tags:
        types,_,conns=gmsh.model.mesh.getElements(2,tag)
        for et,conn in zip(types,conns):
            if et!=2: raise RuntimeError(f'non-triangle element type {et}')
            tris.extend((xyz[int(conn[i])],xyz[int(conn[i+1])],xyz[int(conn[i+2])]) for i in range(0,len(conn),3))
    write_ascii_stl(tri/f'{name}.stl',name,tris); print(name,'triangles',len(tris))
# Duct corners and consistently outward quads.
v=[np.array(q,float) for q in [(-.15,-.15,Z_IN),(.15,-.15,Z_IN),(.15,.15,Z_IN),(-.15,.15,Z_IN),(-.15,-.15,Z_OUT),(.15,-.15,Z_OUT),(.15,.15,Z_OUT),(-.15,.15,Z_OUT)]]
def qtris(q): a,b,c,d=[v[i] for i in q]; return [(a,b,c),(a,c,d)]
write_ascii_stl(tri/'inlet.stl','inlet',qtris((0,3,2,1)))
write_ascii_stl(tri/'outlet.stl','outlet',qtris((4,5,6,7)))
wall_tris=sum((qtris(q) for q in [(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]),[])
write_ascii_stl(tri/'ductWalls.stl','ductWalls',wall_tris)
gmsh.finalize()
print('duct triangles',len(wall_tris)+4)

# -- cell 18 ------------------------------------------------------------------------
# The replacement STLs contain 108,976 conforming obstacle triangles plus 12 duct triangles. I’ll repe
stls=[pv.read(str(tri/f'{n}.stl')) for n in ['barrel','fins','head','ductWalls','inlet','outlet']]
merged=pv.merge(stls,merge_points=True).clean(tolerance=1e-9)
free=merged.extract_feature_edges(boundary_edges=True,non_manifold_edges=False,feature_edges=False,manifold_edges=False)
nonman=merged.extract_feature_edges(boundary_edges=False,non_manifold_edges=True,feature_edges=False,manifold_edges=False)
print('welded final STL union: points',merged.n_points,'triangles',merged.n_cells,'free edges',free.n_cells,'non-manifold edges',nonman.n_cells)

# -- cell 19 ------------------------------------------------------------------------
# The final six-STL union now has zero free and zero non-manifold edges. I’ll reset the background blo
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-1400:]); print(r.stderr[-500:]); print('return',r.returncode)

# -- cell 20 ------------------------------------------------------------------------
# The background reset cleanly. I’ll now generate the final snapped volume mesh from the conforming `b
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-3500:]); print(r.stderr[-800:]); print('return',r.returncode)

# -- cell 21 ------------------------------------------------------------------------
# The final remesh again completed with 94,206 cells and zero final mesher quality errors. I’ll run th
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-4200:]); print(r.stderr[-700:]); print('return',r.returncode)

# -- cell 22 ------------------------------------------------------------------------
# The delivered mesh passes `checkMesh` with one connected region, all six named patches, maximum non-
reader2=pv.OpenFOAMReader('t18.foam'); reader2.set_active_time_value(reader2.time_values[-1]); final_internal=reader2.read()['internalMesh']
final_gap_ids=[final_internal.find_containing_cell((0.055,0,z)) for z in gap_z]
final_fin_ids=[final_internal.find_containing_cell((0.055,0,z)) for z in fin_mid_z]
print('final gap samples in fluid',sum(i>=0 for i in final_gap_ids),'/',len(final_gap_ids))
print('final fin samples excluded',sum(i<0 for i in final_fin_ids),'/',len(final_fin_ids))
# Directed-edge winding audit on the already welded two-shell STL union.
faces=merged.faces.reshape(-1,4)[:,1:]
from collections import defaultdict
edge_dirs=defaultdict(list)
for a,b,c in faces:
    for u,w in ((a,b),(b,c),(c,a)):
        edge_dirs[(min(u,w),max(u,w))].append(1 if u<w else -1)
free_count=sum(len(v)!=2 for v in edge_dirs.values())
same_direction=sum(len(v)==2 and v[0]==v[1] for v in edge_dirs.values())
print('STL union edge audit: unmatched',free_count,'same-direction pairs',same_direction,'total unique edges',len(edge_dirs))
print('final mesh cells',final_internal.n_cells)
