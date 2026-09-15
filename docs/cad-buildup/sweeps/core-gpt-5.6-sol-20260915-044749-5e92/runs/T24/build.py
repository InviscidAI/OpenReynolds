"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first inspection exposed an important issue before meshing: `Cylinder` defaults to axial centeri
import build123d as bd, math, numpy as np
from build123d import Cylinder, Box, Pos, Rot
D=0.120; R=D/2; H=0.200; OUT_D=0.040
PORT_W=0.020; PORT_H=0.030; PORT_Z0=0.040; DUCT_L=0.050; ANG=15.0
chamber=Cylinder(R,H,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
ducts=[]; flow_axes=[]
for a_deg in (0,90,180,270):
    a=math.radians(a_deg); er=np.array([math.cos(a),math.sin(a),0.]); et=np.array([-math.sin(a),math.cos(a),0.])
    flow=math.cos(math.radians(ANG))*et-math.sin(math.radians(ANG))*er; outer=-flow
    wall_center=np.array([R*math.cos(a),R*math.sin(a),PORT_Z0]); rot=math.degrees(math.atan2(outer[1],outer[0]))
    duct=Pos(*wall_center)*Rot(0,0,rot)*Box(DUCT_L,PORT_W,PORT_H,align=(bd.Align.MIN,bd.Align.CENTER,bd.Align.MIN))
    ducts.append(duct); flow_axes.append(flow)
fluid=chamber
for duct in ducts: fluid=fluid+duct
print('solid count:',len(fluid.solids()),'volume:',fluid.volume)
print('z bounds:',fluid.bounding_box().min.Z,fluid.bounding_box().max.Z)
for i,f in enumerate(flow_axes,1):
    a=math.radians((i-1)*90); et=np.array([-math.sin(a),math.cos(a),0.]); measured=math.degrees(math.acos(np.dot(f,et)))
    print(f'inlet{i} axis angle off local tangent: {measured:.6f} deg (requested {ANG})')
bd.export_step(fluid,'coarse_fluid.step'); bd.export_stl(fluid,'coarse_fluid.stl',tolerance=0.002,angular_tolerance=0.2)
import pyvista as pv
m=pv.read('coarse_fluid.stl'); pl=pv.Plotter(off_screen=True,window_size=(900,650)); pl.add_mesh(m,color='lightsteelblue',show_edges=True); pl.view_isometric(); pl.add_axes(); pl.show(screenshot='coarse_fluid.png'); print('wrote coarse_fluid.png')

# -- cell 2 -------------------------------------------------------------------------
# The coarse view confirms the intended tangential arrangement and a single connected fluid solid. Nex
faces=list(fluid.faces())
print('boundary faces:',len(faces),'total area',sum(f.area for f in faces))
for j,f in enumerate(faces):
    c=f.center(); n=f.normal_at(); gt=type(f.geom_type).__name__ if not isinstance(f.geom_type,str) else f.geom_type
    print(j, str(f.geom_type), f'area={f.area:.8f}', f'c=({c.X:.4f},{c.Y:.4f},{c.Z:.4f})', f'n=({n.X:.3f},{n.Y:.3f},{n.Z:.3f})')

# -- cell 3 -------------------------------------------------------------------------
# The union has 27 boundary faces: one cylindrical wall, floor, top, four inlet planes, and five expos
import os
os.makedirs('constant/triSurface',exist_ok=True)
# Identify exact final-boundary faces by geometry/provenance.
cyl_faces=[f for f in fluid.faces() if f.geom_type==bd.GeomType.CYLINDER]
plane_faces=[f for f in fluid.faces() if f.geom_type==bd.GeomType.PLANE]
floor_faces=[f for f in plane_faces if abs(f.center().Z)<1e-9 and f.area>0.005]
full_top=[f for f in plane_faces if abs(f.center().Z-H)<1e-9 and f.area>0.005][0]
inlet_faces=[]
for f in plane_faces:
    if abs(f.area-PORT_W*PORT_H)<1e-8:
        inlet_faces.append(f)
# Sort inlet planes by associated duct azimuth 0,90,180,270 using nearest expected inlet center.
expected=[]
for a_deg in (0,90,180,270):
    a=math.radians(a_deg); er=np.array([math.cos(a),math.sin(a)]); et=np.array([-math.sin(a),math.cos(a)])
    flow=math.cos(math.radians(ANG))*et-math.sin(math.radians(ANG))*er; outer=-flow
    expected.append(R*er+DUCT_L*outer)
inlet_faces=sorted(inlet_faces,key=lambda f:min(range(4),key=lambda k:np.linalg.norm(np.array([f.center().X,f.center().Y])-expected[k])))
# Ensure actual order explicitly by nearest one-to-one expected location.
inlet_faces=[min(inlet_faces,key=lambda f:np.linalg.norm(np.array([f.center().X,f.center().Y])-p)) for p in expected]
# Split top while preserving the original top face orientation.
outlet_tool=Pos(0,0,H-0.001)*Cylinder(OUT_D/2,0.002,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
outlet_face=full_top & outlet_tool
top_face=full_top - outlet_tool
# Remaining planar faces are assigned to the duct whose original solid contains their center (slightly inward-normal offset).
excluded_ids={id(f) for f in floor_faces+inlet_faces+[full_top]}
duct_wall_faces=[[] for _ in range(4)]
for f in plane_faces:
    if id(f) in excluded_ids: continue
    c=f.center(); p=np.array([c.X,c.Y]); k=min(range(4),key=lambda q:np.linalg.norm(p-expected[q]))
    duct_wall_faces[k].append(f)
patch_shapes={'inlet1':inlet_faces[0],'inlet2':inlet_faces[1],'inlet3':inlet_faces[2],'inlet4':inlet_faces[3],
              'outlet':outlet_face,'cylinderWall':bd.Compound(cyl_faces),'floor':bd.Compound(floor_faces),'top':top_face}
for k in range(4): patch_shapes[f'ductWall{k+1}']=bd.Compound(duct_wall_faces[k])
for name,shape in patch_shapes.items():
    bd.export_stl(shape,f'constant/triSurface/{name}.stl',tolerance=0.0005,angular_tolerance=0.08)
    print(name,'faces',len(shape.faces()),'area',shape.area)
print('outlet diameter measured from area:',2*math.sqrt(outlet_face.area/math.pi),'requested',OUT_D)
print('patch area sum:',sum(s.area for s in patch_shapes.values()),'original boundary area:',sum(f.area for f in fluid.faces()))

# -- cell 4 -------------------------------------------------------------------------
# All 12 requested patch STLs are exported (four inlets, outlet, cylinder wall, floor, top, and four s
import os, subprocess, textwrap
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write(textwrap.dedent('''FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
'''))
open('system/blockMeshDict','w').write(textwrap.dedent('''FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices ((-0.10 -0.10 -0.01) (0.10 -0.10 -0.01) (0.10 0.10 -0.01) (-0.10 0.10 -0.01) (-0.10 -0.10 0.21) (0.10 -0.10 0.21) (0.10 0.10 0.21) (-0.10 0.10 0.21));
blocks (hex (0 1 2 3 4 5 6 7) (40 40 44) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });
mergePatchPairs ();
'''))
r=subprocess.run(['blockMesh','-overwrite'],text=True,capture_output=True); print(r.stdout[-1800:]); print(r.stderr); print('return',r.returncode)

# -- cell 5 -------------------------------------------------------------------------
# This OpenFOAM build does not accept `blockMesh -overwrite`; no mesh was written. I’ll rerun the same
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-1800:]); print(r.stderr); print('return',r.returncode); assert r.returncode==0

# -- cell 6 -------------------------------------------------------------------------
# `snappyHexMesh` requires minimal `fvSchemes` and `fvSolution` files even though no flow solution wil
open('system/fvSchemes','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n''')
open('system/fvSolution','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
names=list(patch_shapes.keys()); geom='\n'.join([f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names]); refs='\n'.join([f'        {n} {{ level (1 1); patchInfo {{ type patch; }} }}' for n in names])
snappy=f'''FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry {{\n{geom}\n}}
castellatedMeshControls {{ maxLocalCells 500000; maxGlobalCells 1000000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2; features (); refinementSurfaces {{\n{refs}\n }} resolveFeatureAngle 30; refinementRegions {{}} locationInMesh (0 0 0.10); allowFreeStandingZoneFaces true; }}
snapControls {{ nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
'''; open('system/snappyHexMeshDict','w').write(snappy)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240); print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode); assert r.returncode==0

# -- cell 7 -------------------------------------------------------------------------
# The snapped coarse mesh completed with 51,360 cells and no mesher-level quality errors. I’ll now run
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); open('checkMesh.log','w').write(r.stdout+r.stderr); print(r.stdout[-7000:]); print(r.stderr); print('return',r.returncode); assert r.returncode==0
print('\nBOUNDARY FILE:\n',open('constant/polyMesh/boundary').read())

# -- cell 8 -------------------------------------------------------------------------
# The all-geometry check identified 1,361 concave polyhedral cells, so I will not accept the snapped m
import gmsh
gmsh.initialize(); gmsh.model.add('swirl')
vols=gmsh.model.occ.importShapes('coarse_fluid.step'); gmsh.model.occ.synchronize()
print('imported',vols,'surfaces',len(gmsh.model.getEntities(2)))
# Locate the original horizontal top face by area, centre of mass and z.
top_tag=None
for dim,tag in gmsh.model.getEntities(2):
    com=gmsh.model.occ.getCenterOfMass(dim,tag); area=gmsh.model.occ.getMass(dim,tag)
    if abs(com[2]-H)<1e-8 and area>0.005: top_tag=tag
print('top tag',top_tag)
disk=gmsh.model.occ.addDisk(0,0,H,OUT_D/2,OUT_D/2)
out,mapout=gmsh.model.occ.fragment([(2,top_tag)],[(2,disk)],removeObject=True,removeTool=True)
gmsh.model.occ.synchronize()
print('after imprint volumes',gmsh.model.getEntities(3),'surfaces',len(gmsh.model.getEntities(2)))
for dim,tag in gmsh.model.getEntities(2):
    com=gmsh.model.occ.getCenterOfMass(dim,tag); area=gmsh.model.occ.getMass(dim,tag)
    if abs(com[2]-H)<1e-7: print('top candidate',tag,'area',area,'com',com)
gmsh.write('imprinted_fluid.step'); gmsh.finalize()

# -- cell 9 -------------------------------------------------------------------------
# The imprint operation preserves the single fluid volume and creates exact outlet (0.00125664 m²) and
import gmsh, shutil
gmsh.initialize(); gmsh.model.add('swirlMesh'); imported=gmsh.model.occ.importShapes('imprinted_fluid.step'); gmsh.model.occ.synchronize()
voltag=gmsh.model.getEntities(3)[0][1]
boundary_tags=[t for d,t in gmsh.model.getBoundary([(3,voltag)],oriented=False,recursive=False) if d==2]
print('volume boundary surfaces',len(boundary_tags))
groups={n:[] for n in names}
remaining=[]
for tag in boundary_tags:
    area=gmsh.model.occ.getMass(2,tag); com=np.array(gmsh.model.occ.getCenterOfMass(2,tag)); typ=gmsh.model.getType(2,tag)
    if typ=='Cylinder': groups['cylinderWall'].append(tag)
    elif abs(com[2])<1e-8 and area>0.005: groups['floor'].append(tag)
    elif abs(com[2]-H)<1e-8 and abs(area-math.pi*(OUT_D/2)**2)<1e-7: groups['outlet'].append(tag)
    elif abs(com[2]-H)<1e-8 and area>0.005: groups['top'].append(tag)
    elif abs(area-PORT_W*PORT_H)<1e-8:
        k=min(range(4),key=lambda q:np.linalg.norm(com[:2]-expected[q])); groups[f'inlet{k+1}'].append(tag)
    else: remaining.append((tag,com))
for tag,com in remaining:
    k=min(range(4),key=lambda q:np.linalg.norm(com[:2]-expected[q])); groups[f'ductWall{k+1}'].append(tag)
for name in names:
    pg=gmsh.model.addPhysicalGroup(2,groups[name]); gmsh.model.setPhysicalName(2,pg,name); print(name,len(groups[name]))
pgv=gmsh.model.addPhysicalGroup(3,[voltag]); gmsh.model.setPhysicalName(3,pgv,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0035); gmsh.option.setNumber('Mesh.MeshSizeMax',0.004)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',20); gmsh.option.setNumber('Mesh.Algorithm3D',10)
gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize('Netgen')
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.write('swirl.msh'); gmsh.finalize()
shutil.rmtree('constant/polyMesh',ignore_errors=True)
r=subprocess.run(['gmshToFoam','swirl.msh'],text=True,capture_output=True,timeout=240); print(r.stdout[-4000:]); print(r.stderr); print('return',r.returncode); assert r.returncode==0

# -- cell 10 ------------------------------------------------------------------------
# The tetrahedral trial revealed that exporting the face-only imprint to STEP did not carry the split 
gmsh.initialize(); gmsh.model.add('imprint2'); gmsh.model.occ.importShapes('coarse_fluid.step'); gmsh.model.occ.synchronize(); v=gmsh.model.getEntities(3)[0]
d=gmsh.model.occ.addDisk(0,0,H,OUT_D/2,OUT_D/2)
out,mapout=gmsh.model.occ.fragment([v],[(2,d)],removeObject=True,removeTool=True); gmsh.model.occ.synchronize()
vols2=gmsh.model.getEntities(3); print('volumes',vols2)
bt=[t for dim,t in gmsh.model.getBoundary(vols2,oriented=False,recursive=False) if dim==2]
print('boundary count',len(bt))
for t in bt:
    com=gmsh.model.occ.getCenterOfMass(2,t); area=gmsh.model.occ.getMass(2,t)
    if abs(com[2]-H)<1e-8: print('actual top boundary',t,area,com)
gmsh.write('imprinted_volume.step'); gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# The corrected Boolean now yields one volume with 28 boundary faces, including the exact outlet disk 
gmsh.initialize(); gmsh.model.add('finalSwirl'); gmsh.model.occ.importShapes('imprinted_volume.step'); gmsh.model.occ.synchronize(); voltag=gmsh.model.getEntities(3)[0][1]
boundary_tags=[t for d,t in gmsh.model.getBoundary([(3,voltag)],oriented=False,recursive=False) if d==2]; groups={n:[] for n in names}; remaining=[]
for tag in boundary_tags:
    area=gmsh.model.occ.getMass(2,tag); com=np.array(gmsh.model.occ.getCenterOfMass(2,tag)); typ=gmsh.model.getType(2,tag)
    if typ=='Cylinder': groups['cylinderWall'].append(tag)
    elif abs(com[2])<1e-8 and area>0.005: groups['floor'].append(tag)
    elif abs(com[2]-H)<1e-8 and abs(area-math.pi*(OUT_D/2)**2)<1e-7: groups['outlet'].append(tag)
    elif abs(com[2]-H)<1e-8 and area>0.005: groups['top'].append(tag)
    elif abs(area-PORT_W*PORT_H)<1e-8:
        k=min(range(4),key=lambda q:np.linalg.norm(com[:2]-expected[q])); groups[f'inlet{k+1}'].append(tag)
    else: remaining.append((tag,com))
for tag,com in remaining:
    k=min(range(4),key=lambda q:np.linalg.norm(com[:2]-expected[q])); groups[f'ductWall{k+1}'].append(tag)
for name in names:
    assert groups[name],f'empty physical patch {name}'; pg=gmsh.model.addPhysicalGroup(2,groups[name]); gmsh.model.setPhysicalName(2,pg,name); print(name,len(groups[name]))
pgv=gmsh.model.addPhysicalGroup(3,[voltag]); gmsh.model.setPhysicalName(3,pgv,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0035); gmsh.option.setNumber('Mesh.MeshSizeMax',0.004); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',20); gmsh.option.setNumber('Mesh.Algorithm3D',10)
gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize('Netgen'); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.write('swirl.msh'); gmsh.finalize()
shutil.rmtree('constant/polyMesh',ignore_errors=True); r=subprocess.run(['gmshToFoam','swirl.msh'],text=True,capture_output=True,timeout=240); print(r.stdout[-2500:]); print(r.stderr); assert r.returncode==0

# -- cell 12 ------------------------------------------------------------------------
# The corrected conversion now contains all twelve requested patches, including the outlet. I’ll run f
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); open('checkMesh.log','w').write(r.stdout+r.stderr); print(r.stdout[-6500:]); print(r.stderr); print('return',r.returncode)
btxt=open('constant/polyMesh/boundary').read(); print('\n',btxt); assert 'Mesh OK.' in r.stdout; assert ('defaultFaces' not in btxt) or ('nFaces          0;' in btxt)

# -- cell 13 ------------------------------------------------------------------------
# The final tetrahedral mesh passes all full `checkMesh` checks: 150,398 cells, 12 nonempty named patc
r=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True); print(r.stdout[-1800:]); print(r.stderr); assert r.returncode==0
import glob, pyvista as pv
files=glob.glob('VTK/**/*.vt*',recursive=True); print('VTK files',files[:10])
# Read internal mesh, extract boundary, and render edges on surface.
internal=[f for f in files if 'internal' in f.lower() or f.endswith('.vtu')][0]
ug=pv.read(internal); surf=ug.extract_surface(); pl=pv.Plotter(off_screen=True,window_size=(900,650)); pl.add_mesh(surf,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.3); pl.view_isometric(); pl.add_axes(); pl.show(screenshot='final_mesh.png')
mesh_vol=ug.compute_cell_sizes(volume=True)['Volume'].sum(); print('CAD volume',fluid.volume,'mesh volume',mesh_vol,'relative difference',abs(mesh_vol-fluid.volume)/fluid.volume); print('wrote final_mesh.png')

# -- cell 14 ------------------------------------------------------------------------
# The advisory surface checks found that independently tessellating the CAD faces caused nonconformal 
# Replace independent CAD-face tessellations with the conformal OpenFOAM boundary triangulation.
vtk_base='VTK/t24_0/boundary'
for name in names:
    src=f'{vtk_base}/{name}.vtp'; pd=pv.read(src).extract_surface().triangulate(); out=f'constant/triSurface/{name}.stl'; pd.save(out,binary=False)
    print(name,'triangles',pd.n_cells,'open edges within patch',pd.n_open_edges)
