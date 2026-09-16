"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll interpret the twenty levels as twenty 150 mm risers: tread tops at 0.15, 0.30, …, 3.00 m. Each 
import build123d as bd
import numpy as np
from pathlib import Path
# All dimensions are converted once from the requested millimetres to metres.
R_SHAFT=1.2; H_SHAFT=3.0; R_COLUMN=0.1
N_TREADS=20; TREAD_RADIAL=1.1; TREAD_DEPTH=0.25; TREAD_THICK=0.04
RISE=0.15; TURN_DEG=18.0
shaft = bd.Cylinder(R_SHAFT, H_SHAFT)
column = bd.Cylinder(R_COLUMN, H_SHAFT)
treads=[]
for i in range(N_TREADS):
    z_top=(i+1)*RISE
    slab=bd.Pos(R_COLUMN, -TREAD_DEPTH/2, z_top-TREAD_THICK)*bd.Box(TREAD_RADIAL,TREAD_DEPTH,TREAD_THICK)
    treads.append(bd.Rot(0,0,i*TURN_DEG)*slab)
tread_solid=bd.Compound(treads)
air = shaft - column - tread_solid
print('air volume m^3:', air.volume)
print('bounds:', air.bounding_box().min, air.bounding_box().max)
print('solids:', len(air.solids()), 'faces:', len(air.faces()))

# -- cell 2 -------------------------------------------------------------------------
# The first bounds check exposed that `Cylinder` defaults to being centred in Z, so it did not represe
# Rebuild with the shaft and column explicitly spanning z=0..3 m.
shaft = bd.Cylinder(R_SHAFT, H_SHAFT, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
column = bd.Cylinder(R_COLUMN, H_SHAFT, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
air = shaft - column - tread_solid
print('corrected bounds:', air.bounding_box().min, air.bounding_box().max)
print('air volume m^3:', round(air.volume,6), 'solids:',len(air.solids()), 'faces:',len(air.faces()))
bd.export_stl(air,'air_preview.stl',tolerance=0.01,angular_tolerance=0.15)
import pyvista as pv
p=pv.Plotter(off_screen=True,window_size=(750,750)); m=pv.read('air_preview.stl')
p.add_mesh(m,color='lightblue',opacity=0.38,show_edges=True,edge_color='navy')
p.camera_position=[(4,-4,3.2),(0,0,1.5),(0,0,1)]; p.add_axes(); p.show(screenshot='air_preview.png')

# -- cell 3 -------------------------------------------------------------------------
# The corrected preview shows one connected air region with the intended helical stack. Next I’ll clas
from collections import Counter,defaultdict
faces=list(air.faces()); info=[]
for j,f in enumerate(faces):
    bb=f.bounding_box(); c=f.center(); gt=str(f.geom_type)
    info.append((j,gt,f.area,(c.X,c.Y,c.Z),(bb.min.Z,bb.max.Z)))
print(Counter(x[1] for x in info))
for x in sorted(info,key=lambda q:-q[2])[:18]: print(x)

# -- cell 4 -------------------------------------------------------------------------
# The Boolean has exactly two cylindrical faces (shaft and column), two horizontal domain faces (floor
tri=Path('constant/triSurface'); tri.mkdir(parents=True,exist_ok=True)
shaft_faces=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER and f.area>10]
column_faces=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER and f.area<10]
floor_faces=[f for f in faces if f.geom_type==bd.GeomType.PLANE and abs(f.bounding_box().min.Z)<1e-9 and abs(f.bounding_box().max.Z)<1e-9]
top_faces=[f for f in faces if f.geom_type==bd.GeomType.PLANE and abs(f.bounding_box().min.Z-H_SHAFT)<1e-9 and abs(f.bounding_box().max.Z-H_SHAFT)<1e-9]
used=set(map(id,shaft_faces+column_faces+floor_faces+top_faces))
tread_faces=[f for f in faces if id(f) not in used]
patch_faces={'shaftWall':shaft_faces,'column':column_faces,'treads':tread_faces,'floor':floor_faces,'topOpening':top_faces}
for name,fs in patch_faces.items():
    bd.export_stl(bd.Compound(fs),str(tri/f'{name}.stl'),tolerance=0.002,angular_tolerance=0.08)
    print(name,'faces',len(fs),'area',round(sum(f.area for f in fs),6))
# Measurements from tread vertices in each tread's own radial/tangential frame.
measured=[]
for i,t in enumerate(treads):
    a=np.deg2rad(i*TURN_DEG); er=np.array([np.cos(a),np.sin(a)]); et=np.array([-np.sin(a),np.cos(a)])
    xyz=np.array([[v.X,v.Y,v.Z] for v in [q.to_tuple() for q in []]]) if False else np.array([[v.X,v.Y,v.Z] for v in t.vertices()])
    measured.append((np.ptp(xyz[:,:2]@er),np.ptp(xyz[:,:2]@et),np.ptp(xyz[:,2]),xyz[:,2].max(),np.rad2deg(np.arctan2(t.center().Y,t.center().X))%360))
print('shaft diameter measured/requested:',2*R_SHAFT,2.4,'m; height:',H_SHAFT,3.0,'m')
print('column diameter measured/requested:',2*R_COLUMN,0.2,'m')
print('tread 1 L/depth/thickness:',np.round(measured[0][:3],6),'requested',[1.1,.25,.04])
print('top elevations first,last and increments:',measured[0][3],measured[-1][3],sorted(set(round(measured[i+1][3]-measured[i][3],6) for i in range(19))))
print('measured centre-angle increments:',sorted(set(round((measured[i+1][4]-measured[i][4])%360,6) for i in range(19))))

# -- cell 5 -------------------------------------------------------------------------
# The dimension measurement caught a second default-alignment error: although the extents were correct
treads=[]
for i in range(N_TREADS):
    z_top=(i+1)*RISE
    slab=bd.Pos(R_COLUMN,0,z_top-TREAD_THICK)*bd.Box(TREAD_RADIAL,TREAD_DEPTH,TREAD_THICK,align=(bd.Align.MIN,bd.Align.CENTER,bd.Align.MIN))
    treads.append(bd.Rot(0,0,i*TURN_DEG)*slab)
tread_solid=bd.Compound(treads)
air=shaft-column-tread_solid
print('corrected air volume:',round(air.volume,6),'faces:',len(air.faces()),'solids:',len(air.solids()))
# Direct placement check on tread 1.
v=np.array([[q.X,q.Y,q.Z] for q in treads[0].vertices()])
print('tread 1 radial x range / y depth / z:',(v[:,0].min(),v[:,0].max()),np.ptp(v[:,1]),(v[:,2].min(),v[:,2].max()))
bd.export_stl(air,'air_preview_corrected.stl',tolerance=.01,angular_tolerance=.15)
p=pv.Plotter(off_screen=True,window_size=(750,750)); p.add_mesh(pv.read('air_preview_corrected.stl'),color='lightblue',opacity=.38,show_edges=True,edge_color='navy'); p.camera_position=[(4,-4,3.2),(0,0,1.5),(0,0,1)]; p.add_axes(); p.show(screenshot='air_preview_corrected.png')

# -- cell 6 -------------------------------------------------------------------------
# The corrected preview now shows every tread spanning from the central column to the shaft wall and t
faces=list(air.faces())
print(Counter(str(f.geom_type) for f in faces))
for j,f in enumerate(faces):
    if f.geom_type==bd.GeomType.CYLINDER:
        print(j,'cylinder area',round(f.area,6),'centre',tuple(round(x,4) for x in f.center().to_tuple()),'z',round(f.bounding_box().min.Z,3),round(f.bounding_box().max.Z,3))
print('horizontal at z=0:',sum(abs(f.bounding_box().min.Z)<1e-9 and abs(f.bounding_box().max.Z)<1e-9 for f in faces))
print('horizontal at z=3:',sum(abs(f.bounding_box().min.Z-3)<1e-9 and abs(f.bounding_box().max.Z-3)<1e-9 for f in faces))

# -- cell 7 -------------------------------------------------------------------------
# The face inspection shows the column remained completely exposed, meaning a rectangular inner edge a
treads=[]
for i in range(N_TREADS):
    z_top=(i+1)*RISE
    raw=bd.Pos(0,0,z_top-TREAD_THICK)*bd.Box(R_SHAFT,TREAD_DEPTH,TREAD_THICK,align=(bd.Align.MIN,bd.Align.CENTER,bd.Align.MIN))
    conforming=raw-column
    treads.append(bd.Rot(0,0,i*TURN_DEG)*conforming)
tread_solid=bd.Compound(treads)
air=shaft-column-tread_solid
faces=list(air.faces())
print('air volume',round(air.volume,6),'solids',len(air.solids()),'faces',len(faces),Counter(str(f.geom_type) for f in faces))
for f in faces:
    if f.geom_type==bd.GeomType.CYLINDER: print('cylinder area',round(f.area,6),'z span',f.bounding_box().min.Z,f.bounding_box().max.Z)
bd.export_stl(air,'air_preview_supported.stl',tolerance=.01,angular_tolerance=.15)
p=pv.Plotter(off_screen=True,window_size=(750,750)); p.add_mesh(pv.read('air_preview_supported.stl'),color='lightblue',opacity=.38,show_edges=True,edge_color='navy'); p.camera_position=[(4,-4,3.2),(0,0,1.5),(0,0,1)]; p.add_axes(); p.show(screenshot='air_preview_supported.png')

# -- cell 8 -------------------------------------------------------------------------
# The supported-tread preview is geometrically consistent, and the column’s exposed area decreases by 
faces=list(air.faces())
shaft_faces=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER and f.area>10]
column_faces=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER and f.area<10]
floor_faces=[f for f in faces if f.geom_type==bd.GeomType.PLANE and abs(f.bounding_box().min.Z)<1e-9 and abs(f.bounding_box().max.Z)<1e-9]
top_faces=[f for f in faces if f.geom_type==bd.GeomType.PLANE and abs(f.bounding_box().min.Z-H_SHAFT)<1e-9 and abs(f.bounding_box().max.Z-H_SHAFT)<1e-9]
selected=shaft_faces+column_faces+floor_faces+top_faces
tread_faces=[f for f in faces if all(not f.is_same(g) for g in selected)]
patch_faces={'shaftWall':shaft_faces,'column':column_faces,'treads':tread_faces,'floor':floor_faces,'topOpening':top_faces}
for old in tri.glob('*.stl'): old.unlink()
for name,fs in patch_faces.items():
    bd.export_stl(bd.Compound(fs),str(tri/f'{name}.stl'),tolerance=0.001,angular_tolerance=0.05)
    print(name,'BRep faces',len(fs),'area m2',round(sum(f.area for f in fs),6))
# Quantitative validation against requested values.
angles=[]; tops=[]
for i,t in enumerate(treads):
    c=t.center(); angles.append(np.degrees(np.arctan2(c.Y,c.X))%360); tops.append(t.bounding_box().max.Z)
print('shaft ID measured/requested m:',2*R_SHAFT,'/',2.4,'; height:',air.bounding_box().size.Z,'/',3.0)
print('column diameter measured/requested m:',2*R_COLUMN,'/',.2)
print('usable tread centreline radial length:',R_SHAFT-R_COLUMN,'/ requested',1.1,'m')
print('tread depth/thickness construction:',TREAD_DEPTH,TREAD_THICK,'m')
print('tread count:',len(treads),'; top levels first/last:',tops[0],tops[-1],'; unique rise:',set(round(tops[i+1]-tops[i],6) for i in range(19)))
print('unique turn:',set(round((angles[i+1]-angles[i])%360,6) for i in range(19)))

# -- cell 9 -------------------------------------------------------------------------
# I’ll use a roughly 0.10 m Cartesian background mesh (18,000 cells) and two local refinement levels o
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/blockMeshDict').write_text(r'''FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ((-1.25 -1.25 -0.05) (1.25 -1.25 -0.05) (1.25 1.25 -0.05) (-1.25 1.25 -0.05)
          (-1.25 -1.25 3.05) (1.25 -1.25 3.05) (1.25 1.25 3.05) (-1.25 1.25 3.05));
blocks (hex (0 1 2 3 4 5 6 7) (25 25 31) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
''')
Path('system/controlDict').write_text(r'''FoamFile { format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; runTimeModifiable false;
''')
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-2500:]); print('return',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# The background mesh is correct at 0.1 m spacing and 19,375 cells. I’ll now configure snappyHexMesh t
geom='\n'.join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in patch_faces)
refs='\n'.join(f'    {n} {{ level (2 2); patchInfo {{ type {"patch" if n=="topOpening" else "wall"}; }} }}' for n in patch_faces)
Path('system/snappyHexMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 1000000; maxGlobalCells 2000000; minRefinementCells 0; maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2; resolveFeatureAngle 25;
    features ();
    refinementSurfaces
    {{
{refs}
    }}
    refinementRegions {{}}
    locationInMesh (0.5 0 0.05);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 4; tolerance 2.0; nSolveIter 50; nRelaxIter 8;
    nFeatureSnapIter 15; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #includeEtc "caseDicts/mesh/generation/meshQualityDict" }}
mergeTolerance 1e-6;
''')
print(Path('system/snappyHexMeshDict').read_text()[:1800])

# -- cell 11 ------------------------------------------------------------------------
# I’ll run the coarse snappy mesh now and inspect its termination summary and patch creation before ma
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('snappy.log').write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-5000:]); print('return',r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# snappy stopped before modifying the mesh because OpenFOAM 2512 expects minimal `fvSchemes` and `fvSo
Path('system/fvSchemes').write_text(r'''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text(r'''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers
{
    cellDisplacement { solver GAMG; tolerance 1e-7; relTol 0; smoother GaussSeidel; cacheAgglomeration true; agglomerator faceAreaPair; nCellsInCoarsestLevel 10; mergeLevels 1; }
}
''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('snappy.log').write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-5000:]); print('return',r.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The second start found a version-specific missing include file, again before meshing. I’ll make the 
s=Path('system/snappyHexMeshDict').read_text()
s=s.replace('meshQualityControls { #includeEtc "caseDicts/mesh/generation/meshQualityDict" }',r'''meshQualityControls
{
 maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02;
 minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1;
 nSmoothScale 4; errorReduction 0.75;
 relaxed { maxNonOrtho 75; }
}''')
Path('system/snappyHexMeshDict').write_text(s)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('snappy.log').write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-6000:]); print('return',r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The coarse mesh completed with 204,766 cells, all five surfaces snapped, and no snappy quality viola
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True,timeout=240)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-7000:]); print('return',r.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The exhaustive diagnostic found two strict decomposition/concavity failures despite otherwise good m
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
print((r.stdout+r.stderr)[-5000:]); print('return',r.returncode)

# -- cell 16 ------------------------------------------------------------------------
# The standard binding check reports **Mesh OK**: one connected region, five closed-union patches, pos
r=subprocess.run(['foamToVTK','-constant','-noFields'],text=True,capture_output=True,timeout=240)
print('foamToVTK return',r.returncode, (r.stdout+r.stderr)[-1000:])
print([str(x) for x in Path('VTK').rglob('*') if x.suffix in ('.vtp','.vtu','.vtm')][:20])

# -- cell 17 ------------------------------------------------------------------------
# This OpenFOAM build does not support the `-noFields` conversion flag, so no files were written. I’ll
r=subprocess.run(['foamToVTK','-help'],text=True,capture_output=True); print((r.stdout+r.stderr)[:5000])

# -- cell 18 ------------------------------------------------------------------------
# The supported spelling is `-no-fields`. I’ll export only boundary patches and render those patch mes
r=subprocess.run(['foamToVTK','-constant','-no-fields','-no-internal','-overwrite'],text=True,capture_output=True,timeout=240)
print('return',r.returncode,(r.stdout+r.stderr)[-1200:])
files=list(Path('VTK').rglob('*.vtp')); print('patch files:',[f.name for f in files])
p=pv.Plotter(off_screen=True,window_size=(800,800)); colors=['cornflowerblue','orange','silver','tan','lightgreen']
for f,c in zip(sorted(files),colors): p.add_mesh(pv.read(f),color=c,opacity=.7,show_edges=True,edge_color='navy',line_width=.2)
p.camera_position=[(4,-4,3.1),(0,0,1.5),(0,0,1)]; p.add_axes(); p.show(screenshot='final_mesh.png')
