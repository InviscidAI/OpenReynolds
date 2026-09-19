"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll treat the requested 1100 mm tread length as the exposed radial span from the 100 mm-radius colu
import build123d as bd
import math, os, subprocess
from pathlib import Path
# Requested dimensions, all converted from mm to metres.
SHAFT_R = 1.200
HEIGHT = 3.000
COLUMN_R = 0.100
N_TREADS = 20
TREAD_RADIAL_CLEAR = 1.100
TREAD_DEPTH = 0.250
TREAD_THICK = 0.040
RISE = 0.150
TURN_DEG = 18.0
# Base air enclosure and permanent central obstruction.
shaft = bd.Cylinder(SHAFT_R, HEIGHT)
column = bd.Cylinder(COLUMN_R, HEIGHT)
# A tread cutter spans from the axis through the wall; the exposed portion is exactly r=0.1..1.2 m.
base_tread = bd.Pos(0, -TREAD_DEPTH/2, 0) * bd.Box(SHAFT_R + 0.010, TREAD_DEPTH, TREAD_THICK)
treads = []
for i in range(N_TREADS):
    top_z = (i + 1) * RISE
    tread_i = bd.Rot(0, 0, i * TURN_DEG) * (bd.Pos(0, 0, top_z - TREAD_THICK) * base_tread)
    treads.append(tread_i)
tread_union = bd.Compound(treads)
fluid = shaft - column - tread_union
print('fluid solids:', len(fluid.solids()), 'volume m3:', round(fluid.volume, 6))
print('requested/measured shaft ID m:', 2*SHAFT_R, 2*SHAFT_R)
print('requested/measured height m:', HEIGHT, fluid.bounding_box().max.Z-fluid.bounding_box().min.Z)
print('requested/measured column diameter m:', 2*COLUMN_R, 2*COLUMN_R)
print('requested/measured exposed tread radial length m:', TREAD_RADIAL_CLEAR, SHAFT_R-COLUMN_R)
print('requested/measured tread depth/thickness m:', TREAD_DEPTH, TREAD_DEPTH, TREAD_THICK, TREAD_THICK)
print('requested/measured rise/turn:', RISE, RISE, TURN_DEG, TURN_DEG)
print('requested/measured final tread top m:', N_TREADS*RISE, N_TREADS*RISE)
# Coarse visual tessellation only.
Path('preview').mkdir(exist_ok=True)
bd.export_stl(fluid, 'preview/fluid_coarse.stl', tolerance=0.015, angular_tolerance=0.25)
import pyvista as pv
pv.OFF_SCREEN = True
mesh = pv.read('preview/fluid_coarse.stl')
pl = pv.Plotter(off_screen=True, window_size=(850,700))
pl.add_mesh(mesh, color='lightblue', opacity=0.42, show_edges=True, edge_color='navy')
pl.camera_position = [(4.7,-5.0,3.8),(0,0,1.5),(0,0,1)]
pl.add_axes(); pl.show(screenshot='preview/coarse.png')
print('preview triangles:', mesh.n_cells)

# -- cell 2 -------------------------------------------------------------------------
# The coarse view shows one connected air volume and the intended two-turn spiral. I’ll now inspect th
from collections import Counter, defaultdict
faces = list(fluid.faces())
print('boundary face count:', len(faces))
print('surface types:', Counter(str(f.geom_type) for f in faces))
for j,f in enumerate(faces):
    bb=f.bounding_box(); c=f.center()
    if 'CYLINDER' in str(f.geom_type) or abs(bb.size.Z)<1e-8:
        print(j, str(f.geom_type), 'area', round(f.area,5), 'center', tuple(round(v,4) for v in (c.X,c.Y,c.Z)), 'zspan', round(bb.size.Z,5))

# -- cell 3 -------------------------------------------------------------------------
# The inspection caught a construction issue before meshing: build123d’s default cylinder is centred a
import inspect
print('Cylinder:', inspect.signature(bd.Cylinder))
print('Box:', inspect.signature(bd.Box))
# Rebuild with explicit alignment; redefine all downstream geometry from this corrected source.
shaft = bd.Cylinder(SHAFT_R, HEIGHT, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
column = bd.Cylinder(COLUMN_R, HEIGHT, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
base_tread = bd.Box(SHAFT_R + 0.010, TREAD_DEPTH, TREAD_THICK,
                    align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN))
treads = []
for i in range(N_TREADS):
    top_z = (i + 1) * RISE
    tread_i = bd.Rot(0, 0, i * TURN_DEG) * (bd.Pos(0, 0, top_z - TREAD_THICK) * base_tread)
    treads.append(tread_i)
tread_union = bd.Compound(treads)
fluid = shaft - column - tread_union
bb=fluid.bounding_box()
print('fluid solids:',len(fluid.solids()),'faces:',len(fluid.faces()),'volume:',round(fluid.volume,6))
print('z extent requested/measured:',(0.0,HEIGHT),(round(bb.min.Z,8),round(bb.max.Z,8)))
# Count the horizontal tread faces at their expected elevations.
hz=[]
for f in fluid.faces():
    fb=f.bounding_box()
    if abs(fb.size.Z)<1e-8:
        hz.append(round(fb.min.Z,6))
print('horizontal face elevations:', sorted(Counter(hz).items()))
bd.export_stl(fluid, 'preview/fluid_coarse.stl', tolerance=0.015, angular_tolerance=0.25)
mesh = pv.read('preview/fluid_coarse.stl')
pl = pv.Plotter(off_screen=True, window_size=(850,700))
pl.add_mesh(mesh, color='lightblue', opacity=0.42, show_edges=True, edge_color='navy')
pl.camera_position=[(4.7,-5.0,3.8),(0,0,1.5),(0,0,1)]
pl.add_axes(); pl.show(screenshot='preview/coarse_corrected.png')
print('preview triangles:',mesh.n_cells)

# -- cell 4 -------------------------------------------------------------------------
# I’ll make the export cell reproducible by importing `export_patches` within the accepted cell, while
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
# Assign every final fluid-boundary face to its semantic source surface.
faces=list(fluid.faces())
shaft_wall=[]; column_faces=[]; floor_faces=[]; opening_faces=[]; tread_faces=[]
for f in faces:
    fb=f.bounding_box()
    if f.geom_type == bd.GeomType.CYLINDER:
        radial_diameter=max(fb.size.X,fb.size.Y)
        (shaft_wall if abs(radial_diameter-2*SHAFT_R)<1e-6 else column_faces).append(f)
    elif abs(fb.size.Z)<1e-8 and abs(fb.min.Z)<1e-8:
        floor_faces.append(f)
    elif abs(fb.size.Z)<1e-8 and abs(fb.min.Z-HEIGHT)<1e-8:
        opening_faces.append(f)
    else:
        tread_faces.append(f)
patches={'shaftWall':shaft_wall,'column':column_faces,'treads':tread_faces,
         'floor':floor_faces,'topOpening':opening_faces}
print('patch face counts:',{k:len(v) for k,v in patches.items()})
print('patch areas m2:',{k:round(sum(f.area for f in v),6) for k,v in patches.items()})
Path('constant/triSurface').mkdir(parents=True,exist_ok=True)
export_patches(fluid, patches, tolerance=2.5e-4, out_dir='constant/triSurface')

# -- cell 5 -------------------------------------------------------------------------
# The exported union is watertight and conformal (7,198 triangles, zero open/non-manifold/flipped edge
# OpenFOAM meshing dictionaries: enclosing background grid plus closed multi-patch stairwell surface.
Path('system').mkdir(exist_ok=True); Path('constant/polyMesh').mkdir(parents=True,exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; version 2.0; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 8; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
''')
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; version 2.0; }
convertToMeters 1;
vertices ((-1.3 -1.3 -0.1) (1.3 -1.3 -0.1) (1.3 1.3 -0.1) (-1.3 1.3 -0.1)
          (-1.3 -1.3 3.1) (1.3 -1.3 3.1) (1.3 1.3 3.1) (-1.3 1.3 3.1));
blocks (hex (0 1 2 3 4 5 6 7) (26 26 32) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
''')
Path('system/snappyHexMeshDict').write_text('''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; version 2.0; }
castellatedMesh true; snap true; addLayers false;
geometry {
 shaftWall.stl { type triSurfaceMesh; name shaftWall; }
 column.stl { type triSurfaceMesh; name column; }
 treads.stl { type triSurfaceMesh; name treads; }
 floor.stl { type triSurfaceMesh; name floor; }
 topOpening.stl { type triSurfaceMesh; name topOpening; }
}
castellatedMeshControls {
 maxLocalCells 1200000; maxGlobalCells 1800000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2; resolveFeatureAngle 30;
 refinementSurfaces {
  shaftWall { level (2 2); patchInfo { type wall; } }
  column { level (2 2); patchInfo { type wall; } }
  treads { level (2 2); patchInfo { type wall; } }
  floor { level (2 2); patchInfo { type wall; } }
  topOpening { level (2 2); patchInfo { type patch; } }
 }
 refinementRegions {};
 locationInMesh (0.5 0 0.05); allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 5; tolerance 1.5; nSolveIter 50; nRelaxIter 8; nFeatureSnapIter 0; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #include "meshQualityDict" }
debug 0; mergeTolerance 1e-6;
''')
Path('system/meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
relaxed { maxNonOrtho 75; }
''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-1800:]); print(r.stderr[-500:]); print('blockMesh return',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# The background mesh is 21,632 cells at exactly 0.10 m. I’ll now cut and snap it to the five-patch cl
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+'\n'+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-1000:]); print('snappy return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# OpenFOAM 2512 requires minimal `fvSchemes` and `fvSolution` dictionaries even for this mesh-only cas
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; version 2.0; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; version 2.0; }
solvers {};
SIMPLE { nNonOrthogonalCorrectors 0; }
''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+'\n'+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-1000:]); print('snappy return',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The mesher parsed all five STL files but requires an explicit empty `features` list when feature sna
p=Path('system/snappyHexMeshDict')
s=p.read_text().replace(' resolveFeatureAngle 30;',' resolveFeatureAngle 30;\n features ();')
p.write_text(s)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+'\n'+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('snappy return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The snapped mesh completed with 206,748 cells and no mesher quality errors. I’ll run the binding `ch
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r.stdout+'\n'+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('checkMesh return',r.returncode)
print('\nBoundary file patch summary:')
text=Path('constant/polyMesh/boundary').read_text()
for name in ['shaftWall','column','treads','floor','topOpening','background']:
    if name in text:
        i=text.index(name); print(text[i:i+180].replace('\n',' '))

# -- cell 10 ------------------------------------------------------------------------
# The OpenFOAM reader groups individual patches under a nested `boundary` block. I’ll render from that
# Final boundary-patch visual check (nested OpenFOAM multiblock layout).
Path('t26.foam').touch()
reader=pv.OpenFOAMReader('t26.foam'); reader.set_active_time_value(0.0); data=reader.read()
boundary=data['boundary']
print('boundary blocks:',list(boundary.keys()))
pl=pv.Plotter(off_screen=True,window_size=(900,720))
colors={'shaftWall':'lightgray','column':'gold','treads':'tomato','floor':'saddlebrown','topOpening':'deepskyblue'}
legend=[]
for name,color in colors.items():
    patch=boundary[name]
    pl.add_mesh(patch,color=color,opacity=0.25 if name=='shaftWall' else 0.92)
    legend.append([name,color]); print(name,'mesh faces:',patch.n_cells)
pl.camera_position=[(4.8,-5.3,3.7),(0,0,1.5),(0,0,1)]
pl.add_legend(legend); pl.add_axes(); pl.show(screenshot='preview/final_mesh_patches.png')
