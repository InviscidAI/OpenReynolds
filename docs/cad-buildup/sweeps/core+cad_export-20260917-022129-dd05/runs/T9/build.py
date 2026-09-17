"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the oil as the closed internal cavity: a cylinder of radius 23 mm and height 10 mm, minus
import build123d as bd, inspect
print('Cylinder:', inspect.signature(bd.Cylinder))
print('export_stl:', inspect.signature(bd.export_stl))
print('Shape methods:', [n for n in ('volume','area','bounding_box','faces','center','is_valid') if hasattr(bd.Shape,n)])

# -- cell 2 -------------------------------------------------------------------------
# The dimensions are converted once from millimetres to metres. I’m now creating the fluid cavity dire
MM=1e-3
R_RING_IN=23*MM; R_RING_OUT=30*MM; H_RING=10*MM
R_SPIN_IN=15*MM; R_SPIN_OUT=22*MM; H_SPIN=8*MM
cavity = bd.Cylinder(R_RING_IN,H_RING)
spinner_solid = bd.Cylinder(R_SPIN_OUT,H_SPIN) - bd.Cylinder(R_SPIN_IN,H_SPIN)
fluid = cavity - spinner_solid
bb=fluid.bounding_box()
print('valid:',fluid.is_valid,'solids:',len(fluid.solids()),'faces:',len(fluid.faces()))
print('volume m3:',fluid.volume,'expected:',bd.pi*R_RING_IN**2*H_RING-bd.pi*(R_SPIN_OUT**2-R_SPIN_IN**2)*H_SPIN)
print('bounds m:',tuple(bb.min),tuple(bb.max))
for i,f in enumerate(fluid.faces()):
    c=f.center(); print(i, f.geom_type, 'area',f.area,'center',tuple(round(v,7) for v in c))

# -- cell 3 -------------------------------------------------------------------------
# The solid is valid and single-region, with exactly seven boundary faces: three ring-side faces (wall
import pyvista as pv
bd.export_stl(fluid,'fluid_preview.stl',tolerance=4e-4,angular_tolerance=0.12)
m=pv.read('fluid_preview.stl')
cut=m.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
p=pv.Plotter(off_screen=True,window_size=(900,700))
p.add_mesh(cut,color='#d8a52a',opacity=0.78,show_edges=True,edge_color='#5c4312',line_width=1)
p.add_axes(); p.view_isometric(); p.camera.zoom(1.35)
p.show(screenshot='fluid_cutaway.png')
print('preview triangles:',m.n_cells,'cut triangles:',cut.n_cells)

# -- cell 4 -------------------------------------------------------------------------
# The face classification and every requested dimension checked correctly, but the export call used an
import inspect,sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
print(inspect.signature(export_patches))
print(export_patches.__doc__)

# -- cell 5 -------------------------------------------------------------------------
# I’ll repeat the semantic face assignment, measurements, and conformal export using the helper’s actu
import os
faces=list(fluid.faces()); cyl=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER]; planes=[f for f in faces if f.geom_type==bd.GeomType.PLANE]
outer_ring_wetted=[f for f in cyl if abs(f.area-2*bd.pi*R_RING_IN*H_RING)<1e-10]
spinner_wetted=[f for f in cyl if f not in outer_ring_wetted]+[f for f in planes if abs(abs(f.center().Z)-H_SPIN/2)<1e-10]
ring_end_top=[f for f in planes if abs(f.center().Z-H_RING/2)<1e-10]
ring_end_bottom=[f for f in planes if abs(f.center().Z+H_RING/2)<1e-10]
print('radial clearance:',(R_RING_IN-R_SPIN_OUT)/MM,'mm requested 1.0 mm')
print('axial gaps:',(H_RING-H_SPIN)/2/MM,'mm each requested 1.0 mm')
print('ring radii OD/ID:',R_RING_OUT/MM,R_RING_IN/MM,'mm requested 30/23')
print('spinner radii OD/ID:',R_SPIN_OUT/MM,R_SPIN_IN/MM,'mm requested 22/15')
print('ring/spinner heights:',H_RING/MM,H_SPIN/MM,'mm requested 10/8')
print('patch face counts:',len(outer_ring_wetted),len(spinner_wetted),len(ring_end_top),len(ring_end_bottom))
os.makedirs('constant/triSurface',exist_ok=True)
patch_report=export_patches(fluid,{'outer_ring_wetted':outer_ring_wetted,'spinner_wetted':spinner_wetted,'ring_end_top':ring_end_top,'ring_end_bottom':ring_end_bottom},out_dir='constant/triSurface',tolerance=2.5e-4)

# -- cell 6 -------------------------------------------------------------------------
# The four STLs were exported before meshing, and their welded union is closed with consistent winding
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh;\nstartFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; }\nscale 1;\nvertices ((-0.024 -0.024 -0.006) (0.024 -0.024 -0.006) (0.024 0.024 -0.006) (-0.024 0.024 -0.006) (-0.024 -0.024 0.006) (0.024 -0.024 0.006) (0.024 0.024 0.006) (-0.024 0.024 0.006));\nblocks (hex (0 1 2 3 4 5 6 7) (48 48 12) simpleGrading (1 1 1));\nedges ();\nboundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });\nmergePatchPairs ();\n''')
Path('system/snappyHexMeshDict').write_text('''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }\ncastellatedMesh true; snap true; addLayers false;\ngeometry\n{\n outer_ring_wetted.stl { type triSurfaceMesh; name outer_ring_wetted; }\n spinner_wetted.stl { type triSurfaceMesh; name spinner_wetted; }\n ring_end_top.stl { type triSurfaceMesh; name ring_end_top; }\n ring_end_bottom.stl { type triSurfaceMesh; name ring_end_bottom; }\n}\ncastellatedMeshControls\n{\n maxLocalCells 300000; maxGlobalCells 500000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 1;\n features ();\n refinementSurfaces\n {\n  outer_ring_wetted { level (1 1); patchInfo { type wall; } }\n  spinner_wetted { level (1 1); patchInfo { type wall; } }\n  ring_end_top { level (1 1); patchInfo { type wall; } }\n  ring_end_bottom { level (1 1); patchInfo { type wall; } }\n }\n resolveFeatureAngle 30; refinementRegions {}; locationInMesh (0 0 0); allowFreeStandingZoneFaces true;\n}\nsnapControls { nSmoothPatch 5; tolerance 1.5; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }\naddLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }\nmeshQualityControls { maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-18; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; } }\nwriteFlags (scalarLevels); mergeTolerance 1e-6;\n''')
print('Wrote controlDict, blockMeshDict, snappyHexMeshDict')

# -- cell 7 -------------------------------------------------------------------------
# The coarse meshing dictionaries are ready. I’ll generate the background mesh first and verify its sc
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-1800:]); print(r.stderr[-500:]); print('returncode',r.returncode)
if r.returncode: raise RuntimeError('blockMesh failed')

# -- cell 8 -------------------------------------------------------------------------
# snappyHexMesh requires the standard empty `fvSchemes` and `fvSolution` dictionaries even though no s
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
print('Added mesh-utility fvSchemes and fvSolution')

# -- cell 9 -------------------------------------------------------------------------
# The utility dictionaries are now present; I’ll rerun the coarse surface-conforming mesh.
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-3500:]); print(r.stderr[-500:]); print('returncode',r.returncode)
if r.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 10 ------------------------------------------------------------------------
# The coarse oil mesh has 45,488 cells, with no mesher quality violations. I’ll now run the binding ba
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-4200:]); print(r.stderr[-500:]); print('returncode',r.returncode)
if r.returncode: raise RuntimeError('checkMesh failed')

# -- cell 11 ------------------------------------------------------------------------
# `checkMesh` passes: one connected region, exactly four patches, 45,488 cells, maximum non-orthogonal
r=subprocess.run(['foamToVTK','-no-fields'],text=True,capture_output=True)
print(r.stdout[-1600:]); print(r.stderr[-400:]); print('returncode',r.returncode)
print('\n'.join(str(p) for p in Path('VTK').rglob('*') if p.is_file())[:2000])
if r.returncode: raise RuntimeError('foamToVTK failed')

# -- cell 12 ------------------------------------------------------------------------
# The VTK export confirms the four named boundary datasets. I’ll now make the final cutaway mesh inspe
grid=pv.read('VTK/t9_0/internal.vtu')
cut=grid.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
p=pv.Plotter(off_screen=True,window_size=(1000,760))
p.add_mesh(cut,color='#d9d9d9',show_edges=True,edge_color='#555555',line_width=0.35)
colors={'outer_ring_wetted':'#377eb8','spinner_wetted':'#e41a1c','ring_end_top':'#4daf4a','ring_end_bottom':'#984ea3'}
for name,color in colors.items():
    s=pv.read(f'VTK/t9_0/boundary/{name}.vtp').clip(normal=(1,0,0),origin=(0,0,0),invert=False)
    p.add_mesh(s,color=color,opacity=0.55,label=name)
p.add_legend(bcolor='white',size=(0.30,0.20)); p.add_axes(); p.view_isometric(); p.camera.zoom(1.45)
p.show(screenshot='oil_mesh_cutaway.png')
print('rendered cells:',grid.n_cells,'cut cells:',cut.n_cells)

# -- cell 13 ------------------------------------------------------------------------
# The boundary overlay shows the correct patches, but the oblique cut still hides much of the internal
section=grid.slice(normal=(0,1,0),origin=(0,0,0))
p=pv.Plotter(off_screen=True,window_size=(1100,520))
p.add_mesh(section,color='#f2c14e',show_edges=True,edge_color='#333333',line_width=0.65)
p.add_axes(); p.view_xz(); p.camera.zoom(1.25)
p.show(screenshot='oil_mesh_section.png')
print('section cells:',section.n_cells,'bounds:',section.bounds)
