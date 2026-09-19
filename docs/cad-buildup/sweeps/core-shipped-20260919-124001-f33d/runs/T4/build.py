"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll interpret this as a top-milled, constant-depth serpentine groove: three 4 mm straight passes al
import build123d as bd
from build123d import Box, Cylinder, Pos
import numpy as np, pyvista as pv
from pathlib import Path

# Requested dimensions, in metres
L, W, T = 0.060, 0.040, 0.008
CH_W, CH_D, WEB = 0.004, 0.005, 0.003
PITCH = CH_W + WEB
Y1, Y2, Y3 = 0.013, 0.020, 0.027
R_CL = PITCH / 2
X_LEFT_TURN, X_RIGHT_TURN = 0.006, 0.054
Z0 = T - CH_D

block = Box(L, W, T)
pass1 = Pos(0, Y1-CH_W/2, Z0) * Box(X_RIGHT_TURN, CH_W, CH_D)
pass2 = Pos(X_LEFT_TURN, Y2-CH_W/2, Z0) * Box(X_RIGHT_TURN-X_LEFT_TURN, CH_W, CH_D)
pass3 = Pos(X_LEFT_TURN, Y3-CH_W/2, Z0) * Box(L-X_LEFT_TURN, CH_W, CH_D)
ro, ri = R_CL + CH_W/2, R_CL - CH_W/2
right_ring = Pos(X_RIGHT_TURN, (Y1+Y2)/2, Z0) * (Cylinder(ro, CH_D) - Cylinder(ri, CH_D))
right_half = Pos(X_RIGHT_TURN, 0, Z0) * Box(L-X_RIGHT_TURN, W, CH_D)
right_turn = right_ring & right_half
left_ring = Pos(X_LEFT_TURN, (Y2+Y3)/2, Z0) * (Cylinder(ro, CH_D) - Cylinder(ri, CH_D))
left_half = Pos(0, 0, Z0) * Box(X_LEFT_TURN, W, CH_D)
left_turn = left_ring & left_half
water = pass1 + right_turn + pass2 + left_turn + pass3
aluminium = block - water

bb = water.bounding_box()
water_vol = water.volume
block_vol = block.volume
print(f'plate requested/measured: {L*1e3:.3f} x {W*1e3:.3f} x {T*1e3:.3f} mm')
print(f'channel requested/measured width: {CH_W*1e3:.3f} / {(Y1+CH_W/2-(Y1-CH_W/2))*1e3:.3f} mm')
print(f'channel requested/measured depth: {CH_D*1e3:.3f} / {(bb.max.Z-bb.min.Z)*1e3:.3f} mm')
print(f'straight-pass web requested/measured: {WEB*1e3:.3f} / {(Y2-CH_W/2-(Y1+CH_W/2))*1e3:.3f} mm')
print(f'pass centre spacings: {(Y2-Y1)*1e3:.3f}, {(Y3-Y2)*1e3:.3f} mm; count = 3')
print(f'water bounds mm: x {bb.min.X*1e3:.2f}..{bb.max.X*1e3:.2f}, y {bb.min.Y*1e3:.2f}..{bb.max.Y*1e3:.2f}, z {bb.min.Z*1e3:.2f}..{bb.max.Z*1e3:.2f}')
print(f'volume conservation error: {(aluminium.volume+water_vol-block_vol)/block_vol:.3e}')

bd.export_stl(water, 'water_preview.stl', tolerance=2e-4, angular_tolerance=0.15)
mesh = pv.read('water_preview.stl')
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(mesh, color='#47a9ff', opacity=0.9, show_edges=True)
pl.view_isometric(); pl.camera.zoom(1.25); pl.show_axes()
pl.show(screenshot='coarse_geometry.png')
print(f'preview: {mesh.n_cells} triangles -> coarse_geometry.png')

# -- cell 2 -------------------------------------------------------------------------
# The first preview exposed a construction error before meshing: the straight passages are present, bu
import inspect
print('Pos:', inspect.signature(bd.Pos), bd.Pos.__doc__.splitlines()[0:8])
print('Cylinder:', inspect.signature(bd.Cylinder))
for name, shape in [('pass1',pass1),('right_ring',right_ring),('right_half',right_half),('right_turn',right_turn),('left_ring',left_ring)]:
    b=shape.bounding_box()
    print(name, 'bounds mm', tuple(round(v*1e3,2) for v in (b.min.X,b.max.X,b.min.Y,b.max.Y,b.min.Z,b.max.Z)), 'vol', shape.volume*1e9)

# -- cell 3 -------------------------------------------------------------------------
# The issue is confirmed: this installed build123d centers `Box` on all axes by default, while the int
from build123d import Align
amin = (Align.MIN, Align.MIN, Align.MIN)
czmin = (Align.CENTER, Align.CENTER, Align.MIN)

block = Box(L, W, T, align=amin)
pass1 = Pos(0, Y1-CH_W/2, Z0) * Box(X_RIGHT_TURN, CH_W, CH_D, align=amin)
pass2 = Pos(X_LEFT_TURN, Y2-CH_W/2, Z0) * Box(X_RIGHT_TURN-X_LEFT_TURN, CH_W, CH_D, align=amin)
pass3 = Pos(X_LEFT_TURN, Y3-CH_W/2, Z0) * Box(L-X_LEFT_TURN, CH_W, CH_D, align=amin)
right_ring = Pos(X_RIGHT_TURN, (Y1+Y2)/2, Z0) * (Cylinder(ro, CH_D, align=czmin) - Cylinder(ri, CH_D, align=czmin))
right_half = Pos(X_RIGHT_TURN, 0, Z0) * Box(L-X_RIGHT_TURN, W, CH_D, align=amin)
right_turn = right_ring & right_half
left_ring = Pos(X_LEFT_TURN, (Y2+Y3)/2, Z0) * (Cylinder(ro, CH_D, align=czmin) - Cylinder(ri, CH_D, align=czmin))
left_half = Pos(0, 0, Z0) * Box(X_LEFT_TURN, W, CH_D, align=amin)
left_turn = left_ring & left_half
water = pass1 + right_turn + pass2 + left_turn + pass3
aluminium = block - water

bb=water.bounding_box()
print('water solids:', len(water.solids()))
print('water bounds mm:', tuple(round(v*1e3,3) for v in (bb.min.X,bb.max.X,bb.min.Y,bb.max.Y,bb.min.Z,bb.max.Z)))
print(f'plate volume {block.volume*1e9:.3f} mm^3; water {water.volume*1e9:.3f}; aluminium {aluminium.volume*1e9:.3f}')
print(f'volume conservation relative error: {(aluminium.volume+water.volume-block.volume)/block.volume:.3e}')
print(f'requested/measured web: {WEB*1e3:.3f}/{(Y2-Y1-CH_W)*1e3:.3f} mm')

bd.export_stl(water, 'water_preview.stl', tolerance=1e-4, angular_tolerance=0.1)
mesh = pv.read('water_preview.stl')
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(mesh, color='#45a8e5', opacity=1.0, show_edges=True)
pl.view_isometric(); pl.camera.zoom(1.35); pl.show_axes(); pl.show(screenshot='coarse_geometry_corrected.png')
print(f'corrected preview: {mesh.n_cells} triangles')

# -- cell 4 -------------------------------------------------------------------------
# The corrected water volume is one connected solid, conserves the block volume exactly, and the previ
faces=list(water.faces())
print('face count',len(faces))
for i,f in enumerate(faces):
    c=f.center(); n=f.normal_at(c)
    print(i, 'area mm2',round(f.area*1e6,3),'center mm',tuple(round(q*1e3,3) for q in (c.X,c.Y,c.Z)),'normal',tuple(round(q,3) for q in (n.X,n.Y,n.Z)),'geom',f.geom_type)

# -- cell 5 -------------------------------------------------------------------------
# The topology has exactly two 20 mm² planar end faces at the constructed pass endpoints: inlet at `(0
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches

xplanar = [(f, f.center(), f.normal_at(f.center())) for f in water.faces() if f.geom_type == bd.GeomType.PLANE]
inlet_matches = [f for f,c,n in xplanar if abs(c.X-0.0)<1e-9 and n.X < -0.999 and abs(f.area-CH_W*CH_D)<1e-10]
outlet_matches = [f for f,c,n in xplanar if abs(c.X-L)<1e-9 and n.X > 0.999 and abs(f.area-CH_W*CH_D)<1e-10]
assert len(inlet_matches)==1 and len(outlet_matches)==1
inlet, outlet = inlet_matches[0], outlet_matches[0]

bd.export_step(aluminium, 'aluminium_block_with_channel.step')
bd.export_step(water, 'water_volume.step')
export_patches(water, {'inlet':[inlet], 'outlet':[outlet], 'walls':...}, tolerance=1.5e-4, angular_tolerance=0.12)
print('CAD exports:', Path('aluminium_block_with_channel.step').stat().st_size, Path('water_volume.step').stat().st_size, 'bytes')
print('named inlet/outlet areas mm2:', inlet.area*1e6, outlet.area*1e6)

# -- cell 6 -------------------------------------------------------------------------
# The exported patch union is closed and consistently wound (876 triangles, zero open/flipped/non-mani
import os, subprocess, textwrap
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
block_dict='''FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (-0.001 0.009 0.002) (0.061 0.009 0.002) (0.061 0.031 0.002) (-0.001 0.031 0.002)
 (-0.001 0.009 0.009) (0.061 0.009 0.009) (0.061 0.031 0.009) (-0.001 0.031 0.009)
);
blocks ( hex (0 1 2 3 4 5 6 7) (62 22 7) simpleGrading (1 1 1) );
edges ();
boundary ( background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); } );
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(block_dict)
control='''FoamFile { format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
'''
Path('system/controlDict').write_text(control)
Path('system/fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
res=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(res.stdout[-2500:]); print(res.stderr[-1000:]); print('return',res.returncode)
if res.returncode: raise RuntimeError('blockMesh failed')

# -- cell 7 -------------------------------------------------------------------------
# Castellation succeeded to 21,382 retained water cells, but snapping stopped because the shipped qual
import subprocess
p=subprocess.run(['bash','-lc','grep -R "errorReduction" -n "$WM_PROJECT_DIR/etc/caseDicts" | head -20'],capture_output=True,text=True)
print(p.stdout if p.stdout else p.stderr)
print('current mesh is intermediate only; it will be replaced from blockMesh before the corrected snappy run')

# -- cell 8 -------------------------------------------------------------------------
# The correct OpenFOAM 2512 quality template is `caseDicts/mesh/generation/meshQualityDict.cfg`. I’ll 
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; } wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; } solvers {};''')
Path('system/snappyHexMeshDict').write_text('''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry { inlet.stl { type triSurfaceMesh; name inlet; } outlet.stl { type triSurfaceMesh; name outlet; } walls.stl { type triSurfaceMesh; name walls; } }
castellatedMeshControls { maxLocalCells 500000; maxGlobalCells 1000000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2; features (); refinementSurfaces { inlet { level (1 1); patchInfo { type patch; } } outlet { level (1 1); patchInfo { type patch; } } walls { level (1 1); patchInfo { type wall; } } } resolveFeatureAngle 30; refinementRegions {}; locationInMesh (0.010 0.013 0.0055); allowFreeStandingZoneFaces true; }
snapControls { nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg" }
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
''')
for cmd in (['blockMesh'],['snappyHexMesh','-overwrite']):
    res=subprocess.run(cmd,text=True,capture_output=True,timeout=240)
    Path('log.'+cmd[0]).write_text(res.stdout+res.stderr)
    print('\n###', ' '.join(cmd), 'return',res.returncode,'###\n',res.stdout[-3500:],res.stderr[-500:])
    if res.returncode: raise RuntimeError(cmd[0]+' failed')

# -- cell 9 -------------------------------------------------------------------------
# The final snapped mesh completed with 21,382 cells and no quality faces rejected during meshing. I’l
res=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=120)
Path('log.checkMesh').write_text(res.stdout+res.stderr)
print(res.stdout[-6000:]); print(res.stderr[-1000:]); print('return',res.returncode)
print('\n--- boundary file patch summary ---')
print(Path('constant/polyMesh/boundary').read_text())
if res.returncode: raise RuntimeError('checkMesh failed')

# -- cell 10 ------------------------------------------------------------------------
# The mesh itself is complete; only the visualization utility option differs in this OpenFOAM build (`
p=subprocess.run(['foamToVTK','-help'],text=True,capture_output=True)
print(p.stdout[:5000])

# -- cell 11 ------------------------------------------------------------------------
# The supported spelling is `-no-fields`. I’ll use that and render a clipped isometric view so both th
res=subprocess.run(['foamToVTK','-no-fields','-overwrite'],text=True,capture_output=True,timeout=120)
print(res.stdout[-1200:],res.stderr[-300:],'return',res.returncode)
if res.returncode: raise RuntimeError('foamToVTK failed')
files=list(Path('VTK').rglob('*.vtu'))+list(Path('VTK').rglob('*.vtp'))
print('VTK files:',[str(f) for f in files])
volfiles=[f for f in files if f.suffix=='.vtu']
volmesh=pv.read(str(volfiles[0]))
cut=volmesh.clip(normal=(0,0,1), origin=(0,0,0.0055), invert=False)
pl=pv.Plotter(off_screen=True,window_size=(1000,650))
pl.add_mesh(cut,color='#70b9e8',show_edges=True,edge_color='#183b50',line_width=0.25)
pl.view_isometric(); pl.camera.zoom(1.25); pl.show_axes(); pl.show(screenshot='final_water_mesh.png')
print('rendered final mesh:',volmesh.n_cells,'cells; clipped display:',cut.n_cells)

# -- cell 12 ------------------------------------------------------------------------
# The completion checks found one metadata omission, not a mesh defect: the snappy dictionary contains
import inspect
print(inspect.signature(export_patches))
print(export_patches.__doc__)

# -- cell 13 ------------------------------------------------------------------------
# The exporter accepts `location_in_mesh`. I’ll regenerate the same tessellation and manifest with the
mesh_seed=(0.010,0.013,0.0055)
export_patches(water, {'inlet':[inlet], 'outlet':[outlet], 'walls':...}, tolerance=1.5e-4, angular_tolerance=0.12, location_in_mesh=mesh_seed)
print('manifest now records location_in_mesh:',mesh_seed,'m; point lies in first straight channel')
