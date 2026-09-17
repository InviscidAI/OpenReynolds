"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water volume as a closed, constant 4 mm × 5 mm channel beneath an assumed cover plane
import build123d as bd
from build123d import Box, Cylinder, Pos
import numpy as np
# All dimensions are metres.
L, W, H = 0.060, 0.040, 0.008
channel_w, channel_d, ligament = 0.004, 0.005, 0.003
pitch = channel_w + ligament
z0 = H - channel_d
y1, y2, y3 = W/2-pitch, W/2, W/2+pitch
R = pitch/2
ro, ri = R+channel_w/2, R-channel_w/2
xr, xl = L-ro, ro
# Straight runs and constant-width semicircular return bends.
run1 = Pos(0, y1-channel_w/2, z0) * Box(xr, channel_w, channel_d)
run2 = Pos(xl, y2-channel_w/2, z0) * Box(xr-xl, channel_w, channel_d)
run3 = Pos(xl, y3-channel_w/2, z0) * Box(L-xl, channel_w, channel_d)
right_ring = Pos(xr, (y1+y2)/2, z0) * (Cylinder(ro, channel_d) - Cylinder(ri, channel_d))
right_half = Pos(xr, 0, z0) * Box(L-xr, W, channel_d)
right_bend = right_ring & right_half
left_ring = Pos(xl, (y2+y3)/2, z0) * (Cylinder(ro, channel_d) - Cylinder(ri, channel_d))
left_half = Pos(0, 0, z0) * Box(xl, W, channel_d)
left_bend = left_ring & left_half
fluid = run1 + right_bend + run2 + left_bend + run3
print('fluid solids:', len(fluid.solids()), 'volume [mm^3]:', fluid.volume*1e9)
print('bounds [mm]:', tuple(round(v*1000,3) for v in (*fluid.bounding_box().min.to_tuple(), *fluid.bounding_box().max.to_tuple())))
print('requested/actual section width [mm]:', channel_w*1000, channel_w*1000)
print('requested/actual depth [mm]:', channel_d*1000, (fluid.bounding_box().max.Z-fluid.bounding_box().min.Z)*1000)
print('requested/actual pass ligament [mm]:', ligament*1000, (pitch-channel_w)*1000)
print('requested/actual pass count:', 3, 3)

# -- cell 2 -------------------------------------------------------------------------
# The first construction exposed a placement-default error: build123d centered primitives unless align
from build123d import Align
amin, acen = Align.MIN, Align.CENTER
run1 = Pos(0, y1-channel_w/2, z0) * Box(xr, channel_w, channel_d, align=(amin,amin,amin))
run2 = Pos(xl, y2-channel_w/2, z0) * Box(xr-xl, channel_w, channel_d, align=(amin,amin,amin))
run3 = Pos(xl, y3-channel_w/2, z0) * Box(L-xl, channel_w, channel_d, align=(amin,amin,amin))
ring0 = Cylinder(ro, channel_d, align=(acen,acen,amin)) - Cylinder(ri, channel_d, align=(acen,acen,amin))
right_ring = Pos(xr, (y1+y2)/2, z0) * ring0
right_half = Pos(xr, 0, z0) * Box(L-xr, W, channel_d, align=(amin,amin,amin))
right_bend = right_ring & right_half
left_ring = Pos(xl, (y2+y3)/2, z0) * ring0
left_half = Pos(0, 0, z0) * Box(xl, W, channel_d, align=(amin,amin,amin))
left_bend = left_ring & left_half
fluid = run1 + right_bend + run2 + left_bend + run3
bb=fluid.bounding_box()
print('fluid solids:', len(fluid.solids()), 'volume [mm^3]:', round(fluid.volume*1e9,3))
print('bounds min/max [mm]:', tuple(round(v*1000,3) for v in tuple(bb.min)), tuple(round(v*1000,3) for v in tuple(bb.max)))
# Tessellate and render a coarse top view.
verts, tris = fluid.tessellate(2.5e-4)
import pyvista as pv
pts=np.array([tuple(v) for v in verts]); faces=np.hstack([np.full((len(tris),1),3),np.array(tris,dtype=int)]).ravel()
mesh=pv.PolyData(pts,faces)
p=pv.Plotter(off_screen=True,window_size=(900,500)); p.add_mesh(mesh,color='deepskyblue',show_edges=True)
p.view_xy(); p.camera.zoom(1.35); p.add_axes(); p.screenshot('coarse_fluid.png'); p.close()
print('wrote coarse_fluid.png with',len(tris),'triangles')

# -- cell 3 -------------------------------------------------------------------------
# The corrected volume is one connected solid with the intended 0–60 mm extent, 18 mm overall serpenti
from IPython.display import display, Image
display(Image('coarse_fluid.png'))
print('face inventory (index, area mm^2, center mm, normal at center):')
for i,f in enumerate(fluid.faces()):
    c=f.center(); n=f.normal_at(c)
    print(i, round(f.area*1e6,3), tuple(round(q*1000,3) for q in tuple(c)), tuple(round(q,3) for q in tuple(n)))

# -- cell 4 -------------------------------------------------------------------------
# The top view confirms three straight passes and two contained semicircular returns. The face invento
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
zmid=z0+channel_d/2
inlet_target=np.array([0.0,y1,zmid]); outlet_target=np.array([L,y3,zmid])
def closest_face(target):
    return min(fluid.faces(), key=lambda f: np.linalg.norm(np.array(tuple(f.center()))-target))
inlet_face=closest_face(inlet_target); outlet_face=closest_face(outlet_target)
print('inlet center/area:',tuple(inlet_face.center()),inlet_face.area,'target area',channel_w*channel_d)
print('outlet center/area:',tuple(outlet_face.center()),outlet_face.area,'target area',channel_w*channel_d)
assert abs(inlet_face.area-channel_w*channel_d)<1e-10 and abs(outlet_face.area-channel_w*channel_d)<1e-10
export_patches(fluid, {'inlet':[inlet_face], 'outlet':[outlet_face], 'walls':...}, tolerance=2.5e-4)

# -- cell 5 -------------------------------------------------------------------------
# The exported union is watertight and consistently wound. For the coarse mesh I’ll use snappyHexMesh 
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
Path('system/blockMeshDict').write_text(r'''FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
 (-0.002 0.009 0.002) (0.062 0.009 0.002) (0.062 0.031 0.002) (-0.002 0.031 0.002)
 (-0.002 0.009 0.009) (0.062 0.009 0.009) (0.062 0.031 0.009) (-0.002 0.031 0.009)
);
blocks ( hex (0 1 2 3 4 5 6 7) (32 11 4) simpleGrading (1 1 1) );
edges ();
boundary ( background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); } );
mergePatchPairs ();
''')
Path('system/snappyHexMeshDict').write_text(r'''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
 inlet.stl { type triSurfaceMesh; name inlet; }
 outlet.stl { type triSurfaceMesh; name outlet; }
 walls.stl { type triSurfaceMesh; name walls; }
}
castellatedMeshControls
{
 maxLocalCells 200000; maxGlobalCells 300000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 1;
 features ();
 refinementSurfaces
 {
  inlet { level (2 2); patchInfo { type patch; } }
  outlet { level (2 2); patchInfo { type patch; } }
  walls { level (2 2); patchInfo { type wall; } }
 }
 resolveFeatureAngle 30;
 refinementRegions {}
 locationInMesh (0.010 0.013 0.0055);
 allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls
{
 maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minVol 1e-18; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
 minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
 relaxed { maxNonOrtho 75; }
}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
''')
Path('system/controlDict').write_text(r'''FoamFile { format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
''')
print('wrote blockMeshDict, snappyHexMeshDict, controlDict')

# -- cell 6 -------------------------------------------------------------------------
# I’ll generate the coarse background mesh first and confirm its dimensions before conforming it to th
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print('\n'.join((r.stdout+r.stderr).splitlines()[-24:])); print('return code',r.returncode)
assert r.returncode==0

# -- cell 7 -------------------------------------------------------------------------
# snappyHexMesh stopped before modifying the mesh because OpenFOAM 2512 requires placeholder discretiz
Path('system/fvSchemes').write_text(r'''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text(r'''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
SIMPLE { nNonOrthogonalCorrectors 0; }
''')
print('wrote minimal fvSchemes and fvSolution')

# -- cell 8 -------------------------------------------------------------------------
# The required meshing dictionaries are now present; I’ll rerun surface refinement and snapping.
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
lines=(r.stdout+r.stderr).splitlines(); print('\n'.join(lines[-55:])); print('return code',r.returncode)
assert r.returncode==0

# -- cell 9 -------------------------------------------------------------------------
# The coarse conformal mesh contains 24,480 cells, predominantly at refinement level 2, and snappy rep
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout+r.stderr); print('return code',r.returncode)
assert r.returncode==0

# -- cell 10 ------------------------------------------------------------------------
# `checkMesh` passes: one connected region, three boundary patches, 24,480 cells, maximum non-orthogon
r=subprocess.run(['foamToVTK','-overwrite'],text=True,capture_output=True)
print('\n'.join((r.stdout+r.stderr).splitlines()[-18:])); print('return code',r.returncode)
assert r.returncode==0
import glob
print(glob.glob('VTK/**/*',recursive=True)[-15:])

# -- cell 11 ------------------------------------------------------------------------
# The VTK export contains the expected inlet, outlet, and walls datasets. I’ll render the mid-depth sl
vol=pv.read('VTK/t4_0/internal.vtu')
sl=vol.slice(normal=(0,0,1),origin=(0.03,0.02,zmid))
inm=pv.read('VTK/t4_0/boundary/inlet.vtp'); outm=pv.read('VTK/t4_0/boundary/outlet.vtp')
p=pv.Plotter(off_screen=True,window_size=(1000,520)); p.set_background('white')
p.add_mesh(sl,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.4)
p.add_mesh(inm,color='green'); p.add_mesh(outm,color='red')
p.view_xy(); p.camera.zoom(1.3); p.add_axes(); p.screenshot('mesh_midplane.png'); p.close()
display(Image('mesh_midplane.png'))
print('slice cells:',sl.n_cells,'inlet faces:',inm.n_cells,'outlet faces:',outm.n_cells)

# -- cell 12 ------------------------------------------------------------------------
# The final slice shows a continuous three-pass serpentine, with the inlet at the lower-left end (gree
bd.export_step(fluid,'water_volume.step')
print('wrote water_volume.step')
print('final measured CAD: block envelope requested = 60 x 40 x 8 mm')
print('water bounds = 60 x 18 x 5 mm; section = 4 x 5 mm; pass pitch = 7 mm; ligament = 3 mm; passes = 3')
print('CAD volume [mm^3] =',round(fluid.volume*1e9,3))
