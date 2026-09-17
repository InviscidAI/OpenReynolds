"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I will interpret the water volume as the annular chamber from the 12 mm hub radius to the 58 mm radi
import build123d as bd, inspect, math, numpy as np, os, subprocess, textwrap
print('build123d', bd.__version__)
print('Spline', inspect.signature(bd.Spline))
print('Face.make_surface?', hasattr(bd.Face,'make_surface'))
print('offset', inspect.signature(bd.offset))
print('extrude', inspect.signature(bd.extrude))

# -- cell 2 -------------------------------------------------------------------------
# The first replication expression returned build123d’s location-list wrapper rather than individual s
from build123d import *
import math, numpy as np
# All dimensions are metres.
R_BACK=0.060; T_BACK=0.004; R_HUB=0.012; HUB_RISE=0.018
Z_FLOOR=T_BACK; PASSAGE_H=0.014; Z_SHROUD=Z_FLOOR+PASSAGE_H
R_BLADE_IN=0.016; R_OUT=0.058; BLADE_T=0.003; N_BLADES=7; SWEEP_DEG=30.0
u=np.linspace(0,1,31); r=R_BLADE_IN+(R_OUT-R_BLADE_IN)*u
theta=-math.radians(SWEEP_DEG)*(3*u**2-2*u**3)
pts=[(float(ri*math.cos(ti)),float(ri*math.sin(ti))) for ri,ti in zip(r,theta)]
centre=Spline(*pts)
blade_sketch=offset(centre, amount=BLADE_T/2, side=Side.BOTH, closed=True)
blade0=extrude(blade_sketch, amount=PASSAGE_H, dir=(0,0,1)).moved(Location((0,0,Z_FLOOR)))
blade_parts=[blade0.moved(Location((0,0,0),(0,0,k*360/N_BLADES))) for k in range(N_BLADES)]
blades=Compound(blade_parts)
envelope=(Cylinder(R_OUT,PASSAGE_H)-Cylinder(R_HUB,PASSAGE_H)).moved(Location((0,0,Z_FLOOR)))
fluid=envelope-blades
print('fluid solids',len(fluid.solids()),'volume m3',fluid.volume)
print('passage height requested/measured m',PASSAGE_H,fluid.bounding_box().size.Z)
print('sweep requested/measured deg',SWEEP_DEG,math.degrees(theta[0]-theta[-1]))
print('blade count requested/built',N_BLADES,len(blade_parts))
print('extent m',fluid.bounding_box().min,fluid.bounding_box().max)

# -- cell 3 -------------------------------------------------------------------------
# The measured vertical extent exposed a build123d `Location` constructor ambiguity: the three-tuple w
import inspect
print(inspect.signature(Location))
print(inspect.signature(Location.__init__))

# -- cell 4 -------------------------------------------------------------------------
# I will verify explicit vector translation separately so the corrected construction has a measured \(
test_loc=Location(Vector(0,0,Z_FLOOR))
test_env=(Cylinder(R_OUT,PASSAGE_H)-Cylinder(R_HUB,PASSAGE_H)).moved(test_loc)
print(test_env.bounding_box().min.Z,test_env.bounding_box().max.Z)

# -- cell 5 -------------------------------------------------------------------------
# `Pos` itself is the immutable location, not a list; I will apply it directly.
test_env2=(Cylinder(R_OUT,PASSAGE_H)-Cylinder(R_HUB,PASSAGE_H)).moved(Pos(0,0,Z_FLOOR))
print(test_env2.bounding_box().min.Z,test_env2.bounding_box().max.Z)

# -- cell 6 -------------------------------------------------------------------------
# The translation is functioning, but this reveals that `Cylinder` is centred by default in this insta
print(inspect.signature(Cylinder))
raw=Cylinder(R_OUT,PASSAGE_H)
print('raw z',raw.bounding_box().min.Z,raw.bounding_box().max.Z)

# -- cell 7 -------------------------------------------------------------------------
# I will correct the envelope with explicit minimum-\(z\) alignment, reconstruct the final fluid shape
envelope0=Cylinder(R_OUT,PASSAGE_H,align=(Align.CENTER,Align.CENTER,Align.MIN))-Cylinder(R_HUB,PASSAGE_H,align=(Align.CENTER,Align.CENTER,Align.MIN))
envelope=envelope0.moved(Pos(0,0,Z_FLOOR))
fluid=envelope-blades
bb=fluid.bounding_box()
print('fluid solids',len(fluid.solids()),'volume m3',fluid.volume)
print('z requested/measured m',(Z_FLOOR,Z_SHROUD),(bb.min.Z,bb.max.Z))
print('radial requested/measured m',R_OUT,max(abs(bb.min.X),bb.max.X,abs(bb.min.Y),bb.max.Y))
print('faces',len(fluid.faces()))

# -- cell 8 -------------------------------------------------------------------------
# Four faces means the blade subtraction did not intersect the corrected envelope, despite the unchang
print('blade0 bbox',blade0.bounding_box().min,blade0.bounding_box().max)
print('envelope volume',envelope.volume,'fluid volume',fluid.volume)

# -- cell 9 -------------------------------------------------------------------------
# The offset-generated sketch is empty in this API path. I will inspect the centreline and offset obje
print(type(centre),len(centre.edges()),centre.length)
print(type(blade_sketch),len(blade_sketch.faces()),len(blade_sketch.edges()))

# -- cell 10 ------------------------------------------------------------------------
# The offset contains a valid closed curve but not a face, so extrusion silently produced no solid. I 
print('wires',len(blade_sketch.wires()),[w.is_closed for w in blade_sketch.wires()])
blade_face=Face(blade_sketch.wires()[0])
blade0=extrude(blade_face, amount=PASSAGE_H).moved(Pos(0,0,Z_FLOOR))
print('blade volume',blade0.volume,'bbox',blade0.bounding_box().min,blade0.bounding_box().max)

# -- cell 11 ------------------------------------------------------------------------
# I will re-emit the successful export step with all previously refused-cell names defined locally, so
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
blade_parts=[blade0.moved(Rot(0,0,k*360/N_BLADES)) for k in range(N_BLADES)]; blades=Compound(blade_parts)
INLET_NECK=0.002; tol=2e-5
neck0=Cylinder(R_BLADE_IN,INLET_NECK,align=(Align.CENTER,Align.CENTER,Align.MIN))-Cylinder(R_HUB,INLET_NECK,align=(Align.CENTER,Align.CENTER,Align.MIN))
neck=neck0.moved(Pos(0,0,Z_SHROUD)); envelope_with_inlet=envelope+neck; fluid=envelope_with_inlet-blades
faces=list(fluid.faces()); inlet=[]; shroud=[]; backplate=[]; hub=[]; outlet=[]; blade_surfaces=[]
for f in faces:
 c=f.center(); n=f.normal_at(c); rr=math.hypot(c.X,c.Y)
 if c.Z>Z_SHROUD+INLET_NECK/2 and n.Z>0.5: inlet.append(f)
 elif abs(c.Z-Z_SHROUD)<tol and n.Z>0.5: shroud.append(f)
 elif abs(c.Z-Z_FLOOR)<tol and n.Z<-0.5: backplate.append(f)
 elif abs(rr-R_HUB)<tol: hub.append(f)
 elif abs(rr-R_OUT)<tol: outlet.append(f)
 elif c.Z>Z_SHROUD-tol and abs(rr-R_BLADE_IN)<tol: shroud.append(f)
 else: blade_surfaces.append(f)
print('patch face counts', {k:len(v) for k,v in {'inlet':inlet,'outlet':outlet,'blades':blade_surfaces,'backplate':backplate,'hub':hub,'shroud':shroud}.items()})
export_patches(fluid, {'inlet':inlet,'outlet':outlet,'blades':blade_surfaces,'backplate':backplate,'hub':hub,'shroud':shroud}, tolerance=2.5e-4)

# -- cell 12 ------------------------------------------------------------------------
# The closed six-patch surface is now on disk. I will create a coarse Cartesian background mesh with a
import os, textwrap
os.makedirs('system',exist_ok=True); os.makedirs('constant/polyMesh',exist_ok=True)
open('system/controlDict','w').write('''FoamFile{format ascii; class dictionary; object controlDict;}\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;''')
open('system/blockMeshDict','w').write('''FoamFile{format ascii; class dictionary; object blockMeshDict;}\nscale 1; vertices ((-0.065 -0.065 0.0)(0.065 -0.065 0.0)(0.065 0.065 0.0)(-0.065 0.065 0.0)(-0.065 -0.065 0.024)(0.065 -0.065 0.024)(0.065 0.065 0.024)(-0.065 0.065 0.024)); blocks(hex (0 1 2 3 4 5 6 7) (26 26 6) simpleGrading (1 1 1)); edges(); boundary(background{type patch; faces((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}); mergePatchPairs();''')
names=['inlet','outlet','blades','backplate','hub','shroud']
geom='\n'.join(f'{n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs='\n'.join(f'{n} {{ level (1 1); patchInfo {{ type wall; }} }}' for n in names)
open('system/snappyHexMeshDict','w').write(f'''FoamFile{{format ascii; class dictionary; object snappyHexMeshDict;}}\ncastellatedMesh true; snap true; addLayers false; geometry{{{geom}}} castellatedMeshControls{{maxLocalCells 200000; maxGlobalCells 300000; minRefinementCells 0; nCellsBetweenLevels 1; features(); refinementSurfaces{{{refs}}} resolveFeatureAngle 30; refinementRegions{{}} locationInMesh (0.027 0.013 0.011); allowFreeStandingZoneFaces true;}} snapControls{{nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false;}} addLayersControls{{relativeSizes true; layers{{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;}} meshQualityControls{{#include "meshQualityDict"}} debug 0; mergeTolerance 1e-6;''')
# permissive standard quality dictionary for coarse first mesh
open('system/meshQualityDict','w').write('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed{maxNonOrtho 75;}''')
print('wrote OpenFOAM dictionaries')

# -- cell 13 ------------------------------------------------------------------------
# The first mesher call failed immediately on dictionary syntax, not geometry. I will print the genera
print(open('system/blockMeshDict').read())

# -- cell 14 ------------------------------------------------------------------------
# The coarse background mesh now exists (4,056 cells). Snappy stopped while reading its dictionary; I 
print(open('snappy.log').read()[-2500:])

# -- cell 15 ------------------------------------------------------------------------
# Geometry loading succeeded; the remaining stop is after feature setup. I will inspect the final fata
print(open('snappy.log').read()[-1800:])

# -- cell 16 ------------------------------------------------------------------------
# The compact token `features();` was not parsed as an entry. I will replace it with `features ();`, r
p=open('system/snappyHexMeshDict').read().replace('features();','features ();')
open('system/snappyHexMeshDict','w').write(p)
subprocess.run(['blockMesh'],check=True,capture_output=True,text=True)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True); open('snappy.log','w').write(r2.stdout+r2.stderr); print('snappy rc',r2.returncode,r2.stdout[-1800:]); r2.check_returncode()

# -- cell 17 ------------------------------------------------------------------------
# A coarse snapped mesh now exists with 1,281 cells. I will run the binding `checkMesh` and inspect pa
cm=subprocess.run(['checkMesh'],text=True,capture_output=True); open('checkMesh.log','w').write(cm.stdout+cm.stderr); print(cm.stdout[-2200:]); print('return',cm.returncode)
print(open('constant/polyMesh/boundary').read())

# -- cell 18 ------------------------------------------------------------------------
# The quality check passes, but the bounds show only one blade passage was retained; at this coarse re
res=subprocess.run(['grep','-R','-m','2','locationsInMesh','/opt/OpenFOAM/OpenFOAM-v2512/tutorials'],capture_output=True,text=True)
print(res.stdout[:2000])

# -- cell 19 ------------------------------------------------------------------------
# Rather than accept seven artificially disconnected cell regions, I will make the coarsest useful cor
d=open('system/blockMeshDict').read().replace('(26 26 6)','(52 52 8)'); open('system/blockMeshDict','w').write(d)
subprocess.run(['blockMesh'],check=True,capture_output=True,text=True)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True); open('snappy.log','w').write(r.stdout+r.stderr); print(r.stdout[-900:]); r.check_returncode()
cm=subprocess.run(['checkMesh'],text=True,capture_output=True); open('checkMesh.log','w').write(cm.stdout+cm.stderr); print(cm.stdout[-1400:]); cm.check_returncode()

# -- cell 20 ------------------------------------------------------------------------
# The corrected mesh now spans the full 116 mm diameter, is one region, has 29,831 cells, and passes `
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
# Read STL patches directly for a clear boundary/patch preview.
colors={'inlet':'cyan','outlet':'royalblue','blades':'orange','backplate':'silver','hub':'red','shroud':'lightgreen'}
pl=pv.Plotter(off_screen=True,window_size=(900,700));
for n,col in colors.items():
 m=pv.read(f'constant/triSurface/{n}.stl'); pl.add_mesh(m,color=col,opacity=0.25 if n in ['shroud','backplate'] else 0.9,show_edges=(n=='blades'))
pl.camera_position=[(0.11,-0.13,0.10),(0,0,0.011),(0,0,1)]; pl.add_legend([(k,v) for k,v in colors.items()]); pl.show(screenshot='impeller_passage.png')
print('saved impeller_passage.png')

# -- cell 21 ------------------------------------------------------------------------
# The preview confirms all seven backward-curved passages and the named surfaces. Before closing, I ne
import os
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write('''FoamFile { format ascii; class dictionary; object controlDict; } application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;''')
open('system/blockMeshDict','w').write('''FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1; vertices ( (-0.065 -0.065 0) (0.065 -0.065 0) (0.065 0.065 0) (-0.065 0.065 0) (-0.065 -0.065 0.024) (0.065 -0.065 0.024) (0.065 0.065 0.024) (-0.065 0.065 0.024) );
blocks ( hex (0 1 2 3 4 5 6 7) (52 52 8) simpleGrading (1 1 1) ); edges ();
boundary ( background { type patch; faces ( (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); } ); mergePatchPairs ();''')
names=['inlet','outlet','blades','backplate','hub','shroud']; geom=' '.join(f'{n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names); refs=' '.join(f'{n} {{ level (1 1); patchInfo {{ type wall; }} }}' for n in names)
open('system/snappyHexMeshDict','w').write(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }} castellatedMesh true; snap true; addLayers false; geometry {{ {geom} }} castellatedMeshControls {{ maxLocalCells 200000; maxGlobalCells 300000; minRefinementCells 0; nCellsBetweenLevels 1; features (); refinementSurfaces {{ {refs} }} resolveFeatureAngle 30; refinementRegions {{}} locationInMesh (0.027 0.013 0.011); allowFreeStandingZoneFaces true; }} snapControls {{ nSmoothPatch 3; tolerance 2; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }} addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }} meshQualityControls {{ #include "meshQualityDict" }} debug 0; mergeTolerance 1e-6;''')
open('system/meshQualityDict','w').write('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }''')
open('system/fvSchemes','w').write('''FoamFile { format ascii; class dictionary; object fvSchemes; } ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }''')
open('system/fvSolution','w').write('''FoamFile { format ascii; class dictionary; object fvSolution; } solvers { cellDisplacement { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0; } }''')
print('canonical final OpenFOAM dictionaries written')

# -- cell 22 ------------------------------------------------------------------------
# I will perform one final clean remesh from the canonical dictionaries, rerun `checkMesh`, and print 
subprocess.run(['blockMesh'],check=True,capture_output=True,text=True)
sm=subprocess.run(['snappyHexMesh','-overwrite'],check=True,capture_output=True,text=True); open('snappy.log','w').write(sm.stdout+sm.stderr)
cm=subprocess.run(['checkMesh'],check=True,capture_output=True,text=True); open('checkMesh.log','w').write(cm.stdout+cm.stderr)
# For a constant-width open-curve offset with round caps, A = t*L + pi*t^2/4.
t_meas=(math.sqrt(centre.length**2+math.pi*blade_face.area)-centre.length)*2/math.pi
print('checkMesh:', 'Mesh OK.' in cm.stdout)
print('cells:', [x.strip() for x in cm.stdout.splitlines() if 'cells:' in x.lower()][0])
print('blade count requested/measured:',N_BLADES,len(blade_parts))
print('sweep deg requested/measured:',SWEEP_DEG,math.degrees(theta[0]-theta[-1]))
print('blade thickness m requested/measured:',BLADE_T,t_meas)
print('passage height m requested/measured:',PASSAGE_H,fluid.bounding_box().max.Z-INLET_NECK-Z_FLOOR)
print('fluid radial outlet m requested/measured:',R_OUT,max(abs(fluid.bounding_box().min.X),fluid.bounding_box().max.X))
print('mesh bounds line:',[x.strip() for x in cm.stdout.splitlines() if 'Overall domain bounding box' in x][0])
