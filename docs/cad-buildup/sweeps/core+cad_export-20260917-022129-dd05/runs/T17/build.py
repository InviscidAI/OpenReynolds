"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first probe found the sweep call but stopped on a nonexistent wire convenience method, so I will
import build123d as bd, inspect, os, subprocess, textwrap, math
print('Line', inspect.signature(bd.Line))
print('RadiusArc', inspect.signature(bd.RadiusArc))
print('Plane', inspect.signature(bd.Plane))
print('Circle', inspect.signature(bd.Circle))
print('Box', inspect.signature(bd.Box))
print('Cylinder', inspect.signature(bd.Cylinder))

# -- cell 2 -------------------------------------------------------------------------
# The extrusion API takes a scalar along the profile plane normal, not a vector. I’ll correct only tha
import build123d as bd, inspect, os, subprocess, textwrap, math, numpy as np
from build123d import Align, Vector
L,W,H = 0.300,0.120,0.080
D_THROTTLE, THROTTLE_NECK = 0.070,0.030
D_RUNNER, R_BEND, LEAD = 0.038,0.060,0.030
X_RUNNERS = [0.045,0.115,0.185,0.255]
plenum = bd.Pos(L/2,0,H/2) * bd.Box(L,W,H)
throttle_profile = bd.Plane(origin=(-THROTTLE_NECK,0,H/2),x_dir=(0,1,0),z_dir=(1,0,0))*bd.Circle(D_THROTTLE/2)
throttle = bd.extrude(throttle_profile,amount=THROTTLE_NECK)
runners=[]; runner_paths=[]
for x in X_RUNNERS:
    for side in (-1,1):
        p0=(x,side*W/2,H/2); p1=(x,side*(W/2+LEAD),H/2); p2=(x,side*(W/2+LEAD+R_BEND),H/2-R_BEND)
        path=bd.Line(p0,p1)+bd.RadiusArc(p1,p2,R_BEND)
        profile=bd.Plane(origin=p0,x_dir=(1,0,0),z_dir=(0,side,0))*bd.Circle(D_RUNNER/2)
        runners.append(bd.sweep(profile,path,is_frenet=True)); runner_paths.append(path)
fluid=plenum+throttle
for tube in runners: fluid=fluid+tube
edge_lengths=[e.length for e in runner_paths[0].edges()]
print('solids (required 1):',len(fluid.solids()))
print('volume m3:',fluid.volume,' bbox m:',fluid.bounding_box().min,fluid.bounding_box().max)
print('requested/measured plenum L,W,H m:',(L,W,H),(plenum.bounding_box().size.X,plenum.bounding_box().size.Y,plenum.bounding_box().size.Z))
print('requested/measured throttle diameter m:',D_THROTTLE,2*math.sqrt(min(throttle.faces(),key=lambda f:f.area).area/math.pi))
print('requested/measured runner diameter m:',D_RUNNER,2*math.sqrt(min(runners[0].faces(),key=lambda f:f.area).area/math.pi))
print('requested/measured bend radius m:',R_BEND,R_BEND)
print('requested/measured bend angle deg:',90.0,math.degrees(max(edge_lengths)/R_BEND),'path edge lengths',edge_lengths)
print('requested/measured port lateral distance m:',0.090,abs(runner_paths[0].position_at(1).Y)-W/2)
bd.export_stl(fluid,'coarseFluid.stl',tolerance=0.001,angular_tolerance=0.15)
import pyvista as pv
mesh=pv.read('coarseFluid.stl'); p=pv.Plotter(off_screen=True,window_size=(900,650)); p.add_mesh(mesh,color='lightsteelblue',show_edges=True); p.view_isometric(); p.add_axes(); p.show(screenshot='coarseFluid.png'); p.close()
print('wrote coarseFluid.png')

# -- cell 3 -------------------------------------------------------------------------
# The coarse shape is visibly correct and is one solid, but the path measurement reveals the bend crea
arc=max(runner_paths[0].edges(),key=lambda e:e.length)
print('arc endpoints:',[(v.X,v.Y,v.Z) for v in arc.vertices()])
print('geom type:',arc.geom_type,'length',arc.length)
for attr in ['radius','arc_center','center']:
    try: print(attr,getattr(arc,attr))
    except Exception as e: print(attr,type(e).__name__)
print('CenterArc',inspect.signature(bd.CenterArc))
print('JernArc',inspect.signature(bd.JernArc))

# -- cell 4 -------------------------------------------------------------------------
# The inspection confirms why: the three-dimensional radius-arc constructor chose an unintended obliqu
print(inspect.signature(bd.Edge.make_circle))
print(inspect.getdoc(bd.Edge.make_circle)[:1200])

# -- cell 5 -------------------------------------------------------------------------
# I’ll replace all eight bend paths with exact 90° circular edges whose local plane axes are defined e
runners=[]; runner_paths=[]
for x in X_RUNNERS:
    for side in (-1,1):
        p0=(x,side*W/2,H/2); p1=(x,side*(W/2+LEAD),H/2)
        center=(x,side*(W/2+LEAD),H/2-R_BEND)
        bend_plane=bd.Plane(origin=center,x_dir=(0,side,0),z_dir=(-side,0,0))
        arc=bd.Edge.make_circle(R_BEND,plane=bend_plane,start_angle=-90,end_angle=0,angular_direction=bd.AngularDirection.COUNTER_CLOCKWISE)
        path=bd.Line(p0,p1)+arc
        profile=bd.Plane(origin=p0,x_dir=(1,0,0),z_dir=(0,side,0))*bd.Circle(D_RUNNER/2)
        runners.append(bd.sweep(profile,path,is_frenet=True)); runner_paths.append(path)
fluid=plenum+throttle
for tube in runners: fluid=fluid+tube
arc=max(runner_paths[0].edges(),key=lambda e:e.length)
print('solids (required 1):',len(fluid.solids()),'volume m3:',fluid.volume)
print('requested/measured bend radius m:',R_BEND,arc.radius)
print('requested/measured bend angle deg:',90.0,math.degrees(arc.length/arc.radius))
print('requested/measured port lateral distance m:',0.090,abs(runner_paths[0].position_at(1).Y)-W/2)
print('all port centers:',[(round(p.position_at(1).X,3),round(p.position_at(1).Y,3),round(p.position_at(1).Z,3)) for p in runner_paths])
bd.export_stl(fluid,'coarseFluid.stl',tolerance=0.001,angular_tolerance=0.15)
mesh=pv.read('coarseFluid.stl'); p=pv.Plotter(off_screen=True,window_size=(900,650)); p.add_mesh(mesh,color='lightsteelblue',show_edges=True); p.view_isometric(); p.add_axes(); p.show(screenshot='coarseFluid.png'); p.close()
print('updated coarseFluid.png')

# -- cell 6 -------------------------------------------------------------------------
# The corrected preview shows all eight continuous runners and the throttle neck, and the requested be
faces=list(fluid.faces())
print('face count',len(faces))
for i,f in enumerate(faces):
    c=f.center(); n=f.normal_at();
    print(i,str(f.geom_type),f'area={f.area:.7g}',f'c=({c.X:.4f},{c.Y:.4f},{c.Z:.4f})',f'n=({n.X:.2f},{n.Y:.2f},{n.Z:.2f})')

# -- cell 7 -------------------------------------------------------------------------
# The union has exactly the expected 32 boundary faces: 8 planar port disks, 8 toroidal bend walls, 8 
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
faces=list(fluid.faces())
def nearest_face(point,candidates):
    q=bd.Vector(point); return min(candidates,key=lambda f:(f.center()-q).length)
planar=[f for f in faces if f.geom_type==bd.GeomType.PLANE]
throttle_inlet=nearest_face((-THROTTLE_NECK,0,H/2),planar)
port_faces={}
for bank,side in [('left',-1),('right',1)]:
    for j,x in enumerate(X_RUNNERS,1):
        port_faces[f'port_{bank}_{j}']=nearest_face((x,side*(W/2+LEAD+R_BEND),H/2-R_BEND),planar)
runner_walls=[f for f in faces if f.geom_type==bd.GeomType.TORUS or (f.geom_type==bd.GeomType.CYLINDER and f.area<0.005)]
assigned={throttle_inlet,*port_faces.values(),*runner_walls}
plenum_walls=[f for f in faces if f not in assigned]
patches={'throttle_inlet':[throttle_inlet],'plenum_walls':plenum_walls,'runner_walls':runner_walls}
for name,face in port_faces.items(): patches[name]=[face]
print('tag counts:',{k:len(v) for k,v in patches.items()},'total',sum(map(len,patches.values())),'of',len(faces))
print('port centers:',{k:tuple(round(a,6) for a in (v.center().X,v.center().Y,v.center().Z)) for k,v in port_faces.items()})
export_patches(fluid,patches,tolerance=2.5e-4)

# -- cell 8 -------------------------------------------------------------------------
# The coordinated export is watertight (0 open, flipped, or non-manifold edges). I’ll now create a del
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant/polyMesh').mkdir(parents=True,exist_ok=True)
block='''FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices ((-0.04 -0.18 -0.04) (0.31 -0.18 -0.04) (0.31 0.18 -0.04) (-0.04 0.18 -0.04) (-0.04 -0.18 0.09) (0.31 -0.18 0.09) (0.31 0.18 0.09) (-0.04 0.18 0.09));
blocks (hex (0 1 2 3 4 5 6 7) (44 45 17) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(block)
stls=['throttle_inlet','plenum_walls','runner_walls']+[f'port_{b}_{i}' for b in ['left','right'] for i in range(1,5)]
geometry='\n'.join([f'    {s}.stl {{ type triSurfaceMesh; name {s}; }}' for s in stls])
refine='\n'.join([f'        {s} {{ level (1 1); patchInfo {{ type patch; }} }}' for s in stls])
snappy=f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geometry}
}}
castellatedMeshControls
{{
    maxLocalCells 300000; maxGlobalCells 600000; minRefinementCells 0; nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{refine}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh (0.15 0 0.04);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 40; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls
{{
    #include "meshQualityDict"
}}
mergeTolerance 1e-6;
'''
Path('system/snappyHexMeshDict').write_text(snappy)
Path('system/meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }\n''')
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n''')
print('wrote dictionaries for',len(stls),'named surfaces')

# -- cell 9 -------------------------------------------------------------------------
# The dictionaries are ready for all 11 named surfaces. I’ll generate the background grid first and in
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-3000:]); print(r.stderr[-1000:]); print('return',r.returncode); assert r.returncode==0

# -- cell 10 ------------------------------------------------------------------------
# snappyHexMesh did not start because this OpenFOAM build requires placeholder finite-volume dictionar
Path('system/fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
print('wrote meshing prerequisites')

# -- cell 11 ------------------------------------------------------------------------
# This release also validates the usual scheme subdictionaries. I’ll add only inert defaults required 
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
print('completed required scheme sections')

# -- cell 12 ------------------------------------------------------------------------
# The mesher prerequisites are complete; I’ll run snappyHexMesh again.
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240); Path('log.snappyHexMesh').write_text(r.stdout+r.stderr); print((r.stdout+r.stderr)[-6000:]); print('return',r.returncode); assert r.returncode==0

# -- cell 13 ------------------------------------------------------------------------
# The conformed mesh completed with 37,405 cells (33,078 at the refined level) and no final snappy qua
r=subprocess.run(['checkMesh'],text=True,capture_output=True); Path('log.checkMesh').write_text(r.stdout+r.stderr); print(r.stdout+r.stderr); print('return',r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The binding check passes: one connected region, all 11 requested boundary patches present, 37,405 ce
r=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True); print((r.stdout+r.stderr)[-2000:]); print('return',r.returncode); print(list(Path('VTK').rglob('*'))[:20])

# -- cell 15 ------------------------------------------------------------------------
# The VTK conversion confirms every named patch was written from the actual mesh. I’ll render those me
bdir=Path('VTK/t17_0/boundary'); names=['plenum_walls','runner_walls','throttle_inlet']+[f'port_{b}_{i}' for b in ['left','right'] for i in range(1,5)]
colors=['lightgray','steelblue','crimson','gold','orange','limegreen','cyan','magenta','violet','yellow','salmon']
p=pv.Plotter(off_screen=True,window_size=(1000,700))
for name,color in zip(names,colors): p.add_mesh(pv.read(bdir/f'{name}.vtp'),color=color,show_edges=True,line_width=0.4,label=name)
p.view_isometric(); p.add_axes(); p.add_legend(bcolor='white',size=(0.22,0.42)); p.show(screenshot='finalMesh.png'); p.close()
print('wrote finalMesh.png; rendered patches:',names)
