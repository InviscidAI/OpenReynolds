"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid directly as a closed axisymmetric annulus. The three prescribed stations will b
import build123d as bd, inspect, os, sys, numpy as np
print('build123d', bd.__version__)
for n in ['Spline','Line','Wire','Face','revolve']:
    obj=getattr(bd,n); print(n, inspect.signature(obj))
print('cwd',os.getcwd())

# -- cell 2 -------------------------------------------------------------------------
# A single quadratic Bézier through each set of three stations gives a deterministic smooth blend with
print('Bezier',inspect.signature(bd.Bezier)); print('make_face',inspect.signature(bd.make_face)); print('export_stl',inspect.signature(bd.export_stl))

# -- cell 3 -------------------------------------------------------------------------
# I’ll now construct the closed meridian in metres and revolve it. The source boundary edges are expli
import matplotlib.pyplot as plt
L=0.900
x_st=np.array([0.0,0.450,0.900])
r_outer_req=np.array([0.450,0.430,0.410])
r_core_req=np.array([0.280,0.310,0.250])
# Quadratic Bezier control radii chosen to interpolate the half-length station.
r_outer_ctrl=2*r_outer_req[1]-0.5*(r_outer_req[0]+r_outer_req[2])
r_core_ctrl=2*r_core_req[1]-0.5*(r_core_req[0]+r_core_req[2])
outer_edge=bd.Bezier((0,0,r_outer_req[0]),(L/2,0,r_outer_ctrl),(L,0,r_outer_req[2])).edge()
outlet_edge=bd.Line((L,0,r_outer_req[2]),(L,0,r_core_req[2])).edge()
core_edge=bd.Bezier((L,0,r_core_req[2]),(L/2,0,r_core_ctrl),(0,0,r_core_req[0])).edge()
inlet_edge=bd.Line((0,0,r_core_req[0]),(0,0,r_outer_req[0])).edge()
meridian_wire=bd.Wire([outer_edge,outlet_edge,core_edge,inlet_edge])
meridian_face=bd.Face(meridian_wire)
fluid=bd.revolve(meridian_face,axis=bd.Axis.X)
print('solid volume m3',fluid.volume,'faces',len(fluid.faces()))
def qbez(r0,rc,r2,t): return (1-t)**2*r0+2*(1-t)*t*rc+t*t*r2
for name,req,ctrl in [('outer',r_outer_req,r_outer_ctrl),('core',r_core_req,r_core_ctrl)]:
    got=np.array([qbez(req[0],ctrl,req[2],t) for t in (0,.5,1)])
    print(name,'requested m',req,'measured/evaluated m',got,'max error',abs(got-req).max())
# Coarse meridian preview is sufficient to expose the axial/radial shape before meshing.
t=np.linspace(0,1,41); x=L*t
ro=qbez(r_outer_req[0],r_outer_ctrl,r_outer_req[2],t); ri=qbez(r_core_req[0],r_core_ctrl,r_core_req[2],t)
fig,ax=plt.subplots(figsize=(9,4)); ax.fill_between(x,ri,ro,alpha=.35,label='bypass air'); ax.fill_between(x,-ro,-ri,alpha=.35)
ax.plot(x,ro,'b-',x,ri,'r-',x,-ro,'b-',x,-ri,'r-'); ax.scatter(np.r_[x_st,x_st],np.r_[r_outer_req,r_core_req],c='k',s=20)
ax.set_aspect('equal'); ax.set_xlabel('axial x [m]'); ax.set_ylabel('radius [m]'); ax.grid(); ax.legend(); ax.set_title('Coarse axisymmetric bypass-air geometry'); plt.show()

# -- cell 4 -------------------------------------------------------------------------
# The coarse section has the intended converging outer wall and the inner cowl’s smooth rise and contr
for i,f in enumerate(fluid.faces()):
    c=f.center();
    try: n=f.normal_at()
    except Exception as e: n=e
    print(i,'geom',f.geom_type,'area',f.area,'center',tuple(round(v,6) for v in c),'normal',n)

# -- cell 5 -------------------------------------------------------------------------
# The face order follows the four source meridian edges exactly: outer revolution, outlet plane, core 
sys.path.insert(0,'.reference')
from cad_export import export_patches
help(export_patches)

# -- cell 6 -------------------------------------------------------------------------
# For the final surface I’ll use 0.5 mm chord tolerance, much finer than the planned ~25–30 mm coarse 
outer_cowl=fluid.faces()[0]
nozzle_outlet=fluid.faces()[1]
core_cowl=fluid.faces()[2]
fan_exit_inlet=fluid.faces()[3]
assert outer_cowl.geom_type==bd.GeomType.REVOLUTION and core_cowl.geom_type==bd.GeomType.REVOLUTION
assert nozzle_outlet.geom_type==bd.GeomType.PLANE and fan_exit_inlet.geom_type==bd.GeomType.PLANE
assert outer_cowl.area>core_cowl.area>fan_exit_inlet.area>nozzle_outlet.area
patch_report=export_patches(fluid,{'outer_cowl':outer_cowl,'core_cowl':core_cowl,'fan_exit_inlet':fan_exit_inlet,'nozzle_outlet':nozzle_outlet},directory='constant/triSurface',tolerance=5e-4,angular_tolerance=0.12)
print(patch_report)

# -- cell 7 -------------------------------------------------------------------------
# I’ll use `snappyHexMesh` because the exported closed patch union is the authoritative geometry and p
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
def write(path,text): Path(path).write_text(text)
foam='FoamFile\n{\n version 2.0; format ascii; class dictionary; object %s;\n}\n'
write('system/controlDict',(foam%'controlDict')+'''application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 8; writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable true;
''')
write('system/blockMeshDict',(foam%'blockMeshDict')+'''scale 1;
vertices
(
 (-0.03 -0.48 -0.48) (0.93 -0.48 -0.48) (0.93 0.48 -0.48) (-0.03 0.48 -0.48)
 (-0.03 -0.48  0.48) (0.93 -0.48  0.48) (0.93 0.48  0.48) (-0.03 0.48  0.48)
);
blocks ( hex (0 1 2 3 4 5 6 7) (32 32 32) simpleGrading (1 1 1) );
edges ();
boundary ( background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); } );
mergePatchPairs ();
''')
write('system/snappyHexMeshDict',(foam%'snappyHexMeshDict')+'''castellatedMesh true;
snap true;
addLayers false;
geometry
{
 outer_cowl.stl { type triSurfaceMesh; name outer_cowl; }
 core_cowl.stl { type triSurfaceMesh; name core_cowl; }
 fan_exit_inlet.stl { type triSurfaceMesh; name fan_exit_inlet; }
 nozzle_outlet.stl { type triSurfaceMesh; name nozzle_outlet; }
}
castellatedMeshControls
{
 maxLocalCells 300000; maxGlobalCells 600000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2;
 features ();
 refinementSurfaces
 {
  outer_cowl { level (0 0); patchInfo { type wall; } }
  core_cowl { level (0 0); patchInfo { type wall; } }
  fan_exit_inlet { level (0 0); patchInfo { type patch; } }
  nozzle_outlet { level (0 0); patchInfo { type patch; } }
 }
 resolveFeatureAngle 30;
 refinementRegions {}
 locationInMesh (0.10 0.35 0);
 allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls
{
 relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3;
 minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
 nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
 maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
 nLayerIter 50;
}
meshQualityControls
{
 #include "meshQualityDict"
}
debug 0; mergeTolerance 1e-6;
''')
write('system/meshQualityDict','''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
minVol 1e-13; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
relaxed { maxNonOrtho 75; }
''')
print('dictionaries written')

# -- cell 8 -------------------------------------------------------------------------
# I’ll generate and inspect the background mesh first, before surface cutting.
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('blockMesh failed')

# -- cell 9 -------------------------------------------------------------------------
# `snappyHexMesh` stopped before changing the mesh because this OpenFOAM build requires minimal `fvSch
write('system/fvSchemes',(foam%'fvSchemes')+'ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
write('system/fvSolution',(foam%'fvSolution')+'solvers {}\n')
print('minimal meshing runtime dictionaries written')

# -- cell 10 ------------------------------------------------------------------------
# I’ll rerun the previously blocked coarse cut-and-snap operation.
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r2.stdout+r2.stderr)
print((r2.stdout+r2.stderr)[-5000:]); print('return',r2.returncode)
if r2.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 11 ------------------------------------------------------------------------
# The coarse cut mesh has 10,123 cells and completed all mesher quality checks. I’ll now run the bindi
cm=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(cm.stdout+cm.stderr)
print((cm.stdout+cm.stderr)[-5000:]); print('return',cm.returncode)
if cm.returncode: raise RuntimeError('checkMesh failed')

# -- cell 12 ------------------------------------------------------------------------
# The coarse mesh passes `checkMesh`, has one connected region and all four named patches. Its total v
import pyvista as pv
Path('t16.foam').touch()
reader=pv.OpenFOAMReader('t16.foam'); reader.set_active_time_value(reader.time_values[-1]); mb=reader.read()
print(mb)
for i in range(mb.n_blocks): print(i,mb.get_block_name(i),mb[i])

# -- cell 13 ------------------------------------------------------------------------
# The OpenFOAM reader confirms 10,123 volume cells and four boundary blocks. I’ll display a half-secti
grid=mb['internalMesh']; bnd=mb['boundary']
cut=grid.clip(normal=(0,-1,0),origin=(0,0,0),invert=False)
p=pv.Plotter(off_screen=True,window_size=(1000,650)); p.set_background('white')
p.add_mesh(cut,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.4,opacity=0.75)
colors={'outer_cowl':'royalblue','core_cowl':'firebrick','fan_exit_inlet':'seagreen','nozzle_outlet':'darkorange'}
for name,col in colors.items(): p.add_mesh(bnd[name],color=col,label=name,opacity=0.9)
p.add_legend(bcolor='white',face='circle'); p.view_isometric(); p.camera.zoom(1.25); p.show_axes()
p.show(screenshot='mesh_preview.png')

# -- cell 14 ------------------------------------------------------------------------
# The cutaway confirms the correct annular region, continuous core exclusion, and distinct end patches
bmd=Path('system/blockMeshDict').read_text().replace('(32 32 32)','(48 48 48)')
write('system/blockMeshDict',bmd)
print('final background resolution set to 48 x 48 x 48 = 20 mm isotropic')

# -- cell 15 ------------------------------------------------------------------------
# I’ll regenerate the refined background mesh.
rb=subprocess.run(['blockMesh'],text=True,capture_output=True)
print((rb.stdout+rb.stderr)[-2200:]); print('return',rb.returncode)
if rb.returncode: raise RuntimeError('refined blockMesh failed')

# -- cell 16 ------------------------------------------------------------------------
# The final background is confirmed at 20 mm isotropic spacing. I’ll cut and snap it to the annulus.
rf=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(rf.stdout+rf.stderr)
print((rf.stdout+rf.stderr)[-4200:]); print('return',rf.returncode)
if rf.returncode: raise RuntimeError('final snappyHexMesh failed')

# -- cell 17 ------------------------------------------------------------------------
# The final mesh contains 33,502 cells and the mesher reports no remaining quality violations. I’ll ru
cmf=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(cmf.stdout+cmf.stderr)
print((cmf.stdout+cmf.stderr)[-5000:]); print('return',cmf.returncode)
if cmf.returncode: raise RuntimeError('final checkMesh failed')

# -- cell 18 ------------------------------------------------------------------------
# The final mesh passes `checkMesh`: one region, four patches, 33,502 cells, maximum non-orthogonality
print(inspect.signature(bd.export_step)); bd.export_step(fluid,'bypass_air_fluid.step'); print('wrote bypass_air_fluid.step, bytes',Path('bypass_air_fluid.step').stat().st_size)

# -- cell 19 ------------------------------------------------------------------------
# The exporter signature reveals that its default STEP declaration is millimetres, so the preceding au
print(list(bd.Unit)); bd.export_step(fluid,'bypass_air_fluid.step',unit=bd.Unit.M); print('rewrote STEP with explicit unit',bd.Unit.M)

# -- cell 20 ------------------------------------------------------------------------
# The completion gate correctly notes that the meshing-side point was not recorded in the STL manifest
patch_report=export_patches(fluid,{'outer_cowl':outer_cowl,'core_cowl':core_cowl,'fan_exit_inlet':fan_exit_inlet,'nozzle_outlet':nozzle_outlet},directory='constant/triSurface',tolerance=5e-4,angular_tolerance=0.12,location_in_mesh=(0.10,0.35,0.0))
print('recorded location_in_mesh',patch_report['location_in_mesh'])
