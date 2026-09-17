"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the barrel on the duct axis, with its base at \(z=0\). The stated fin stack is 102 mm tal
import build123d as bd
from build123d import Box, Cylinder, Sphere, Pos
import numpy as np, os, subprocess, pathlib
# All dimensions in metres
BARREL_D = 0.090
BARREL_H = 0.120
FIN_OD = 0.140
FIN_T = 0.003
FIN_GAP = 0.006
N_FINS = 12
DUCT_W = 0.300
DUCT_BELOW = 0.150
DUCT_ABOVE = 0.300
HEAD_R = BARREL_D/2
FIN_STACK_H = N_FINS*FIN_T + (N_FINS-1)*FIN_GAP
FIN_MARGIN = (BARREL_H-FIN_STACK_H)/2
fin_z = [FIN_MARGIN+i*(FIN_T+FIN_GAP) for i in range(N_FINS)]
barrel_tool = Cylinder(BARREL_D/2, BARREL_H)
fin_tools = [Pos(0,0,z)*Cylinder(FIN_OD/2, FIN_T) for z in fin_z]
# Upper half-sphere; its equator closes the barrel at z=BARREL_H
head_tool = (Pos(0,0,BARREL_H)*Sphere(HEAD_R)) & (Pos(-HEAD_R,-HEAD_R,BARREL_H)*Box(2*HEAD_R,2*HEAD_R,HEAD_R))
engine = barrel_tool
for ft in fin_tools:
    engine = engine + ft
engine = engine + head_tool
duct = Pos(-DUCT_W/2,-DUCT_W/2,-DUCT_BELOW)*Box(DUCT_W,DUCT_W,DUCT_BELOW+BARREL_H+DUCT_ABOVE)
fluid = duct - engine
print(f"fin stack requested/actual: {FIN_STACK_H:.6f} m; margins: {FIN_MARGIN:.6f} m")
print(f"duct extent z: {-DUCT_BELOW:.3f} to {BARREL_H+DUCT_ABOVE:.3f} m")
print(f"engine volume={engine.volume:.9g} m^3, fluid volume={fluid.volume:.9g} m^3")
print(f"engine faces={len(engine.faces())}, fluid faces={len(fluid.faces())}")
print('fluid bounds:', fluid.bounding_box().min, fluid.bounding_box().max)

# -- cell 2 -------------------------------------------------------------------------
# The first bounds check exposed a placement convention issue: `Box` is centred by default, so the duc
# Correct centred primitive placement explicitly; cylinders retain their z-min alignment.
duct_zmin = -DUCT_BELOW
duct_zmax = BARREL_H + DUCT_ABOVE
duct_center_z = (duct_zmin + duct_zmax)/2
duct = Pos(0,0,duct_center_z)*Box(DUCT_W, DUCT_W, duct_zmax-duct_zmin)
# Clip the sphere with a box centred over the required upper hemisphere.
head_clip = Pos(0,0,BARREL_H+HEAD_R/2)*Box(2*HEAD_R,2*HEAD_R,HEAD_R)
head_tool = (Pos(0,0,BARREL_H)*Sphere(HEAD_R)) & head_clip
engine = barrel_tool
for ft in fin_tools:
    engine = engine + ft
engine = engine + head_tool
fluid = duct - engine
print('duct bounds:', duct.bounding_box().min, duct.bounding_box().max)
print('engine bounds:', engine.bounding_box().min, engine.bounding_box().max)
print(f"head height measured={engine.bounding_box().max.Z-BARREL_H:.6f} m (assumed {HEAD_R:.6f})")
print(f"fin OD measured={2*max(abs(engine.bounding_box().min.X),abs(engine.bounding_box().max.X)):.6f} m (requested {FIN_OD:.6f})")
print(f"barrel axial height={BARREL_H:.6f} m (requested {BARREL_H:.6f})")
os.makedirs('constant/triSurface',exist_ok=True)
bd.export_stl(engine,'engine_preview.stl',tolerance=8e-4,angular_tolerance=0.15)
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
p = pv.Plotter(off_screen=True, window_size=(800,650))
m = pv.read('engine_preview.stl')
p.add_mesh(m,color='silver',show_edges=True)
# Wireframe duct gives spatial context.
p.add_mesh(pv.Box(bounds=(-.15,.15,-.15,.15,duct_zmin,duct_zmax)),style='wireframe',color='navy',opacity=.35)
p.camera_position=[(.42,-.50,.33),(0,0,.08),(0,0,1)]
p.show(screenshot='coarse_preview.png')
print(f"engine faces={len(engine.faces())}, fluid faces={len(fluid.faces())}, fluid volume={fluid.volume:.9g} m^3")

# -- cell 3 -------------------------------------------------------------------------
# The preview also reveals that `Cylinder` is centred axially in this installed build123d version: the
barrel_tool = Pos(0,0,BARREL_H/2)*Cylinder(BARREL_D/2, BARREL_H)
fin_tools = [Pos(0,0,z+FIN_T/2)*Cylinder(FIN_OD/2, FIN_T) for z in fin_z]
engine = barrel_tool
for ft in fin_tools:
    engine = engine + ft
engine = engine + head_tool
fluid = duct - engine
print('corrected engine bounds:',engine.bounding_box().min,engine.bounding_box().max)
print(f"lowest/highest fin faces measured={fin_z[0]:.6f}, {fin_z[-1]+FIN_T:.6f} m; intended centered stack 0.009000, 0.111000")
print(f"neighbour clear gap measured={fin_z[1]-(fin_z[0]+FIN_T):.6f} m (requested {FIN_GAP:.6f})")
from collections import Counter, defaultdict
summary=defaultdict(list)
for i,f in enumerate(fluid.faces()):
    gt=str(f.geom_type)
    c=f.center()
    summary[gt].append((i,round(f.area,8),(round(c.X,5),round(c.Y,5),round(c.Z,5))))
for k,v in summary.items():
    print(k,len(v),v[:8])
bd.export_stl(engine,'engine_preview_corrected.stl',tolerance=8e-4,angular_tolerance=.15)
p = pv.Plotter(off_screen=True, window_size=(800,650))
p.add_mesh(pv.read('engine_preview_corrected.stl'),color='silver',show_edges=True)
p.add_mesh(pv.Box(bounds=(-.15,.15,-.15,.15,duct_zmin,duct_zmax)),style='wireframe',color='navy',opacity=.3)
p.camera_position=[(.42,-.50,.33),(0,0,.08),(0,0,1)]
p.show(screenshot='coarse_preview_corrected.png')

# -- cell 4 -------------------------------------------------------------------------
# The corrected preview now shows the intended 12-fin stack entirely on the 120 mm barrel. The fluid b
import inspect, sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
print(inspect.signature(export_patches))
print(export_patches.__doc__.splitlines()[0:12])
# Show analytic properties available on representative planar/cylindrical/spherical faces.
for idx in [4,6,7,9,56]:
    f=fluid.faces()[idx]
    attrs={a:getattr(f,a) for a in ['geom_type','area','radius','center_location','axis_of_rotation'] if hasattr(f,a)}
    print(idx,attrs,'bb=',f.bounding_box().min,f.bounding_box().max)

# -- cell 5 -------------------------------------------------------------------------
# I’ll now export all six requested patch classes in one coordinated tessellation. The fin group is id
from build123d import GeomType
ff=list(fluid.faces())
# Duct faces survive subtraction unchanged; map each source face by OCC identity.
duct_source=list(duct.faces())
def mapped(src):
    hits=[f for f in ff if f.is_same(src)]
    if len(hits)!=1: raise RuntimeError(f'duct-face mapping yielded {len(hits)} matches')
    return hits[0]
duct_fluid=[mapped(s) for s in duct_source]
# Name axial duct planes from their exact construction coordinates (not bounding-box extrema).
inlet_faces=[f for f in duct_fluid if abs(f.center().Z-duct_zmin)<1e-9]
outlet_faces=[f for f in duct_fluid if abs(f.center().Z-duct_zmax)<1e-9]
duct_wall_faces=[f for f in duct_fluid if f not in inlet_faces+outlet_faces]
annulus_area=np.pi*((FIN_OD/2)**2-(BARREL_D/2)**2)
fin_faces=[f for f in ff if (f.geom_type==GeomType.CYLINDER and abs(f.radius-FIN_OD/2)<1e-9) or (f.geom_type==GeomType.PLANE and abs(f.area-annulus_area)<1e-9)]
head_faces=[f for f in ff if f.geom_type==GeomType.SPHERE and abs(f.radius-HEAD_R)<1e-9]
barrel_faces=[f for f in ff if (f.geom_type==GeomType.CYLINDER and abs(f.radius-BARREL_D/2)<1e-9) or (f.geom_type==GeomType.PLANE and abs(f.area-np.pi*(BARREL_D/2)**2)<1e-9)]
patches={'inlet':inlet_faces,'outlet':outlet_faces,'ductWalls':duct_wall_faces,'barrel':barrel_faces,'fins':fin_faces,'head':head_faces}
print({k:(len(v),sum(f.area for f in v)) for k,v in patches.items()})
print('covered faces:',sum(len(v) for v in patches.values()),'of',len(ff),'unique:',len({id(f) for v in patches.values() for f in v}))
export_info=export_patches(fluid,patches,tolerance=2.5e-4,angular_tolerance=0.08,location_in_mesh=(0.12,0.12,0.30))

# -- cell 6 -------------------------------------------------------------------------
# For meshing, I’ll use a 10 mm Cartesian duct background (30 × 30 × 57 cells) and let `snappyHexMesh`
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
block_dict='''FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (-0.15 -0.15 -0.15) (0.15 -0.15 -0.15) (0.15 0.15 -0.15) (-0.15 0.15 -0.15)
 (-0.15 -0.15 0.42) (0.15 -0.15 0.42) (0.15 0.15 0.42) (-0.15 0.15 0.42)
);
blocks ( hex (0 1 2 3 4 5 6 7) (30 30 57) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet { type patch; faces ((0 3 2 1)); }
 outlet { type patch; faces ((4 5 6 7)); }
 ductWalls { type wall; faces ((0 1 5 4)(1 2 6 5)(2 3 7 6)(3 0 4 7)); }
);
mergePatchPairs ();
'''
control='''FoamFile { format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 10; writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable true;
'''
open('system/blockMeshDict','w').write(block_dict)
open('system/controlDict','w').write(control)
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-1800:]); print(r.stderr[-500:]); print('return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The background mesh is correctly 0.30 × 0.30 × 0.57 m with 51,300 cells. I’ll now refine a compact b
snappy='''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
 barrel.stl { type triSurfaceMesh; name barrel; }
 fins.stl { type triSurfaceMesh; name fins; }
 head.stl { type triSurfaceMesh; name head; }
 engineRefine { type searchableBox; min (-0.075 -0.075 -0.005); max (0.075 0.075 0.170); }
}
castellatedMeshControls
{
 maxLocalCells 1200000; maxGlobalCells 1800000; minRefinementCells 0;
 maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ();
 refinementSurfaces
 {
  barrel { level (2 2); patchInfo { type wall; } }
  fins { level (2 2); patchInfo { type wall; } }
  head { level (2 2); patchInfo { type wall; } }
 }
 resolveFeatureAngle 25;
 refinementRegions { engineRefine { mode inside; levels ((1E15 2)); } }
 locationInMesh (0.12 0.12 0.30);
 allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
 multiRegionFeatureSnap false; detectNearSurfacesSnap true;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls
{
 #includeEtc "caseDicts/mesh/generation/meshQualityDict"
 relaxed { maxNonOrtho 75; }
}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
'''
open('system/snappyHexMeshDict','w').write(snappy)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-800:]); print('return',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# `snappyHexMesh` stopped before changing the block because this OpenFOAM release requires the standar
fv_schemes='''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default none; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
'''
fv_solution='''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers
{
 "cellDisplacement.*"
 {
  solver GAMG; tolerance 1e-7; relTol 0; smoother GaussSeidel;
  cacheAgglomeration true; nCellsInCoarsestLevel 10;
  agglomerator faceAreaPair; mergeLevels 1;
 }
}
'''
open('system/fvSchemes','w').write(fv_schemes); open('system/fvSolution','w').write(fv_solution)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The installed distribution does not provide the usual shared `meshQualityDict` include at that path.
r=subprocess.run(['bash','-lc','find ${WM_PROJECT_DIR:-/usr/lib/openfoam} -iname "*meshQuality*" -o -iname "snappyHexMeshDict" | head -30'],text=True,capture_output=True)
print(r.stdout); print('WM_PROJECT_DIR=',os.environ.get('WM_PROJECT_DIR'))

# -- cell 10 ------------------------------------------------------------------------
# The packaged file is `meshQualityDict.cfg`; I’ll correct that single include and rerun.
snappy=snappy.replace('caseDicts/mesh/generation/meshQualityDict"','caseDicts/mesh/generation/meshQualityDict.cfg"')
open('system/snappyHexMeshDict','w').write(snappy)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 11 ------------------------------------------------------------------------
# The snapped mesh completed with 234,184 cells and retained all three engine patch names. I’ll run th
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
open('log.checkMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-500:]); print('return',r.returncode)
print('\nBOUNDARY FILE:\n',open('constant/polyMesh/boundary').read())

# -- cell 12 ------------------------------------------------------------------------
# The binding check found four highly skew faces (maximum 7.34), so this version is not acceptable. Th
snappy_fine=snappy.replace('level (2 2); patchInfo','level (3 3); patchInfo')
open('system/snappyHexMeshDict','w').write(snappy_fine)
r=subprocess.run(['blockMesh'],text=True,capture_output=True,timeout=240)
print(r.stdout[-1200:]); print('return',r.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The original block has been restored; I’ll run the level-3 surface snap and inspect its final qualit
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-6500:]); print(r.stderr[-700:]); print('return',r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The targeted refinement produced 610,902 cells, with 1.25 mm cells adjacent to the engine and 2.5 mm
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
open('log.checkMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-500:]); print('return',r.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The refined mesh passes `checkMesh`: one region, six patches, maximum non-orthogonality 56.0°, maxim
open('case.foam','w').close()
reader=pv.OpenFOAMReader('case.foam')
meshdata=reader.read()
print(meshdata.keys())
internal=meshdata['internalMesh']
cut=internal.slice(normal=(0,1,0),origin=(0,0,0.06))
print('internal cells',internal.n_cells,'slice cells',cut.n_cells,'slice bounds',cut.bounds)
p=pv.Plotter(off_screen=True,window_size=(1000,750))
p.add_mesh(cut,color='white',show_edges=True,edge_color='black',line_width=.35)
p.camera_position='xz'
p.camera.zoom(2.0)
p.show(screenshot='mesh_longitudinal_cut.png')

# -- cell 16 ------------------------------------------------------------------------
# The advisory report confirms complete one-to-one patch coverage but still labels that confirmation a
pt=bd.Vector(0.12,0.12,0.30)
print('fluid contains locationInMesh:',fluid.is_inside(pt))
print('duct contains locationInMesh:',duct.is_inside(pt))
print('engine contains locationInMesh:',engine.is_inside(pt))
print('radial distance from engine axis:',float(np.hypot(pt.X,pt.Y)),'m; max engine radius:',FIN_OD/2,'m')
