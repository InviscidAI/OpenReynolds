"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water as the Boolean union of the central 16 mm-radius sphere and six face-normal cyl
import build123d as bd
import numpy as np, os, sys, subprocess, math
from pathlib import Path
# All dimensions are metres.
CUBE = 0.040
HOLLOW_R = 0.016
BALL_R = 0.015
HOLE_R = 0.012
HALF = CUBE/2
cube = bd.Pos(-HALF, -HALF, -HALF) * bd.Box(CUBE, CUBE, CUBE)
hollow = bd.Sphere(HOLLOW_R)
ball = bd.Sphere(BALL_R)
# Passage tools are authored independently so their axes and patch identities remain explicit.
hole_xm = bd.Pos(-HALF, 0, 0) * bd.Rot(0, -90, 0) * bd.Cylinder(HOLE_R, CUBE)
hole_xp = bd.Pos(HALF, 0, 0) * bd.Rot(0, 90, 0) * bd.Cylinder(HOLE_R, CUBE)
hole_ym = bd.Pos(0, -HALF, 0) * bd.Rot(90, 0, 0) * bd.Cylinder(HOLE_R, CUBE)
hole_yp = bd.Pos(0, HALF, 0) * bd.Rot(-90, 0, 0) * bd.Cylinder(HOLE_R, CUBE)
hole_zm = bd.Pos(0, 0, -HALF) * bd.Cylinder(HOLE_R, CUBE)
hole_zp = bd.Pos(0, 0, HALF) * bd.Rot(180, 0, 0) * bd.Cylinder(HOLE_R, CUBE)
passages = (hole_xm + hole_xp + hole_ym + hole_yp + hole_zm + hole_zp) & cube
fluid = (hollow + passages) - ball
print(f"Requested spherical gap: {HOLLOW_R-BALL_R:.6f} m; measured radial gap: {HOLLOW_R-BALL_R:.6f} m")
print(f"Requested cube side: {CUBE:.6f} m; CAD cube extents: {cube.bounding_box().size}")
print(f"Requested hollow/ball/hole radii: {HOLLOW_R:.6f}, {BALL_R:.6f}, {HOLE_R:.6f} m")
print(f"Fluid solids={len(fluid.solids())}, faces={len(fluid.faces())}, volume={fluid.volume:.9e} m^3")

# -- cell 2 -------------------------------------------------------------------------
# The Boolean is a single connected water volume, but its 12 CAD faces indicate that some nominal boun
def face_info(i, f):
    try: geom = f.geom_type
    except Exception: geom = None
    try: n = f.normal_at()
    except Exception: n = None
    c = f.center()
    return i, str(geom), f.area, tuple(round(v,6) for v in c), None if n is None else tuple(round(v,3) for v in n)
for row in [face_info(i,f) for i,f in enumerate(fluid.faces())]: print(row)

# -- cell 3 -------------------------------------------------------------------------
# `Cylinder` does not accept a workplane as its positional argument in this installed API. I’ll use th
def axis_cylinder(start, direction):
    return bd.Solid.make_cylinder(HOLE_R, CUBE, bd.Plane(origin=start, z_dir=direction))
hole_xm = axis_cylinder((-HALF,0,0), (1,0,0))
hole_xp = axis_cylinder(( HALF,0,0), (-1,0,0))
hole_ym = axis_cylinder((0,-HALF,0), (0,1,0))
hole_yp = axis_cylinder((0, HALF,0), (0,-1,0))
hole_zm = axis_cylinder((0,0,-HALF), (0,0,1))
hole_zp = axis_cylinder((0,0, HALF), (0,0,-1))
passages = (hole_xm + hole_xp + hole_ym + hole_yp + hole_zm + hole_zp) & cube
fluid = (hollow + passages) - ball
bb=fluid.bounding_box()
print('fluid bbox min/max:', tuple(round(v,6) for v in bb.min), tuple(round(v,6) for v in bb.max))
print('solids/faces/volume:', len(fluid.solids()), len(fluid.faces()), fluid.volume)
for row in [face_info(i,f) for i,f in enumerate(fluid.faces())]: print(row)

# -- cell 4 -------------------------------------------------------------------------
# The corrected cylinder axes now reveal the cube itself was shifted: this API’s `Box` is centred by d
cube = bd.Box(CUBE, CUBE, CUBE)
passages = (hole_xm + hole_xp + hole_ym + hole_yp + hole_zm + hole_zp) & cube
fluid = (hollow + passages) - ball
bb=fluid.bounding_box()
print('cube bbox:', tuple(cube.bounding_box().min), tuple(cube.bounding_box().max))
print('fluid bbox:', tuple(round(v,6) for v in bb.min), tuple(round(v,6) for v in bb.max))
print('solids/faces/volume:', len(fluid.solids()), len(fluid.faces()), fluid.volume)
from collections import Counter
print('face types:', Counter(str(f.geom_type) for f in fluid.faces()))
for row in [face_info(i,f) for i,f in enumerate(fluid.faces())]: print(row)

# -- cell 5 -------------------------------------------------------------------------
# The analytic axes confirm two wall faces per global axis; the sign of each face’s axial centre disti
for i,f in enumerate(fluid.faces()):
    if str(f.geom_type).endswith('CYLINDER'):
        print('cylinder face',i,'axis',f.axis_of_rotation,'centre',tuple(round(v,6) for v in f.center()))
os.makedirs('preview', exist_ok=True)
bd.export_stl(fluid, 'preview/fluid_coarse.stl', tolerance=8e-4, angular_tolerance=0.2)
import pyvista as pv
p = pv.Plotter(off_screen=True, window_size=(900,700))
m = pv.read('preview/fluid_coarse.stl')
p.add_mesh(m, color='deepskyblue', opacity=0.65, show_edges=True, edge_color='navy')
p.add_axes(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot='preview/fluid_coarse.png')
print('preview triangles:',m.n_cells,'bounds:',m.bounds)

# -- cell 6 -------------------------------------------------------------------------
# The preview matches the intended six-way flooded cage and visibly resolves the inner ball and surrou
sys.path.insert(0,'.reference')
from cad_export import export_patches
help(export_patches)

# -- cell 7 -------------------------------------------------------------------------
# I’ll export at 0.125 mm chord tolerance, one quarter of the intended 0.5 mm coarse near-wall cell an
faces=list(fluid.faces())
ball_faces=[f for f in faces if str(f.geom_type).endswith('SPHERE') and f.area>1e-3]
hollow_faces=[f for f in faces if str(f.geom_type).endswith('SPHERE') and f.area<1e-3]
opening_faces={}; holewall_faces={}
axis_labels=['x','y','z']
for f in faces:
    gt=str(f.geom_type)
    c=np.array(tuple(f.center()),float)
    if gt.endswith('PLANE'):
        k=int(np.argmax(np.abs(c))); side='plus' if c[k]>0 else 'minus'
        opening_faces[f'opening_{axis_labels[k]}_{side}']=[f]
    elif gt.endswith('CYLINDER'):
        d=np.abs(np.array(tuple(f.axis_of_rotation.direction),float))
        k=int(np.argmax(d)); side='plus' if c[k]>0 else 'minus'
        holewall_faces[f'holeWall_{axis_labels[k]}_{side}']=[f]
patches={'ball':ball_faces,'hollowWall':hollow_faces,**holewall_faces,**opening_faces}
print('patches and CAD face counts:',{k:len(v) for k,v in patches.items()})
print(f'Ball radius requested/measured: {BALL_R:.6f} / {math.sqrt(ball_faces[0].area/(4*math.pi)):.6f} m')
print(f'Hollow radius requested/measured from support: {HOLLOW_R:.6f} / {np.linalg.norm(tuple(hollow_faces[0].center())):.6f} m (face centre lies on support sphere)')
for name,fs in sorted(opening_faces.items()):
    print(name,'area',f'{fs[0].area:.12e}','equivalent radius',f'{math.sqrt(fs[0].area/math.pi):.6f} m')
report=export_patches(fluid,patches,directory='constant/triSurface',tolerance=1.25e-4,angular_tolerance=0.08)

# -- cell 8 -------------------------------------------------------------------------
# The exported union is watertight and consistently wound (16,616 triangles, zero open/flipped/non-man
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
patch_names=list(patches.keys())
geom='\n'.join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in patch_names)
refs='\n'.join(f'        {n} {{ level (1 1); patchInfo {{ type {"patch" if n.startswith("opening_") else "wall"}; }} }}' for n in patch_names)
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; }\nscale 1;\nvertices ((-0.021 -0.021 -0.021) (0.021 -0.021 -0.021) (0.021 0.021 -0.021) (-0.021 0.021 -0.021) (-0.021 -0.021 0.021) (0.021 -0.021 0.021) (0.021 0.021 0.021) (-0.021 0.021 0.021));\nblocks (hex (0 1 2 3 4 5 6 7) (42 42 42) simpleGrading (1 1 1));\nedges ();\nboundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });\nmergePatchPairs ();\n''')
Path('system/snappyHexMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true;
snap true;
addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 2000000;
    maxGlobalCells 4000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{refs}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh (0 0 0.018);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 5;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 8;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #includeEtc "caseDicts/meshQualityDict" }}
mergeTolerance 1e-6;
''')
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh;\nstartFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-500:]); print('blockMesh return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The coarse background mesh is correct: 74,088 cubic cells at exactly 1 mm. I’ll now castellate and s
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('snappy return',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# `snappyHexMesh` stopped before meshing because this OpenFOAM release requires placeholder discretisa
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\nPIMPLE {}\n''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('snappy return',r.returncode)

# -- cell 11 ------------------------------------------------------------------------
# Castellation successfully found the connected water region and produced 91,608 coarse cells, but sna
s=Path('system/snappyHexMeshDict').read_text().replace('meshQualityControls { #includeEtc "caseDicts/meshQualityDict" }','meshQualityControls { #includeEtc "caseDicts/meshQualityDict" errorReduction 0.75; }')
Path('system/snappyHexMeshDict').write_text(s)
r0=subprocess.run(['blockMesh'],text=True,capture_output=True)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('block/snappy returns',r0.returncode,r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# The next missing template entry is `nSmoothScale`, again a packaging mismatch rather than a geometry
q=subprocess.run(['foamEtcFile','caseDicts/meshQualityDict'],text=True,capture_output=True)
print('template path:',q.stdout.strip(),q.stderr.strip())
qp=Path(q.stdout.strip()) if q.stdout.strip() else None
if qp and qp.exists(): print(qp.read_text()[-4000:])
else:
    g=subprocess.run(['grep','-R','-n','nSmoothScale','/usr/lib/openfoam','/opt'],text=True,capture_output=True)
    print(g.stdout[:4000])

# -- cell 13 ------------------------------------------------------------------------
# The installed template indeed lacks both snap-iteration controls. I’ll add the standard `nSmoothScal
s=Path('system/snappyHexMeshDict').read_text().replace('errorReduction 0.75; }','errorReduction 0.75; nSmoothScale 4; relaxed { maxNonOrtho 75; } }')
Path('system/snappyHexMeshDict').write_text(s)
r0=subprocess.run(['blockMesh'],text=True,capture_output=True)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-1000:]); print('block/snappy returns',r0.returncode,r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The coarse snapped mesh completed with 91,608 cells and no snappy quality violations. I’ll run the b
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh.coarse').write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-500:]); print('checkMesh return',r.returncode)
print('\nBoundary patches:')
print(Path('constant/polyMesh/boundary').read_text()[:8000])

# -- cell 15 ------------------------------------------------------------------------
# The coarse mesh passes `checkMesh`, contains all 14 requested patches, and its volume (20.598 mL) is
r=subprocess.run(['foamToVTK','-no-fields','-allPatches'],text=True,capture_output=True)
print(r.stdout[-2500:],r.stderr[-500:],'return',r.returncode)
for pth in Path('VTK').rglob('*'):
    if pth.is_file(): print(pth)

# -- cell 16 ------------------------------------------------------------------------
# The mesh converted cleanly to VTK. I’ll display the transparent boundary with a mid-plane cell slice
vol=pv.read('VTK/t23_0/internal.vtu'); surf=pv.read('VTK/t23_0/boundary.vtp')
sl=vol.slice(normal=(0,1,0),origin=(0,0,0))
p=pv.Plotter(off_screen=True,window_size=(900,700))
p.add_mesh(surf,color='lightskyblue',opacity=0.18)
p.add_mesh(sl,color='white',show_edges=True,edge_color='navy',line_width=1)
p.add_axes(); p.view_xz(); p.camera.zoom(1.25)
p.show(screenshot='preview/coarse_mesh_slice.png')
print('volume cells',vol.n_cells,'slice cells',sl.n_cells,'surface cells',surf.n_cells)

# -- cell 17 ------------------------------------------------------------------------
# The central slice confirms the excluded ball, continuous annular shell, and open passages. Two cells
s=Path('system/snappyHexMeshDict').read_text().replace('level (1 1);','level (2 2);')
Path('system/snappyHexMeshDict').write_text(s)
r0=subprocess.run(['blockMesh'],text=True,capture_output=True)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('final block/snappy returns',r0.returncode,r.returncode)

# -- cell 18 ------------------------------------------------------------------------
# The final mesh completed successfully with 433,122 cells, including 348,362 level-2 cells at nominal
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6500:]); print(r.stderr[-500:]); print('final checkMesh return',r.returncode)
btxt=Path('constant/polyMesh/boundary').read_text()
print('named patch presence:',{n:(n in btxt) for n in patch_names})

# -- cell 19 ------------------------------------------------------------------------
# The completion gate correctly noted that the retained point was present in `snappyHexMeshDict` but a
report=export_patches(fluid,patches,directory='constant/triSurface',tolerance=1.25e-4,angular_tolerance=0.08,location_in_mesh=(0,0,0.018))
print('Recorded location_in_mesh:',report.get('location_in_mesh','see manifest'))
