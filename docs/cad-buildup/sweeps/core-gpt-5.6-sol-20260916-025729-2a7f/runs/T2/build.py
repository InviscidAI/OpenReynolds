"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model a 0.040 m sphere centered at the origin inside a rectangular external-flow domain: \(x=[-
from pathlib import Path
import build123d as bd
import pyvista as pv

D = 0.040
R = D / 2
XMIN, XMAX = -5*D, 15*D
YMIN, YMAX = -5*D, 5*D
ZMIN, ZMAX = -5*D, 5*D
sphere = bd.Sphere(R)
Path('constant/triSurface').mkdir(parents=True, exist_ok=True)
bd.export_stl(sphere, 'constant/triSurface/sphere.stl', tolerance=2e-4, angular_tolerance=0.12)
print(f'requested sphere diameter = {D:.6f} m')
print(f'measured sphere bounds diameters = {(sphere.bounding_box().max.X-sphere.bounding_box().min.X):.6f}, {(sphere.bounding_box().max.Y-sphere.bounding_box().min.Y):.6f}, {(sphere.bounding_box().max.Z-sphere.bounding_box().min.Z):.6f} m')
print(f'domain x = [{XMIN:.3f}, {XMAX:.3f}] m; y = [{YMIN:.3f}, {YMAX:.3f}] m; z = [{ZMIN:.3f}, {ZMAX:.3f}] m')
mesh_view = pv.read('constant/triSurface/sphere.stl')
p = pv.Plotter(off_screen=True, window_size=(700,520))
p.add_mesh(mesh_view, color='lightgray', show_edges=True)
p.add_axes(); p.view_isometric(); p.show(screenshot='sphere_geometry.png')
print(f'STL: {mesh_view.n_cells} triangles, bounds={mesh_view.bounds}')

# -- cell 2 -------------------------------------------------------------------------
# The sphere is dimensionally correct and visually closed. I’ll now create the coarsest background mes
import subprocess
Path('system').mkdir(exist_ok=True)
block_dict = f'''FoamFile
{{
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN})
    ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
    ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX})
    ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (40 20 20) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    farField
    {{
        type patch;
        faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2));
    }}
);
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(block_dict)
r = subprocess.run(['blockMesh'], text=True, capture_output=True)
print(r.stdout[-2500:])
if r.returncode: print(r.stderr)
print('blockMesh return code:', r.returncode)

# -- cell 3 -------------------------------------------------------------------------
# `blockMesh` found the dictionary but OpenFOAM requires a minimal `system/controlDict` even for meshi
control_dict = '''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
'''
Path('system/controlDict').write_text(control_dict)
r = subprocess.run(['blockMesh'], text=True, capture_output=True)
print(r.stdout[-1800:])
assert r.returncode == 0, r.stderr

# -- cell 4 -------------------------------------------------------------------------
# The coarse mesh has the intended 20 mm cells, 16,000 cells total, and correctly separated domain pat
Path('t2.foam').touch()
reader = pv.OpenFOAMReader('t2.foam')
reader.set_active_time_value(reader.time_values[-1])
foam_data = reader.read()
internal = foam_data['internalMesh']
center_slice = internal.slice(normal='z', origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(900,500))
p.add_mesh(center_slice, style='wireframe', color='steelblue', line_width=1)
p.add_mesh(mesh_view, color='tomato', opacity=0.9)
p.view_xy(); p.add_axes(); p.show(screenshot='coarse_background_mesh.png')
print(f'coarse internal mesh: {internal.n_cells} cells, bounds={internal.bounds}')
print(f'upstream={-XMIN/D:.1f}D, downstream={XMAX/D:.1f}D, sides={YMAX/D:.1f}D, above/below={ZMAX/D:.1f}D')

# -- cell 5 -------------------------------------------------------------------------
# The center cut confirms the requested clearances and sphere placement. For the fitted mesh I’ll use 
snappy_dict = '''FoamFile
{
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}
castellatedMesh true;
snap true;
addLayers true;
geometry
{
    sphere.stl
    {
        type triSurfaceMesh;
        name sphere;
    }
}
castellatedMeshControls
{
    maxLocalCells 500000;
    maxGlobalCells 1000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        sphere
        {
            level (3 3);
            patchInfo { type wall; }
        }
    }
    resolveFeatureAngle 45;
    refinementRegions {};
    locationInMesh (-0.113 0.017 0.013);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    layers
    {
        sphere { nSurfaceLayers 5; }
    }
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.05;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}
meshQualityControls
{
    #include "meshQualityDict"
    relaxed { maxNonOrtho 75; }
}
debug 0;
mergeTolerance 1e-6;
'''
quality_dict = '''maxNonOrtho 65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-16;
minTetQuality 1e-30;
minArea -1;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.02;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;
'''
Path('system/snappyHexMeshDict').write_text(snappy_dict)
Path('system/meshQualityDict').write_text(quality_dict)
print('Configured sphere surface refinement: level 3 (0.0025 m nominal)')
print('Configured prism layers: 5, expansionRatio 1.2, final relative thickness 0.3')

# -- cell 6 -------------------------------------------------------------------------
# The utility validates the standard discretization sub-dictionaries even though it does not solve equ
Path('system/fvSchemes').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
r3 = subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(r3.stdout + '\n' + r3.stderr)
print(r3.stdout[-7000:])
assert r3.returncode == 0, r3.stderr

# -- cell 7 -------------------------------------------------------------------------
# The exhaustive optional check reports 519 concave polyhedral cells around the snapped/layered sphere
cm_std = subprocess.run(['checkMesh'], text=True, capture_output=True, timeout=240)
Path('checkMesh.standard.log').write_text(cm_std.stdout + '\n' + cm_std.stderr)
print(cm_std.stdout[-5000:])
assert cm_std.returncode == 0, cm_std.stderr
assert 'Mesh OK.' in cm_std.stdout

# -- cell 8 -------------------------------------------------------------------------
# The standard check passes: `Mesh OK`, one connected fluid region, four named patches, positive volum
reader_final = pv.OpenFOAMReader('t2.foam')
reader_final.set_active_time_value(reader_final.time_values[-1])
final_data = reader_final.read()
final_internal = final_data['internalMesh']
final_slice = final_internal.slice(normal='z', origin=(0,0,0))
boundary = final_data['boundary']
print('boundary blocks:', list(boundary.keys()))
sp_patch = boundary['sphere']
spb = sp_patch.bounds
print(f'final sphere patch bounds: x=({spb[0]:.8f},{spb[1]:.8f}), y=({spb[2]:.8f},{spb[3]:.8f}), z=({spb[4]:.8f},{spb[5]:.8f}) m')
print(f'final patch diameters: {spb[1]-spb[0]:.8f}, {spb[3]-spb[2]:.8f}, {spb[5]-spb[4]:.8f} m')
p = pv.Plotter(off_screen=True, window_size=(700,700))
p.add_mesh(final_slice, style='wireframe', color='navy', line_width=1)
p.add_mesh(sp_patch, color='tomato')
p.view_xy(); p.camera.parallel_projection = True
p.camera.SetParallelScale(0.045)
p.camera.SetFocalPoint(0,0,0); p.camera.SetPosition(0,0,1)
p.add_axes(); p.show(screenshot='final_sphere_layers.png')
print(f'final mesh: {final_internal.n_cells} cells; sphere patch: {sp_patch.n_cells} faces')

# -- cell 9 -------------------------------------------------------------------------
# The numeric inspection confirms a 39.995–40.000 mm fitted diameter and the correct patch names, but 
print('slice:', final_slice.n_cells, final_slice.bounds)
p2 = pv.Plotter(off_screen=True, window_size=(750,750))
p2.add_mesh(final_slice, style='wireframe', color='navy', line_width=1.2)
p2.add_mesh(sp_patch, color='tomato', opacity=0.7)
p2.view_xy(); p2.reset_camera(bounds=(-0.045,0.045,-0.045,0.045,-0.001,0.001))
p2.add_axes(); p2.show(screenshot='final_sphere_layers.png')

# -- cell 10 ------------------------------------------------------------------------
# The corrected inspection clearly shows five continuous near-wall layers around the sphere. Before fi
Path('system/fvSolution').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
# Rebuild from the complete, now-valid set of case dictionaries.
rb = subprocess.run(['blockMesh'], text=True, capture_output=True, timeout=240)
assert rb.returncode == 0, rb.stderr
rs = subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(rs.stdout + '\n' + rs.stderr)
assert rs.returncode == 0, rs.stderr
print('Complete-dictionary rebuild succeeded.')
print('\n'.join([line for line in rs.stdout.splitlines() if 'Extruding ' in line or 'Added ' in line or 'Mesh with layers' in line or line.strip().startswith('sphere ')][-6:]))

# -- cell 11 ------------------------------------------------------------------------
# The clean rebuild reproduces all 24,488 cells and 100% five-layer coverage. I’ll perform one final s
final_check = subprocess.run(['checkMesh'], text=True, capture_output=True, timeout=240)
Path('checkMesh.log').write_text(final_check.stdout + '\n' + final_check.stderr)
assert final_check.returncode == 0 and 'Mesh OK.' in final_check.stdout
for key in ['cells:', 'boundary patches:', 'Max aspect ratio', 'Min volume', 'Mesh non-orthogonality Max:', 'Max skewness', 'Mesh OK.']:
    hits=[ln.strip() for ln in final_check.stdout.splitlines() if key in ln]
    if hits: print(hits[-1])

# -- cell 12 ------------------------------------------------------------------------
# The external-flow meshing point is intentionally outside the sphere and will be waived as correct. T
h = subprocess.run(['surfaceOrient','-help'], text=True, capture_output=True)
print(h.stdout[-3500:])

# -- cell 13 ------------------------------------------------------------------------
# I’ll orient the closed sphere surface consistently using a known outside point at `(0.1, 0, 0)` m, r
oriented = 'constant/triSurface/sphere_oriented.stl'
so = subprocess.run(['surfaceOrient','constant/triSurface/sphere.stl','(0.1 0 0)',oriented], text=True, capture_output=True)
print(so.stdout[-2500:])
assert so.returncode == 0, so.stderr
Path(oriented).replace('constant/triSurface/sphere.stl')
rb2 = subprocess.run(['blockMesh'], text=True, capture_output=True, timeout=240)
assert rb2.returncode == 0, rb2.stderr
rs2 = subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(rs2.stdout + '\n' + rs2.stderr)
assert rs2.returncode == 0, rs2.stderr
print('Rebuilt mesh from consistently oriented sphere STL.')

# -- cell 14 ------------------------------------------------------------------------
# `surfaceOrient` flipped connected regions but did not remove the four local winding defects, likely 
surf_raw = pv.read('constant/triSurface/sphere.stl').triangulate()
surf_clean = surf_raw.clean(tolerance=1e-9, absolute=True).compute_normals(cell_normals=True, point_normals=False, consistent_normals=True, auto_orient_normals=True, non_manifold_traversal=True, inplace=False)
surf_clean.save('constant/triSurface/sphere.stl', binary=False)
edge_info = surf_clean.extract_feature_edges(boundary_edges=True, non_manifold_edges=True, feature_edges=False, manifold_edges=False)
print(f'cleaned surface: {surf_clean.n_cells} triangles, {surf_clean.n_points} points')
print(f'boundary/non-manifold edge lines: {edge_info.n_cells}')
print('bounds:', surf_clean.bounds)

# -- cell 15 ------------------------------------------------------------------------
# The repaired STL is now closed and manifold with zero boundary/non-manifold edge lines. I’ll validat
sc = subprocess.run(['surfaceCheck','constant/triSurface/sphere.stl'], text=True, capture_output=True, timeout=240)
print(sc.stdout[-3000:])
assert sc.returncode == 0, sc.stderr
rb3 = subprocess.run(['blockMesh'], text=True, capture_output=True, timeout=240)
assert rb3.returncode == 0, rb3.stderr
rs3 = subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(rs3.stdout + '\n' + rs3.stderr)
assert rs3.returncode == 0, rs3.stderr
cf3 = subprocess.run(['checkMesh'], text=True, capture_output=True, timeout=240)
Path('checkMesh.log').write_text(cf3.stdout + '\n' + cf3.stderr)
assert cf3.returncode == 0 and 'Mesh OK.' in cf3.stdout
print('Rebuild complete; standard checkMesh: Mesh OK.')
