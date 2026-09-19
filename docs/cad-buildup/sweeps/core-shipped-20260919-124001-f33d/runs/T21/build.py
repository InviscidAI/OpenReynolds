"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the internal air volume in metres as a 160 × 100 × 40 mm cavity, extended through the 3 m
import build123d as bd
from build123d import Box, Cylinder, Pos, export_stl
import os, math
# All dimensions are metres.
L, W, H = 0.160, 0.100, 0.040
WALL = 0.003
STAND_D, STAND_H = 0.010, 0.025
HOLE_D, HOLE_DEPTH = 0.004, 0.012
VENT_W, VENT_H, VENT_Z0 = 0.030, 0.020, 0.010
# Main cavity and wall-thickness vent passage on the x-min short wall.
cavity = Box(L, W, H)
vent_passage = Pos(-WALL, (W-VENT_W)/2, VENT_Z0) * Box(WALL, VENT_W, VENT_H)
post_centres = [(0.015,0.015),(0.015,W-0.015),(L-0.015,0.015),(L-0.015,W-0.015)]
outer_posts = [Pos(x,y,0) * Cylinder(STAND_D/2, STAND_H) for x,y in post_centres]
blind_airs = [Pos(x,y,STAND_H-HOLE_DEPTH) * Cylinder(HOLE_D/2, HOLE_DEPTH) for x,y in post_centres]
fluid = cavity + vent_passage
for post in outer_posts:
    fluid = fluid - post
for bore in blind_airs:
    fluid = fluid + bore
os.makedirs('constant/triSurface', exist_ok=True)
export_stl(fluid, 'coarse_geometry_preview.stl', tolerance=5e-4, angular_tolerance=0.2)
print(f'Fluid volume: {fluid.volume:.9e} m^3')
print(f'Expected cavity+vent-posts+bores: {(L*W*H + WALL*VENT_W*VENT_H - 4*math.pi*(STAND_D/2)**2*STAND_H + 4*math.pi*(HOLE_D/2)**2*HOLE_DEPTH):.9e} m^3')
print('Post centres (m):', post_centres)
print('Faces:', len(fluid.faces()), 'bounds:', fluid.bounding_box().min, fluid.bounding_box().max)
# Render the coarse tessellation for inspection.
import pyvista as pv
pv.OFF_SCREEN = True
mesh_preview = pv.read('coarse_geometry_preview.stl')
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(mesh_preview, color='lightblue', show_edges=True, edge_color='navy', opacity=0.82)
pl.view_isometric(); pl.camera.zoom(1.25)
pl.add_axes(); pl.show(screenshot='coarse_geometry_preview.png')
print('Preview:', mesh_preview.n_cells, 'triangles; saved coarse_geometry_preview.png')

# -- cell 2 -------------------------------------------------------------------------
# The coarse preview exposed an origin-placement error: `Box` was centred by default, so three standof
import inspect
print('Box:', inspect.signature(Box))
print('Cylinder:', inspect.signature(Cylinder))

# -- cell 3 -------------------------------------------------------------------------
# I’ll now apply explicit `(MIN, MIN, MIN)` alignment to the boxes and `(CENTER, CENTER, MIN)` alignme
A = bd.Align
cavity = Box(L, W, H, align=(A.MIN,A.MIN,A.MIN))
vent_passage = Pos(-WALL, (W-VENT_W)/2, VENT_Z0) * Box(WALL, VENT_W, VENT_H, align=(A.MIN,A.MIN,A.MIN))
outer_posts = [Pos(x,y,0) * Cylinder(STAND_D/2, STAND_H, align=(A.CENTER,A.CENTER,A.MIN)) for x,y in post_centres]
blind_airs = [Pos(x,y,STAND_H-HOLE_DEPTH) * Cylinder(HOLE_D/2, HOLE_DEPTH, align=(A.CENTER,A.CENTER,A.MIN)) for x,y in post_centres]
fluid = cavity + vent_passage
for post in outer_posts:
    fluid = fluid - post
for bore in blind_airs:
    fluid = fluid + bore
expected_volume = L*W*H + WALL*VENT_W*VENT_H - 4*math.pi*(STAND_D/2)**2*STAND_H + 4*math.pi*(HOLE_D/2)**2*HOLE_DEPTH
print(f'Measured fluid volume {fluid.volume:.12e} m^3; analytic {expected_volume:.12e}; rel.err {(fluid.volume/expected_volume-1):.3e}')
bb=fluid.bounding_box(); print('Measured bounds:', tuple(bb.min), tuple(bb.max), 'target:',(-WALL,0,0),(L,W,H))
export_stl(fluid, 'coarse_geometry_preview.stl', tolerance=3e-4, angular_tolerance=0.15)
mesh_preview = pv.read('coarse_geometry_preview.stl')
pl = pv.Plotter(off_screen=True, window_size=(900,650)); pl.add_mesh(mesh_preview,color='lightblue',show_edges=True,edge_color='navy',opacity=.8)
pl.view_isometric(); pl.camera.zoom(1.2); pl.add_axes(); pl.show(screenshot='coarse_geometry_preview_aligned.png')
print('Faces:',len(fluid.faces()),'preview triangles:',mesh_preview.n_cells)

# -- cell 4 -------------------------------------------------------------------------
# The corrected model now matches the analytic fluid volume exactly and has the intended bounds. Next 
for i,f in enumerate(fluid.faces()):
    c=f.center();
    try: n=f.normal_at()
    except Exception: n=None
    print(i, 'type=',f.geom_type, 'area=',f'{f.area:.9g}', 'center=',tuple(round(v,6) for v in c), 'normal=',None if n is None else tuple(round(v,3) for v in n))

# -- cell 5 -------------------------------------------------------------------------
# The face inventory cleanly separates the requested regions: the top and vent are single planar faces
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
help(export_patches)

# -- cell 6 -------------------------------------------------------------------------
faces = list(fluid.faces())
open_top = [faces[2]]
floor_faces = [faces[3]]
vent = [faces[14]]
standoff_outer = [faces[i] for i in [10,11,12,13,15,16,17,18]]
blind_hole_surfaces = [faces[i] for i in [19,20,21,22,23,24,25,26]]
enclosure_walls = [faces[i] for i in [0,1,4,5,6,7,8,9]]
patches = {'open_top':open_top, 'vent':vent, 'enclosure_walls':enclosure_walls,
           'floor':floor_faces, 'standoff_outer':standoff_outer,
           'blind_hole_surfaces':blind_hole_surfaces}
def area(fs): return sum(f.area for f in fs)
expected_areas = {
 'open_top':L*W,
 'vent':VENT_W*VENT_H,
 'floor':L*W-4*math.pi*(STAND_D/2)**2,
 'standoff_outer':4*(math.pi*STAND_D*STAND_H + math.pi*((STAND_D/2)**2-(HOLE_D/2)**2)),
 'blind_hole_surfaces':4*(math.pi*HOLE_D*HOLE_DEPTH + math.pi*(HOLE_D/2)**2)}
for name,fs in patches.items():
    measured=area(fs); target=expected_areas.get(name)
    print(name, 'faces',len(fs),'measured area',f'{measured:.12g}', '' if target is None else f'target {target:.12g}, error {measured-target:.3e}')
report = export_patches(fluid, patches, directory='constant/triSurface', tolerance=2.5e-4, angular_tolerance=0.1)

# -- cell 7 -------------------------------------------------------------------------
# The six STL patches are conformal and their welded union has zero open, flipped, or non-manifold edg
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ((-0.005 -0.005 -0.005) (0.165 -0.005 -0.005) (0.165 0.105 -0.005) (-0.005 0.105 -0.005)
          (-0.005 -0.005 0.045) (0.165 -0.005 0.045) (0.165 0.105 0.045) (-0.005 0.105 0.045));
blocks (hex (0 1 2 3 4 5 6 7) (34 22 10) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
''')
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr); print('blockMesh rc',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The 7,480-cell background grid is correct at 5 mm isotropic spacing. I’ll now run the coarse surface
stl_names=['open_top','vent','enclosure_walls','floor','standoff_outer','blind_hole_surfaces']
geom='\n'.join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in stl_names)
refs='\n'.join(f'    {n} {{ level ({2 if n in ["standoff_outer","blind_hole_surfaces"] else 1} {2 if n in ["standoff_outer","blind_hole_surfaces"] else 1}); patchInfo {{ type patch; }} }}' for n in stl_names)
snappy=f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls {{
 maxLocalCells 300000; maxGlobalCells 500000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2; features ();
 refinementSurfaces {{
{refs}
 }}
 resolveFeatureAngle 30; refinementRegions {{}}
 locationInMesh (0.080 0.050 0.035); allowFreeStandingZoneFaces true;
}}
snapControls {{ nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #include "meshQualityDict" }}
mergeTolerance 1e-6;
'''
Path('system/snappyHexMeshDict').write_text(snappy)
Path('system/meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }\n''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('snappyHexMesh.log').write_text(r.stdout+r.stderr)
print(r.stdout[-5000:]); print('snappy rc',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The coarse snapped mesh completed with 27,772 cells and no mesh-quality errors reported during snapp
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print(r.stdout); print(r.stderr); print('checkMesh rc',r.returncode)
print('\nBoundary file patch summary:')
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 10 ------------------------------------------------------------------------
# `checkMesh` passes: one connected region, six requested patches, 27,772 cells, maximum non-orthogona
r=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True)
print(r.stdout[-1200:],r.stderr,'rc',r.returncode)
print([str(p) for p in Path('VTK').rglob('*') if p.is_file()][:20])

# -- cell 11 ------------------------------------------------------------------------
# The VTK export contains all six patches. I’ll render the boundary with transparent enclosure surface
vdir=Path('VTK/t21_0/boundary')
parts={n:pv.read(vdir/f'{n}.vtp') for n in stl_names}
bh=parts['blind_hole_surfaces']
conn=bh.connectivity(); region_ids=set(conn.cell_data['RegionId'].tolist())
print('Blind-hole mesh bounds:',bh.bounds,'connected components:',len(region_ids),'target z extent:',STAND_H-HOLE_DEPTH,'to',STAND_H)
print('Standoff mesh bounds:',parts['standoff_outer'].bounds,'target z extent: 0 to',STAND_H)
pl=pv.Plotter(off_screen=True,window_size=(1000,700))
pl.add_mesh(parts['enclosure_walls'],color='lightgray',opacity=.16,show_edges=True,edge_color='gray')
pl.add_mesh(parts['floor'],color='tan',opacity=.35)
pl.add_mesh(parts['open_top'],color='cyan',opacity=.08)
pl.add_mesh(parts['vent'],color='red',opacity=.9,show_edges=True)
pl.add_mesh(parts['standoff_outer'],color='royalblue',opacity=.75,show_edges=True,edge_color='navy')
pl.add_mesh(bh,color='orange',opacity=1,show_edges=True,edge_color='darkorange')
pl.view_isometric(); pl.camera.zoom(1.35); pl.add_axes(); pl.show(screenshot='mesh_boundary_inspection.png')
print('Saved mesh_boundary_inspection.png')

# -- cell 12 ------------------------------------------------------------------------
# The inspection confirms four disconnected blind-hole surface components spanning exactly \(z=13\)–25
# Measurements extracted from resulting CAD faces/edges.
def circle_radii(face):
    vals=[]
    for e in face.edges():
        if e.geom_type==bd.GeomType.CIRCLE:
            vals.append(e.radius)
    return vals
outer_r = circle_radii(faces[10])[0]
hole_r = circle_radii(faces[19])[0]
outer_cyl_z=[v.Z for v in faces[10].vertices()]
hole_cyl_z=[v.Z for v in faces[19].vertices()]
vent_bb=faces[14].bounding_box(); top_bb=faces[2].bounding_box()
centres_measured=[]
for i in [15,16,17,18]:
    c=faces[i].center(); centres_measured.append((c.X,c.Y))
print(f'Internal plan measured: {(top_bb.max.X-top_bb.min.X)*1000:.3f} x {(top_bb.max.Y-top_bb.min.Y)*1000:.3f} mm; depth {(top_bb.min.Z-faces[3].bounding_box().max.Z)*1000:.3f} mm')
print(f'Outer wall/vent passage thickness measured: {(0-vent_bb.min.X)*1000:.3f} mm')
print(f'Vent measured: {(vent_bb.max.Y-vent_bb.min.Y)*1000:.3f} x {(vent_bb.max.Z-vent_bb.min.Z)*1000:.3f} mm; lower edge {vent_bb.min.Z*1000:.3f} mm above floor')
print(f'Standoff diameter measured: {2*outer_r*1000:.3f} mm; height {(max(outer_cyl_z)-min(outer_cyl_z))*1000:.3f} mm')
print(f'Blind-hole diameter measured: {2*hole_r*1000:.3f} mm; depth {(max(hole_cyl_z)-min(hole_cyl_z))*1000:.3f} mm')
print('Measured standoff centres (mm):',[(round(x*1000,3),round(y*1000,3)) for x,y in centres_measured])
print('Offsets from nearest internal corners (mm):',[(round(min(x,L-x)*1000,3),round(min(y,W-y)*1000,3)) for x,y in centres_measured])

# -- cell 13 ------------------------------------------------------------------------
# The completion gate found one metadata omission: although snappyHexMesh used the valid interior poin
report = export_patches(fluid, patches, directory='constant/triSurface', tolerance=2.5e-4, angular_tolerance=0.1, location_in_mesh=(0.080,0.050,0.035))
print('Recorded location_in_mesh:',report.get('location_in_mesh','see manifest'))
