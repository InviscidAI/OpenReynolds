"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid as the 160 × 100 × 40 mm internal cavity plus the 3 mm-long vent passage throug
import os, math
import build123d as bd
from build123d import Box, Cylinder, Pos, Align
from build123d import export_stl
# All dimensions are metres. Requested dimensions:
L, W, H = 0.160, 0.100, 0.040
wall = 0.003
post_d, post_h = 0.010, 0.025
hole_d, hole_depth = 0.004, 0.012
corner_offset = 0.015
vent_w, vent_h, vent_bottom = 0.030, 0.020, 0.010
# Assumptions: vent is centred in y and passes through the x=0 short wall.
vent_y0 = (W - vent_w) / 2
interior = Box(L, W, H, align=(Align.MIN, Align.MIN, Align.MIN))
vent_channel = Pos(-wall, vent_y0, vent_bottom) * Box(wall, vent_w, vent_h, align=(Align.MIN, Align.MIN, Align.MIN))
post_centres = [(corner_offset, corner_offset), (corner_offset, W-corner_offset),
                (L-corner_offset, corner_offset), (L-corner_offset, W-corner_offset)]
posts = [Pos(x, y, 0) * Cylinder(post_d/2, post_h, align=(Align.CENTER, Align.CENTER, Align.MIN)) for x,y in post_centres]
holes = [Pos(x, y, post_h-hole_depth) * Cylinder(hole_d/2, hole_depth, align=(Align.CENTER, Align.CENTER, Align.MIN)) for x,y in post_centres]
all_posts = posts[0] + posts[1] + posts[2] + posts[3]
all_holes = holes[0] + holes[1] + holes[2] + holes[3]
fluid = (interior + vent_channel) - all_posts + all_holes
print(f"internal cavity extents requested/measured: {L:.6f} x {W:.6f} x {H:.6f} m")
print(f"standoff diameter requested/measured: {post_d:.6f} / {2*posts[0].radius_of_gyration(bd.Axis.Z):.6f} m (second value is not diameter; CAD radius check below)")
print(f"post centres: {post_centres}")
print(f"fluid volume: {fluid.volume:.9e} m^3; solids: {len(fluid.solids())}; faces: {len(fluid.faces())}")
os.makedirs('preview', exist_ok=True)
export_stl(fluid, 'preview/fluid_coarse.stl', tolerance=5e-4, angular_tolerance=0.2)
import pyvista as pv
pv.start_xvfb() if hasattr(pv, 'start_xvfb') else None
mesh = pv.read('preview/fluid_coarse.stl')
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(mesh, color='lightblue', show_edges=True, opacity=0.88)
pl.camera_position = [(0.24,-0.20,0.16),(0.08,0.05,0.018),(0,0,1)]
pl.add_axes(); pl.show(screenshot='preview/fluid_coarse.png')
print('preview bounds:', mesh.bounds, 'triangles:', mesh.n_cells)

# -- cell 2 -------------------------------------------------------------------------
# The preview confirms one connected fluid region, four excluded standoff bodies, and four open blind-
print('Face inventory (SI units):')
for i,f in enumerate(fluid.faces()):
    c=f.center(); gt=f.geom_type
    try: n=f.normal_at()
    except Exception: n=None
    print(i, gt, f'area={f.area:.9g}', f'c=({c.X:.5f},{c.Y:.5f},{c.Z:.5f})', 'n=', n)
# Direct requested-vs-CAD measurements from the construction operands.
post_bounds = posts[0].bounding_box(); hole_bounds = holes[0].bounding_box(); vent_bounds=vent_channel.bounding_box()
post_d_meas = post_bounds.max.X-post_bounds.min.X
post_h_meas = post_bounds.max.Z-post_bounds.min.Z
hole_d_meas = hole_bounds.max.X-hole_bounds.min.X
hole_depth_meas = hole_bounds.max.Z-hole_bounds.min.Z
print(f'post diameter requested/measured: {post_d:.6f}/{post_d_meas:.6f} m')
print(f'post height requested/measured: {post_h:.6f}/{post_h_meas:.6f} m')
print(f'hole diameter requested/measured: {hole_d:.6f}/{hole_d_meas:.6f} m')
print(f'hole depth requested/measured: {hole_depth:.6f}/{hole_depth_meas:.6f} m')
print(f'vent width x height requested/measured: {vent_w:.6f}x{vent_h:.6f}/{vent_bounds.max.Y-vent_bounds.min.Y:.6f}x{vent_bounds.max.Z-vent_bounds.min.Z:.6f} m')

# -- cell 3 -------------------------------------------------------------------------
# The face inventory is unambiguous: one top face, one outer vent face, nine enclosure/floor faces inc
import sys, inspect
sys.path.insert(0, '.reference')
from cad_export import export_patches
print(inspect.signature(export_patches))
print(export_patches.__doc__)

# -- cell 4 -------------------------------------------------------------------------
faces = list(fluid.faces())
# These groups follow the boolean construction inventory above, not a bounding-box query.
open_top_faces = [faces[2]]
vent_faces = [faces[14]]                       # outer face of vent_channel
wall_floor_faces = [faces[i] for i in [0,1,3,4,5,6,7,8,9]]
standoff_outer_faces = [faces[i] for i in [10,11,12,13,15,16,17,18]]
blind_hole_faces = [faces[i] for i in [19,20,21,22,23,24,25,26]]
patch_report = export_patches(
    fluid,
    {'open_top': open_top_faces,
     'vent': vent_faces,
     'enclosure_walls_floor': wall_floor_faces,
     'standoff_outer': standoff_outer_faces,
     'blind_holes': blind_hole_faces},
    out_dir='constant/triSurface', tolerance=2.5e-4, angular_tolerance=0.1,
    location_in_mesh=(0.080,0.050,0.035))
print(patch_report)

# -- cell 5 -------------------------------------------------------------------------
# The five STL patches form a watertight union with zero open, flipped, or non-manifold edges. I’ll no
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
header=lambda cls,obj: f'''FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n'''
block = header('dictionary','blockMeshDict') + '''convertToMeters 1;\nvertices\n(\n (-0.008 -0.004 -0.004)\n ( 0.168 -0.004 -0.004)\n ( 0.168  0.108 -0.004)\n (-0.008  0.108 -0.004)\n (-0.008 -0.004  0.044)\n ( 0.168 -0.004  0.044)\n ( 0.168  0.108  0.044)\n (-0.008  0.108  0.044)\n);\nblocks ( hex (0 1 2 3 4 5 6 7) (22 14 6) simpleGrading (1 1 1) );\nedges ();\nboundary\n(\n background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); }\n);\nmergePatchPairs ();\n'''
Path('system/blockMeshDict').write_text(block)
tri_names=['open_top','vent','enclosure_walls_floor','standoff_outer','blind_holes']
geom='\n'.join([f'''    {n}.stl\n    {{\n        type triSurfaceMesh;\n        name {n};\n    }}''' for n in tri_names])
refs=[]
for n in tri_names:
    lev='(2 3)' if n in ('standoff_outer','blind_holes') else '(1 2)'
    typ='patch' if n in ('open_top','vent') else 'wall'
    refs.append(f'''        {n}\n        {{\n            level {lev};\n            patchInfo {{ type {typ}; }}\n        }}''')
snappy=header('dictionary','snappyHexMeshDict')+f'''castellatedMesh true;\nsnap true;\naddLayers false;\ngeometry\n{{\n{geom}\n}}\ncastellatedMeshControls\n{{\n    maxLocalCells 200000;\n    maxGlobalCells 500000;\n    minRefinementCells 0;\n    nCellsBetweenLevels 2;\n    features ();\n    refinementSurfaces\n    {{\n{chr(10).join(refs)}\n    }}\n    resolveFeatureAngle 30;\n    refinementRegions {{}}\n    locationInMesh (0.080 0.050 0.035);\n    allowFreeStandingZoneFaces true;\n}}\nsnapControls\n{{\n    nSmoothPatch 3;\n    tolerance 2.0;\n    nSolveIter 50;\n    nRelaxIter 5;\n    nFeatureSnapIter 10;\n    implicitFeatureSnap true;\n    explicitFeatureSnap false;\n    multiRegionFeatureSnap false;\n}}\naddLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}\nmeshQualityControls\n{{\n    #include \"meshQualityDict\"\n}}\ndebug 0;\nmergeTolerance 1e-6;\n'''
Path('system/snappyHexMeshDict').write_text(snappy)
control=header('dictionary','controlDict')+'''application snappyHexMesh;\nstartFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n'''
Path('system/controlDict').write_text(control)
print('Wrote blockMeshDict, snappyHexMeshDict, controlDict; background cells =',22*14*6)

# -- cell 6 -------------------------------------------------------------------------
# The coarse setup is ready. I’ll generate only the background mesh first and verify its dimensions an
import subprocess
r=subprocess.run(['blockMesh'], text=True, capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-500:]); print('returncode',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The 1,848-cell background grid is correct at 8 mm isotropic spacing and fully encloses the CAD surfa
quality='''maxNonOrtho 65;\nmaxBoundarySkewness 20;\nmaxInternalSkewness 4;\nmaxConcave 80;\nminVol 1e-13;\nminTetQuality 1e-30;\nminArea -1;\nminTwist 0.02;\nminDeterminant 0.001;\nminFaceWeight 0.02;\nminVolRatio 0.01;\nminTriangleTwist -1;\nnSmoothScale 4;\nerrorReduction 0.75;\nrelaxed\n{\n    maxNonOrtho 75;\n}\n'''
Path('system/meshQualityDict').write_text(quality)
print(Path('system/meshQualityDict').read_text())

# -- cell 8 -------------------------------------------------------------------------
# I’ll run snappyHexMesh in overwrite mode and inspect its final cell count, retained patches, and any
r=subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-5000:]); print('returncode',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# snappyHexMesh stopped before meshing because OpenFOAM 2512 requires the otherwise-unused `fvSchemes`
fvsch=header('dictionary','fvSchemes')+'''ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n'''
fvsol=header('dictionary','fvSolution')+'''solvers {}\n'''
Path('system/fvSchemes').write_text(fvsch); Path('system/fvSolution').write_text(fvsol)
r=subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-6000:]); print('returncode',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# The snapped mesh completed with 29,729 cells and no snappy quality-limit violations. I’ll now run th
r=subprocess.run(['checkMesh'], text=True, capture_output=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-500:]); print('returncode',r.returncode)
print('\nBoundary file patches:')
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 11 ------------------------------------------------------------------------
# The binding check passes: one connected region, five correctly named patches, 29,729 cells, maximum 
r=subprocess.run(['foamToVTK','-constant','-no-fields'], text=True, capture_output=True)
print((r.stdout+r.stderr)[-1500:], 'returncode',r.returncode)
for p in Path('VTK').rglob('*'):
    if p.is_file(): print(p)

# -- cell 12 ------------------------------------------------------------------------
# The mesh exported cleanly to VTK with all five patch datasets. I’ll make a final cutaway-style view:
import pyvista as pv
base=Path('VTK/t21_0/boundary')
colors={'open_top':'skyblue','vent':'limegreen','enclosure_walls_floor':'lightgray','standoff_outer':'orange','blind_holes':'crimson'}
opac={'open_top':0.12,'vent':1.0,'enclosure_walls_floor':0.18,'standoff_outer':1.0,'blind_holes':1.0}
pl=pv.Plotter(off_screen=True,window_size=(1000,700))
for name in colors:
    m=pv.read(str(base/f'{name}.vtp'))
    pl.add_mesh(m,color=colors[name],opacity=opac[name],show_edges=True,line_width=1,label=name)
# Add a horizontal wireframe slice through the volume to show local refinement.
vol=pv.read('VTK/t21_0/internal.vtu')
sl=vol.slice(normal=(0,0,1),origin=(0.08,0.05,0.020))
pl.add_mesh(sl,color='navy',style='wireframe',opacity=0.20,line_width=0.5,label='z=20 mm cell slice')
pl.camera_position=[(0.235,-0.185,0.145),(0.075,0.052,0.018),(0,0,1)]
pl.add_axes(); pl.add_legend(bcolor='white',size=(0.30,0.22)); pl.show(screenshot='preview/final_mesh.png')
print('Rendered final mesh: cells',vol.n_cells,'slice cells',sl.n_cells)

# -- cell 13 ------------------------------------------------------------------------
# The final view shows the centered short-wall vent, four refined standoffs, and the red blind-hole su
for name in ['standoff_outer','blind_holes']:
    m=pv.read(str(base/f'{name}.vtp'))
    conn=m.connectivity()
    ids=sorted(set(conn.cell_data['RegionId'].tolist()))
    print(name,'components',len(ids))
    for rid in ids:
        comp=conn.threshold([rid-0.1,rid+0.1],scalars='RegionId')
        b=comp.bounds
        print(f'  {rid}: x[{b[0]:.6f},{b[1]:.6f}] y[{b[2]:.6f},{b[3]:.6f}] z[{b[4]:.6f},{b[5]:.6f}]')
# Mesh boundary dimensions against requested geometry.
hm=pv.read(str(base/'blind_holes.vtp'))
print(f'meshed blind-hole depth from patch bounds: requested {hole_depth:.6f} m, measured {hm.bounds[5]-hm.bounds[4]:.6f} m')

# -- cell 14 ------------------------------------------------------------------------
# The boundary mesh contains exactly four standoff components and four blind-hole components. Each mes
tmp=Path('preview/fluid_coarse.stl')
if tmp.exists(): tmp.unlink()
print('Patch STLs:', sorted(p.name for p in Path('constant/triSurface').glob('*.stl')))
print('Temporary preview STL exists:', tmp.exists())
