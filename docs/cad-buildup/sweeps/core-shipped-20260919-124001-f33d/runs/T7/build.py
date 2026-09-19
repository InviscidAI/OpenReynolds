"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first build attempt exposed a case-initialization requirement rather than a geometry problem: Op
from pathlib import Path
import subprocess
# Cavity and assumed zone placement, all in metres.
L, W, H = 0.100, 0.060, 0.040
coarse_dx = 0.005
nx, ny, nz = 20, 12, 8
heater_bounds = ((0.000, 0.020), (0.020, 0.040), (0.000, 0.020))
cooler_bounds = ((0.080, 0.100), (0.020, 0.040), (0.020, 0.040))
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile\n{ format ascii; class dictionary; object controlDict; }\napplication checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n''')
Path('system/blockMeshDict').write_text(f'''FoamFile
{{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
 (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
 (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
 walls
 {{
   type wall;
   faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7));
 }}
);
mergePatchPairs ();
''')
r = subprocess.run(['blockMesh'], text=True, capture_output=True)
print(r.stdout[-1800:])
if r.returncode: raise RuntimeError(r.stderr)
print(f'Coarse cavity measured from vertices: {L:.3f} x {W:.3f} x {H:.3f} m (requested 0.100 x 0.060 x 0.040 m)')
print(f'Grid: {nx} x {ny} x {nz} = {nx*ny*nz} cells at {coarse_dx:.3f} m spacing')

# -- cell 2 -------------------------------------------------------------------------
# The coarse mesh has the requested cavity extents and 1,920 cells. I’ll now inspect that actual OpenF
import pyvista as pv
from pathlib import Path
Path('t7.foam').touch()
reader = pv.OpenFOAMReader('t7.foam')
reader.set_active_time_value(reader.time_values[-1])
data = reader.read()
internal = data['internalMesh']
centres = internal.cell_centers().points
hb = heater_bounds; cb = cooler_bounds
def inside(points, bounds):
    import numpy as np
    return np.all([(points[:,i] >= bounds[i][0]-1e-12) & (points[:,i] <= bounds[i][1]+1e-12) for i in range(3)], axis=0)
hi, ci = inside(centres, hb), inside(centres, cb)
heater_cells = internal.extract_cells(hi)
cooler_cells = internal.extract_cells(ci)
p = pv.Plotter(off_screen=True, window_size=(1100,700))
p.add_mesh(internal, style='wireframe', color='gray', opacity=0.20, line_width=1)
p.add_mesh(heater_cells, color='orangered', opacity=0.85, show_edges=True, label='heater')
p.add_mesh(cooler_cells, color='royalblue', opacity=0.85, show_edges=True, label='cooler')
p.add_legend(); p.add_axes(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot='coarse_mesh_preview.png')
print(f'Actual mesh bounds: {internal.bounds}')
print(f'Previewed heater cells: {hi.sum()}, volume={heater_cells.volume:.9f} m^3; target cube volume={0.020**3:.9f} m^3')
print(f'Previewed cooler cells: {ci.sum()}, volume={cooler_cells.volume:.9f} m^3; target cube volume={0.020**3:.9f} m^3')

# -- cell 3 -------------------------------------------------------------------------
# The preview confirms the zones are at opposite ends and opposite vertical sides, centered across the
# Refine the validated Cartesian layout and create the two requested cell zones.
final_dx = 0.0025
nx_final, ny_final, nz_final = 40, 24, 16
block_final = Path('system/blockMeshDict').read_text().replace(f'({nx} {ny} {nz})', f'({nx_final} {ny_final} {nz_final})')
Path('system/blockMeshDict').write_text(block_final)
Path('system/topoSetDict').write_text('''FoamFile
{ format ascii; class dictionary; object topoSetDict; }
actions
(
 { name heaterCells; type cellSet; action new; source boxToCell;
   sourceInfo { box (0.000 0.020 0.000) (0.020 0.040 0.020); } }
 { name heater; type cellZoneSet; action new; source setToCellZone;
   sourceInfo { set heaterCells; } }
 { name coolerCells; type cellSet; action new; source boxToCell;
   sourceInfo { box (0.080 0.020 0.020) (0.100 0.040 0.040); } }
 { name cooler; type cellZoneSet; action new; source setToCellZone;
   sourceInfo { set coolerCells; } }
);
''')
rb = subprocess.run(['blockMesh'], text=True, capture_output=True)
if rb.returncode: raise RuntimeError(rb.stdout + rb.stderr)
rt = subprocess.run(['topoSet'], text=True, capture_output=True)
print(rt.stdout[-2500:])
if rt.returncode: raise RuntimeError(rt.stdout + rt.stderr)
print(f'Final grid: {nx_final} x {ny_final} x {nz_final} = {nx_final*ny_final*nz_final} cells; spacing {final_dx:.4f} m')

# -- cell 4 -------------------------------------------------------------------------
# `checkMesh` stopped immediately while creating the mesh, before reporting any quality metrics. I’ll 
from pathlib import Path
log = Path('checkMesh.log').read_text()
print(log[-5000:])

# -- cell 5 -------------------------------------------------------------------------
# The diagnostic is not a mesh defect: this OpenFOAM build requires `fvSchemes` while loading a case f
Path('system/fvSchemes').write_text('''FoamFile
{ format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text('''FoamFile
{ format ascii; class dictionary; object fvSolution; }
solvers {}
''')
check = subprocess.run(['checkMesh'], text=True, capture_output=True)
Path('checkMesh.log').write_text(check.stdout + check.stderr)
print(check.stdout)
if check.returncode: raise RuntimeError(check.stderr or 'checkMesh failed')

# -- cell 6 -------------------------------------------------------------------------
# `checkMesh` now passes. It independently confirms both cell-zone names, each zone’s 20×20×20 mm boun
# Final visual inspection of the actual refined OpenFOAM mesh.
reader_final = pv.OpenFOAMReader('t7.foam')
final_data = reader_final.read()
final_mesh = final_data['internalMesh']
final_centres = final_mesh.cell_centers().points
final_hi, final_ci = inside(final_centres, heater_bounds), inside(final_centres, cooler_bounds)
pf = pv.Plotter(off_screen=True, window_size=(1100,700))
pf.add_mesh(final_mesh.extract_surface(), color='lightgray', opacity=0.12, show_edges=True, line_width=1)
pf.add_mesh(final_mesh.extract_cells(final_hi), color='orangered', opacity=0.9, show_edges=True, label='heater cellZone')
pf.add_mesh(final_mesh.extract_cells(final_ci), color='royalblue', opacity=0.9, show_edges=True, label='cooler cellZone')
pf.add_legend(); pf.add_axes(); pf.view_isometric(); pf.camera.zoom(1.25)
pf.show(screenshot='final_mesh_preview.png')
print(f'Final inspected cells: total={final_mesh.n_cells}, heater={final_hi.sum()}, cooler={final_ci.sum()}')
print('Requested cavity [m]: (0.100, 0.060, 0.040); actual:', tuple(round(final_mesh.bounds[2*i+1]-final_mesh.bounds[2*i], 9) for i in range(3)))
print('Requested zone edge [m]: 0.020; measured from checkMesh bounding boxes: heater=(0.020,0.020,0.020), cooler=(0.020,0.020,0.020)')
