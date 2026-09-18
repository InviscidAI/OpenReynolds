"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the air itself as one 0.100 × 0.060 × 0.040 m hexahedral block. For the coarse mesh I’ll 
from pathlib import Path

# Requested cavity dimensions, in metres.
L, W, H = 0.100, 0.060, 0.040
cube = 0.020
cell = 0.005
nx, ny, nz = int(L/cell), int(W/cell), int(H/cell)

# Assumption: 5 mm clearance from the nearby end and floor/ceiling;
# both cubes are centred in the cavity width.
heater_min, heater_max = (0.005, 0.020, 0.005), (0.025, 0.040, 0.025)
cooler_min, cooler_max = (0.075, 0.020, 0.015), (0.095, 0.040, 0.035)

Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
Path('0').mkdir(exist_ok=True)

block_mesh_dict = f'''FoamFile
{{
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
    (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 4 7 3) (1 2 6 5)
            (0 1 5 4) (3 7 6 2)
            (0 3 2 1) (4 5 6 7)
        );
    }}
);
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(block_mesh_dict)

def vec(v): return '(' + ' '.join(f'{x:.6f}' for x in v) + ')'
topo_set_dict = f'''FoamFile
{{
    format ascii;
    class dictionary;
    object topoSetDict;
}}
actions
(
    {{ name heaterSet; type cellSet; action new; source boxToCell;
       box {vec(heater_min)} {vec(heater_max)}; }}
    {{ name heater; type cellZoneSet; action new; source setToCellZone;
       set heaterSet; }}
    {{ name coolerSet; type cellSet; action new; source boxToCell;
       box {vec(cooler_min)} {vec(cooler_max)}; }}
    {{ name cooler; type cellZoneSet; action new; source setToCellZone;
       set coolerSet; }}
);
'''
Path('system/topoSetDict').write_text(topo_set_dict)
print(f'Cavity requested/built: {L:.3f} x {W:.3f} x {H:.3f} m')
print(f'Coarse mesh: {nx} x {ny} x {nz} = {nx*ny*nz} cells, nominal edge {cell:.3f} m')
print(f'Heater bounds: {heater_min} to {heater_max}')
print(f'Cooler bounds: {cooler_min} to {cooler_max}')

# -- cell 2 -------------------------------------------------------------------------
# `blockMesh` needs a minimal case `controlDict`; the geometry dictionary itself was not reached. I’ll
control_dict = '''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application     none;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
'''
Path('system/controlDict').write_text(control_dict)
print('Wrote minimal system/controlDict required by OpenFOAM mesh utilities.')

# -- cell 3 -------------------------------------------------------------------------
# The mesh run succeeded interactively but was not accepted into the reproducible script because `subp
import subprocess
for command in (["blockMesh"], ["topoSet"]):
    result = subprocess.run(command, text=True, capture_output=True)
    print(f"$ {' '.join(command)}  return={result.returncode}")
    print(result.stdout[-2600:])
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f"{' '.join(command)} failed")

# -- cell 4 -------------------------------------------------------------------------
# The coarse mesh contains 1,920 cubic cells and both zones contain 64 cells, as expected for 20 mm cu
import pyvista as pv
Path('t7.foam').touch()
reader = pv.OpenFOAMReader('t7.foam')
reader.set_active_time_value(0.0)
mesh_data = reader.read()
print(mesh_data)
print('Blocks:', list(mesh_data.keys()) if hasattr(mesh_data, 'keys') else 'single dataset')

# -- cell 5 -------------------------------------------------------------------------
# The reader confirms the actual mesh bounds are exactly 0–100 mm, 0–60 mm, and 0–40 mm. I’ll render a
import numpy as np
from IPython.display import display, Image
internal = mesh_data['internalMesh']
centres = internal.cell_centers().points

def cells_in_box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    return np.flatnonzero(np.all((centres >= lo-1e-12) & (centres <= hi+1e-12), axis=1))

heater_ids = cells_in_box(heater_min, heater_max)
cooler_ids = cells_in_box(cooler_min, cooler_max)
heater_mesh = internal.extract_cells(heater_ids)
cooler_mesh = internal.extract_cells(cooler_ids)

plotter = pv.Plotter(off_screen=True, window_size=(1000, 700))
plotter.set_background('white')
plotter.add_mesh(internal, style='wireframe', color='gray', opacity=0.16, line_width=1)
plotter.add_mesh(heater_mesh, color='orangered', show_edges=True, label='heater')
plotter.add_mesh(cooler_mesh, color='royalblue', show_edges=True, label='cooler')
plotter.add_legend(bcolor='white', face=None)
plotter.add_axes()
plotter.view_isometric()
plotter.camera.zoom(1.25)
plotter.show(screenshot='coarse_mesh.png', auto_close=True)
print(f'Actual rendered zone cells: heater={len(heater_ids)}, cooler={len(cooler_ids)}')
display(Image('coarse_mesh.png'))

# -- cell 6 -------------------------------------------------------------------------
# The coarse rendering confirms the zone placement, separation, and cell alignment. For the delivered 
# Final mesh-resolution assumption: uniform 2.5 mm cubic cells.
final_cell = 0.0025
final_nx, final_ny, final_nz = int(L/final_cell), int(W/final_cell), int(H/final_cell)
block_mesh_dict = f'''FoamFile
{{
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
    (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({final_nx} {final_ny} {final_nz}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 4 7 3) (1 2 6 5)
            (0 1 5 4) (3 7 6 2)
            (0 3 2 1) (4 5 6 7)
        );
    }}
);
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(block_mesh_dict)
print(f'Final mesh specification: {final_nx} x {final_ny} x {final_nz} = '
      f'{final_nx*final_ny*final_nz} cells at {final_cell*1000:g} mm')

# -- cell 7 -------------------------------------------------------------------------
# I’ll generate the final refined mesh and recreate the zones on it.
import subprocess
for command in (["blockMesh"], ["topoSet"]):
    result = subprocess.run(command, text=True, capture_output=True)
    print(f"$ {' '.join(command)}  return={result.returncode}")
    print(result.stdout[-1800:])
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f"{' '.join(command)} failed")

# -- cell 8 -------------------------------------------------------------------------
# The refined zones contain 512 cells each as intended. I’ll now measure the final mesh itself—cavity 
import pyvista as pv
final_reader = pv.OpenFOAMReader('t7.foam')
final_reader.set_active_time_value(0.0)
final_data = final_reader.read()
final_internal = final_data['internalMesh']
final_centres = final_internal.cell_centers().points

def measured_zone(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    ids = np.flatnonzero(np.all((final_centres >= lo-1e-12) & (final_centres <= hi+1e-12), axis=1))
    selected = final_internal.extract_cells(ids)
    volume = float(selected.compute_cell_sizes(length=False, area=False, volume=True)['Volume'].sum())
    bounds = np.array(selected.bounds).reshape(3, 2)
    return ids, selected, bounds, volume

heater_ids_f, heater_final, hb, hv = measured_zone(heater_min, heater_max)
cooler_ids_f, cooler_final, cb, cv = measured_zone(cooler_min, cooler_max)
cavity_bounds = np.array(final_internal.bounds).reshape(3, 2)
cavity_dims = cavity_bounds[:,1] - cavity_bounds[:,0]
print('Requested cavity dimensions (m):', np.array([L, W, H]))
print('Measured cavity dimensions  (m):', cavity_dims)
for name, bounds, volume, ids in [('heater', hb, hv, heater_ids_f), ('cooler', cb, cv, cooler_ids_f)]:
    print(f'{name}: cells={len(ids)}, measured bounds={bounds.tolist()}')
    print(f'  measured side lengths={bounds[:,1]-bounds[:,0]} m; volume={volume:.9g} m^3')
print('Requested cube side/volume:', cube, cube**3)

# -- cell 9 -------------------------------------------------------------------------
# `checkMesh` reached the mesh but requires the standard empty numerical dictionaries to instantiate t
Path('system/fvSchemes').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default none; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path('system/fvSolution').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
print('Wrote minimal mesh-check support dictionaries.')

# -- cell 10 ------------------------------------------------------------------------
# I’ll rerun bare `checkMesh`.
import subprocess
check = subprocess.run(['checkMesh'], text=True, capture_output=True)
print(check.stdout)
if check.returncode:
    print(check.stderr)
    raise RuntimeError('checkMesh failed')
