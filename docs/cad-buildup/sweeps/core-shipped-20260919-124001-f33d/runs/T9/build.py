"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the finite oil region as the cylindrical space inside the ring bore (`r ≤ 23 mm`, `|z| ≤ 
import build123d as bd
from build123d import Cylinder, Pos
import math
# All dimensions are metres.
R_RING_IN, R_RING_OUT, H_RING = 0.023, 0.030, 0.010
R_SPIN_IN, R_SPIN_OUT, H_SPIN = 0.015, 0.022, 0.008
ring_bore = Pos(0, 0, -H_RING/2) * Cylinder(R_RING_IN, H_RING)
spinner_outer = Pos(0, 0, -H_SPIN/2) * Cylinder(R_SPIN_OUT, H_SPIN)
spinner_bore = Pos(0, 0, -H_SPIN/2) * Cylinder(R_SPIN_IN, H_SPIN)
spinner = spinner_outer - spinner_bore
fluid = ring_bore - spinner
expected_volume = math.pi*R_RING_IN**2*H_RING - math.pi*(R_SPIN_OUT**2-R_SPIN_IN**2)*H_SPIN
print(f'Fluid volume: {fluid.volume:.12e} m^3; expected {expected_volume:.12e} m^3')
print(f'Radial clearance: {(R_RING_IN-R_SPIN_OUT)*1e3:.3f} mm; requested 1.000 mm')
print(f'Axial gap each end: {(H_RING-H_SPIN)/2*1e3:.3f} mm; requested 1.000 mm')
print(f'Fluid solids: {len(fluid.solids())}; faces: {len(fluid.faces())}; bbox:', fluid.bounding_box().min, fluid.bounding_box().max)
bd.export_stl(fluid, 'coarse_fluid_preview.stl', tolerance=2.5e-4, angular_tolerance=0.15)

# -- cell 2 -------------------------------------------------------------------------
# The first measurement exposed a placement error: this build123d `Cylinder` is already centred axiall
# build123d cylinders are centred by default; rebuild without an axial translation.
ring_bore = Cylinder(R_RING_IN, H_RING)
spinner_outer = Cylinder(R_SPIN_OUT, H_SPIN)
spinner_bore = Cylinder(R_SPIN_IN, H_SPIN)
spinner = spinner_outer - spinner_bore
fluid = ring_bore - spinner
bb = fluid.bounding_box()
print(f'Corrected axial extent: {bb.min.Z*1e3:.3f} to {bb.max.Z*1e3:.3f} mm; requested -5.000 to 5.000 mm')
print(f'Connected solids: {len(fluid.solids())}; volume: {fluid.volume:.12e} m^3')
bd.export_stl(fluid, 'coarse_fluid_preview.stl', tolerance=2.5e-4, angular_tolerance=0.15)
import pyvista as pv
mesh_preview = pv.read('coarse_fluid_preview.stl')
clipped = mesh_preview.clip(normal=(1,0,0), origin=(0,0,0), invert=False)
p = pv.Plotter(off_screen=True, window_size=(850,650))
p.add_mesh(clipped, color='goldenrod', show_edges=True, edge_color='black')
p.add_axes(); p.view_isometric(); p.show(screenshot='coarse_fluid_preview.png')
print('Rendered coarse_fluid_preview.png')

# -- cell 3 -------------------------------------------------------------------------
# The corrected CAD is one solid oil region with the requested dimensions. I’ll now inspect its seven 
for i, face in enumerate(fluid.faces()):
    c = face.center()
    print(i, 'geom=', face.geom_type, f'area={face.area:.10e}', f'center=({c.X:.6g},{c.Y:.6g},{c.Z:.6g})')

# -- cell 4 -------------------------------------------------------------------------
# The areas identify the patches unambiguously: the `r=23 mm` cylinder is the outer-ring wetted wall; 
import os, sys
sys.path.insert(0, '.reference')
from cad_export import export_patches
faces = list(fluid.faces())
outer_ring_wetted = [faces[0]]
ring_top_end = [faces[1]]
ring_bottom_end = [faces[2]]
spinner_wetted = [faces[3], faces[4], faces[5], faces[6]]
os.makedirs('constant/triSurface', exist_ok=True)
export_patches(fluid, {
    'outer_ring_wetted': outer_ring_wetted,
    'spinner_wetted': spinner_wetted,
    'ring_top_end': ring_top_end,
    'ring_bottom_end': ring_bottom_end,
}, directory='constant/triSurface', tolerance=2.5e-4)

# -- cell 5 -------------------------------------------------------------------------
# The patch export is closed and consistently wound (1,508 triangles, zero open/non-manifold/flipped e
bd.export_step(fluid, 'fluid.step')
import gmsh
gmsh.initialize()
gmsh.model.add('oil')
vols = gmsh.model.occ.importShapes('fluid.step', highestDimOnly=False)
gmsh.model.occ.synchronize()
print('Imported volumes:', gmsh.model.getEntities(3))
for dim, tag in gmsh.model.getEntities(2):
    a = gmsh.model.occ.getMass(dim, tag)
    c = gmsh.model.occ.getCenterOfMass(dim, tag)
    print(tag, f'area={a:.10e}', f'COM=({c[0]:.6g},{c[1]:.6g},{c[2]:.6g})')
gmsh.finalize()

# -- cell 6 -------------------------------------------------------------------------
# The imported STEP preserves all seven CAD faces and their measured areas exactly. I’ll generate a de
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 1)
gmsh.model.add('oil')
gmsh.model.occ.importShapes('fluid.step', highestDimOnly=False)
gmsh.model.occ.synchronize()
# Surface tags are verified above from CAD area and centre of mass, not bounding-box location.
patch_tags = {
    'outer_ring_wetted': [1],
    'spinner_wetted': [4, 5, 6, 7],
    'ring_top_end': [2],
    'ring_bottom_end': [3],
}
for name, tags in patch_tags.items():
    pg = gmsh.model.addPhysicalGroup(2, tags)
    gmsh.model.setPhysicalName(2, pg, name)
oil_pg = gmsh.model.addPhysicalGroup(3, [1])
gmsh.model.setPhysicalName(3, oil_pg, 'oil')
for dim, tag in gmsh.model.getEntities(0):
    gmsh.model.mesh.setSize([(dim, tag)], 5.0e-4)
gmsh.option.setNumber('Mesh.MeshSizeMin', 3.5e-4)
gmsh.option.setNumber('Mesh.MeshSizeMax', 6.0e-4)
gmsh.option.setNumber('Mesh.Algorithm3D', 10)
gmsh.option.setNumber('Mesh.Optimize', 1)
gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
gmsh.model.mesh.generate(3)
gmsh.write('oil.msh')
node_count = len(gmsh.model.mesh.getNodes()[0])
tet_count = len(gmsh.model.mesh.getElementsByType(4)[0])
print(f'Gmsh coarse mesh: {node_count} nodes, {tet_count} linear tetrahedra')
gmsh.finalize()

# -- cell 7 -------------------------------------------------------------------------
# The coarse CAD-conformal mesh completed with about 354k exported tetrahedra. I’ll convert it to Open
import subprocess, pathlib, textwrap
pathlib.Path('system').mkdir(exist_ok=True)
pathlib.Path('constant').mkdir(exist_ok=True)
pathlib.Path('system/controlDict').write_text(textwrap.dedent('''\
FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application foamRun;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
'''))
res = subprocess.run(['gmshToFoam', 'oil.msh'], text=True, capture_output=True)
print(res.stdout[-3000:])
if res.returncode: print(res.stderr); raise RuntimeError('gmshToFoam failed')
print(pathlib.Path('constant/polyMesh/boundary').read_text())

# -- cell 8 -------------------------------------------------------------------------
# The utility additionally parses the standard scheme subdictionaries. I’ll provide a complete minimal
pathlib.Path('system/fvSchemes').write_text(textwrap.dedent('''\
FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
'''))
chk = subprocess.run(['checkMesh'], text=True, capture_output=True)
print(chk.stdout)
if chk.returncode != 0:
    print(chk.stderr)
    raise RuntimeError(f'checkMesh failed with code {chk.returncode}')

# -- cell 9 -------------------------------------------------------------------------
# The binding check passes: one connected region, correct `46 × 46 × 10 mm` bounds, volume within 0.01
pathlib.Path('t9.foam').touch()
reader = pv.OpenFOAMReader('t9.foam')
reader.set_active_time_value(reader.time_values[-1])
mesh_foam = reader.read()['internalMesh']
cut = mesh_foam.clip(normal=(1,0,0), origin=(0,0,0), invert=False)
p = pv.Plotter(off_screen=True, window_size=(900,700))
p.add_mesh(cut, color='lightsteelblue', show_edges=True, edge_color='navy', line_width=0.25)
p.add_axes(); p.view_isometric(); p.show(screenshot='openfoam_mesh_cutaway.png')
print('Rendered actual OpenFOAM mesh cutaway:', mesh_foam.n_cells, 'cells,', mesh_foam.n_points, 'points')

# -- cell 10 ------------------------------------------------------------------------
# The isometric cutaway confirms the converted mesh exists, but its viewing angle hides the narrow int
p = pv.Plotter(off_screen=True, window_size=(1000,550))
p.add_mesh(cut, color='lightsteelblue', show_edges=True, edge_color='navy', line_width=0.2)
p.view_yz(); p.camera.parallel_projection = True
p.show(screenshot='openfoam_mesh_cross_section.png')
print('Rendered openfoam_mesh_cross_section.png (view along the X axis)')
