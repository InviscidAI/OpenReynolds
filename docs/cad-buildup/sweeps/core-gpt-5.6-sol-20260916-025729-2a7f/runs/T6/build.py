"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The import succeeded, but the validity API is a Boolean property rather than a method. I’ll rerun th
from pathlib import Path
import build123d as bd
import pyvista as pv
step_path = Path('/home/qiuzi/.openreynolds-buildup/work/T6-20260916-033127-a3a6/geometry/customer_part.step')
header = step_path.read_text(errors='ignore')[:200000]
unit_context = [ln.strip() for ln in header.splitlines() if any(f'#{i} =' in ln for i in range(420,432))]
shape = bd.import_step(step_path)
solids = list(shape.solids()); bb = shape.bounding_box()
print('STEP records near unit assignment:\n' + '\n'.join(unit_context))
print(f'Imported type={type(shape).__name__}; solids={len(solids)}; shells={len(shape.shells())}; faces={len(shape.faces())}; valid={shape.is_valid}')
print(f'Bounds: min=({bb.min.X:.6g},{bb.min.Y:.6g},{bb.min.Z:.6g}), max=({bb.max.X:.6g},{bb.max.Y:.6g},{bb.max.Z:.6g}), size=({bb.size.X:.6g},{bb.size.Y:.6g},{bb.size.Z:.6g})')
print(f'Volume={shape.volume:.9g} coordinate_unit^3')
bd.export_stl(shape, 'customer_import_raw.stl', tolerance=0.15, angular_tolerance=0.15)
mesh = pv.read('customer_import_raw.stl')
pl = pv.Plotter(off_screen=True, window_size=(900,650)); pl.add_mesh(mesh, color='lightsteelblue', show_edges=True); pl.add_axes(); pl.view_isometric(); pl.show(screenshot='customer_import.png')
print('Rendered customer_import.png')

# -- cell 2 -------------------------------------------------------------------------
# The model is one valid solid, apparently a faceted transition duct. The unit record is split across 
lines = header.splitlines()
for i, ln in enumerate(lines):
    if ln.startswith('#425'):
        print('Complete STEP context:')
        print('\n'.join(lines[i:i+12]))
print('\nCAD face inventory:')
for i, f in enumerate(shape.faces(), 1):
    c = f.center(); n = f.normal_at(); fb = f.bounding_box()
    print(f'{i}: geom={f.geom_type}, area={f.area:.9g}, edges={len(f.edges())}, center=({c.X:.6g},{c.Y:.6g},{c.Z:.6g}), normal=({n.X:.5g},{n.Y:.5g},{n.Z:.5g}), size=({fb.size.X:.6g},{fb.size.Y:.6g},{fb.size.Z:.6g})')

# -- cell 3 -------------------------------------------------------------------------
# The STEP explicitly assigns `#426` as its length unit, but `#426` is only `LENGTH_UNIT() NAMED_UNIT(
from pathlib import Path
for diagnostic in ('customer_import_raw.stl', 'customer_import.png'):
    Path(diagnostic).unlink(missing_ok=True)
print('Removed temporary inspection artifacts; no geometry or mesh retained.')
