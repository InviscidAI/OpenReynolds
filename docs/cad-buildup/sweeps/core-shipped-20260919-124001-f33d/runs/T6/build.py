"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The import itself succeeded, but the validity API is a property rather than a method. I’ll repeat th
from pathlib import Path
import re
import build123d as bd
step_path = Path('/home/qiuzi/.openreynolds-buildup/work/T6-20260919-135229-afb8/geometry/customer_part.step')
step_text = step_path.read_text(errors='ignore')
unit_records = re.findall(r'#[0-9]+\s*=\s*[^;]*(?:LENGTH_UNIT|SI_UNIT|CONVERSION_BASED_UNIT)[^;]*;', step_text, flags=re.I)
print('STEP unit records:')
print('\n'.join(unit_records) or 'none found')
customer = bd.import_step(str(step_path))
print('Imported type:', type(customer).__name__)
print('Solids:', len(customer.solids()), 'Shells:', len(customer.shells()), 'Faces:', len(customer.faces()))
print('OCCT valid:', customer.is_valid)
print('Volume in declared CAD units^3:', customer.volume)
bb = customer.bounding_box()
print('Bounds in declared CAD units:', tuple(bb.min), tuple(bb.max))
print('Extent in declared CAD units:', bb.size.X, bb.size.Y, bb.size.Z)

# -- cell 2 -------------------------------------------------------------------------
# The length unit is defined indirectly, so I’ll inspect its referencing records before scaling. I’ll 
lines = step_text.splitlines()
for i, line in enumerate(lines):
    if '#426' in line or 'GLOBAL_UNIT_ASSIGNED_CONTEXT' in line or 'LENGTH_MEASURE_WITH_UNIT' in line:
        print('\n'.join(lines[max(0,i-3):min(len(lines),i+4)]))
print('\nFaces (index, geom type, area, center):')
for i, f in enumerate(customer.faces()):
    print(i, f.geom_type, round(f.area,6), tuple(round(v,6) for v in f.center()))
