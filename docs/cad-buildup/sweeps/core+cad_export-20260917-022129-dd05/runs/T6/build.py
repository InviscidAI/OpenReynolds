"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The import reveals one seven-face solid; the first probe only failed because this build123d version 
from pathlib import Path
import re
import build123d as bd
step_path = Path('/home/qiuzi/.openreynolds-buildup/work/T6-20260917-030305-5c06/geometry/customer_part.step')
step_text = step_path.read_text(errors='ignore')
unit_lines = [line.strip() for line in step_text.splitlines() if 'SI_UNIT' in line or 'CONVERSION_BASED_UNIT' in line or 'LENGTH_UNIT' in line]
print('STEP unit declarations:')
print('\n'.join(unit_lines[-20:]))
customer = bd.import_step(step_path)
print('imported type:', type(customer).__name__)
print('solid/shell/face counts:', len(customer.solids()), len(customer.shells()), len(customer.faces()))
print('OCCT valid:', customer.is_valid)
bb = customer.bounding_box()
print('bounds:', tuple(round(v,6) for v in (*bb.min.to_tuple(), *bb.max.to_tuple())))
print('size:', tuple(round(v,6) for v in bb.size.to_tuple()))
print('volume:', round(customer.volume,6))

# -- cell 2 -------------------------------------------------------------------------
# The declaration line is part of a complex STEP entity, so I’ll inspect its surrounding records to de
lines = step_text.splitlines()
for i,line in enumerate(lines):
    if '#425' in line or '#426' in line:
        print('\n'.join(lines[max(0,i-3):min(len(lines),i+8)]))
