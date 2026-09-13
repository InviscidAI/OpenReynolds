"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by looking at the raw STEP file's header to see what unit it declares.
import subprocess
p="/home/qiuzi/.openreynolds-buildup/work/T6-20260913-140447-9a29/geometry/customer_part.step"
print(subprocess.run(["head","-60",p],capture_output=True,text=True).stdout)
print(subprocess.run(["grep","-c","","%s"%p],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
print(subprocess.run(["grep","-n","UNIT\\|CONVERSION\\|SI_UNIT\\|MANIFOLD_SOLID_BREP\\|SHAPE_REPRESENTATION",p],capture_output=True,text=True).stdout)

# -- cell 3 -------------------------------------------------------------------------
print("".join(open(p).readlines()[495:512]))

# -- cell 4 -------------------------------------------------------------------------
# I need to send that through the tool.
from build123d import *

STEP_PATH = "/home/qiuzi/.openreynolds-buildup/work/T6-20260913-140447-9a29/geometry/customer_part.step"

s = import_step(STEP_PATH)
print("solids:", len(s.solids()), " faces:", len(s.faces()), " valid:", s.is_valid)
bb = s.bounding_box()
print("bbox min", bb.min, "max", bb.max, "size", bb.size)
print("volume", s.volume, "area", s.area)
for i, f in enumerate(s.faces()):
    print(i, f.geom_type, round(f.area, 1), f.center())

import re
txt = open(STEP_PATH).read()
for m in re.findall(r"#\d+ = \([^;]*UNIT[^;]*\);", txt, re.S):
    print(" ".join(m.split()))
