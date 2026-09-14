"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by looking at the file itself before deciding anything.
import subprocess
p="/home/qiuzi/.openreynolds-buildup/work/T6-20260914-111441-ab8a/geometry/customer_part.step"
print(subprocess.run(["head","-40",p],capture_output=True,text=True).stdout)
print(subprocess.run(["grep","-c","","-e","",p],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
print(subprocess.run(["grep","-n","-i","-e","UNIT","-e","MANIFOLD_SOLID","-e","CYLINDRICAL","-e","CONICAL",p],capture_output=True,text=True).stdout[-3000:])

# -- cell 3 -------------------------------------------------------------------------
print(subprocess.run(["sed","-n","495,512p",p],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# I need to send that through the tool.
import subprocess
import build123d as bd

step_path = "/home/qiuzi/.openreynolds-buildup/work/T6-20260914-111441-ab8a/geometry/customer_part.step"

shape = bd.import_step(step_path)
solid_list = shape.solids()
print("type:", type(shape).__name__)
print("solids:", len(solid_list))
print("valid:", shape.is_valid)
print("bbox (file units):", shape.bounding_box())
print("volume (file units^3):", [sl.volume for sl in solid_list])
print("faces:", len(shape.faces()))

print(subprocess.run(["sed", "-n", "500,510p", step_path],
                     capture_output=True, text=True).stdout)

# -- cell 5 -------------------------------------------------------------------------
import os
print(os.getcwd(), sorted(os.listdir(".")))
