"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll verify the case remains untouched, then formally refuse rather than inventing an engine scale a
from pathlib import Path
case = Path.cwd()
contents = sorted(p.name for p in case.iterdir() if p.name != '.reference')
print(f"Case: {case}")
print(f"User-created contents: {contents}")
print("No geometry created: the request specifies no dimensions or length scale.")
