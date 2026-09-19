"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’m confirming that the case remains untouched because the request gives no physical dimension or le
from pathlib import Path
case_dir = Path.cwd()
entries = sorted(p.name for p in case_dir.iterdir() if p.name != "build.py")
print(f"Case directory: {case_dir}")
print(f"Existing case entries: {entries}")
print("Refusal: no dimensions or physical length scale were specified; no geometry or mesh was built.")
