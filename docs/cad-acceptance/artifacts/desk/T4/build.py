"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Let me start by setting up parameters and building the channel footprint as a 2D sketch (union of re
import build123d as bd
from build123d import *
import math

# ---- Parameters (mm, as given in the request) ----
BLOCK_L_MM = 60.0   # X
BLOCK_W_MM = 40.0   # Y
BLOCK_H_MM = 8.0    # Z

CH_W_MM = 4.0       # channel width
CH_D_MM = 5.0       # channel depth (vertical extent of duct cross-section)
METAL_BETWEEN_MM = 3.0   # metal between passes (Y direction wall)
N_PASSES = 3

# Wall thickness left at the U-turn ends (not specified in the request;
# chosen equal to METAL_BETWEEN_MM for a consistent wall thickness). Flagged in closing notes.
TURN_END_WALL_MM = 3.0

# ---- Y layout of the 3 passes, centered in the 40 mm width ----
total_span = N_PASSES * CH_W_MM + (N_PASSES - 1) * METAL_BETWEEN_MM
margin = (BLOCK_W_MM - total_span) / 2.0
pass_y0 = []
y = margin
for i in range(N_PASSES):
    pass_y0.append(y)
    y += CH_W_MM + METAL_BETWEEN_MM
# pass i occupies y in [pass_y0[i], pass_y0[i]+CH_W_MM]
print("pass y-ranges (mm):", [(p, p+CH_W_MM) for p in pass_y0])
print("margin each side (mm):", margin)

# ---- X layout ----
x_right_turn = BLOCK_L_MM - TURN_END_WALL_MM   # right end of pass1/pass2 (turn1 region)
x_left_turn  = TURN_END_WALL_MM                # left end of pass2/pass3 (turn2 region)

print("x_right_turn:", x_right_turn, "x_left_turn:", x_left_turn)

# pass1: inlet at x=0, right end merges into turn1
p1_y0, p1_y1 = pass_y0[0], pass_y0[0] + CH_W_MM
pass1 = Rectangle(x_right_turn, CH_W_MM).moved(Location((x_right_turn/2, (p1_y0+p1_y1)/2, 0)))

# pass2: spans between turn2 and turn1 (both ends internal)
p2_y0, p2_y1 = pass_y0[1], pass_y0[1] + CH_W_MM
pass2_len = x_right_turn - x_left_turn
pass2 = Rectangle(pass2_len, CH_W_MM).moved(Location(((x_left_turn+x_right_turn)/2, (p2_y0+p2_y1)/2, 0)))

# pass3: outlet at x=60, left end merges into turn2
p3_y0, p3_y1 = pass_y0[2], pass_y0[2] + CH_W_MM
pass3_len = BLOCK_L_MM - x_left_turn
pass3 = Rectangle(pass3_len, CH_W_MM).moved(Location(((x_left_turn+BLOCK_L_MM)/2, (p3_y0+p3_y1)/2, 0)))

# turn1: near x = x_right_turn, bridges pass1 (y0..) to pass2 (..y1)
turn1 = Rectangle(CH_W_MM, p2_y1 - p1_y0).moved(
    Location((x_right_turn - CH_W_MM/2, (p1_y0 + p2_y1)/2, 0)))

# turn2: near x = x_left_turn, bridges pass2 to pass3
turn2 = Rectangle(CH_W_MM, p3_y1 - p2_y0).moved(
    Location((x_left_turn + CH_W_MM/2, (p2_y0 + p3_y1)/2, 0)))

footprint = pass1 + pass2 + pass3 + turn1 + turn2
print("footprint area (mm^2):", footprint.area)
print("footprint is single face/region:", len(footprint.faces()))

# -- cell 2 -------------------------------------------------------------------------
# You're right — the previous cell errored, so it wasn't accepted and `block_mm`/`channel_mm` aren't i
# Extrude footprint into the fluid (channel) solid, centered vertically in block thickness
Z_CENTER_MM = BLOCK_H_MM / 2.0
z0 = Z_CENTER_MM - CH_D_MM/2.0
z1 = Z_CENTER_MM + CH_D_MM/2.0
print("channel z-range (mm):", z0, z1, "block wall top/bottom (mm):", BLOCK_H_MM - z1, z0)

channel_mm = extrude(footprint, amount=CH_D_MM).moved(Location((0,0,z0)))

# Block (context only, not meshed) - aluminium solid, for the picture
block_mm = Box(BLOCK_L_MM, BLOCK_W_MM, BLOCK_H_MM).moved(
    Location((BLOCK_L_MM/2, BLOCK_W_MM/2, BLOCK_H_MM/2)))

print("channel volume (mm^3):", channel_mm.volume, " expected ~", footprint.area*CH_D_MM)
print("channel bbox:", channel_mm.bounding_box())
print("block bbox:", block_mm.bounding_box())

# sanity: channel must be fully inside block
inside = channel_mm & block_mm
inside_vol = sum(s.volume for s in inside.solids())
print("channel volume vs intersect-with-block volume (should match if fully enclosed):",
      channel_mm.volume, inside_vol)

# -- cell 3 -------------------------------------------------------------------------
# Good — the channel is fully enclosed within the block. Now let's look at the shape before going furt
import pyvista as pv
pv.set_jupyter_backend(None)
plotter = pv.Plotter(off_screen=True, window_size=(1000, 700))

def bd_to_pv(shape, color, opacity=1.0):
    verts, tris = shape.tessellate(tolerance=0.1)
    pts = np.array([(v.X, v.Y, v.Z) for v in verts])
    faces = np.hstack([[3, *t] for t in tris])
    mesh = pv.PolyData(pts, faces)
    plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=False)

import numpy as np
bd_to_pv(block_mm, "lightgray", opacity=0.25)
bd_to_pv(channel_mm, "dodgerblue", opacity=1.0)

plotter.camera_position = "iso"
plotter.camera.azimuth = 30
plotter.camera.elevation = 20
plotter.reset_camera()
plotter.screenshot("look_channel_in_block.png")
print("saved")

# -- cell 4 -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread("look_channel_in_block.png")
plt.figure(figsize=(10,7))
plt.imshow(img)
plt.axis("off")
plt.show()

# -- cell 5 -------------------------------------------------------------------------
# ShapeList doesn't have `.filter`; I'll use Python's `filter()` builtin / list comprehension instead,
from build123d import Axis, ShapeList

faces = channel_mm.faces()
faces_x = faces.filter_by(Axis.X)  # faces whose normal is along X

inlet_faces = ShapeList([f for f in faces_x if abs(f.center().X - 0.0) < 1e-3])
outlet_faces = ShapeList([f for f in faces_x if abs(f.center().X - BLOCK_L_MM) < 1e-3])
wall_faces = ShapeList([f for f in faces if f not in inlet_faces and f not in outlet_faces])

print("num inlet faces:", len(inlet_faces))
print("num outlet faces:", len(outlet_faces))
print("num wall faces:", len(wall_faces))
print("total faces:", len(faces))

for name, fl in [("inlet", inlet_faces), ("outlet", outlet_faces)]:
    for f in fl:
        print(name, "center:", f.center(), "area(mm^2):", f.area, "normal:", f.normal_at(f.center()))

print("expected inlet/outlet area = CH_W*CH_D (mm^2):", CH_W_MM*CH_D_MM)
print("sum of wall face areas (mm^2):", sum(f.area for f in wall_faces))

# -- cell 6 -------------------------------------------------------------------------
# Inlet and outlet faces measure exactly 4×5 = 20 mm², centered at the correct pass locations (y=13 fo
import subprocess
result = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/cad_convert.py", "--help"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# -- cell 7 -------------------------------------------------------------------------
# The CLI takes a STEP/IGES file path, not in-memory faces. The mention of `export_patches()` suggests
import sys
sys.path.insert(0, "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/")
import cad_convert
help(cad_convert.export_patches)

# -- cell 8 -------------------------------------------------------------------------
import inspect
print(inspect.signature(channel_mm.scale))
print(channel_mm.scale.__doc__)

# -- cell 9 -------------------------------------------------------------------------
from build123d import export_step
from pathlib import Path

MM_TO_M = 0.001
CASE_DIR = Path("/work/accept-t4")

channel_m = channel_mm.scale(MM_TO_M, about=(0, 0, 0))
print("channel_m bbox (m):", channel_m.bounding_box())
print("channel_m volume (m^3):", channel_m.volume, " expected:", channel_mm.volume * MM_TO_M**3)

step_path = CASE_DIR / "channel_fluid.step"
export_step(channel_m, str(step_path))
print("wrote", step_path, step_path.exists(), step_path.stat().st_size)

# -- cell 10 ------------------------------------------------------------------------
# This is the classic STEP-units gotcha: `export_step` writes the file with a millimetre header regard
import subprocess
result = subprocess.run(
    ["grep", "-n", "-i", "-A", "8", "unit", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/prep/README.md"],
    capture_output=True, text=True
)
print(result.stdout[:4000])

# -- cell 11 ------------------------------------------------------------------------
# Found it — I need `export_step(shape, path, unit=Unit.M)` to declare the file truthfully as metres o
from build123d import export_step, Unit
from pathlib import Path

CASE_DIR = Path("/work/accept-t4")
step_path = CASE_DIR / "channel_fluid.step"

export_step(channel_m, str(step_path), unit=Unit.M)
print("wrote", step_path, step_path.stat().st_size)

import cad_convert
from build123d import CenterOf

BLOCK_L_M = 0.060
faces_m = channel_m.faces()
faces_x_m = faces_m.filter_by(Axis.X)
inlet_faces_m = [f for f in faces_x_m if abs(f.center(CenterOf.MASS).X - 0.0) < 1e-6]
outlet_faces_m = [f for f in faces_x_m if abs(f.center(CenterOf.MASS).X - BLOCK_L_M) < 1e-6]

inlet_centroids = [tuple(f.center(CenterOf.MASS)) for f in inlet_faces_m]
outlet_centroids = [tuple(f.center(CenterOf.MASS)) for f in outlet_faces_m]

groups = {
    "inlet": {"role": "inlet", "faces": inlet_centroids},
    "outlet": {"role": "outlet", "faces": outlet_centroids},
    "wall": {"role": "wall", "faces": None},
}

CLMAX_M = 0.0015  # 1.5 mm facets, coarse first pass

out_dir = CASE_DIR / "constant" / "triSurface"
out_dir.mkdir(parents=True, exist_ok=True)

report = cad_convert.export_patches(
    step_path, groups, out_dir, clmax=CLMAX_M, unit="m", binary=False
)
print(report)

# -- cell 12 ------------------------------------------------------------------------
# Patch export succeeded: 14/14 faces claimed, inlet/outlet areas exactly 20 mm² each (2e-5 m²), match
result = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/cad_audit.py",
     str(CASE_DIR / "constant" / "triSurface"), "--json"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# -- cell 13 ------------------------------------------------------------------------
# The audit is fully clean: watertight union, manifold, consistent normals, no degenerate/self-interse
result = subprocess.run(
    ["python3", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/domain_probe.py",
     str(CASE_DIR / "constant" / "triSurface"), "--suggest", "--json"],
    capture_output=True, text=True
)
print(result.stdout[:4000])
print(result.stderr[:2000])

# -- cell 14 ------------------------------------------------------------------------
# Good — the suggested `locationInMesh` point is confirmed inside, with min channel width ~4.0 mm matc
result = subprocess.run(
    ["cat", "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/templates/snappy/README.md"],
    capture_output=True, text=True
)
print(result.stdout[:6000])

# -- cell 15 ------------------------------------------------------------------------
# Now let's copy the templates into the case, run `patch_entries.py` to fill in the dictionaries, and 
import subprocess, shutil
from pathlib import Path

CASE_DIR = Path("/work/accept-t4")
TOOLBOX = Path("/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox")

(CASE_DIR / "system").mkdir(parents=True, exist_ok=True)
for f in (TOOLBOX / "templates" / "snappy").glob("*Dict"):
    shutil.copy(f, CASE_DIR / "system" / f.name)
    print("copied", f.name)

result = subprocess.run(
    ["python3", str(TOOLBOX / "patch_entries.py"), str(CASE_DIR),
     "--insert", "system/snappyHexMeshDict",
     "--insert", "system/surfaceFeatureExtractDict"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# -- cell 16 ------------------------------------------------------------------------
# Now let's fill the two `CHANGE_ME` values: the `blockMeshDict` background box and the `locationInMes
print((CASE_DIR / "system" / "blockMeshDict").read_text())
print("=" * 60)
print((CASE_DIR / "system" / "snappyHexMeshDict").read_text()[:3000])

# -- cell 17 ------------------------------------------------------------------------
print((CASE_DIR / "system" / "snappyHexMeshDict").read_text()[3000:7000])

# -- cell 18 ------------------------------------------------------------------------
# Now let's set the two `CHANGE_ME` values and adjust the background cell size and surface refinement 
import re

CASE_DIR = Path("/work/accept-t4")

# Domain bounds from the audit: x 0..0.06, y 0.011..0.029, z 0.0015..0.0065 (metres)
DOM_X = (0.0, 0.060)
DOM_Y = (0.011, 0.029)
DOM_Z = (0.0015, 0.0065)

DX = 0.002          # background cell size (m); channel is 4x5mm so this gives ~2 cells across at level 0
PAD = 2 * DX         # padding beyond the surface, in background cells

box_min = (DOM_X[0]-PAD, DOM_Y[0]-PAD, DOM_Z[0]-PAD)
box_max = (DOM_X[1]+PAD, DOM_Y[1]+PAD, DOM_Z[1]+PAD)
print("boxMin:", box_min, "boxMax:", box_max)

LOC_IN_MESH = (0.00536089, 0.0266661, 0.00413173)  # from domain_probe.py --suggest, confirmed inside

bmd_path = CASE_DIR / "system" / "blockMeshDict"
bmd = bmd_path.read_text()
bmd = bmd.replace(
    "boxMin  (CHANGE_ME CHANGE_ME CHANGE_ME);",
    f"boxMin  ({box_min[0]:.6f} {box_min[1]:.6f} {box_min[2]:.6f});"
)
bmd = bmd.replace(
    "boxMax  (CHANGE_ME CHANGE_ME CHANGE_ME);",
    f"boxMax  ({box_max[0]:.6f} {box_max[1]:.6f} {box_max[2]:.6f});"
)
bmd = bmd.replace("dx      0.005;", f"dx      {DX};")
bmd_path.write_text(bmd)

shd_path = CASE_DIR / "system" / "snappyHexMeshDict"
shd = shd_path.read_text()
shd = shd.replace(
    "locationInMesh (CHANGE_ME CHANGE_ME CHANGE_ME);",
    f"locationInMesh ({LOC_IN_MESH[0]:.8f} {LOC_IN_MESH[1]:.8f} {LOC_IN_MESH[2]:.8f});"
)
# Coarse-first surface refinement: level (1 1) at the wall surfaces -> 1 mm cells there,
# enough to get a few cells across the 4mm channel width for a first look.
shd = shd.replace("level (0 0);\n            patchInfo { type patch; }",
                   "level (1 1);\n            patchInfo { type patch; }")
shd = shd.replace("level (0 0);\n            patchInfo { type wall; }",
                   "level (1 1);\n            patchInfo { type wall; }")
shd_path.write_text(shd)

print("--- blockMeshDict box+dx lines ---")
for line in bmd.splitlines():
    if "boxMin" in line or "boxMax" in line or line.strip().startswith("dx "):
        print(line)
print("--- snappyHexMeshDict locationInMesh + levels ---")
for line in shd.splitlines():
    if "locationInMesh" in line or "level (" in line:
        print(line)

# -- cell 19 ------------------------------------------------------------------------
# Now let's check with `preflight.py` that nothing obviously wrong remains (no leftover `CHANGE_ME`, b
# minimal fvSchemes / fvSolution, required by snappyHexMesh though it solves nothing
fvSchemes = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes {}
gradSchemes {}
divSchemes {}
laplacianSchemes {}
interpolationSchemes {}
snGradSchemes {}
"""
fvSolution = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
"""
(CASE_DIR / "system" / "fvSchemes").write_text(fvSchemes)
(CASE_DIR / "system" / "fvSolution").write_text(fvSolution)

result = subprocess.run(
    ["python3", str(TOOLBOX / "preflight.py"), str(CASE_DIR), "--json"],
    capture_output=True, text=True
)
print(result.stdout[:5000])
print(result.stderr[:2000])
