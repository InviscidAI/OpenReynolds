"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess
for d in ["system","constant/polyMesh","constant/triSurface"]: os.makedirs(d, exist_ok=True)
head = "FoamFile{ version 2.0; format ascii; class dictionary; object %s; }\n"
open("system/controlDict","w").write(head%"controlDict"+
"application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
"deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head%"fvSchemes"+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head%"fvSolution"+"solvers{}\n")
print(sorted(os.listdir(".")))

# -- cell 2 -------------------------------------------------------------------------
# Now the blockMeshDict, in metres, with each of the six walls named explicitly on the faces as they a
L,W,H = 0.100, 0.060, 0.040          # request: 100 x 60 x 40 mm
CELL  = 0.004                         # coarse first: 4 mm cells (my choice)
nx,ny,nz = int(round(L/CELL)), int(round(W/CELL)), int(round(H/CELL))

verts = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),(0,0,H),(L,0,H),(L,W,H),(0,W,H)]
faces = {  # named where the face is defined
 "floor":   "(0 3 2 1)",
 "ceiling": "(4 5 6 7)",
 "xMin":    "(0 4 7 3)",
 "xMax":    "(1 2 6 5)",
 "yMin":    "(0 1 5 4)",
 "yMax":    "(3 7 6 2)",
}
bm = ["FoamFile{ version 2.0; format ascii; class dictionary; object blockMeshDict; }",
      "scale 1;", "vertices (", *[f"  ({x} {y} {z})" for x,y,z in verts], ");",
      f"blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );",
      "edges ();", "boundary ("]
for n,f in faces.items():
    bm += [f"  {n} {{ type wall; faces ( {f} ); }}"]
bm += [");","mergePatchPairs ();"]
open("system/blockMeshDict","w").write("\n".join(bm)+"\n")
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-900:])

# -- cell 3 -------------------------------------------------------------------------
# Zones via `topoSet`: heater 20 mm cube 4 mm above the floor at the x-min end, cooler the same 4 mm b
C = 0.020                 # 20 mm cube (given)
GAP = 0.004               # my choice: 4 mm clearance from floor / ceiling
X0_H, X0_C = 0.010, 0.070 # my choice: 10 mm inset from each end wall
Y0 = (W - C)/2.0
boxes = {"heater": (X0_H, Y0, GAP), "cooler": (X0_C, Y0, H-GAP-C)}
ts = ["FoamFile{ version 2.0; format ascii; class dictionary; object topoSetDict; }","actions ("]
for n,(x0,y0,z0) in boxes.items():
    ts += [f"  {{ name {n}Cells; type cellSet; action new; source boxToCell;"
           f" box ({x0} {y0} {z0}) ({x0+C} {y0+C} {z0+C}); }}",
           f"  {{ name {n}; type cellZoneSet; action new; source setToCellZone; set {n}Cells; }}"]
ts += [");"]
open("system/topoSetDict","w").write("\n".join(ts)+"\n")
r=subprocess.run(["topoSet"],capture_output=True,text=True)
print(r.returncode, "\n".join(l for l in r.stdout.splitlines() if "cells" in l.lower() or "zone" in l.lower())[-1200:])
print(boxes)

# -- cell 4 -------------------------------------------------------------------------
# 133/129 instead of 125: the x-faces of the boxes land exactly on cell centres. Shrink the *query* bo
eps = CELL/4
ts = ["FoamFile{ version 2.0; format ascii; class dictionary; object topoSetDict; }","actions ("]
for n,(x0,y0,z0) in boxes.items():
    a=(x0+eps,y0+eps,z0+eps); b=(x0+C-eps,y0+C-eps,z0+C-eps)
    ts += [f"  {{ name {n}Cells; type cellSet; action new; source boxToCell;"
           f" box ({a[0]} {a[1]} {a[2]}) ({b[0]} {b[1]} {b[2]}); }}",
           f"  {{ name {n}; type cellZoneSet; action new; source setToCellZone; set {n}Cells; }}"]
ts += [");"]
open("system/topoSetDict","w").write("\n".join(ts)+"\n")
r=subprocess.run(["topoSet"],capture_output=True,text=True); print(r.returncode)
print("\n".join(l for l in r.stdout.splitlines() if "now size" in l))

# -- cell 5 -------------------------------------------------------------------------
# 4×5×5 — the 10 mm inset puts the cube's x-faces mid-cell. Move the inset to 12 mm (heater 12–32 mm, 
X0_H, X0_C = 0.012, 0.068   # my choice: 12 mm inset, cube faces on 4 mm cell faces
boxes = {"heater": (X0_H, Y0, GAP), "cooler": (X0_C, Y0, H-GAP-C)}
ts = ["FoamFile{ version 2.0; format ascii; class dictionary; object topoSetDict; }","actions ("]
for n,(x0,y0,z0) in boxes.items():
    a=(x0+eps,y0+eps,z0+eps); b=(x0+C-eps,y0+C-eps,z0+C-eps)
    ts += [f"  {{ name {n}Cells; type cellSet; action new; source boxToCell;"
           f" box ({a[0]} {a[1]} {a[2]}) ({b[0]} {b[1]} {b[2]}); }}",
           f"  {{ name {n}; type cellZoneSet; action new; source setToCellZone; set {n}Cells; }}"]
ts += [");"]
open("system/topoSetDict","w").write("\n".join(ts)+"\n")
r=subprocess.run(["topoSet"],capture_output=True,text=True); print(r.returncode)
print("\n".join(l for l in r.stdout.splitlines() if "now size" in l))

# -- cell 6 -------------------------------------------------------------------------
# Coarse shape is right (125 cells = 5³ per zone). Refine to 2 mm cells and rebuild both mesh and zone
CELL = 0.002   # final resolution: 2 mm -> 50 x 30 x 20
nx,ny,nz = int(round(L/CELL)), int(round(W/CELL)), int(round(H/CELL))
bm = ["FoamFile{ version 2.0; format ascii; class dictionary; object blockMeshDict; }",
      "scale 1;", "vertices (", *[f"  ({x} {y} {z})" for x,y,z in verts], ");",
      f"blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );",
      "edges ();", "boundary ("]
for n,f in faces.items(): bm += [f"  {n} {{ type wall; faces ( {f} ); }}"]
bm += [");","mergePatchPairs ();"]
open("system/blockMeshDict","w").write("\n".join(bm)+"\n")
eps = CELL/4
ts = ["FoamFile{ version 2.0; format ascii; class dictionary; object topoSetDict; }","actions ("]
for n,(x0,y0,z0) in boxes.items():
    a=(x0+eps,y0+eps,z0+eps); b=(x0+C-eps,y0+C-eps,z0+C-eps)
    ts += [f"  {{ name {n}Cells; type cellSet; action new; source boxToCell;"
           f" box ({a[0]} {a[1]} {a[2]}) ({b[0]} {b[1]} {b[2]}); }}",
           f"  {{ name {n}; type cellZoneSet; action new; source setToCellZone; set {n}Cells; }}"]
ts += [");"]
open("system/topoSetDict","w").write("\n".join(ts)+"\n")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).returncode)
r=subprocess.run(["topoSet"],capture_output=True,text=True); print(r.returncode)
print("\n".join(l for l in r.stdout.splitlines() if "now size" in l))

# -- cell 7 -------------------------------------------------------------------------
# 1000 = 10³ cells per zone. Now measure the zones' actual bounding boxes from the mesh itself and ren
import pyvista as pv, numpy as np
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_cell_arrays()
try: rd.enable_all_patch_arrays()
except Exception: pass
m = rd.read()
print(m.keys() if hasattr(m,'keys') else m)

# -- cell 8 -------------------------------------------------------------------------
# Zones aren't exposed by the reader, so parse `cellZones` labels directly and index the mesh cell cen
import re
txt = open("constant/polyMesh/cellZones").read()
blocks = re.findall(r"(\w+)\s*\{[^{}]*?cellLabels\s+List<label>\s*\d+\s*\((.*?)\)\s*;", txt, re.S)
im = m["internalMesh"]; cc = im.cell_centers().points
half = CELL/2
for name, body in blocks:
    idx = np.fromstring(body, sep=" ", dtype=int)
    p = cc[idx]
    lo, hi = p.min(0)-half, p.max(0)+half
    print(f"{name}: n={len(idx)} lo={np.round(lo*1000,3)} hi={np.round(hi*1000,3)} size={np.round((hi-lo)*1000,3)} mm")
print("cavity bbox mm:", np.round(np.array(im.bounds)*1000,3), "nCells", im.n_cells)

# -- cell 9 -------------------------------------------------------------------------
# Numbers match the request. A picture to confirm placement (heater low at the x-min end, cooler high 
zones = {name: np.fromstring(b, sep=" ", dtype=int) for name,b in blocks}
p = pv.Plotter(off_screen=True, window_size=(900,650))
p.add_mesh(im.outline(), color="k")
p.add_mesh(im.extract_cells(zones["heater"]), color="red", show_edges=True)
p.add_mesh(im.extract_cells(zones["cooler"]), color="blue", show_edges=True)
p.add_mesh(im.extract_surface(), color="gray", opacity=0.12)
p.add_axes(); p.camera_position="iso"; p.screenshot("zones.png")
print("ok")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("zones.png"))

# -- cell 11 ------------------------------------------------------------------------
# The iso view's axis orientation is ambiguous; take a straight x–z side view to confirm heater is low
p = pv.Plotter(off_screen=True, window_size=(900,420))
p.add_mesh(im.outline(), color="k")
p.add_mesh(im.extract_cells(zones["heater"]), color="red")
p.add_mesh(im.extract_cells(zones["cooler"]), color="blue")
p.view_xz(); p.camera.up=(0,0,1); p.add_axes(); p.screenshot("side.png")
display(Image("side.png"))

# -- cell 12 ------------------------------------------------------------------------
# Heater low at the near end, cooler high at the far end. Run `checkMesh`.
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-2200:])
