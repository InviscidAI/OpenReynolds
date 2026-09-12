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
CASE = "/home/qiuzi/.openreynolds-buildup/work/T7-20260912-125423-1f17/t7"
os.chdir(CASE)
for d in ["system", "constant"]:
    os.makedirs(d, exist_ok=True)

# All dimensions in METRES (request was mm).
LX, LY, LZ = 0.100, 0.060, 0.040      # cavity
CUBE = 0.020                           # heater/cooler cube edge
GAP = 0.005                            # gap from floor/ceiling and from end walls
HEAT = (GAP, GAP + CUBE, (LY - CUBE) / 2, (LY + CUBE) / 2, GAP, GAP + CUBE)
COOL = (LX - GAP - CUBE, LX - GAP, (LY - CUBE) / 2, (LY + CUBE) / 2, LZ - GAP - CUBE, LZ - GAP)
NX, NY, NZ = 20, 12, 8                 # coarse: 5 mm cells

head = "FoamFile\n{ version 2.0; format ascii; class dictionary; object %s; }\n"
bm = head % "blockMeshDict" + f"""
scale 1;
vertices
(
 (0 0 0) ({LX} 0 0) ({LX} {LY} 0) (0 {LY} 0)
 (0 0 {LZ}) ({LX} 0 {LZ}) ({LX} {LY} {LZ}) (0 {LY} {LZ})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2) (0 4 7 3) (1 2 6 5)
        );
    }}
);
mergePatchPairs ();
"""
open("system/blockMeshDict", "w").write(bm)

ctrl = head % "controlDict" + """
application scalarTransportFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
"""
open("system/controlDict", "w").write(ctrl)
open("system/fvSchemes", "w").write(head % "fvSchemes" + "ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\ninterpolationSchemes{default linear;}\nsnGradSchemes{default corrected;}\n")
open("system/fvSolution", "w").write(head % "fvSolution" + "solvers{}\n")

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-800:], r.returncode)
print("HEAT", HEAT, "\nCOOL", COOL)

# -- cell 2 -------------------------------------------------------------------------
# Mesh box is right. Now cut the two cellZones with `topoSet` (boxToCell → cellSet → cellZoneSet).
def topo_dict(heat, cool):
    def blk(name, b):
        return f"""
    {{ name {name}Set; type cellSet; action new; source boxToCell;
      box ({b[0]} {b[2]} {b[4]}) ({b[1]} {b[3]} {b[5]}); }}
    {{ name {name}; type cellZoneSet; action new; source setToCellZone; set {name}Set; }}
"""
    return head % "topoSetDict" + "actions (\n" + blk("heater", heat) + blk("cooler", cool) + ");\n"

open("system/topoSetDict", "w").write(topo_dict(HEAT, COOL))
r = subprocess.run(["topoSet"], capture_output=True, text=True)
print(r.stdout[-1200:], r.returncode)

# -- cell 3 -------------------------------------------------------------------------
# 64 cells each = 4×4×4 × 5 mm = exactly a 20 mm cube. Let me look at it before refining.
import pyvista as pv, numpy as np
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
from pyvista import examples
mesh = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").write("")
r = pv.OpenFOAMReader("case.foam"); r.set_active_time_value(0.0)
blk = r.read()
internal = blk["internalMesh"]
cz = np.zeros(internal.n_cells)
import re
txt = open("constant/polyMesh/cellZones").read()
def zone_ids(name):
    m = re.search(name + r"\s*\{.*?List<label>\s*\d+\s*\((.*?)\)", txt, re.S)
    return np.array([int(x) for x in m.group(1).split()])
h, c = zone_ids("heater"), zone_ids("cooler")
cz[h] = 1; cz[c] = 2
internal.cell_data["zone"] = cz
p = pv.Plotter(off_screen=True, window_size=(900,650))
p.add_mesh(internal.extract_cells(np.concatenate([h,c])), scalars="zone", cmap=["red","blue"], show_edges=True)
p.add_mesh(internal.extract_all_edges(), color="grey", opacity=0.25)
p.show_axes(); p.camera_position='iso'
p.screenshot("coarse.png"); print(len(h), len(c))

# -- cell 4 -------------------------------------------------------------------------
from IPython.display import Image, display
print(os.path.exists("coarse.png"), os.path.getsize("coarse.png") if os.path.exists("coarse.png") else 0)
display(Image("coarse.png"))

# -- cell 5 -------------------------------------------------------------------------
# Picture confirms two separated cubes; orientation is ambiguous by eye, so measure the zone bounds nu
cc = internal.cell_centers().points
for nm, ids in [("heater", h), ("cooler", c)]:
    pts = cc[ids]
    lo, hi = pts.min(0) - 0.0025, pts.max(0) + 0.0025   # cell half-size 2.5mm
    print(nm, "extent mm:", np.round(lo*1000,3), np.round(hi*1000,3), "edge mm:", np.round((hi-lo)*1000,3))
print("cavity bbox mm:", np.round(np.array(internal.bounds)*1000,3))

# -- cell 6 -------------------------------------------------------------------------
# Shape and zones are correct. Refine to 2.5 mm cells (40×24×16 = 15 360) — every zone boundary (5, 25
NX, NY, NZ = 40, 24, 16   # final: 2.5 mm cells
open("system/blockMeshDict", "w").write(bm.replace("(20 12 8)", f"({NX} {NY} {NZ})"))
for cmd in (["blockMesh"], ["topoSet"], ["checkMesh"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("###", cmd[0], r.returncode)
    print(r.stdout[-900:])
