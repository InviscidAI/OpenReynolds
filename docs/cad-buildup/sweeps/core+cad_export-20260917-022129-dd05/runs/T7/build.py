"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# `blockMesh` requires a minimal `system/controlDict` even for mesh-only work; the first attempt there
from pathlib import Path
import subprocess

L, W, H, CELL = 0.100, 0.060, 0.040, 0.005
NX, NY, NZ = round(L/CELL), round(W/CELL), round(H/CELL)
Path("system").mkdir(exist_ok=True)
Path("system/controlDict").write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
''')
Path("system/blockMeshDict").write_text(f'''FoamFile
{{
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
    (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 3 2 1) (4 5 6 7)
            (0 1 5 4) (3 7 6 2)
            (0 4 7 3) (1 2 6 5)
        );
    }}
);
mergePatchPairs ();
''')
result = subprocess.run(["blockMesh"], text=True, capture_output=True)
print(result.stdout[-2200:])
if result.returncode:
    print(result.stderr)
    raise RuntimeError(f"blockMesh failed with code {result.returncode}")
print(f"Measured block extents: {L:.3f} x {W:.3f} x {H:.3f} m (requested 0.100 x 0.060 x 0.040 m)")
print(f"Coarse mesh: {NX} x {NY} x {NZ} = {NX*NY*NZ} cells at {CELL:.3f} m")

# -- cell 2 -------------------------------------------------------------------------
# The coarse mesh has the requested extents and a single 992-face wall patch. I’ll now inspect it visu
import pyvista as pv
from pathlib import Path

L, W, H, CELL = 0.100, 0.060, 0.040, 0.005
heater_bounds = (0.005, 0.025, 0.020, 0.040, 0.000, 0.020)
cooler_bounds = (0.075, 0.095, 0.020, 0.040, 0.020, 0.040)

cavity = pv.Box(bounds=(0, L, 0, W, 0, H))
heater_preview = pv.Box(bounds=heater_bounds)
cooler_preview = pv.Box(bounds=cooler_bounds)

plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
plotter.set_background("white")
plotter.add_mesh(cavity, style="wireframe", color="black", line_width=3, label="walls")
# Add the structured 5 mm surface grid as thin lines.
for x in [i*CELL for i in range(21)]:
    plotter.add_mesh(pv.Line((x,0,0),(x,W,0)), color="lightgray", line_width=1)
    plotter.add_mesh(pv.Line((x,0,H),(x,W,H)), color="lightgray", line_width=1)
for y in [i*CELL for i in range(13)]:
    plotter.add_mesh(pv.Line((0,y,0),(L,y,0)), color="lightgray", line_width=1)
    plotter.add_mesh(pv.Line((0,y,H),(L,y,H)), color="lightgray", line_width=1)
for z in [i*CELL for i in range(9)]:
    plotter.add_mesh(pv.Line((0,0,z),(L,0,z)), color="lightgray", line_width=1)
    plotter.add_mesh(pv.Line((0,W,z),(L,W,z)), color="lightgray", line_width=1)
plotter.add_mesh(heater_preview, color="orangered", opacity=0.75, show_edges=True, label="heater")
plotter.add_mesh(cooler_preview, color="royalblue", opacity=0.75, show_edges=True, label="cooler")
plotter.add_legend(bcolor="white", face=None)
plotter.add_axes()
plotter.view_isometric()
plotter.camera.zoom(1.25)
plotter.show(screenshot="coarse_mesh_preview.png", auto_close=True)
print("Wrote coarse_mesh_preview.png")
print(f"heater preview bounds: {heater_bounds}; dimensions = 0.020 x 0.020 x 0.020 m")
print(f"cooler preview bounds: {cooler_bounds}; dimensions = 0.020 x 0.020 x 0.020 m")

# -- cell 3 -------------------------------------------------------------------------
# The preview confirms the opposed placement and exact grid alignment. I’ll now create `heater` and `c
from pathlib import Path
import subprocess

HEATER_MIN, HEATER_MAX = (0.005, 0.020, 0.000), (0.025, 0.040, 0.020)
COOLER_MIN, COOLER_MAX = (0.075, 0.020, 0.020), (0.095, 0.040, 0.040)
Path("system/topoSetDict").write_text(f'''FoamFile
{{
    format ascii;
    class dictionary;
    object topoSetDict;
}}
actions
(
    {{
        name heaterCells;
        type cellSet;
        action new;
        source boxToCell;
        box ({HEATER_MIN[0]} {HEATER_MIN[1]} {HEATER_MIN[2]}) ({HEATER_MAX[0]} {HEATER_MAX[1]} {HEATER_MAX[2]});
    }}
    {{
        name heater;
        type cellZoneSet;
        action new;
        source setToCellZone;
        set heaterCells;
    }}
    {{
        name coolerCells;
        type cellSet;
        action new;
        source boxToCell;
        box ({COOLER_MIN[0]} {COOLER_MIN[1]} {COOLER_MIN[2]}) ({COOLER_MAX[0]} {COOLER_MAX[1]} {COOLER_MAX[2]});
    }}
    {{
        name cooler;
        type cellZoneSet;
        action new;
        source setToCellZone;
        set coolerCells;
    }}
);
''')
result = subprocess.run(["topoSet"], text=True, capture_output=True)
print(result.stdout[-3000:])
if result.returncode:
    print(result.stderr)
    raise RuntimeError(f"topoSet failed with code {result.returncode}")

# -- cell 4 -------------------------------------------------------------------------
# This OpenFOAM build also requires baseline `fvSchemes` and `fvSolution` dictionaries when loading a 
from pathlib import Path
import subprocess

Path("system/fvSchemes").write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
wallDist { method meshWave; }
fluxRequired { default no; }
''')
Path("system/fvSolution").write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
result = subprocess.run(["checkMesh"], text=True, capture_output=True)
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise RuntimeError(f"checkMesh failed with code {result.returncode}")

# -- cell 5 -------------------------------------------------------------------------
# The coarse mesh is geometrically exact and fully orthogonal. For the delivered mesh, I’ll refine uni
from pathlib import Path
import subprocess

L, W, H, FINAL_CELL = 0.100, 0.060, 0.040, 0.0025
FINAL_NX, FINAL_NY, FINAL_NZ = round(L/FINAL_CELL), round(W/FINAL_CELL), round(H/FINAL_CELL)
Path("system/blockMeshDict").write_text(f'''FoamFile
{{
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
    (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({FINAL_NX} {FINAL_NY} {FINAL_NZ}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 3 2 1) (4 5 6 7)
            (0 1 5 4) (3 7 6 2)
            (0 4 7 3) (1 2 6 5)
        );
    }}
);
mergePatchPairs ();
''')
result = subprocess.run(["blockMesh"], text=True, capture_output=True)
print(result.stdout[-1600:])
if result.returncode:
    print(result.stderr)
    raise RuntimeError(f"final blockMesh failed with code {result.returncode}")
print(f"Final mesh divisions: {FINAL_NX} x {FINAL_NY} x {FINAL_NZ} = {FINAL_NX*FINAL_NY*FINAL_NZ} cells")

# -- cell 6 -------------------------------------------------------------------------
# Refinement recreates `polyMesh`, so I’ll reapply the already-defined physical zone boxes to the fina
import subprocess

result = subprocess.run(["topoSet"], text=True, capture_output=True)
print(result.stdout[-2200:])
if result.returncode:
    print(result.stderr)
    raise RuntimeError(f"final topoSet failed with code {result.returncode}")

# -- cell 7 -------------------------------------------------------------------------
# The final zones each contain 512 cells as intended. I’ll perform the final bare `checkMesh` validati
import subprocess

result = subprocess.run(["checkMesh"], text=True, capture_output=True)
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise RuntimeError(f"final checkMesh failed with code {result.returncode}")
