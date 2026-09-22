"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, os
os.chdir('/work/accept-t7')
print(subprocess.run(['ls','-la'], capture_output=True, text=True).stdout)
print(subprocess.run(['python3', '/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/b123d_api.py','--help'], capture_output=True, text=True).stdout[:2000])

# -- cell 2 -------------------------------------------------------------------------
# Let me check the toolbox templates for guidance on the simple-box + cellZone pattern before committi
import subprocess
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
print(subprocess.run(["ls", toolbox], capture_output=True, text=True).stdout)
print(subprocess.run(["cat", toolbox+"templates/prep/README.md"], capture_output=True, text=True).stdout[:3000])

# -- cell 3 -------------------------------------------------------------------------
import subprocess
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
r = subprocess.run(["grep","-ril","cellZone", toolbox], capture_output=True, text=True)
print(r.stdout)
r2 = subprocess.run(["grep","-rn","cellZone", toolbox+"templates"], capture_output=True, text=True)
print(r2.stdout[:3000])

# -- cell 4 -------------------------------------------------------------------------
import subprocess
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
r = subprocess.run(["grep","-n","-B3","-A15","cellZone", toolbox+"notes/openfoam-field-notes.md"], capture_output=True, text=True)
print(r.stdout[:6000])

# -- cell 5 -------------------------------------------------------------------------
# Now let me set up the geometry parameters and build a build123d model purely to visualize the box an
import subprocess, os
os.chdir('/work/accept-t7')

from build123d import Box, Pos, Compound, export_step

# ---- Cavity dimensions (metres) ----
LX_M = 0.100   # length (x)
LY_M = 0.060   # width  (y)
LZ_M = 0.040   # height (z), z=0 floor, z=LZ ceiling

# ---- Heater / cooler cube (metres) ----
CUBE_M = 0.020

# Heater: near floor, near x=0 end, centred in y
heater_x0, heater_x1 = 0.010, 0.030
heater_y0, heater_y1 = 0.020, 0.040
heater_z0, heater_z1 = 0.000, 0.020   # touches floor

# Cooler: near ceiling, near x=LX end, centred in y
cooler_x0, cooler_x1 = 0.070, 0.090
cooler_y0, cooler_y1 = 0.020, 0.040
cooler_z0, cooler_z1 = 0.020, 0.040   # touches ceiling

cavity = Pos(LX_M/2, LY_M/2, LZ_M/2) * Box(LX_M, LY_M, LZ_M)

heater_box = Pos((heater_x0+heater_x1)/2, (heater_y0+heater_y1)/2, (heater_z0+heater_z1)/2) * Box(CUBE_M, CUBE_M, CUBE_M)
cooler_box = Pos((cooler_x0+cooler_x1)/2, (cooler_y0+cooler_y1)/2, (cooler_z0+cooler_z1)/2) * Box(CUBE_M, CUBE_M, CUBE_M)

print("cavity bbox:", cavity.bounding_box())
print("heater bbox:", heater_box.bounding_box())
print("cooler bbox:", cooler_box.bounding_box())

# -- cell 6 -------------------------------------------------------------------------
# Now let's render a quick look at the geometry (box plus the two marker cubes) before committing to t
import subprocess, os
os.chdir('/work/accept-t7')
from build123d import export_stl

os.makedirs("look_stl", exist_ok=True)
export_stl(cavity, "look_stl/cavity.stl")
export_stl(heater_box, "look_stl/heater.stl")
export_stl(cooler_box, "look_stl/cooler.stl")

import pyvista as pv
pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=(900,600))
cav = pv.read("look_stl/cavity.stl")
hea = pv.read("look_stl/heater.stl")
coo = pv.read("look_stl/cooler.stl")
plotter.add_mesh(cav, color="lightgray", opacity=0.25, show_edges=True)
plotter.add_mesh(hea, color="red", opacity=0.9)
plotter.add_mesh(coo, color="blue", opacity=0.9)
plotter.add_axes()
plotter.camera_position = "iso"
plotter.reset_camera()
plotter.screenshot("look_geometry.png")
print("wrote look_geometry.png")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image
Image(filename="look_geometry.png")

# -- cell 8 -------------------------------------------------------------------------
# The iso view makes height differences hard to judge. Let me check with axis-aligned views (front and
import pyvista as pv
pv.OFF_SCREEN = True

plotter = pv.Plotter(off_screen=True, window_size=(1200,500), shape=(1,2))

cav = pv.read("look_stl/cavity.stl")
hea = pv.read("look_stl/heater.stl")
coo = pv.read("look_stl/cooler.stl")

plotter.subplot(0,0)
plotter.add_mesh(cav, color="lightgray", opacity=0.15, show_edges=True)
plotter.add_mesh(hea, color="red")
plotter.add_mesh(coo, color="blue")
plotter.add_axes()
plotter.camera_position = 'xz'   # looking along -y, see x (length) vs z (height)
plotter.add_text("front: x (length) vs z (height)", font_size=10)
plotter.reset_camera()

plotter.subplot(0,1)
plotter.add_mesh(cav, color="lightgray", opacity=0.15, show_edges=True)
plotter.add_mesh(hea, color="red")
plotter.add_mesh(coo, color="blue")
plotter.add_axes()
plotter.camera_position = 'yz'   # looking along x, see y (width) vs z (height)
plotter.add_text("side: y (width) vs z (height)", font_size=10)
plotter.reset_camera()

plotter.screenshot("look_geometry_ortho.png")
print("wrote look_geometry_ortho.png")

# -- cell 9 -------------------------------------------------------------------------
from IPython.display import Image
Image(filename="look_geometry_ortho.png")

# -- cell 10 ------------------------------------------------------------------------
# The geometry checks out: heater sits at the floor near x=0, cooler at the ceiling near x=100mm, both
import os
os.chdir('/work/accept-t7')

for d in ["system", "constant", "0"]:
    os.makedirs(d, exist_ok=True)

# ---- Grid: 5 mm cells so the 20 mm cube boundaries land exactly on nodes ----
CELL_M = 0.005
NX = round(LX_M / CELL_M)   # 20
NY = round(LY_M / CELL_M)   # 12
NZ = round(LZ_M / CELL_M)   # 8
print("cells:", NX, NY, NZ, "total", NX*NY*NZ)

# sanity: cube boundaries must be multiples of CELL_M
for name, (a,b) in {
    "heater_x": (heater_x0, heater_x1), "heater_y": (heater_y0, heater_y1), "heater_z": (heater_z0, heater_z1),
    "cooler_x": (cooler_x0, cooler_x1), "cooler_y": (cooler_y0, cooler_y1), "cooler_z": (cooler_z0, cooler_z1),
}.items():
    ra, rb = a/CELL_M, b/CELL_M
    assert abs(ra-round(ra)) < 1e-9 and abs(rb-round(rb)) < 1e-9, (name, a, b)
print("cube boundaries all align with the 5 mm grid")

blockMeshDict = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}

convertToMeters 1.0;

vertices
(
    (0        0        0)
    ({LX_M}   0        0)
    ({LX_M}   {LY_M}   0)
    (0        {LY_M}   0)
    (0        0        {LZ_M})
    ({LX_M}   0        {LZ_M})
    ({LX_M}   {LY_M}   {LZ_M})
    (0        {LY_M}   {LZ_M})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    floor
    {{
        type wall;
        faces ((0 1 2 3));
    }}
    ceiling
    {{
        type wall;
        faces ((4 5 6 7));
    }}
    xmin
    {{
        type wall;
        faces ((0 4 7 3));
    }}
    xmax
    {{
        type wall;
        faces ((1 5 6 2));
    }}
    ymin
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    ymax
    {{
        type wall;
        faces ((3 2 6 7));
    }}
);

mergePatchPairs
(
);
"""

with open("system/blockMeshDict", "w") as f:
    f.write(blockMeshDict)

print("wrote system/blockMeshDict")
print(blockMeshDict)

# -- cell 11 ------------------------------------------------------------------------
# Now the minimal controlDict/fvSchemes/fvSolution that OpenFOAM utilities need to run, and topoSetDic
import os
os.chdir('/work/accept-t7')

controlDict = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application     buoyantSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
with open("system/controlDict", "w") as f:
    f.write(controlDict)

fvSchemes = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
with open("system/fvSchemes", "w") as f:
    f.write(fvSchemes)

fvSolution = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers
{
}
"""
with open("system/fvSolution", "w") as f:
    f.write(fvSolution)

# ---- topoSetDict: two cellZones aligned to the grid ----
topoSetDict = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object topoSetDict;
}}

actions
(
    {{
        name    heater;
        type    cellZoneSet;
        action  new;
        source  boxToCell;
        box     ({heater_x0} {heater_y0} {heater_z0}) ({heater_x1} {heater_y1} {heater_z1});
    }}
    {{
        name    cooler;
        type    cellZoneSet;
        action  new;
        source  boxToCell;
        box     ({cooler_x0} {cooler_y0} {cooler_z0}) ({cooler_x1} {cooler_y1} {cooler_z1});
    }}
);
"""
with open("system/topoSetDict", "w") as f:
    f.write(topoSetDict)

print("wrote system/controlDict, fvSchemes, fvSolution, topoSetDict")

# -- cell 12 ------------------------------------------------------------------------
# Now run blockMesh, then topoSet to tag the two cell zones.
import subprocess, os
os.chdir('/work/accept-t7')

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-3000:])
print("STDERR:", r.stderr[-2000:])
r.check_returncode()

# -- cell 13 ------------------------------------------------------------------------
# blockMesh created 1920 cells and the expected bounding box (0.1×0.06×0.04 m). Now run topoSet to tag
import subprocess, os
os.chdir('/work/accept-t7')

r = subprocess.run(["topoSet"], capture_output=True, text=True)
print(r.stdout[-3000:])
print("STDERR:", r.stderr[-1000:])
r.check_returncode()

# -- cell 14 ------------------------------------------------------------------------
# Each zone got exactly 64 cells (4×4×4 at 5 mm resolution = the 20 mm cube, exactly), confirming clea
import subprocess, os
os.chdir('/work/accept-t7')

r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-4000:])
print("STDERR:", r.stderr[-1000:])

# -- cell 15 ------------------------------------------------------------------------
# checkMesh reports "Mesh OK." and both cellZones have exactly the volume of a 20 mm cube (8×10⁻⁸ m³..
import subprocess, os
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
os.chdir('/work/accept-t7')

r = subprocess.run(["python3", toolbox+"mesh_look.py", "--case", ".", "--json"],
                    capture_output=True, text=True)
print(r.stdout[-4000:])
print("STDERR:", r.stderr[-2000:])

# -- cell 16 ------------------------------------------------------------------------
import subprocess, os
toolbox = "/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/b0fa9eb3-70e4-4403-8113-987906f361e4/scratchpad/trials3/work3/.toolbox/"
os.chdir('/work/accept-t7')

r = subprocess.run(["python3", toolbox+"mesh_look.py", ".", "--json", "look.json"],
                    capture_output=True, text=True)
print(r.stdout[-4000:])
print("STDERR:", r.stderr[-2000:])

# -- cell 17 ------------------------------------------------------------------------
from IPython.display import Image
Image(filename="look.png")

# -- cell 18 ------------------------------------------------------------------------
# Let me add one more visual check: render the actual mesh with the two cellZones highlighted (not jus
import subprocess, os
os.chdir('/work/accept-t7')

r = subprocess.run(["foamToVTK", "-cellSet", "heater", "-cellSet", "cooler"], capture_output=True, text=True)
print(r.stdout[-2000:])
print("STDERR:", r.stderr[-1500:])
print(subprocess.run(["find", "VTK", "-maxdepth", "3"], capture_output=True, text=True).stdout)

# -- cell 19 ------------------------------------------------------------------------
# That only exported cellSet cooler (the second `-cellSet` flag overrode the first, it seems). Let me 
import subprocess, os
os.chdir('/work/accept-t7')

r = subprocess.run(["foamToVTK", "-cellSet", "heater"], capture_output=True, text=True)
print(r.stdout[-1200:])
print("STDERR:", r.stderr[-500:])
print(subprocess.run(["find", "VTK", "-maxdepth", "2"], capture_output=True, text=True).stdout)

# -- cell 20 ------------------------------------------------------------------------
# Also export the full domain internal mesh so I can show the zones in context.
import subprocess, os
os.chdir('/work/accept-t7')

r = subprocess.run(["foamToVTK"], capture_output=True, text=True)
print(r.stdout[-800:])
print("STDERR:", r.stderr[-500:])
print(subprocess.run(["find", "VTK", "-maxdepth", "2"], capture_output=True, text=True).stdout)

# -- cell 21 ------------------------------------------------------------------------
import pyvista as pv
pv.OFF_SCREEN = True

whole = pv.read("VTK/accept-t7_0/internal.vtu")
heater_v = pv.read("VTK/heater_0/internal.vtu")
cooler_v = pv.read("VTK/cooler_0/internal.vtu")

print("whole cells:", whole.n_cells, "heater cells:", heater_v.n_cells, "cooler cells:", cooler_v.n_cells)
print("heater bounds:", heater_v.bounds)
print("cooler bounds:", cooler_v.bounds)

plotter = pv.Plotter(off_screen=True, window_size=(1000,700))
plotter.add_mesh(whole, color="lightgray", opacity=0.12, show_edges=True, edge_color="gray")
plotter.add_mesh(heater_v, color="red", show_edges=True, label="heater")
plotter.add_mesh(cooler_v, color="blue", show_edges=True, label="cooler")
plotter.add_axes()
plotter.camera_position = "iso"
plotter.reset_camera()
plotter.add_legend()
plotter.screenshot("look_cellzones.png")
print("wrote look_cellzones.png")

# -- cell 22 ------------------------------------------------------------------------
from IPython.display import Image
Image(filename="look_cellzones.png")
