"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess
result = subprocess.run(["grep", "-i", "-A", "8", "-B", "2", "class CenterArc\\|class RadiusArc\\|class Line\\b",
                          "/work/.toolbox/b123d_api.md"], capture_output=True, text=True)
print(result.stdout[:4000])
print("----STDERR----")
print(result.stderr[:2000])

# -- cell 2 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["ls", "/work/.toolbox/"], capture_output=True, text=True)
print(r.stdout, r.stderr)
r2 = subprocess.run(["grep", "-in", "arc", "/work/.toolbox/b123d_api.md"], capture_output=True, text=True)
print(r2.stdout[:4000])

# -- cell 3 -------------------------------------------------------------------------
# Endpoints line up correctly. `is_closed` is a property, not a method — fix that and continue.
from build123d import *
import math

WIDTH_M   = 0.010
LEG_LEN_M = 0.120
BEND_R_M  = 0.015
HALF_W    = WIDTH_M / 2.0

OUTER_R = BEND_R_M + HALF_W
INNER_R = BEND_R_M - HALF_W
BEND_CTR = (LEG_LEN_M, BEND_R_M)

y_lo_outer = -HALF_W
y_lo_inner =  HALF_W
y_hi_inner = 2*BEND_R_M - HALF_W
y_hi_outer = 2*BEND_R_M + HALF_W

e1 = Line((0, y_lo_outer), (LEG_LEN_M, y_lo_outer))
e2 = CenterArc(BEND_CTR, OUTER_R, -90, 180)
e3 = Line((LEG_LEN_M, y_hi_outer), (0, y_hi_outer))
e4 = Line((0, y_hi_outer), (0, y_hi_inner))
e5 = Line((0, y_hi_inner), (LEG_LEN_M, y_hi_inner))
e6 = CenterArc(BEND_CTR, INNER_R, 90, -180)
e7 = Line((LEG_LEN_M, y_lo_inner), (0, y_lo_inner))
e8 = Line((0, y_lo_inner), (0, y_lo_outer))

wire = Wire([e1, e2, e3, e4, e5, e6, e7, e8])
print("wire closed:", wire.is_closed)

face = Face(wire)
expected_area = WIDTH_M * (2*LEG_LEN_M + math.pi*BEND_R_M)
print("face area (m^2):", face.area, " expected approx:", expected_area)

# -- cell 4 -------------------------------------------------------------------------
# Face area matches the expected centreline-length × width to within rounding. Now extrude this planar
THICKNESS_M = 0.001   # slab thickness in z, one cell later -- arbitrary for a 2D case

solid = extrude(face, amount=THICKNESS_M)
print("solid volume (m^3):", solid.volume, " expected:", face.area*THICKNESS_M)
print("bounding box:", solid.bounding_box())

from build123d import export_step
export_step(solid, "ubend_solid.step")
print("exported ubend_solid.step")

# -- cell 5 -------------------------------------------------------------------------
# Volume matches, bounding box is as expected (x: 0–140mm, y: -5–35mm, z: 0–1mm). Now let's tag the fa
faces = solid.faces()

# front/back: the two z-normal faces (the extrusion caps) -> empty patches for 2D
front = faces.filter_by(Axis.Z).sort_by(Axis.Z)[0]
back  = faces.filter_by(Axis.Z).sort_by(Axis.Z)[-1]

# inlet / outlet: faces whose plane is x=0 (the two end caps of the duct)
x0_faces = faces.filter_by(Axis.X).filter_by(lambda f: abs(f.center().X) < 1e-9)
print("num x=0 faces:", len(x0_faces))
for f in x0_faces:
    print("  center:", f.center(), "area:", f.area)

# -- cell 6 -------------------------------------------------------------------------
inlet_face  = x0_faces.filter_by(lambda f: abs(f.center().Y - 0.0) < 1e-6)[0]
outlet_face = x0_faces.filter_by(lambda f: abs(f.center().Y - 2*BEND_R_M) < 1e-6)[0]

print("inlet center:", inlet_face.center(), "area:", inlet_face.area)
print("outlet center:", outlet_face.center(), "area:", outlet_face.area)

named_ids = {front.center().to_tuple(), back.center().to_tuple(),
             inlet_face.center().to_tuple(), outlet_face.center().to_tuple()}
wall_faces = faces.filter_by(lambda f: f.center().to_tuple() not in named_ids)
print("num wall faces:", len(wall_faces))
for f in wall_faces:
    print("  center:", f.center(), "area:", f.area)

# -- cell 7 -------------------------------------------------------------------------
# All 6 wall faces are identified and their areas match expectation (leg walls = 0.120×0.001 m², outer
import subprocess
r = subprocess.run(["python3", "/work/.toolbox/cad_convert.py", "--help"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)

# -- cell 8 -------------------------------------------------------------------------
# This is the CLI STEP→STL path. `export_patches()` was mentioned as a python function — let's inspect
import sys
sys.path.insert(0, "/work/.toolbox")
import cad_convert
help(cad_convert.export_patches)

# -- cell 9 -------------------------------------------------------------------------
print(cad_convert.PATCH_ROLES)

# -- cell 10 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["cat", "/work/.toolbox/templates/snappy/README.md"], capture_output=True, text=True)
print(r.stdout)

# -- cell 11 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["find", "/work/.toolbox", "-ipath", "*duct*"], capture_output=True, text=True)
print(r.stdout)

# -- cell 12 ------------------------------------------------------------------------
# There's a ready template for exactly this kind of case. Let's read it before continuing — it likely 
import subprocess
r = subprocess.run(["cat", "/work/.toolbox/templates/duct2d.py"], capture_output=True, text=True)
print(r.stdout)

# -- cell 13 ------------------------------------------------------------------------
# This confirms the far better-suited path: gmsh OCC quad-mesh + extrude for a genuinely 2D case, not 
import subprocess
r = subprocess.run(["cat", "/work/.toolbox/templates/duct2d.py"], capture_output=True, text=True)
print(r.stdout)

# -- cell 14 ------------------------------------------------------------------------
# The full text seems to get truncated by the transport. Let me read the file directly in Python and s
text = open("/work/.toolbox/templates/duct2d.py").read()
print(len(text))
print(text[1800:5200])

# -- cell 15 ------------------------------------------------------------------------
# Good, let's see the rest.
print(text[5200:9000])

# -- cell 16 ------------------------------------------------------------------------
import gmsh
import numpy as np
import matplotlib.pyplot as plt

WIDTH_M   = 0.010
LEG_LEN_M = 0.120
BEND_R_M  = 0.015
HALF_W    = WIDTH_M / 2.0
BEND_CTR  = (LEG_LEN_M, BEND_R_M)

y_lo_outer = -HALF_W
y_lo_inner =  HALF_W
y_hi_inner = 2*BEND_R_M - HALF_W
y_hi_outer = 2*BEND_R_M + HALF_W

CELL_SIZE_TEST = WIDTH_M / 5.0

gmsh.initialize()
gmsh.model.add("outline_check")

p1 = gmsh.model.geo.addPoint(0, y_lo_outer, 0, CELL_SIZE_TEST)
p2 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_outer, 0, CELL_SIZE_TEST)
p3 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_outer, 0, CELL_SIZE_TEST)
p4 = gmsh.model.geo.addPoint(0, y_hi_outer, 0, CELL_SIZE_TEST)
p5 = gmsh.model.geo.addPoint(0, y_hi_inner, 0, CELL_SIZE_TEST)
p6 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_inner, 0, CELL_SIZE_TEST)
p7 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_inner, 0, CELL_SIZE_TEST)
p8 = gmsh.model.geo.addPoint(0, y_lo_inner, 0, CELL_SIZE_TEST)
pc = gmsh.model.geo.addPoint(BEND_CTR[0], BEND_CTR[1], 0, CELL_SIZE_TEST)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addCircleArc(p2, pc, p3, nx=0, ny=0, nz=1)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p5)
l5 = gmsh.model.geo.addLine(p5, p6)
l6 = gmsh.model.geo.addCircleArc(p6, pc, p7, nx=0, ny=0, nz=-1)
l7 = gmsh.model.geo.addLine(p7, p8)
l8 = gmsh.model.geo.addLine(p8, p1)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
surf = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords = np.array(node_coords).reshape(-1, 3)
gmsh.finalize()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(coords[:, 0], coords[:, 1], s=4)
ax.set_aspect("equal")
ax.set_title("outline check: node cloud")
plt.savefig("outline_check.png", dpi=120)
plt.show()
print("num nodes:", len(coords))
print("x range:", coords[:,0].min(), coords[:,0].max())
print("y range:", coords[:,1].min(), coords[:,1].max())

# -- cell 17 ------------------------------------------------------------------------
# The outline is correct: a proper U-bend shape, x from 0 to 0.140 m, y from -0.005 to 0.035 m, bend b
import subprocess, sys
from pathlib import Path

WIDTH_M   = 0.010
LEG_LEN_M = 0.120
BEND_R_M  = 0.015
HALF_W    = WIDTH_M / 2.0
BEND_CTR  = (LEG_LEN_M, BEND_R_M)

y_lo_outer = -HALF_W
y_lo_inner =  HALF_W
y_hi_inner = 2*BEND_R_M - HALF_W
y_hi_outer = 2*BEND_R_M + HALF_W

CELL_SIZE_M = WIDTH_M / 5.0   # coarse first pass: ~5 cells across the passage width
THICK_M = CELL_SIZE_M         # one layer, kept roughly cubic

def build_mesh(msh_path: Path, cell_size: float, thick: float) -> None:
    import gmsh
    gmsh.initialize()
    gmsh.model.add("ubend2d")

    p1 = gmsh.model.geo.addPoint(0, y_lo_outer, 0, cell_size)
    p2 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_outer, 0, cell_size)
    p3 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_outer, 0, cell_size)
    p4 = gmsh.model.geo.addPoint(0, y_hi_outer, 0, cell_size)
    p5 = gmsh.model.geo.addPoint(0, y_hi_inner, 0, cell_size)
    p6 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_inner, 0, cell_size)
    p7 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_inner, 0, cell_size)
    p8 = gmsh.model.geo.addPoint(0, y_lo_inner, 0, cell_size)
    pc = gmsh.model.geo.addPoint(BEND_CTR[0], BEND_CTR[1], 0, cell_size)

    l_wall_lo_outer = gmsh.model.geo.addLine(p1, p2)
    l_wall_bend_outer = gmsh.model.geo.addCircleArc(p2, pc, p3, nx=0, ny=0, nz=1)
    l_wall_hi_outer = gmsh.model.geo.addLine(p3, p4)
    l_outlet = gmsh.model.geo.addLine(p4, p5)
    l_wall_hi_inner = gmsh.model.geo.addLine(p5, p6)
    l_wall_bend_inner = gmsh.model.geo.addCircleArc(p6, pc, p7, nx=0, ny=0, nz=-1)
    l_wall_lo_inner = gmsh.model.geo.addLine(p7, p8)
    l_inlet = gmsh.model.geo.addLine(p8, p1)

    loop = gmsh.model.geo.addCurveLoop([l_wall_lo_outer, l_wall_bend_outer, l_wall_hi_outer,
                                         l_outlet, l_wall_hi_inner, l_wall_bend_inner,
                                         l_wall_lo_inner, l_inlet])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.model.mesh.setRecombine(2, surf)

    out = gmsh.model.geo.extrude([(2, surf)], 0, 0, thick, numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    # out order: [top_surf, volume, side(l_wall_lo_outer), side(l_wall_bend_outer),
    #             side(l_wall_hi_outer), side(l_outlet), side(l_wall_hi_inner),
    #             side(l_wall_bend_inner), side(l_wall_lo_inner), side(l_inlet)]
    top_surf, volume = out[0][1], out[1][1]
    (side_lo_outer, side_bend_outer, side_hi_outer, side_outlet,
     side_hi_inner, side_bend_inner, side_lo_inner, side_inlet) = (e[1] for e in out[2:10])

    wall_sides = [side_lo_outer, side_bend_outer, side_hi_outer,
                  side_hi_inner, side_bend_inner, side_lo_inner]

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(2, [side_outlet], name="outlet")
    gmsh.model.addPhysicalGroup(2, wall_sides, name="wall")
    gmsh.model.addPhysicalGroup(3, [volume], name="internal")

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(msh_path))
    gmsh.finalize()

msh_path = Path("ubend2d.msh")
build_mesh(msh_path, CELL_SIZE_M, THICK_M)
print("wrote", msh_path, msh_path.stat().st_size, "bytes")

# -- cell 18 ------------------------------------------------------------------------
# Now write the OpenFOAM system stubs and convert with `gmshToFoam`, then retype patches (`wall`, `emp
import subprocess, sys
from pathlib import Path

CONTROL_DICT_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     foamRun;
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

FV_SCHEMES_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

FV_SOLUTION_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
}
"""

def write_system_stubs(case: Path) -> None:
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(CONTROL_DICT_STUB)
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES_STUB)
    (case / "system" / "fvSolution").write_text(FV_SOLUTION_STUB)

def run(cmd, **kw):
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode})")
    return proc

def retype(patch: str, kind: str) -> None:
    run(["foamDictionary", "constant/polyMesh/boundary",
         "-entry", f"entry0.{patch}.type", "-set", kind])

case = Path.cwd()
write_system_stubs(case)
run(["gmshToFoam", "ubend2d.msh"])
retype("wall", "wall")
retype("frontAndBack", "empty")
run(["checkMesh", "-case", "."])

# -- cell 19 ------------------------------------------------------------------------
# The mesh reports 0 hexahedra and 1458 prisms — the 2D surface came out triangulated despite the quad
import gmsh
import numpy as np

WIDTH_M   = 0.010
LEG_LEN_M = 0.120
BEND_R_M  = 0.015
HALF_W    = WIDTH_M / 2.0
BEND_CTR  = (LEG_LEN_M, BEND_R_M)
y_lo_outer = -HALF_W
y_lo_inner =  HALF_W
y_hi_inner = 2*BEND_R_M - HALF_W
y_hi_outer = 2*BEND_R_M + HALF_W
CELL_SIZE_TEST = WIDTH_M / 5.0

gmsh.initialize()
gmsh.model.add("quad_check")
p1 = gmsh.model.geo.addPoint(0, y_lo_outer, 0, CELL_SIZE_TEST)
p2 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_outer, 0, CELL_SIZE_TEST)
p3 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_outer, 0, CELL_SIZE_TEST)
p4 = gmsh.model.geo.addPoint(0, y_hi_outer, 0, CELL_SIZE_TEST)
p5 = gmsh.model.geo.addPoint(0, y_hi_inner, 0, CELL_SIZE_TEST)
p6 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_inner, 0, CELL_SIZE_TEST)
p7 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_inner, 0, CELL_SIZE_TEST)
p8 = gmsh.model.geo.addPoint(0, y_lo_inner, 0, CELL_SIZE_TEST)
pc = gmsh.model.geo.addPoint(BEND_CTR[0], BEND_CTR[1], 0, CELL_SIZE_TEST)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addCircleArc(p2, pc, p3, nx=0, ny=0, nz=1)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p5)
l5 = gmsh.model.geo.addLine(p5, p6)
l6 = gmsh.model.geo.addCircleArc(p6, pc, p7, nx=0, ny=0, nz=-1)
l7 = gmsh.model.geo.addLine(p7, p8)
l8 = gmsh.model.geo.addLine(p8, p1)
loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
surf = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.setRecombine(2, surf)
gmsh.model.mesh.generate(2)

elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2, surf)
for et, tags in zip(elem_types, elem_tags):
    name, dim, order, nnodes, *_ = gmsh.model.mesh.getElementProperties(et)
    print(name, "count:", len(tags))
gmsh.finalize()

# -- cell 20 ------------------------------------------------------------------------
# `Mesh.RecombineAll = 1` was the missing option — with it set, recombination gives 872 quads, 0 trian
import subprocess, sys
from pathlib import Path

WIDTH_M   = 0.010
LEG_LEN_M = 0.120
BEND_R_M  = 0.015
HALF_W    = WIDTH_M / 2.0
BEND_CTR  = (LEG_LEN_M, BEND_R_M)

y_lo_outer = -HALF_W
y_lo_inner =  HALF_W
y_hi_inner = 2*BEND_R_M - HALF_W
y_hi_outer = 2*BEND_R_M + HALF_W

CELL_SIZE_M = WIDTH_M / 5.0   # coarse first pass: ~5 cells across the passage width
THICK_M = CELL_SIZE_M

def build_mesh(msh_path: Path, cell_size: float, thick: float) -> None:
    import gmsh
    gmsh.initialize()
    gmsh.model.add("ubend2d")

    p1 = gmsh.model.geo.addPoint(0, y_lo_outer, 0, cell_size)
    p2 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_outer, 0, cell_size)
    p3 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_outer, 0, cell_size)
    p4 = gmsh.model.geo.addPoint(0, y_hi_outer, 0, cell_size)
    p5 = gmsh.model.geo.addPoint(0, y_hi_inner, 0, cell_size)
    p6 = gmsh.model.geo.addPoint(LEG_LEN_M, y_hi_inner, 0, cell_size)
    p7 = gmsh.model.geo.addPoint(LEG_LEN_M, y_lo_inner, 0, cell_size)
    p8 = gmsh.model.geo.addPoint(0, y_lo_inner, 0, cell_size)
    pc = gmsh.model.geo.addPoint(BEND_CTR[0], BEND_CTR[1], 0, cell_size)

    l_wall_lo_outer = gmsh.model.geo.addLine(p1, p2)
    l_wall_bend_outer = gmsh.model.geo.addCircleArc(p2, pc, p3, nx=0, ny=0, nz=1)
    l_wall_hi_outer = gmsh.model.geo.addLine(p3, p4)
    l_outlet = gmsh.model.geo.addLine(p4, p5)
    l_wall_hi_inner = gmsh.model.geo.addLine(p5, p6)
    l_wall_bend_inner = gmsh.model.geo.addCircleArc(p6, pc, p7, nx=0, ny=0, nz=-1)
    l_wall_lo_inner = gmsh.model.geo.addLine(p7, p8)
    l_inlet = gmsh.model.geo.addLine(p8, p1)

    loop = gmsh.model.geo.addCurveLoop([l_wall_lo_outer, l_wall_bend_outer, l_wall_hi_outer,
                                         l_outlet, l_wall_hi_inner, l_wall_bend_inner,
                                         l_wall_lo_inner, l_inlet])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.setRecombine(2, surf)

    out = gmsh.model.geo.extrude([(2, surf)], 0, 0, thick, numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    top_surf, volume = out[0][1], out[1][1]
    (side_lo_outer, side_bend_outer, side_hi_outer, side_outlet,
     side_hi_inner, side_bend_inner, side_lo_inner, side_inlet) = (e[1] for e in out[2:10])

    wall_sides = [side_lo_outer, side_bend_outer, side_hi_outer,
                  side_hi_inner, side_bend_inner, side_lo_inner]

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(2, [side_outlet], name="outlet")
    gmsh.model.addPhysicalGroup(2, wall_sides, name="wall")
    gmsh.model.addPhysicalGroup(3, [volume], name="internal")

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(msh_path))
    gmsh.finalize()

msh_path = Path("ubend2d.msh")
build_mesh(msh_path, CELL_SIZE_M, THICK_M)
print("wrote", msh_path, msh_path.stat().st_size, "bytes")

# -- cell 21 ------------------------------------------------------------------------
# Now convert to OpenFOAM, retype patches, and run checkMesh.
import subprocess, sys
from pathlib import Path

CONTROL_DICT_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     foamRun;
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

FV_SCHEMES_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

FV_SOLUTION_STUB = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
}
"""

def write_system_stubs(case: Path) -> None:
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(CONTROL_DICT_STUB)
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES_STUB)
    (case / "system" / "fvSolution").write_text(FV_SOLUTION_STUB)

def run(cmd, **kw):
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{cmd[0]} failed (exit {proc.returncode})")
    return proc

def retype(patch: str, kind: str) -> None:
    run(["foamDictionary", "constant/polyMesh/boundary",
         "-entry", f"entry0.{patch}.type", "-set", kind])

case = Path.cwd()
write_system_stubs(case)
run(["gmshToFoam", "ubend2d.msh"])
retype("wall", "wall")
retype("frontAndBack", "empty")
run(["checkMesh", "-case", "."])

# -- cell 22 ------------------------------------------------------------------------
# Now we have 872 hexahedra, 0 prisms — "Mesh OK." Let's look at it with `mesh_look.py` to confirm the
import subprocess
r = subprocess.run(["python3", "/work/.toolbox/mesh_look.py", ".", "--out", "look_coarse.png", "--json", "look_coarse.json"],
                    capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-2000:])

# -- cell 23 ------------------------------------------------------------------------
# All patch areas and positions match expectation: inlet at (0,0), outlet at (0,0.03), wall length mat
from PIL import Image
img = Image.open("look_coarse.png")
print(img.size)
img

# -- cell 24 ------------------------------------------------------------------------
# The coarse mesh matches the requested U-bend shape well: 120 mm legs, 15 mm-radius bend, inlet/outle
import subprocess, sys
from pathlib import Path

CELL_SIZE_M = WIDTH_M / 20.0   # refine: ~20 cells across the passage width
THICK_M = CELL_SIZE_M

msh_path = Path("ubend2d.msh")
build_mesh(msh_path, CELL_SIZE_M, THICK_M)
print("wrote", msh_path, msh_path.stat().st_size, "bytes")

case = Path.cwd()
write_system_stubs(case)
run(["gmshToFoam", "ubend2d.msh"])
retype("wall", "wall")
retype("frontAndBack", "empty")
r = run(["checkMesh", "-case", "."])

# -- cell 25 ------------------------------------------------------------------------
# checkMesh passes cleanly at the finer resolution: 11,696 hexahedra, 0 prisms, max non-orthogonality 
import subprocess
r = subprocess.run(["python3", "/work/.toolbox/mesh_look.py", ".", "--out", "look_fine.png", "--json", "look_fine.json"],
                    capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-1500:])

# -- cell 26 ------------------------------------------------------------------------
from PIL import Image
img = Image.open("look_fine.png")
print(img.size)
img

# -- cell 27 ------------------------------------------------------------------------
# Let's do a final sanity listing of the case directory and confirm the mesh files are in place before
import subprocess
r = subprocess.run(["find", ".", "-maxdepth", "3", "-type", "f"], capture_output=True, text=True)
print(r.stdout)
r2 = subprocess.run(["ls", "constant/polyMesh"], capture_output=True, text=True)
print(r2.stdout)
