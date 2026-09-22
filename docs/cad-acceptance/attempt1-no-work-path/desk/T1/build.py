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
# Inspect build123d API for arc construction helpers before using them
r = subprocess.run(["grep", "-n", "-A", "3", "-B", "1", "CenterArc\\|RadiusArc",
                     "/work/.toolbox/b123d_api.md"], capture_output=True, text=True)
print(r.stdout[:4000])
print("----STDERR----")
print(r.stderr[:2000])

# -- cell 2 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["python3", "/work/.toolbox/b123d_api.py"], capture_output=True, text=True)
print(r.stdout[-3000:])
print("STDERR", r.stderr[-2000:])

r2 = subprocess.run(["grep", "-n", "-A", "3", "-B", "1", "CenterArc\\|RadiusArc",
                     "/work/.toolbox/b123d_api.md"], capture_output=True, text=True)
print(r2.stdout[:4000])
print("STDERR2", r2.stderr[:2000])

# -- cell 3 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["ls", "-la", "/work/.toolbox/"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)

# -- cell 4 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["find", "/", "-iname", "*b123d_api*", "-o", "-iname", "mesh_look.py"],
                    capture_output=True, text=True, timeout=60)
print(r.stdout)
print(r.stderr[:2000])

# -- cell 5 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["ls", "-la", "/work/"], capture_output=True, text=True)
print(r.stdout)
print(r.stderr)
r2 = subprocess.run(["pwd"], capture_output=True, text=True)
print(r2.stdout)

# -- cell 6 -------------------------------------------------------------------------
import subprocess, os
print(os.getcwd())
r = subprocess.run(["ls", "-la", "../.toolbox"], capture_output=True, text=True)
print(r.stdout, r.stderr)
r2 = subprocess.run(["ls", "-la", "."], capture_output=True, text=True)
print(r2.stdout, r2.stderr)

# -- cell 7 -------------------------------------------------------------------------
import subprocess
TOOLBOX = "../.toolbox"
r = subprocess.run(["grep", "-n", "-A", "4", "-B", "1", "CenterArc\\|RadiusArc",
                     f"{TOOLBOX}/b123d_api.md"], capture_output=True, text=True)
print(r.stdout[:4000])
print(r.stderr[:1000])

# -- cell 8 -------------------------------------------------------------------------
# Good, `CenterArc(center, radius, start_angle, arc_size)` confirmed. Now let's build the U-bend duct 
import math
from build123d import *

# --- Parameters (metres) ---
W_M       = 0.010          # passage width
HALF_W_M  = W_M / 2.0
LEG_L_M   = 0.120          # horizontal leg length (open-end to bend start)
R_C_M     = 0.015          # bend centreline radius
THICK_M   = 0.002          # extrusion thickness in Z ("2D" -> one cell thick)

R_IN_M  = R_C_M - HALF_W_M   # 0.010
R_OUT_M = R_C_M + HALF_W_M   # 0.020

LOWER_OUTER_Y = -HALF_W_M                # -0.005
LOWER_INNER_Y =  HALF_W_M                #  0.005
UPPER_INNER_Y =  2*R_C_M - HALF_W_M      #  0.025
UPPER_OUTER_Y =  2*R_C_M + HALF_W_M      #  0.035
BEND_CENTER   = (LEG_L_M, R_C_M)         # (0.120, 0.015)

A = (0.0,      LOWER_OUTER_Y)
B = (LEG_L_M,  LOWER_OUTER_Y)
C = (LEG_L_M,  UPPER_OUTER_Y)
D = (0.0,      UPPER_OUTER_Y)
E = (0.0,      UPPER_INNER_Y)
F = (LEG_L_M,  UPPER_INNER_Y)
G = (LEG_L_M,  LOWER_INNER_Y)
H = (0.0,      LOWER_INNER_Y)

e_lower_outer = Line(A, B)
e_bend_outer  = CenterArc(BEND_CENTER, R_OUT_M, -90, 180)   # B -> C, bulging to +x
e_upper_outer = Line(C, D)
e_outlet_cap  = Line(D, E)
e_upper_inner = Line(E, F)
e_bend_inner  = CenterArc(BEND_CENTER, R_IN_M, 90, -180)    # F -> G, bulging to +x
e_lower_inner = Line(F if False else G, H)  # placeholder, fixed below

# fix ordering explicitly (avoid the placeholder above being wrong)
e_lower_inner = Line(G, H)
e_inlet_cap   = Line(H, A)

wire = (e_lower_outer + e_bend_outer + e_upper_outer + e_outlet_cap +
        e_upper_inner + e_bend_inner + e_lower_inner + e_inlet_cap)

duct_face = make_face(wire)
duct_solid = extrude(duct_face, amount=THICK_M/2.0, both=True)

print("face area (m^2):", duct_face.area, " expected approx:", (2*LEG_L_M + math.pi*R_C_M) * W_M)
print("solid volume (m^3):", duct_solid.volume, " expected approx:", (2*LEG_L_M + math.pi*R_C_M) * W_M * THICK_M)
bb = duct_solid.bounding_box()
print("bbox min", bb.min, "max", bb.max)

# -- cell 9 -------------------------------------------------------------------------
# Area and volume match the expected values closely, and the bounding box (0.14 × 0.04 × 0.002 m) is c
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
from build123d import export_stl

export_stl(duct_solid, "look_duct.stl")
mesh = pv.read("look_duct.stl")

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.view_xy()
plotter.camera.parallel_projection = True
plotter.screenshot("look_duct.png")
print("saved look_duct.png")

# -- cell 10 ------------------------------------------------------------------------
import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.2")
import pyvista as pv
pv.OFF_SCREEN = True

mesh = pv.read("look_duct.stl")
plotter = pv.Plotter(off_screen=True, window_size=(900, 500))
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.view_xy()
plotter.camera.parallel_projection = True
plotter.screenshot("look_duct.png")
plotter.close()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread("look_duct.png")
plt.figure(figsize=(9,5))
plt.imshow(img)
plt.axis("off")
plt.title("U-bend duct (look)")
plt.show()

# -- cell 11 ------------------------------------------------------------------------
# The shape is exactly right: two 120 mm horizontal legs, 10 mm passage width, joined by a smooth 180°
from build123d import Axis, GeomType

faces = duct_solid.faces()

# front/back: planar faces normal to Z
front_faces = faces.filter_by(Axis.Z)
print("front/back candidate count:", len(front_faces))
for f in front_faces:
    print("  z-center:", f.center().Z, "area:", f.area)

# planar faces normal to X at x=0 (the two caps: inlet + outlet)
x0_faces = faces.filter_by(Axis.X).filter_by(lambda f: abs(f.center().X) < 1e-9)
print("x=0 candidate count:", len(x0_faces))
for f in x0_faces:
    print("  y-center:", f.center().Y, "area:", f.area)

# remaining faces -> walls (curved + straight side walls)
wall_faces = faces - front_faces - x0_faces
print("wall candidate count:", len(wall_faces))
for f in wall_faces:
    print("  geom_type:", f.geom_type, "center:", f.center(), "area:", f.area)

# -- cell 12 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["python3", "../.toolbox/cad_convert.py", "--help"], capture_output=True, text=True)
print(r.stdout[:5000])
print(r.stderr[:2000])

# -- cell 13 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-n", "def export_patches\\|^def \\|^class ", "../.toolbox/cad_convert.py"],
                    capture_output=True, text=True)
print(r.stdout[:3000])

# -- cell 14 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["sed", "-n", "511,600p", "../.toolbox/cad_convert.py"], capture_output=True, text=True)
print(r.stdout)

# -- cell 15 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-n", "-B2", "-A15", "^def export_step\\|^    def export_step", 
                     "../.toolbox/b123d_api.md"], capture_output=True, text=True)
print(r.stdout[:3000])

# -- cell 16 ------------------------------------------------------------------------
from build123d import export_step
help(export_step)

# -- cell 17 ------------------------------------------------------------------------
from build123d import Unit
print(list(Unit))

# -- cell 18 ------------------------------------------------------------------------
from build123d import export_step

STEP_PATH = "duct.step"
ok = export_step(duct_solid, STEP_PATH, unit=Unit.M)
print("export ok:", ok)

# sanity check: re-import and compare bounding box to what we measured directly (metres)
from build123d import import_step
reimported = import_step(STEP_PATH)
bb = reimported.bounding_box()
print("reimported bbox min", bb.min, "max", bb.max)

# -- cell 19 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["python3", "-c",
    "import sys; sys.path.insert(0,'../.toolbox'); import cad_convert; "
    "print(cad_convert.declared_unit('duct.step'))"],
    capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-2000:])

# -- cell 20 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["python3", "-c",
    "import sys; from pathlib import Path; sys.path.insert(0,'../.toolbox'); import cad_convert; "
    "print(cad_convert.declared_unit(Path('duct.step')))"],
    capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-2000:])

# -- cell 21 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-m", "5", "-i", "CARTESIAN_POINT", "duct.step"], capture_output=True, text=True)
print(r.stdout)
r2 = subprocess.run(["grep", "-i", "SI_UNIT\\|CONVERSION_BASED_UNIT\\|LENGTH_UNIT", "duct.step"], capture_output=True, text=True)
print(r2.stdout)

# -- cell 22 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep", "-n", "PATCH_ROLES", "../.toolbox/cad_convert.py"], capture_output=True, text=True)
print(r.stdout[:2000])

# -- cell 23 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["cat", "../.toolbox/templates/snappy/README.md"], capture_output=True, text=True)
print(r.stdout)

# -- cell 24 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["find", "../.toolbox/templates", "-maxdepth", "3"], capture_output=True, text=True)
print(r.stdout)
r2 = subprocess.run(["find", "/", "-path", "*/tests/data/snappy/duct*", "-not", "-path", "*/node_modules/*"],
                     capture_output=True, text=True, timeout=30)
print(r2.stdout[:3000])
print(r2.stderr[-1500:])

# -- cell 25 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["cat", "../.toolbox/templates/duct2d.py"], capture_output=True, text=True)
print(r.stdout)

# -- cell 26 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["sed", "-n", "1,140p", "../.toolbox/templates/duct2d.py"], capture_output=True, text=True)
print(r.stdout)

# -- cell 27 ------------------------------------------------------------------------
import subprocess, sys
from pathlib import Path

# --- Parameters (metres) --- coarse first pass
W_M      = 0.010     # passage width
HALF_W_M = W_M / 2.0
LEG_L_M  = 0.120     # horizontal leg length
R_C_M    = 0.015     # bend centreline radius

CELL_SIZE_M = W_M / 5.0     # coarse: ~5 cells across the passage width
THICK_M     = CELL_SIZE_M   # one cell thick, roughly cubic cells

R_IN_M  = R_C_M - HALF_W_M
R_OUT_M = R_C_M + HALF_W_M
LOWER_OUTER_Y = -HALF_W_M
LOWER_INNER_Y =  HALF_W_M
UPPER_INNER_Y =  2*R_C_M - HALF_W_M
UPPER_OUTER_Y =  2*R_C_M + HALF_W_M

MSH_PATH = Path("duct2d.msh")


def build_mesh(msh_path: Path) -> None:
    import gmsh
    gmsh.initialize()
    gmsh.model.add("ubend2d")

    cs = CELL_SIZE_M
    pA = gmsh.model.geo.addPoint(0.0,      LOWER_OUTER_Y, 0, cs)
    pB = gmsh.model.geo.addPoint(LEG_L_M,  LOWER_OUTER_Y, 0, cs)
    pC = gmsh.model.geo.addPoint(LEG_L_M,  UPPER_OUTER_Y, 0, cs)
    pD = gmsh.model.geo.addPoint(0.0,      UPPER_OUTER_Y, 0, cs)
    pE = gmsh.model.geo.addPoint(0.0,      UPPER_INNER_Y, 0, cs)
    pF = gmsh.model.geo.addPoint(LEG_L_M,  UPPER_INNER_Y, 0, cs)
    pG = gmsh.model.geo.addPoint(LEG_L_M,  LOWER_INNER_Y, 0, cs)
    pH = gmsh.model.geo.addPoint(0.0,      LOWER_INNER_Y, 0, cs)
    pO = gmsh.model.geo.addPoint(LEG_L_M,  R_C_M,         0, cs)  # bend centre

    l_lower_outer = gmsh.model.geo.addLine(pA, pB)
    l_bend_outer  = gmsh.model.geo.addCircleArc(pB, pO, pC)
    l_upper_outer = gmsh.model.geo.addLine(pC, pD)
    l_outlet      = gmsh.model.geo.addLine(pD, pE)
    l_upper_inner = gmsh.model.geo.addLine(pE, pF)
    l_bend_inner  = gmsh.model.geo.addCircleArc(pF, pO, pG)
    l_lower_inner = gmsh.model.geo.addLine(pG, pH)
    l_inlet       = gmsh.model.geo.addLine(pH, pA)

    loop = gmsh.model.geo.addCurveLoop([l_lower_outer, l_bend_outer, l_upper_outer,
                                         l_outlet, l_upper_inner, l_bend_inner,
                                         l_lower_inner, l_inlet])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.model.mesh.setRecombine(2, surf)

    out = gmsh.model.geo.extrude([(2, surf)], 0, 0, THICK_M,
                                  numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    top_surf, volume = out[0][1], out[1][1]
    (side_lower_outer, side_bend_outer, side_upper_outer, side_outlet,
     side_upper_inner, side_bend_inner, side_lower_inner, side_inlet) = (
        e[1] for e in out[2:10])

    gmsh.model.addPhysicalGroup(2, [surf, top_surf], name="frontAndBack")
    gmsh.model.addPhysicalGroup(2, [side_inlet], name="inlet")
    gmsh.model.addPhysicalGroup(2, [side_outlet], name="outlet")
    gmsh.model.addPhysicalGroup(2, [side_lower_outer, side_bend_outer, side_upper_outer,
                                     side_upper_inner, side_bend_inner, side_lower_inner],
                                 name="wall")
    gmsh.model.addPhysicalGroup(3, [volume], name="internal")

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(msh_path))
    gmsh.finalize()


build_mesh(MSH_PATH)
print("wrote", MSH_PATH, MSH_PATH.stat().st_size, "bytes")
