"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll use a structured Cartesian `blockMesh`: 5 mm cells for the coarse mesh (20×12×8 = 1,920 cells),
from pathlib import Path
import build123d as bd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# All dimensions are metres. User dimensions: cavity 0.100 x 0.060 x 0.040 m; zones are 0.020 m cubes.
L, W, H = 0.100, 0.060, 0.040
ZONE = 0.020
END_CLEARANCE = 0.005  # assumption: "near ... at one end"
Y0 = (W - ZONE) / 2
heater_min = (END_CLEARANCE, Y0, 0.0)
cooler_min = (L - END_CLEARANCE - ZONE, Y0, H - ZONE)

cavity = bd.Box(L, W, H, align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN))
heater = bd.Pos(*heater_min) * bd.Box(ZONE, ZONE, ZONE, align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN))
cooler = bd.Pos(*cooler_min) * bd.Box(ZONE, ZONE, ZONE, align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN))
bd.export_step(cavity, "cavity.step")
bd.export_step(heater, "heater_zone.step")
bd.export_step(cooler, "cooler_zone.step")

print(f"Cavity requested/measured: {L:.3f} x {W:.3f} x {H:.3f} m; volume={cavity.volume:.9f} m^3")
print(f"Heater requested/measured edge: {ZONE:.3f} m; volume={heater.volume:.9f} m^3; min={heater_min}")
print(f"Cooler requested/measured edge: {ZONE:.3f} m; volume={cooler.volume:.9f} m^3; min={cooler_min}")

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection="3d")
def add_box(origin, size, color, alpha, label):
    x,y,z=origin; dx,dy,dz=size
    p=[(x,y,z),(x+dx,y,z),(x+dx,y+dy,z),(x,y+dy,z),(x,y,z+dz),(x+dx,y,z+dz),(x+dx,y+dy,z+dz),(x,y+dy,z+dz)]
    faces=[[p[i] for i in q] for q in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=color, alpha=alpha, label=label))
add_box((0,0,0),(L,W,H),"steelblue",0.08,"cavity")
add_box(heater_min,(ZONE,)*3,"orangered",0.75,"heater")
add_box(cooler_min,(ZONE,)*3,"deepskyblue",0.75,"cooler")
ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", xlim=(0,L), ylim=(0,W), zlim=(0,H))
ax.set_box_aspect((L,W,H)); ax.view_init(23,-58); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("geometry_overview.png", dpi=160); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The coarse geometry matches the requested dimensions and the assumed placement. I’ll now create the 
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True)
os.makedirs("constant", exist_ok=True)
block_mesh_dict = r'''FoamFile
{
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 1;
vertices
(
    (0 0 0)       (0.100 0 0)       (0.100 0.060 0)       (0 0.060 0)
    (0 0 0.040)   (0.100 0 0.040)   (0.100 0.060 0.040)   (0 0.060 0.040)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 12 8) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {
        type wall;
        faces
        (
            (0 4 7 3) (1 2 6 5)
            (0 1 5 4) (3 7 6 2)
            (0 3 2 1) (4 5 6 7)
        );
    }
);
mergePatchPairs ();
'''
topo_set_dict = r'''FoamFile
{
    format ascii;
    class dictionary;
    object topoSetDict;
}
actions
(
    {
        name heaterCells;
        type cellSet;
        action new;
        source boxToCell;
        box (0.005 0.020 0.000) (0.025 0.040 0.020);
    }
    {
        name heater;
        type cellZoneSet;
        action new;
        source setToCellZone;
        set heaterCells;
    }
    {
        name coolerCells;
        type cellSet;
        action new;
        source boxToCell;
        box (0.075 0.020 0.020) (0.095 0.040 0.040);
    }
    {
        name cooler;
        type cellZoneSet;
        action new;
        source setToCellZone;
        set coolerCells;
    }
);
'''
control_dict = r'''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
'''
Path("system/blockMeshDict").write_text(block_mesh_dict)
Path("system/topoSetDict").write_text(topo_set_dict)
Path("system/controlDict").write_text(control_dict)
for cmd in (["blockMesh"], ["topoSet"]):
    r=subprocess.run(cmd, text=True, capture_output=True)
    print(f"$ {' '.join(cmd)}\n" + "\n".join(r.stdout.splitlines()[-14:]))
    if r.returncode: print(r.stderr); raise RuntimeError(cmd)

# -- cell 3 -------------------------------------------------------------------------
# The actual mesh bounds are correct to reader precision. The screenshot call failed only because this
import pyvista as pv
pv.OFF_SCREEN = True
rdr = pv.OpenFOAMReader("case.foam"); rdr.skip_zero_time = False
actual_mesh = rdr.read()["internalMesh"]
p = pv.Plotter(off_screen=True, window_size=(1000,600)); p.set_background("white")
p.add_mesh(actual_mesh, style="wireframe", color="slategray", opacity=0.22, line_width=1)
p.add_mesh(pv.Box(bounds=(0.005,0.025,0.020,0.040,0.000,0.020)), color="orangered", opacity=0.85, label="heater: 64 cells")
p.add_mesh(pv.Box(bounds=(0.075,0.095,0.020,0.040,0.020,0.040)), color="deepskyblue", opacity=0.85, label="cooler: 64 cells")
p.add_legend(bcolor="white", face="rectangle"); p.add_axes(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot="coarse_mesh.png", auto_close=True)
print(f"Rendered generated mesh: {actual_mesh.n_cells} cells, dimensions {actual_mesh.bounds.x_max-actual_mesh.bounds.x_min:.6f} x {actual_mesh.bounds.y_max-actual_mesh.bounds.y_min:.6f} x {actual_mesh.bounds.z_max-actual_mesh.bounds.z_min:.6f} m")

# -- cell 4 -------------------------------------------------------------------------
# `checkMesh` stopped while reading the mesh, before performing any checks. I’ll inspect its diagnosti
log_text = Path("checkMesh.log").read_text()
print(log_text[-5000:])

# -- cell 5 -------------------------------------------------------------------------
# The mesh is readable; OpenFOAM 2512’s extended `checkMesh` mode additionally requires placeholder nu
Path("system/fvSchemes").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default none; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path("system/fvSolution").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
validation = subprocess.run(["checkMesh","-allGeometry","-allTopology"], text=True, capture_output=True)
Path("checkMesh.log").write_text(validation.stdout + validation.stderr)
print("\n".join((validation.stdout+validation.stderr).splitlines()[-55:]))
if validation.returncode != 0 or "Failed" in validation.stdout:
    raise RuntimeError("checkMesh did not pass")
