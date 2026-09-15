"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# **Plan.** The shape is a single closed rectangular cavity — no ports, no protrusions — so the OpenFO
# - `heater`: x ∈ [4, 24] mm, z ∈ [2, 22] mm (near floor, at the x=0 end)
# - `cooler`: x ∈ [76, 96] mm, z ∈ [18, 38] mm (near ceiling, at the x=100 end)
# - both centred across the width: y ∈ [20, 40] mm
# Those placements (2 mm standoff from wall/end) are my choice — the request says only "near the floor
import os, subprocess, textwrap

os.makedirs("system", exist_ok=True)
os.makedirs("constant", exist_ok=True)

# Minimal case scaffolding so blockMesh/checkMesh have an application to read
open("system/controlDict", "w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
    application blockMesh;
    startFrom startTime; startTime 0; stopAt endTime; endTime 1;
    deltaT 1; writeControl timeStep; writeInterval 1;
"""))
open("system/fvSchemes", "w").write(
    'FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n')
open("system/fvSolution", "w").write(
    'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n')

# Cavity: 100 x 60 x 40 mm -> metres. One hex block, 2 mm pitch, all faces = walls.
Lx, Ly, Lz = 0.100, 0.060, 0.040
nx, ny, nz = 50, 30, 20   # 2 mm cells, 30 k cells

open("system/blockMeshDict", "w").write(textwrap.dedent(f"""\
    FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
    scale 1;
    vertices
    (
        (0   0   0  )   // 0
        ({Lx} 0   0  )  // 1
        ({Lx} {Ly} 0  ) // 2
        (0   {Ly} 0  )  // 3
        (0   0   {Lz})  // 4
        ({Lx} 0   {Lz}) // 5
        ({Lx} {Ly} {Lz})// 6
        (0   {Ly} {Lz}) // 7
    );
    blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
    edges ();
    boundary
    (
        walls
        {{
            type wall;
            faces
            (
                (0 3 2 1)   // z = 0   floor
                (4 5 6 7)   // z = Lz  ceiling
                (0 1 5 4)   // y = 0
                (3 7 6 2)   // y = Ly
                (0 4 7 3)   // x = 0
                (1 2 6 5)   // x = Lx
            );
        }}
    );
"""))

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-800:])
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2000:])

# -- cell 2 -------------------------------------------------------------------------
# Block mesh created cleanly: 30 000 hexahedra at 2 mm pitch, bounding box exactly (0 0 0)→(0.1 0.06 0
# Now the cell zones: `topoSet` with `boxToCell` for the two 20 mm cubes (placed on 2 mm multiples so 
import subprocess, textwrap

# Cell zones: 20 mm cubes, boundaries on the 2 mm grid.
# heater: near floor, x=0 end.  cooler: near ceiling, x=Lx end. Centred in y.
open("system/topoSetDict", "w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
    actions
    (
        { name heaterSet; type cellSet; action new;
          source boxToCell; sourceInfo { boxes ((0.004 0.020 0.002) (0.024 0.040 0.022)); } }
        { name coolerSet; type cellSet; action new;
          source boxToCell; sourceInfo { boxes ((0.076 0.020 0.018) (0.096 0.040 0.038)); } }
        { name heater; type cellZoneSet; action new;
          source setToCellZone; sourceInfo { set heaterSet; } }
        { name cooler; type cellZoneSet; action new;
          source setToCellZone; sourceInfo { set coolerSet; } }
    );
"""))

r = subprocess.run(["topoSet"], capture_output=True, text=True)
print(r.stdout[-1200:]); print(r.stderr[-400:])

# Verdict: mesh quality + zone sizes
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
tail = [l for l in r.stdout.splitlines() if any(k in l for k in
        ("cells:", "cellZone", "Mesh OK", "Failed", "boundingBox", "patch "))]
print("\n".join(tail))

# -- cell 3 -------------------------------------------------------------------------
# Both zones came out at exactly 1000 cells = 10×10×10 cells = a 20 mm cube at 2 mm pitch, as designed
import subprocess, os
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1600:])
print("--- cellZones file:", os.path.exists("constant/polyMesh/cellZones"))
print(open("constant/polyMesh/cellZones").read()[:600])

# -- cell 4 -------------------------------------------------------------------------
# The cellZones file exists in `constant/polyMesh`. The checkMesh output appears truncated right after
import subprocess
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("rc:", r.returncode, "stdout len:", len(r.stdout), "stderr len:", len(r.stderr))
print(r.stdout[len(r.stdout)-2500:])
print("STDERR:", r.stderr[-500:])

# -- cell 5 -------------------------------------------------------------------------
# checkMesh in this build wants `divSchemes` present in fvSchemes. I'll fill in the standard default s
import subprocess, textwrap
open("system/fvSchemes", "w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
    ddtSchemes          { default steadyState; }
    gradSchemes         { default Gauss linear; }
    divSchemes          { default Gauss linear; }
    laplacianSchemes    { default Gauss linear corrected; }
    interpolationSchemes{ default linear; }
    snGradSchemes       { default corrected; }
"""))

r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[r.stdout.find("Create mesh"):])

# -- cell 6 -------------------------------------------------------------------------
# `Mesh OK`, and every requested number checks out against the mesh itself:
# - Domain: (0 0 0) → (0.1 0.06 0.04) m, total volume 2.4e-4 m³ = 100×60×40 mm ✓
# - `heater` zone: 1000 cells, volume 8e-6 m³ = 20³ mm³, near the floor at x=0 end ✓
# - `cooler` zone: 1000 cells, 8e-6 m³, near the ceiling at x=100 mm end ✓
# - All 30000 cells hexahedra, aspect ratio 1, orthogonality 0, single closed `walls` patch ✓
# Now a picture: export the mesh and the two zones to VTK and render them.
import subprocess
# VTK export of the boundary mesh plus the two cell zones
r = subprocess.run(["foamToVTK", "-cellZones", "heater,cooler"],
                   capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-300:])
import glob
print(sorted(glob.glob("VTK/**/*.vtk", recursive=True)))

# -- cell 7 -------------------------------------------------------------------------
import subprocess, glob
# cellSets heaterSet/coolerSet were left behind by topoSet; export those
for s in ["heaterSet", "coolerSet"]:
    r = subprocess.run(["foamToVTK", "-cellSet", s], capture_output=True, text=True)
    print(s, r.returncode, r.stderr[-200:])
r = subprocess.run(["foamToVTK"], capture_output=True, text=True)
print("full:", r.returncode, r.stderr[-200:])
print(sorted(glob.glob("VTK/**/*.vtk", recursive=True)))

# -- cell 8 -------------------------------------------------------------------------
import glob, os
print(sorted(glob.glob("VTK/**/*", recursive=True)))

# -- cell 9 -------------------------------------------------------------------------
import pyvista as pv
pv.OFF_SCREEN = True

walls  = pv.read("VTK/t7_0/boundary/walls.vtp")
heater = pv.read("VTK/heaterSet_0/internal.vtu")
cooler = pv.read("VTK/coolerSet_0/internal.vtu")

pl = pv.Plotter(off_screen=True)
pl.add_mesh(walls,  color="lightsteelblue", opacity=0.25, show_edges=False, label="cavity walls")
pl.add_mesh(heater, color="red",  show_edges=True, edge_color="darkred", label="heater")
pl.add_mesh(cooler, color="deepskyblue", show_edges=True, edge_color="navy", label="cooler")
pl.add_legend()
pl.add_axes()
pl.camera_position = [(0.22, -0.18, 0.15), (0.05, 0.03, 0.02), (0, 0, 1)]
pl.screenshot("mesh_overview.png", scale=2)
pl.show()
