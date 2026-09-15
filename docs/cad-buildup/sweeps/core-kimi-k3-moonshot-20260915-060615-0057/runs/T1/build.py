"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The shape: a planar U-duct — two 120 mm horizontal legs, 10 mm apart-centreline-offset 30 mm, joined
# Coarse first: 60×10 in the legs, 10 radial × 30 around in the bend (~2700 cells). All dimensions in 
import os, subprocess, textwrap

# --- Parameters (metres) ---
W  = 0.010   # passage width
L  = 0.120   # leg length
Rc = 0.015   # bend centreline radius
Ri, Ro = Rc - W/2, Rc + W/2          # 0.010 / 0.020
t  = 0.001   # z thickness (one cell, 2D)
# coarse cell counts
nx_leg, ny = 60, 10      # leg: along x, across width
nr, nth  = 10, 30        # bend: radial, circumferential

os.makedirs("system", exist_ok=True)

controlDict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     none;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
"""
open("system/controlDict","w").write(controlDict)

def p(x, y, z): return f"    ({x} {y} {z})"

# vertex list, z=0 then z=t
xy = [(-L,-Ro),(0,-Ro),(0,-Ri),(-L,-Ri),(0,Ri),(0,Ro),(-L,Ro),(-L,Ri)]
verts = [p(x,y,0) for (x,y) in xy] + [p(x,y,t) for (x,y) in xy]

bmd = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

vertices
(
{chr(10).join(verts)}
);

blocks
(
    hex (0 1 2 3 8 9 10 11)      ({nx_leg} {ny} 1) simpleGrading (1 1 1)   // lower leg
    hex (2 1 5 4 10 9 13 12)     ({nr} {nth} 1) simpleGrading (1 1 1)      // 180 deg bend
    hex (7 4 5 6 15 12 13 14)    ({nx_leg} {ny} 1) simpleGrading (1 1 1)   // upper leg
);

edges
(
    arc 1 5  ({Ro} 0 0)
    arc 4 2  ({Ri} 0 0)
    arc 9 13 ({Ro} 0 {t})
    arc 12 10 ({Ri} 0 {t})
);

boundary
(
    inlet
    {{
        type patch;
        faces ( (0 8 11 3) );
    }}
    outlet
    {{
        type patch;
        faces ( (7 15 14 6) );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 9 8)      // lower leg bottom
            (6 5 13 14)    // upper leg top
            (1 5 13 9)     // bend outer
            (2 4 12 10)    // bend inner
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1) (2 4 5 1) (7 6 5 4)
            (8 9 10 11) (10 9 13 12) (15 12 13 14)
        );
    }}
);
"""
open("system/blockMeshDict","w").write(bmd)

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-400:])
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1600:])

# -- cell 2 -------------------------------------------------------------------------
# The 180° bend as a single block is degenerate — its flat parametric faces collapse (all four corner 
import numpy as np

# extra vertices at the 0-degree section (angle 0, +x side)
xy = [(-L,-Ro),(0,-Ro),(0,-Ri),(-L,-Ri),(0,Ri),(0,Ro),(-L,Ro),(-L,Ri),(Ro,0),(Ri,0)]
verts = [p(x,y,0) for (x,y) in xy] + [p(x,y,t) for (x,y) in xy]

c, s = np.cos(np.pi/4), np.sin(np.pi/4)   # arc mid-points at +/-45 deg

bmd = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

vertices
(
{chr(10).join(verts)}
);

blocks
(
    hex (0 1 2 3 10 11 12 13)     ({nx_leg} {ny} 1) simpleGrading (1 1 1)      // lower leg
    hex (2 1 8 9 12 11 18 19)     ({nr} {nth//2} 1) simpleGrading (1 1 1)      // bend -90..0 deg
    hex (9 8 5 4 19 18 15 14)     ({nr} {nth//2} 1) simpleGrading (1 1 1)      // bend 0..90 deg
    hex (7 4 5 6 17 14 15 16)     ({nx_leg} {ny} 1) simpleGrading (1 1 1)      // upper leg
);

edges
(
    arc 1 8  ({Ro*c} {-Ro*s} 0)
    arc 9 2  ({Ri*c} {-Ri*s} 0)
    arc 8 5  ({Ro*c} {Ro*s} 0)
    arc 4 9  ({Ri*c} {Ri*s} 0)
    arc 11 18 ({Ro*c} {-Ro*s} {t})
    arc 19 12 ({Ri*c} {-Ri*s} {t})
    arc 18 15 ({Ro*c} {Ro*s} {t})
    arc 14 19 ({Ri*c} {Ri*s} {t})
);

boundary
(
    inlet
    {{
        type patch;
        faces ( (0 10 13 3) );
    }}
    outlet
    {{
        type patch;
        faces ( (7 17 16 6) );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 11 10)     // lower leg bottom
            (6 5 15 16)     // upper leg top
            (1 8 18 11)     // bend outer, lower half
            (8 5 15 18)     // bend outer, upper half
            (2 9 19 12)     // bend inner, lower half
            (9 4 14 19)     // bend inner, upper half
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1) (2 9 8 1) (9 4 5 8) (7 6 5 4)
            (10 11 12 13) (12 11 18 19) (19 18 15 14) (17 14 15 16)
        );
    }}
);
"""
open("system/blockMeshDict","w").write(bmd)

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc:", r.returncode, r.stderr[-300:])
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1800:])

# -- cell 3 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print(out[out.find("Create mesh"):])
print("rc:", r.returncode)

# -- cell 4 -------------------------------------------------------------------------
print("STDOUT tail:\n", r.stdout[-3000:])
print("STDERR:\n", r.stderr[-2000:])
import os
print(os.listdir("constant/polyMesh") if os.path.exists("constant/polyMesh") else "no polyMesh")

# -- cell 5 -------------------------------------------------------------------------
open("system/fvSchemes","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes  { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes  { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open("system/fvSolution","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {}
""")

r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print(out[out.find("Create mesh"):])
print("rc:", r.returncode)

# -- cell 6 -------------------------------------------------------------------------
bmd = bmd.replace(
"""            (2 9 19 12)     // bend inner, lower half
            (9 4 14 19)     // bend inner, upper half""",
"""            (2 9 19 12)     // bend inner, lower half
            (9 4 14 19)     // bend inner, upper half
            (3 2 12 13)     // lower leg inner wall (y=-Ri)
            (7 4 14 17)     // upper leg inner wall (y=+Ri)""")
open("system/blockMeshDict","w").write(bmd)

subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
seg = out[out.find("Mesh stats"):]
# print key lines
import re
for line in seg.splitlines():
    if any(k in line for k in ["cells:", "directions", "not aligned", "OK", "Failed", "Mesh OK", "Patch ", "inlet", "outlet", "walls", "frontAndBack", "defaultFaces", "aspect", "non-orthogonality Max", "skewness"]):
        print(line)
print("rc:", r.returncode)

# -- cell 7 -------------------------------------------------------------------------
import glob, pyvista as pv

m  = pv.read("VTK/t1_0/internal.vtu")
pi = pv.read("VTK/t1_0/boundary/inlet.vtp").points
po = pv.read("VTK/t1_0/boundary/outlet.vtp").points
pw = pv.read("VTK/t1_0/boundary/walls.vtp").points

b = m.bounds
print(f"bounds: x [{b[0]:.4f},{b[1]:.4f}]  y [{b[2]:.4f},{b[3]:.4f}]  z [{b[4]:.4f},{b[5]:.4f}]")
print(f"cells = {m.n_cells}")

print(f"inlet width  = {(pi[:,1].max()-pi[:,1].min())*1000:.3f} mm  (asked 10)   at x = {pi[:,0].mean():.4f} m")
print(f"outlet width = {(po[:,1].max()-po[:,1].min())*1000:.3f} mm  (asked 10)   at x = {po[:,0].mean():.4f} m")
print(f"leg length (open end to bend tangent x=0) = {(0-pi[:,0].mean())*1000:.3f} mm (asked 120)")
print(f"leg centreline offset = {(po[:,1].mean()-pi[:,1].mean())*1000:.3f} mm (asked 2*Rc = 30)")

bw = pw[pw[:,0] > 1e-6]
rb = np.hypot(bw[:,0], bw[:,1])
print(f"bend inner radius = {rb.min()*1000:.3f} mm (asked {Ri*1000:.1f})")
print(f"bend outer radius = {rb.max()*1000:.3f} mm (asked {Ro*1000:.1f})")
print(f"bend centreline radius = {(rb.min()+rb.max())/2*1000:.3f} mm (asked {Rc*1000:.1f})")

pl = pv.Plotter(off_screen=True, window_size=(1100,620))
pl.add_mesh(m, style="wireframe", color="k", line_width=0.7)
pl.view_xy(); pl.camera.zoom(1.05)
pl.add_title(f"U-bend: 2D structured hex mesh, {m.n_cells} cells (coarse)")
pl.show()

# -- cell 8 -------------------------------------------------------------------------
nx_leg, ny = 240, 40     # leg: 0.5 mm streamwise, 0.25 mm across
nr, nth    = 40, 144     # bend: 0.25 mm radial, ~0.33 mm circumferential
# regenerate dict with the refined counts (same vertices/edges/boundary text)
import re
bmd_r = re.sub(r"\(\d+ \d+ 1\) simpleGrading",
               lambda mo, it=iter([(nx_leg,ny),(nr,nth//2),(nr,nth//2),(nx_leg,ny)]):
               f"({next(it)[0]} {ny if False else ''}", bmd)  # placeholder, do it plainly below
# plain rebuild: just patch the four block lines
lines = bmd.splitlines()
blocks_new = [
    f"    hex (0 1 2 3 10 11 12 13)     ({nx_leg} {ny} 1) simpleGrading (1 1 1)      // lower leg",
    f"    hex (2 1 8 9 12 11 18 19)     ({nr} {nth//2} 1) simpleGrading (1 1 1)      // bend -90..0 deg",
    f"    hex (9 8 5 4 19 18 15 14)     ({nr} {nth//2} 1) simpleGrading (1 1 1)      // bend 0..90 deg",
    f"    hex (7 4 5 6 17 14 15 16)     ({nx_leg} {ny} 1) simpleGrading (1 1 1)      // upper leg",
]
out_lines, bi = [], 0
for ln in lines:
    if ln.strip().startswith("hex"):
        out_lines.append(blocks_new[bi]); bi += 1
    else:
        out_lines.append(ln)
bmd = "\n".join(out_lines) + "\n"
open("system/blockMeshDict","w").write(bmd)

subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
seg = r.stdout[r.stdout.find("Mesh stats"):]
for line in seg.splitlines():
    if any(k in line for k in ["cells:", "directions", "not aligned", "Failed", "Mesh OK",
                               "aspect", "non-orthogonality Max", "skewness", "Min volume"]):
        print(line)
print("checkMesh rc:", r.returncode)

# -- cell 9 -------------------------------------------------------------------------
import subprocess, pyvista as pv
subprocess.run(["foamToVTK"], capture_output=True, text=True)
m = pv.read("VTK/t1_0/internal.vtu")
print("cells:", m.n_cells)
pl = pv.Plotter(off_screen=True, window_size=(1100,620))
pl.add_mesh(m, style="wireframe", color="k", line_width=0.4)
pl.view_xy(); pl.camera.zoom(1.02)
pl.add_title(f"U-bend final mesh: {m.n_cells} cells, hex, 2D (empty front/back)")
pl.show()
