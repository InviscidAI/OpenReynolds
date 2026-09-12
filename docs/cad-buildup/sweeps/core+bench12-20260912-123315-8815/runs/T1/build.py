"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, numpy as np
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

W, LEG, RC, TH = 0.010, 0.120, 0.015, 0.001
Ri, Ro = RC - W/2, RC + W/2
CY = RC
def P(r, th_deg):
    t = np.radians(th_deg)
    return (r*np.sin(t), CY - r*np.cos(t))

pts2d = [(-LEG,-W/2), (0.0,-W/2), (0.0,W/2), (-LEG,W/2),
         P(Ro,90), P(Ri,90), P(Ro,180), P(Ri,180),
         (-LEG, CY+Ro), (-LEG, CY+Ri)]
for i,p in enumerate(pts2d): print(i, tuple(round(v,5) for v in p))
print("Ri,Ro =", Ri, Ro, "| 90deg arc len =", round(RC*np.pi/2,5))

# -- cell 2 -------------------------------------------------------------------------
# Vertices check out. Now write `blockMeshDict` with the orderings I derived (each boundary face order
import subprocess
nW, nLeg, nBend = 6, 24, 8   # coarse first

verts = []
for z in (0.0, TH):
    for (x,y) in pts2d:
        verts.append((round(x,9)+0.0, round(y,9)+0.0, z))
vtxt = "\n".join(f"    ({v[0]:.6f} {v[1]:.6f} {v[2]:.6f})" for v in verts)

def arc(a, b, r, th):
    p = P(r, th)
    return f"    arc {a} {b} ({p[0]:.6f} {p[1]:.6f} 0.000000)\n    arc {a+10} {b+10} ({p[0]:.6f} {p[1]:.6f} {TH:.6f})"

blocks = f"""
    hex (0 1 2 3 10 11 12 13) ({nLeg} {nW} 1) simpleGrading (1 1 1)
    hex (1 4 5 2 11 14 15 12) ({nBend} {nW} 1) simpleGrading (1 1 1)
    hex (4 6 7 5 14 16 17 15) ({nBend} {nW} 1) simpleGrading (1 1 1)
    hex (6 8 9 7 16 18 19 17) ({nLeg} {nW} 1) simpleGrading (1 1 1)
"""
edges = "\n".join([arc(1,4,Ro,45), arc(2,5,Ri,45), arc(4,6,Ro,135), arc(5,7,Ri,135)])

bnd = """
    inlet  { type patch; faces ( (0 10 13 3) ); }
    outlet { type patch; faces ( (9 19 18 8) ); }
    walls  { type wall;  faces (
        (0 1 11 10) (3 13 12 2)
        (1 4 14 11) (2 12 15 5)
        (4 6 16 14) (5 15 17 7)
        (6 8 18 16) (7 17 19 9)
    ); }
    frontAndBack { type empty; faces (
        (0 3 2 1) (1 2 5 4) (4 5 7 6) (6 7 9 8)
        (10 11 12 13) (11 14 15 12) (14 16 17 15) (16 18 19 17)
    ); }
"""
head = """FoamFile{version 2.0; format ascii; class dictionary; object blockMeshDict;}
scale 1;
"""
open("system/blockMeshDict","w").write(
    head + "vertices (\n"+vtxt+"\n);\nblocks (" + blocks + ");\nedges (\n" + edges +
    "\n);\nboundary (" + bnd + ");\nmergePatchPairs ();\n")

open("system/controlDict","w").write("""FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-1500:])

# -- cell 3 -------------------------------------------------------------------------
# Mesh builds. Look at it and check it.
import subprocess, pyvista as pv
pv.start_xvfb() if False else None
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1200:])

# -- cell 4 -------------------------------------------------------------------------
print(len(r.stdout)); print(r.stdout[1500:])

# -- cell 5 -------------------------------------------------------------------------
print("RC", r.returncode); print(r.stderr[-2000:])

# -- cell 6 -------------------------------------------------------------------------
# checkMesh needs `fvSchemes`/`fvSolution`; add minimal incompressible-solve dicts and rerun.
open("system/fvSchemes","w").write("""FoamFile{version 2.0; format ascii; class dictionary; object fvSchemes;}
ddtSchemes{default Euler;} gradSchemes{default Gauss linear;}
divSchemes{default none; div(phi,U) Gauss linearUpwind grad(U);}
laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;} snGradSchemes{default corrected;}
""")
open("system/fvSolution","w").write("""FoamFile{version 2.0; format ascii; class dictionary; object fvSolution;}
solvers{ p{solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0.05;}
         U{solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;} }
PISO{nCorrectors 2; nNonOrthogonalCorrectors 1;}
""")
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Checking geometry"):])

# -- cell 7 -------------------------------------------------------------------------
# Mesh OK. Now look at it and measure the requested dimensions from the mesh itself.
import pyvista as pv, numpy as np, matplotlib.pyplot as plt
pv.OFF_SCREEN = True
m = pv.OpenFOAMReader("system/controlDict")  # placeholder; use POLY reader below
m = pv.POpenFOAMReader("case.foam") if False else None
open("case.foam","w").close()
rd = pv.POpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.time_values else None
msh = rd.read()["internalMesh"]
p = pv.Plotter(off_screen=True, window_size=(1000,500))
p.add_mesh(msh.extract_all_edges(), color="k", line_width=1)
p.view_xy(); p.camera.zoom(1.3); p.screenshot("mesh_coarse.png")
print(msh.n_cells, msh.bounds)

# -- cell 8 -------------------------------------------------------------------------
# pyvista has no GL here; draw the mesh edges with matplotlib instead.
import matplotlib.pyplot as plt, numpy as np
ed = msh.extract_all_edges()
pts = ed.points; lines = ed.lines.reshape(-1,3)[:,1:]
fig, ax = plt.subplots(figsize=(11,4))
for a,b in lines:
    if pts[a][2] < 1e-9 and pts[b][2] < 1e-9:
        ax.plot([pts[a][0],pts[b][0]],[pts[a][1],pts[b][1]], 'k-', lw=0.6)
ax.set_aspect('equal'); ax.set_title("U-bend, coarse (384 cells), z=0 plane")
fig

# -- cell 9 -------------------------------------------------------------------------
# Shape is right. Measure the requested numbers from the mesh boundary patches before refining.
bl = rd.read()["boundary"]
def patch(n): return bl[n]
wp = patch("walls").points
inl, out = patch("inlet"), patch("outlet")
C = np.array([0.0, 0.015])
r = np.hypot(wp[:,0]-C[0], wp[:,1]-C[1])
bend = wp[wp[:,0] > 1e-6]
rb = np.hypot(bend[:,0]-C[0], bend[:,1]-C[1])
print("bend wall radii: inner", round(rb[rb<0.015].min(),5), round(rb[rb<0.015].max(),5),
      "| outer", round(rb[rb>0.015].min(),5), round(rb[rb>0.015].max(),5))
print("centreline radius =", round(0.5*(rb[rb<0.015].mean()+rb[rb>0.015].mean()),5), "target 0.015")
print("passage width from radii =", round(rb[rb>0.015].mean()-rb[rb<0.015].mean(),5), "target 0.010")
print("inlet area", inl.area, "=> width", round(inl.area/0.001,5), " y-range", inl.bounds[2:4])
print("outlet area", out.area, " y-range", out.bounds[2:4])
print("lower leg length: x from", wp[:,0].min(), "to bend start 0.0 =>", round(0-wp[:,0].min(),4), "target 0.120")

# -- cell 10 ------------------------------------------------------------------------
# Numbers match the request exactly. Refine to the production resolution (0.5 mm across the 10 mm pass
nW, nLeg, nBend = 20, 96, 24   # 0.5 mm across passage, ~1.25 mm along legs, ~1.0 mm along bend
blocks = f"""
    hex (0 1 2 3 10 11 12 13) ({nLeg} {nW} 1) simpleGrading (1 1 1)
    hex (1 4 5 2 11 14 15 12) ({nBend} {nW} 1) simpleGrading (1 1 1)
    hex (4 6 7 5 14 16 17 15) ({nBend} {nW} 1) simpleGrading (1 1 1)
    hex (6 8 9 7 16 18 19 17) ({nLeg} {nW} 1) simpleGrading (1 1 1)
"""
open("system/blockMeshDict","w").write(
    head + "vertices (\n"+vtxt+"\n);\nblocks (" + blocks + ");\nedges (\n" + edges +
    "\n);\nboundary (" + bnd + ");\nmergePatchPairs ();\n")
rb_ = subprocess.run(["blockMesh"], capture_output=True, text=True)
rc_ = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(rb_.stdout[rb_.stdout.find("Mesh Information"):])
print(rc_.stdout[rc_.stdout.find("Checking geometry"):])

# -- cell 11 ------------------------------------------------------------------------
# Final look at the refined mesh, with the analytic fluid volume as a cross-check.
rd2 = pv.POpenFOAMReader("case.foam"); msh2 = rd2.read()["internalMesh"]
ed2 = msh2.extract_all_edges(); pts = ed2.points; lines = ed2.lines.reshape(-1,3)[:,1:]
fig, ax = plt.subplots(figsize=(11,4))
for a,b in lines:
    if pts[a][2] < 1e-9 and pts[b][2] < 1e-9:
        ax.plot([pts[a][0],pts[b][0]],[pts[a][1],pts[b][1]], 'k-', lw=0.25)
ax.set_aspect('equal'); ax.set_title(f"U-bend, {msh2.n_cells} cells")
V = (2*LEG*W + W*np.pi*RC)*TH
print("volume analytic", round(V,10), "mesh", round(msh2.volume,10), "rel diff", round(abs(V-msh2.volume)/V,5))
fig
