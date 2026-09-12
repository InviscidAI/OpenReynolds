"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll build a plane U-bend: lower leg centreline at y=0, upper at y=2·Rc=30 mm, bend centred at (0, 1
import numpy as np, matplotlib.pyplot as plt
from build123d import *

# --- parameters (metres) ---
W   = 0.010    # passage width
L   = 0.120    # straight leg length
Rc  = 0.015    # bend centreline radius
THK = 0.001    # spanwise thickness (one cell, 2D)
Ri, Ro = Rc - W/2, Rc + W/2

lower = Pos(-L/2, 0.0)    * Rectangle(L, W)
upper = Pos(-L/2, 2*Rc)   * Rectangle(L, W)
annul = Pos(0.0,  Rc)     * (Circle(Ro) - Circle(Ri))
halfx = Pos(Ro/2, Rc)     * Rectangle(Ro, 2*Ro)     # keep x >= 0
bend  = annul & halfx
profile = lower + upper + bend

print("area  =", profile.area, " expected", 2*L*W + np.pi*Rc*W)
print("bbox  =", profile.bounding_box())

fig, ax = plt.subplots(figsize=(7,3))
for e in profile.edges():
    p = np.array([tuple(e @ t) for t in np.linspace(0,1,60)])
    ax.plot(p[:,0]*1e3, p[:,1]*1e3, 'k-')
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.grid(alpha=.3); plt.tight_layout(); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# Profile area matches analytic to 1e-18. Now verify the arc radii explicitly from the geometry, extru
solid = extrude(profile, amount=THK)

arcs = [e for e in profile.edges() if e.geom_type == GeomType.CIRCLE]
for e in arcs:
    c = e.arc_center
    print(f"arc R = {e.radius*1e3:7.3f} mm  centre = ({c.X*1e3:.3f}, {c.Y*1e3:.3f}) mm  len = {e.length*1e3:.3f} mm")

print("solid volume =", solid.volume, " expected", profile.area*THK)
print("solid bbox   =", solid.bounding_box())
export_step(solid, "ubend.step")
print("wrote ubend.step")

# -- cell 3 -------------------------------------------------------------------------
# Geometry confirmed: outer arc R=20 mm, inner R=10 mm about (0,15) mm → centreline 15 mm, width 10 mm
import os, subprocess, numpy as np
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

# named 2D points (metres)
P = {"A0":(-L,-W/2), "A1":(-L, W/2), "B0":(0.0,-W/2), "B1":(0.0, W/2),
     "C0":(Ro, Rc),  "C1":(Ri, Rc),  "D0":(0.0, Rc+Ro), "D1":(0.0, Rc+Ri),
     "E0":(-L, Rc+Ro), "E1":(-L, Rc+Ri)}
names = list(P); idx = {n:i for i,n in enumerate(names)}
NP = len(names)
v0 = lambda n: idx[n]; v1 = lambda n: idx[n]+NP

Nw, NL, Nb = 6, 24, 8          # coarse first
blocks = [(["A0","B0","B1","A1"], (NL,Nw)),
          (["B0","C0","C1","B1"], (Nb,Nw)),
          (["C0","D0","D1","C1"], (Nb,Nw)),
          (["D0","E0","E1","D1"], (NL,Nw))]
patch_of_edge = {frozenset(("A1","A0")):"inlet", frozenset(("E0","E1")):"outlet"}

# arc midpoints on the bend (angles measured from bend centre (0,Rc))
def arcpt(R, deg): return (R*np.cos(np.radians(deg)), Rc + R*np.sin(np.radians(deg)))
arcs = [("B0","C0",arcpt(Ro,-45)), ("B1","C1",arcpt(Ri,-45)),
        ("C0","D0",arcpt(Ro, 45)), ("C1","D1",arcpt(Ri, 45))]

# collect boundary edges (CCW traversal per block)
from collections import Counter
cnt = Counter()
for q,_ in blocks:
    for i in range(4): cnt[frozenset((q[i], q[(i+1)%4]))] += 1
faces = {"inlet":[], "outlet":[], "walls":[], "frontAndBack":[]}
for q,_ in blocks:
    for i in range(4):
        p, r = q[i], q[(i+1)%4]
        if cnt[frozenset((p,r))] == 1:
            nm = patch_of_edge.get(frozenset((p,r)), "walls")
            faces[nm].append([v0(p), v0(r), v1(r), v1(p)])
    faces["frontAndBack"].append([v0(n) for n in q[::-1]])   # z=0, normal -z
    faces["frontAndBack"].append([v1(n) for n in q])          # z=THK, normal +z
print({k:len(v) for k,v in faces.items()})

# -- cell 4 -------------------------------------------------------------------------
# Edge bookkeeping checks out (10 boundary edges: inlet, outlet, 8 wall). Now write the dict and run b
def vstr(n, z): return f"({P[n][0]:.9g} {P[n][1]:.9g} {z:.9g})"
lines = ["FoamFile{version 2.0; format ascii; class dictionary; object blockMeshDict;}",
         "scale 1;", "vertices (",
         *[f"    {vstr(n,0.0)}" for n in names],
         *[f"    {vstr(n,THK)}" for n in names], ");", "", "blocks ("]
for q,(n1,n2) in blocks:
    vs = " ".join(str(v0(n)) for n in q) + " " + " ".join(str(v1(n)) for n in q)
    lines.append(f"    hex ({vs}) ({n1} {n2} 1) simpleGrading (1 1 1)")
lines += [");", "", "edges ("]
for a,b,m in arcs:
    lines.append(f"    arc {v0(a)} {v0(b)} ({m[0]:.9g} {m[1]:.9g} 0)")
    lines.append(f"    arc {v1(a)} {v1(b)} ({m[0]:.9g} {m[1]:.9g} {THK:.9g})")
lines += [");", "", "boundary ("]
for nm, fl in faces.items():
    typ = "empty" if nm == "frontAndBack" else ("wall" if nm == "walls" else "patch")
    lines.append(f"    {nm} {{ type {typ}; faces (")
    lines += [f"        ({' '.join(map(str,f))})" for f in fl]
    lines.append("    ); }")
lines += [");", "mergePatchPairs ();"]
open("system/blockMeshDict","w").write("\n".join(lines)+"\n")

open("system/controlDict","w").write(
 'FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}\n'
 'application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n'
 'deltaT 1e-3; writeControl timeStep; writeInterval 100;\n')
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:])

# -- cell 5 -------------------------------------------------------------------------
# `foamToVTK -latestTime` wrote nothing (no time directories). Read the polyMesh through pyvista's Ope
import pyvista as pv, glob
pv.global_theme.background = 'white'
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
mb = rd.read(); m = mb["internalMesh"]
print(m.n_cells, m.bounds)
p = pv.Plotter(off_screen=True, window_size=(1100,450))
p.add_mesh(m, show_edges=True, color='lightsteelblue', line_width=1)
p.view_xy(); p.screenshot("mesh_coarse.png")

# -- cell 6 -------------------------------------------------------------------------
# Rendering has no GL context — blank image. Fall back to matplotlib: slice the mesh at mid-span and d
import matplotlib.pyplot as plt
sl = m.slice(normal='z', origin=(0,0,THK/2))
pts = sl.points
fig, ax = plt.subplots(figsize=(11,4))
i = 0; f = sl.faces
while i < len(f):
    n = f[i]; poly = pts[f[i+1:i+1+n]]; i += n+1
    ax.fill(poly[:,0]*1e3, poly[:,1]*1e3, facecolor='lightsteelblue', edgecolor='k', lw=.5)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title(f'coarse mesh, {m.n_cells} cells'); plt.tight_layout(); plt.show()

# -- cell 7 -------------------------------------------------------------------------
# Only one block — resending just the refinement cell.
Nw, NL, Nb = 24, 120, 32
blocks = [(["A0","B0","B1","A1"], (NL,Nw)),
          (["B0","C0","C1","B1"], (Nb,Nw)),
          (["C0","D0","D1","C1"], (Nb,Nw)),
          (["D0","E0","E1","D1"], (NL,Nw))]
old = open("system/blockMeshDict").read()
txt = old.split("blocks (")[0] + "blocks (\n"
for q,(n1,n2) in blocks:
    vs = " ".join(str(v0(n)) for n in q) + " " + " ".join(str(v1(n)) for n in q)
    txt += f"    hex ({vs}) ({n1} {n2} 1) simpleGrading (1 1 1)\n"
txt += ");\n\nedges (" + old.split("edges (",1)[1]
open("system/blockMeshDict","w").write(txt)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
# Run `checkMesh` and measure the mesh's own volume and patch areas against the requested dimensions.
# exit 0 (0 s)
# Create time
# Create polyMesh for time = 0
# Time = 0
# Mesh stats
# points:           15250
# cells:            7296
# faces:            29512
# nCells:           7296
# nFaces:           29512
# nInternalFaces:   14264
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 9 -------------------------------------------------------------------------
# Look at what the log actually contains.
log = open("log.checkMesh").read()
print(len(log)); print(log[-3000:])

# -- cell 10 ------------------------------------------------------------------------
# `checkMesh` needs `fvSchemes`/`fvSolution` present. Write minimal ones (incompressible, transient-ca
open("system/fvSchemes","w").write(
 'FoamFile{version 2.0; format ascii; class dictionary; object fvSchemes;}\n'
 'ddtSchemes{default Euler;}\ngradSchemes{default Gauss linear;}\n'
 'divSchemes{default none; div(phi,U) Gauss linear;}\n'
 'laplacianSchemes{default Gauss linear orthogonal;}\n'
 'interpolationSchemes{default linear;}\nsnGradSchemes{default orthogonal;}\n')
open("system/fvSolution","w").write(
 'FoamFile{version 2.0; format ascii; class dictionary; object fvSolution;}\n'
 'solvers{p{solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0.05;}\n'
 '        U{solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;}}\n'
 'PISO{nCorrectors 2; nNonOrthogonalCorrectors 0;}\n')
with open("log.checkMesh","w") as f:
    subprocess.run(["checkMesh"], stdout=f, stderr=subprocess.STDOUT)
log = open("log.checkMesh").read()
print(log[log.index("Mesh stats"):])

# -- cell 11 ------------------------------------------------------------------------
# Mesh OK. Final check: picture of the refined mesh plus measurements taken from the mesh itself (patc
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
rd.enable_all_patch_arrays(); mb = rd.read()
mesh = mb["internalMesh"]; bnd = mb["boundary"]
for nm in ["inlet","outlet","walls"]:
    a = bnd[nm].compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{nm:7s} area = {a:.6e} m^2")
print(f"inlet/outlet area target W*THK = {W*THK:.6e} m^2")
print(f"mesh volume {mesh.volume:.6e} vs CAD {solid.volume:.6e} m^3")
wp = bnd["walls"].points; r = np.hypot(wp[:,0], wp[:,1]-Rc)
sel = wp[:,0] > 1e-9
print(f"bend wall radii: inner {r[sel].min()*1e3:.3f} mm, outer {r[sel].max()*1e3:.3f} mm "
      f"-> centreline {(r[sel].min()+r[sel].max())/2*1e3:.3f} mm, width {(r[sel].max()-r[sel].min())*1e3:.3f} mm")

sl = mesh.slice(normal='z', origin=(0,0,THK/2)); pts = sl.points; f = sl.faces
fig, ax = plt.subplots(figsize=(11,4)); i = 0
while i < len(f):
    n = f[i]; poly = pts[f[i+1:i+1+n]]; i += n+1
    ax.fill(poly[:,0]*1e3, poly[:,1]*1e3, facecolor='lightsteelblue', edgecolor='k', lw=.25)
ax.set_aspect('equal'); ax.set_title(f'{mesh.n_cells} cells'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
plt.tight_layout(); plt.show()
