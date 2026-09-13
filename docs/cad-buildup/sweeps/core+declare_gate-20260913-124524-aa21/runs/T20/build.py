"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess, numpy as np
from build123d import *

D_BORE = 0.060; T_FL = 0.020; T_GK = 0.002
L = 2*T_FL + T_GK
R = D_BORE/2

flangeA = Pos(0,0,T_FL/2) * Cylinder(R, T_FL)
gasket  = Pos(0,0,T_FL+T_GK/2) * Cylinder(R, T_GK)
flangeB = Pos(0,0,T_FL+T_GK+T_FL/2) * Cylinder(R, T_FL)
water = flangeA + gasket + flangeB
print(water.volume, np.pi*R**2*L, water.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Volume matches πR²L exactly. Now check whether the fuse kept the three lateral faces separate (I nee
for i,f in enumerate(water.faces()):
    c = f.center()
    print(i, type(f.geom_type), f.geom_type, round(f.area,6), [round(v,4) for v in (c.X,c.Y,c.Z)])

# -- cell 3 -------------------------------------------------------------------------
# Fuse merged the laterals, so I take faces from the three unfused solids instead — patch identity dec
os.makedirs("constant/triSurface", exist_ok=True)
def lateral(s): return [f for f in s.faces() if f.geom_type == GeomType.CYLINDER][0]
def flat(s, z): return [f for f in s.faces() if f.geom_type == GeomType.PLANE and abs(f.center().Z - z) < 1e-9][0]

patches = {
 "inlet":        flat(flangeA, 0.0),
 "outlet":       flat(flangeB, L),
 "wall_flangeA": lateral(flangeA),
 "gasket_inner": lateral(gasket),
 "wall_flangeB": lateral(flangeB),
}
for n,f in patches.items():
    export_stl(Compound(children=[Solid.make_solid(Shell(faces=[f]))]) if False else f,
               f"constant/triSurface/{n}.stl", tolerance=1e-4, angular_tolerance=0.1)
    print(n, round(f.area,6), os.path.getsize(f"constant/triSurface/{n}.stl"))

# -- cell 4 -------------------------------------------------------------------------
# Binary STL; re-export ASCII so the solid name carries the patch name, then re-check.
import re
for n,f in patches.items():
    p=f"constant/triSurface/{n}.stl"
    export_stl(f, p, tolerance=1e-4, angular_tolerance=0.1, ascii_format=True)
    txt = open(p).read()
    txt = re.sub(r"solid\s*\S*", f"solid {n}", txt, count=1).replace("endsolid", f"endsolid {n}")
    open(p,"w").write(txt)
    print(n, open(p).readline().strip(), txt.count("facet normal"))

# -- cell 5 -------------------------------------------------------------------------
# The regex swallowed the first `facet` keyword. Fix headers line-wise.
for n,f in patches.items():
    p=f"constant/triSurface/{n}.stl"
    export_stl(f, p, tolerance=1e-4, angular_tolerance=0.1, ascii_format=True)
    lines = open(p).read().splitlines()
    lines[0] = f"solid {n}"
    lines[-1] = f"endsolid {n}"
    open(p,"w").write("\n".join(lines)+"\n")
    print(n, lines[0], lines[1].strip()[:20], lines[-1], sum(1 for l in lines if "facet normal" in l))

# -- cell 6 -------------------------------------------------------------------------
# Now the watertightness check on the welded union.
from collections import Counter
def read_stl(p):
    v=[float(x) for l in open(p) if l.strip().startswith("vertex") for x in l.split()[1:]]
    return np.array(v).reshape(-1,3,3)
allt = np.vstack([read_stl(f"constant/triSurface/{n}.stl") for n in patches])
V = np.round(allt.reshape(-1,3), 9)
uniq, inv = np.unique(V, axis=0, return_inverse=True)
tri = inv.reshape(-1,3)
c = Counter(tuple(sorted(e)) for t in tri for e in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])))
print(len(tri), len(uniq), "free edges:", sum(1 for k,v in c.items() if v!=2))

# -- cell 7 -------------------------------------------------------------------------
# Better mesher for this shape than snappy: a blockMesh O-grid whose z block boundaries sit exactly at
import math, os
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
ZL = [0.0, T_FL, T_FL+T_GK, L]          # 0, 20, 22, 42 mm
RI = 0.5*R                               # inner O-grid radius
NA, NR, NZ = 8, 5, [10, 2, 10]           # coarse
th = [math.radians(45+90*i) for i in range(4)]
verts=[]
for z in ZL:
    for r in (RI, R):
        for t in th: verts.append((r*math.cos(t), r*math.sin(t), z))
def vi(k, ring, i): return k*8 + ring*4 + (i % 4)   # ring 0=inner,1=outer
blocks=[]; edges=[]; faces={"inlet":[], "outlet":[], "wall_flangeA":[], "gasket_inner":[], "wall_flangeB":[]}
wallname = ["wall_flangeA", "gasket_inner", "wall_flangeB"]
for k in range(3):
    b = [vi(k,0,i) for i in range(4)]; t_ = [vi(k+1,0,i) for i in range(4)]
    blocks.append((b+t_, (NA,NA,NZ[k])))
    if k==0: faces["inlet"].append(b[::-1])
    if k==2: faces["outlet"].append(t_)
    for i in range(4):
        bf = [vi(k,0,i), vi(k,1,i), vi(k,1,i+1), vi(k,0,i+1)]
        tf = [vi(k+1,0,i), vi(k+1,1,i), vi(k+1,1,i+1), vi(k+1,0,i+1)]
        blocks.append((bf+tf, (NR,NA,NZ[k])))
        if k==0: faces["inlet"].append(bf[::-1])
        if k==2: faces["outlet"].append(tf)
        faces[wallname[k]].append([vi(k,1,i), vi(k,1,i+1), vi(k+1,1,i+1), vi(k+1,1,i)])
for k in range(4):
    for i in range(4):
        tm = math.radians(45+90*i+45)
        edges.append((vi(k,1,i), vi(k,1,i+1), (R*math.cos(tm), R*math.sin(tm), ZL[k])))
print(len(verts), len(blocks), len(edges), {k:len(v) for k,v in faces.items()})

# -- cell 8 -------------------------------------------------------------------------
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object blockMeshDict;}\n"
s = [hdr, "scale 1;\n\nvertices\n(\n"]
s += [f"    ({v[0]:.9g} {v[1]:.9g} {v[2]:.9g})\n" for v in verts]
s.append(");\n\nblocks\n(\n")
for vs,n in blocks:
    s.append("    hex (" + " ".join(map(str,vs)) + f") ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n")
s.append(");\n\nedges\n(\n")
for a,b,m in edges:
    s.append(f"    arc {a} {b} ({m[0]:.9g} {m[1]:.9g} {m[2]:.9g})\n")
s.append(");\n\nboundary\n(\n")
for name, fl in faces.items():
    typ = "patch" if name in ("inlet","outlet") else "wall"
    s.append(f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n")
    s += ["            (" + " ".join(map(str,f)) + ")\n" for f in fl]
    s.append("        );\n    }\n")
s.append(");\n\nmergePatchPairs ();\n")
open("system/blockMeshDict","w").writelines(s)

open("system/controlDict","w").write(hdr.replace("blockMeshDict","controlDict")+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(hdr.replace("blockMeshDict","fvSchemes")+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr.replace("blockMeshDict","fvSolution")+"solvers{}\n")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 10 ------------------------------------------------------------------------
# Mesh OK; volume 1.17991e-4 vs exact 1.18752e-4 m³ (0.64 % low from the faceted circle). Refine once 
NA, NR, NZ = 16, 8, [16, 4, 16]
blocks=[]; edges=[]; faces={k:[] for k in faces}
for k in range(3):
    b = [vi(k,0,i) for i in range(4)]; t_ = [vi(k+1,0,i) for i in range(4)]
    blocks.append((b+t_, (NA,NA,NZ[k])))
    if k==0: faces["inlet"].append(b[::-1])
    if k==2: faces["outlet"].append(t_)
    for i in range(4):
        bf = [vi(k,0,i), vi(k,1,i), vi(k,1,i+1), vi(k,0,i+1)]
        tf = [vi(k+1,0,i), vi(k+1,1,i), vi(k+1,1,i+1), vi(k+1,0,i+1)]
        blocks.append((bf+tf, (NR,NA,NZ[k])))
        if k==0: faces["inlet"].append(bf[::-1])
        if k==2: faces["outlet"].append(tf)
        faces[wallname[k]].append([vi(k,1,i), vi(k,1,i+1), vi(k+1,1,i+1), vi(k+1,1,i)][:2]+[vi(k+1,1,i+1), vi(k+1,1,i)])
for k in range(4):
    for i in range(4):
        tm = math.radians(45+90*i+45)
        edges.append((vi(k,1,i), vi(k,1,i+1), (R*math.cos(tm), R*math.sin(tm), ZL[k])))
s = [hdr, "scale 1;\n\nvertices\n(\n"] + [f"    ({v[0]:.9g} {v[1]:.9g} {v[2]:.9g})\n" for v in verts]
s.append(");\n\nblocks\n(\n")
for vs,n in blocks: s.append("    hex (" + " ".join(map(str,vs)) + f") ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n")
s.append(");\n\nedges\n(\n")
for a,b,m in edges: s.append(f"    arc {a} {b} ({m[0]:.9g} {m[1]:.9g} {m[2]:.9g})\n")
s.append(");\n\nboundary\n(\n")
for name, fl in faces.items():
    typ = "patch" if name in ("inlet","outlet") else "wall"
    s.append(f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n")
    s += ["            (" + " ".join(map(str,f)) + ")\n" for f in fl]
    s.append("        );\n    }\n")
s.append(");\n\nmergePatchPairs ();\n")
open("system/blockMeshDict","w").writelines(s)
r = subprocess.run(["blockMesh"], capture_output=True, text=True); print(r.returncode, r.stdout[-600:])

# -- cell 11 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if any(x in l for x in ("Max skew","non-orth","Total volume","Max aspect","Mesh OK","***","Failed"))])

# -- cell 12 ------------------------------------------------------------------------
# Now measure each named patch from the mesh itself — z-extent, radius, area — and render.
import pyvista as pv
pv.start_xvfb() if False else None
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.enable_all_patch_arrays(); m = rd.read()
bnd = m["boundary"]
for name in bnd.keys():
    p = bnd[name].extract_surface(); b = p.bounds
    rad = np.hypot(p.points[:,0], p.points[:,1])
    print(f"{name:14s} area={p.area:.6f}  z=[{b[4]:.4f},{b[5]:.4f}]  r=[{rad.min():.4f},{rad.max():.4f}]")
print("exact: disk 0.0028274, flange wall 0.0037699, gasket 0.00037699")

# -- cell 13 ------------------------------------------------------------------------
# Every patch lands exactly where the request says. One picture to confirm.
pl = pv.Plotter(off_screen=True, window_size=(900,700))
cols = {"inlet":"blue","outlet":"red","wall_flangeA":"lightgray","gasket_inner":"orange","wall_flangeB":"tan"}
for n,c in cols.items():
    pl.add_mesh(bnd[n].extract_surface(), color=c, show_edges=True, line_width=1, label=n)
pl.add_legend(); pl.camera_position = [(0.12,-0.12,0.10),(0,0,0.021),(0,0,1)]
pl.screenshot("mesh_view.png"); print("ok")

# -- cell 14 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("mesh_view.png"))
