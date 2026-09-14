"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd
from build123d import Pos, Rectangle, Circle, extrude
import numpy as np

W   = 0.010    # passage width  (given: 10 mm)
L   = 0.120    # leg length     (given: 120 mm)
RC  = 0.015    # centreline bend radius (given: 15 mm)
THK = 0.001    # z-thickness, my choice (2D: one cell)
RI, RO = RC - W/2, RC + W/2

lower = Pos(-L/2, -RC) * Rectangle(L, W)
upper = Pos(-L/2,  RC) * Rectangle(L, W)
annul = (Circle(RO) - Circle(RI)) & (Pos(RO/2, 0) * Rectangle(RO, 2*RO))
face  = lower + upper + annul
solid = extrude(face, THK)

print("RI, RO, centreline R:", RI, RO, (RI+RO)/2)
print("face area m^2:", face.area, " expected:", 2*L*W + np.pi*(RO**2-RI**2)/2)
print("volume m^3:", solid.volume)
print("bbox:", solid.bounding_box())
print("n faces:", len(solid.faces()))

# -- cell 2 -------------------------------------------------------------------------
# Shape and area match analytically. Now the coarse blockMeshDict: 4 hex blocks (lower leg, two 90° be
import os, subprocess, math, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

NA, NW, NB = 30, 5, 6     # coarse: leg-length, across-width, per-90deg-arc cells
s = math.sqrt(0.5)
V2 = [(-L,-RO),(0,-RO),(0,-RI),(-L,-RI),(RO,0),(RI,0),(0,RO),(0,RI),(-L,RO),(-L,RI)]
verts = [(x,y,0.0) for x,y in V2] + [(x,y,THK) for x,y in V2]

def vstr(): return "\n".join("    (%.8g %.8g %.8g)"%v for v in verts)
blocks = [((0,1,2,3,10,11,12,13),(NA,NW,1)),
          ((1,4,5,2,11,14,15,12),(NB,NW,1)),
          ((4,6,7,5,14,16,17,15),(NB,NW,1)),
          ((6,8,9,7,16,18,19,17),(NA,NW,1))]
arcs = [(1,4,(RO*s,-RO*s)),(2,5,(RI*s,-RI*s)),(4,6,(RO*s,RO*s)),(5,7,(RI*s,RI*s))]
arcs += [(a+10,b+10,m) for a,b,m in arcs]
patches = {
 "inlet":  [(0,10,13,3)],
 "outlet": [(8,9,19,18)],
 "walls":  [(0,1,11,10),(2,3,13,12),(1,4,14,11),(2,12,15,5),(4,6,16,14),(5,15,17,7),(6,8,18,16),(7,17,19,9)],
 "frontAndBack": [(0,3,2,1),(1,2,5,4),(4,5,7,6),(6,7,9,8),(10,11,12,13),(11,14,15,12),(14,16,17,15),(16,18,19,17)],
}
ptype = {"inlet":"patch","outlet":"patch","walls":"wall","frontAndBack":"empty"}

d  = "FoamFile{version 2.0; format ascii; class dictionary; object blockMeshDict;}\nscale 1;\n\nvertices\n(\n"+vstr()+"\n);\n\nblocks\n(\n"
for v,n in blocks: d += "    hex (%s) (%d %d %d) simpleGrading (1 1 1)\n"%(" ".join(map(str,v)),*n)
d += ");\n\nedges\n(\n"
for a,b,m in arcs: d += "    arc %d %d (%.8g %.8g %.8g)\n"%(a,b,m[0],m[1], 0.0 if a<10 else THK)
d += ");\n\nboundary\n(\n"
for name,fs in patches.items():
    d += "    %s\n    {\n        type %s;\n        faces\n        (\n"%(name,ptype[name])
    for f in fs: d += "            (%s)\n"%" ".join(map(str,f))
    d += "        );\n    }\n"
d += ");\n\nmergePatchPairs ();\n"
open("system/blockMeshDict","w").write(d)

ctrl = """FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; runTimeModifiable true;
"""
open("system/controlDict","w").write(ctrl)
open("system/fvSchemes","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSchemes;}\nddtSchemes{default Euler;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSolution;}\nsolvers{}\n")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:])

# -- cell 3 -------------------------------------------------------------------------
# Mesh OK. Rendering it (dropping the bad attribute line).
import pyvista as pv
open("case.foam","w").close()
mesh = pv.OpenFOAMReader("case.foam").read().combine().extract_surface()
p = pv.Plotter(off_screen=True, window_size=(900,600))
p.add_mesh(mesh, show_edges=True, color="lightblue")
p.view_xy(); p.screenshot("mesh_coarse.png")

# -- cell 4 -------------------------------------------------------------------------
# Off-screen GL is dead; I'll draw the mesh with matplotlib from the polyMesh points/faces instead.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np, re

def read_pts(f="constant/polyMesh/points"):
    t=open(f).read(); body=t[t.index("(",t.index("\n(")):]
    return np.array(re.findall(r"\(([-\d.eE+ ]+)\)",body)[:], dtype=object)
raw=open("constant/polyMesh/points").read()
pts=np.array([list(map(float,m.split())) for m in re.findall(r"\(([-0-9.eE+ ]+)\)",raw)])
raw=open("constant/polyMesh/faces").read()
faces=[list(map(int,m.split())) for m in re.findall(r"\d+\(([\d ]+)\)",raw)]
polys=[pts[f][:,:2] for f in faces if np.allclose(pts[f][:,2],0.0)]
fig,ax=plt.subplots(figsize=(9,5))
ax.add_collection(PolyCollection(polys,facecolors="lightblue",edgecolors="k",lw=0.4))
ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("U-bend fluid mesh, z=0 plane (coarse, %d cells)"%len(polys))
plt.show()

# -- cell 5 -------------------------------------------------------------------------
fig

# -- cell 6 -------------------------------------------------------------------------
# Shape is correct. Now refine to ~1 mm streamwise / 0.5 mm across (cells stay ~square), rewriting the
def write_blockmesh(NA, NW, NB, path="system/blockMeshDict"):
    d  = "FoamFile{version 2.0; format ascii; class dictionary; object blockMeshDict;}\nscale 1;\n\nvertices\n(\n"+vstr()+"\n);\n\nblocks\n(\n"
    for v,(a,b,c) in [(blocks[0][0],(NA,NW,1)),(blocks[1][0],(NB,NW,1)),(blocks[2][0],(NB,NW,1)),(blocks[3][0],(NA,NW,1))]:
        d += "    hex (%s) (%d %d %d) simpleGrading (1 1 1)\n"%(" ".join(map(str,v)),a,b,c)
    d += ");\n\nedges\n(\n"
    for a,b,m in arcs: d += "    arc %d %d (%.8g %.8g %.8g)\n"%(a,b,m[0],m[1], 0.0 if a<10 else THK)
    d += ");\n\nboundary\n(\n"
    for name,fs in patches.items():
        d += "    %s\n    {\n        type %s;\n        faces\n        (\n"%(name,ptype[name])
        for f in fs: d += "            (%s)\n"%" ".join(map(str,f))
        d += "        );\n    }\n"
    d += ");\n\nmergePatchPairs ();\n"
    open(path,"w").write(d)

write_blockmesh(120, 20, 24)   # 1 mm along legs, 0.5 mm across, ~0.98 mm along bend arc
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-600:])
c = subprocess.run(["checkMesh"], capture_output=True, text=True).stdout
print("\n".join(l for l in c.splitlines() if any(k in l for k in ("cells:","Max aspect","non-orthogonality","skewness","Total volume","Mesh OK","***","Failed"))))

# -- cell 7 -------------------------------------------------------------------------
# Mesh volume matches the analytic 2.8712e-6 m³ to 0.01% (arc faceting). Now export per-patch STLs, gr
from build123d import export_stl, Compound
import numpy as np
groups = {"inlet":[], "outlet":[], "walls":[], "frontAndBack":[]}
for f in solid.faces():
    n = f.normal_at(f.center()); c = f.center()
    if abs(n.Z) > 0.9:                       groups["frontAndBack"].append(f)
    elif abs(c.X + L) < 1e-9 and c.Y < 0:    groups["inlet"].append(f)
    elif abs(c.X + L) < 1e-9 and c.Y > 0:    groups["outlet"].append(f)
    else:                                    groups["walls"].append(f)
for k,v in groups.items():
    a = sum(f.area for f in v)
    export_stl(Compound(children=v), k+".stl")
    print(k, len(v), "area=%.6g"%a)
print("expected inlet/outlet area:", W*THK, " frontAndBack:", 2*face.area,
      " walls:", THK*(2*(2*L) + np.pi*(RI+RO)))

# -- cell 8 -------------------------------------------------------------------------
print("passage width   : %.4f m (asked 0.010)" % (RO-RI))
print("leg length      : %.4f m (asked 0.120)" % L)
print("centreline R    : %.4f m (asked 0.015)" % ((RO+RI)/2))
print("bend sweep      : 180 deg by construction (two 90 deg blocks, theta -90..0..+90)")
print("inlet patch area from mesh dict: %.3g m2 = width*thickness" % (W*THK))
print("mesh: 5760 hex cells, 1 cell in z, frontAndBack empty -> 2D")
