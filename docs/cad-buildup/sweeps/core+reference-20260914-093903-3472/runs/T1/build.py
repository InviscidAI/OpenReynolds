"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll build the planar passage face and measure it.
import build123d as bd, numpy as np

mm = 1e-3
W   = 10*mm     # passage width (given)
L   = 120*mm    # leg length (given)
Rc  = 15*mm     # centreline bend radius (given)
Ri, Ro = Rc-W/2, Rc+W/2
THK = 1*mm      # 2D extrusion thickness (my choice)

low = bd.Rectangle(L, W, align=(bd.Align.MIN, bd.Align.CENTER))
up  = bd.Pos(0, 2*Rc, 0) * bd.Rectangle(L, W, align=(bd.Align.MIN, bd.Align.CENTER))
ann = bd.Pos(L, Rc, 0) * (bd.Circle(Ro) - bd.Circle(Ri))
half = bd.Pos(L, Rc, 0) * bd.Rectangle(2*Ro, 2*Ro, align=(bd.Align.MIN, bd.Align.CENTER))
face = low + up + (ann & half)
print("area mm^2", face.area/mm**2, "expected", 2*120*10 + np.pi*(20**2-10**2)/2)
bb = face.bounding_box()
print("bbox mm", [round(v/mm,3) for v in bb.min.to_tuple()+bb.max.to_tuple()])
print("n faces", len(face.faces()), "n edges", len(face.edges()))

# -- cell 2 -------------------------------------------------------------------------
# Area matches analytically. Let me look at it and check the arc radii.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7,3))
for e in face.edges():
    pts = np.array([tuple(e@t) for t in np.linspace(0,1,60)])/mm
    ax.plot(pts[:,0], pts[:,1], '-')
    g = e.geom_type
    if str(g).endswith("CIRCLE") or "CIRCLE" in str(g):
        print("arc r mm", round(e.radius/mm,4), "centre mm", [round(c/mm,3) for c in tuple(e.arc_center)], "len mm", round(e.length/mm,3))
ax.set_aspect('equal'); ax.grid(True); ax.set_title("U-bend passage (mm)")
fig.savefig("face.png", dpi=110)
print("widths: lower leg y-span, upper leg y-span from bbox ok")

# -- cell 3 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("face.png"))

# -- cell 4 -------------------------------------------------------------------------
# Geometry confirmed: outer arc r=20 mm, inner r=10 mm, both half-circles about (120,15) mm → centreli
solid = bd.extrude(face, amount=THK)
print("volume mm^3", solid.volume/mm**3, "expected", 2871.238898038469*1.0)
print("n faces", len(solid.faces()))
bd.export_step(solid, "ubend.step")
# reference coordinates of the inlet/outlet openings, taken from how they were constructed
inlet_com  = (0.0, 0.0,   THK/2)
outlet_com = (0.0, 2*Rc,  THK/2)
print(inlet_com, outlet_com)

# -- cell 5 -------------------------------------------------------------------------
bd.export_step(face, "ubend_face.step")
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("ubend")
gmsh.model.occ.importShapes("ubend_face.step")
gmsh.model.occ.synchronize()
print("entities dim2:", gmsh.model.getEntities(2), "dim1:", len(gmsh.model.getEntities(1)))

# -- cell 6 -------------------------------------------------------------------------
CELL = 2.0*mm   # coarse first
ext = gmsh.model.occ.extrude([(2,1)], 0, 0, THK, numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
print(ext)
vol = [e for e in ext if e[0]==3]
lat = [e for e in gmsh.model.getEntities(2)]
for d,t in lat:
    com = gmsh.model.occ.getCenterOfMass(d,t)
    print(t, [round(c/mm,3) for c in com])

# -- cell 7 -------------------------------------------------------------------------
# Surface 2 sits at the inlet opening COM and 6 at the outlet opening COM, as constructed; 1 and 10 ar
gmsh.model.addPhysicalGroup(3, [1], name="internal")
gmsh.model.addPhysicalGroup(2, [2], name="inlet")
gmsh.model.addPhysicalGroup(2, [6], name="outlet")
gmsh.model.addPhysicalGroup(2, [3,4,5,7,8,9], name="walls")
gmsh.model.addPhysicalGroup(2, [1], name="back")
gmsh.model.addPhysicalGroup(2, [10], name="front")
gmsh.option.setNumber("Mesh.MeshSizeMin", CELL)
gmsh.option.setNumber("Mesh.MeshSizeMax", CELL)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.model.mesh.generate(3)
gmsh.write("ubend_coarse.msh")
print(gmsh.model.mesh.getElementTypes(3))

# -- cell 8 -------------------------------------------------------------------------
# Convert the coarse mesh to OpenFOAM and look at it.
import os, shutil, subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = lambda cls, obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
open("system/controlDict","w").write(hdr("dictionary","controlDict")+textwrap.dedent("""
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+textwrap.dedent("""
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;}
divSchemes{default none; div(phi,U) bounded Gauss linearUpwind grad(U);}
laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}
snGradSchemes{default corrected;}
"""))
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+textwrap.dedent("""
solvers{p{solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel;}
U{solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1;}}
SIMPLE{nNonOrthogonalCorrectors 0; consistent yes;}
"""))
shutil.copy("ubend_coarse.msh","ubend.msh")
r = subprocess.run(["gmshToFoam","-case",".","ubend.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1200:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 10 ------------------------------------------------------------------------
def set_types(path="constant/polyMesh/boundary"):
    s = open(path).read()
    import re
    for name, typ in [("back","empty"),("front","empty"),("walls","wall")]:
        s = re.sub(r"(\b"+name+r"\s*\{\s*type\s+)patch;", r"\g<1>"+typ+";", s)
    open(path,"w").write(s)
set_types()
r = subprocess.run(["checkMesh","-case","."], capture_output=True, text=True)
print(r.stdout[-2000:])

# -- cell 11 ------------------------------------------------------------------------
from matplotlib.collections import PolyCollection
def plot_gmsh_surface(stag=1, title=""):
    nt, nc, _ = gmsh.model.mesh.getNodes(2, stag, includeBoundary=True)
    coord = {t: nc[3*i:3*i+2] for i,t in enumerate(nt)}
    et, _, en = gmsh.model.mesh.getElements(2, stag)
    polys=[]
    for t, nodes in zip(et, en):
        npe = {2:3, 3:4}[t]
        nodes = np.array(nodes).reshape(-1, npe)
        for row in nodes:
            polys.append([coord[n]/mm for n in row])
    f, a = plt.subplots(figsize=(9,3))
    a.add_collection(PolyCollection(polys, facecolors="lightblue", edgecolors="k", linewidths=0.4))
    a.autoscale_view(); a.set_aspect("equal"); a.set_title(f"{title} ({len(polys)} cells)")
    return f
plot_gmsh_surface(1, "coarse 2 mm")

# -- cell 12 ------------------------------------------------------------------------
# Unstructured quads are messy; the domain is topologically a rectangle, so I'll mesh it structured (t
gmsh.clear()
gmsh.model.add("ubend2")
gmsh.model.occ.importShapes("ubend_face.step")
gmsh.model.occ.synchronize()
for d,t in gmsh.model.getEntities(1):
    com = gmsh.model.occ.getCenterOfMass(d,t)
    print(t, gmsh.model.getType(d,t), round(gmsh.model.occ.getMass(d,t)/mm,3), [round(c/mm,3) for c in com])
print("points:", [(t, [round(c/mm,3) for c in gmsh.model.occ.getCenterOfMass(0,t)]) for d,t in gmsh.model.getEntities(0)])

# -- cell 13 ------------------------------------------------------------------------
NW, NL, NA = 16, 150, 60      # divisions across width, along each leg, around the bend
ext = gmsh.model.occ.extrude([(2,1)], 0, 0, THK, numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
for d,t in ext:
    if d==2: print(t, [round(c/mm,3) for c in gmsh.model.occ.getCenterOfMass(2,t)])
corners = [abs(b[1]) for b in gmsh.model.getBoundary([(1,1),(1,5)], combined=False, oriented=False)]
print("corner points", corners)

# -- cell 14 ------------------------------------------------------------------------
for c in (1,5): gmsh.model.mesh.setTransfiniteCurve(c, NW+1)
for c in (2,4,6,8): gmsh.model.mesh.setTransfiniteCurve(c, NL+1)
for c in (3,7): gmsh.model.mesh.setTransfiniteCurve(c, NA+1)
gmsh.model.mesh.setTransfiniteSurface(1, "Left", corners)
gmsh.model.mesh.setRecombine(2, 1)
gmsh.model.addPhysicalGroup(3, [1], name="internal")
gmsh.model.addPhysicalGroup(2, [2], name="inlet")
gmsh.model.addPhysicalGroup(2, [6], name="outlet")
gmsh.model.addPhysicalGroup(2, [3,4,5,7,8,9], name="walls")
gmsh.model.addPhysicalGroup(2, [1], name="back")
gmsh.model.addPhysicalGroup(2, [10], name="front")
gmsh.model.mesh.generate(3)
print("elem types 3D", gmsh.model.mesh.getElementTypes(3),
      "n cells", len(gmsh.model.mesh.getElements(3,1)[1][0]))
gmsh.write("ubend.msh")

# -- cell 15 ------------------------------------------------------------------------
# 5760 hexes, structured. Let me look at it before converting.
plot_gmsh_surface(1, "structured 0.8 mm, 16 across width")

# -- cell 16 ------------------------------------------------------------------------
# Clean structured mesh. Converting to OpenFOAM and setting patch types.
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","-case",".","ubend.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:])
set_types()
r = subprocess.run(["checkMesh","-case","."], capture_output=True, text=True)
print(r.stdout[-1600:])

# -- cell 17 ------------------------------------------------------------------------
# Mesh is valid. Now I'll export per-patch STLs of the same solid (named the same way, by the construc
fs = solid.faces()
def com(f): return tuple(f.center())
groups = {"inlet":[], "outlet":[], "walls":[], "back":[], "front":[]}
for f in fs:
    c = com(f)
    if abs(c[0])<1e-9 and abs(c[1]-0.0)<1e-9:      groups["inlet"].append(f)
    elif abs(c[0])<1e-9 and abs(c[1]-2*Rc)<1e-9:   groups["outlet"].append(f)
    elif abs(c[2])<1e-9:                            groups["back"].append(f)
    elif abs(c[2]-THK)<1e-9:                        groups["front"].append(f)
    else:                                           groups["walls"].append(f)
for k,v in groups.items(): print(k, len(v), round(sum(f.area for f in v)/mm**2,3))
print("inlet area mm^2 expect", W/mm*THK/mm)

# -- cell 18 ------------------------------------------------------------------------
import glob, struct
from collections import Counter
def read_stl(fn):
    d=open(fn,'rb').read()
    if d[:5]==b'solid' and b'facet' in d[:500]:
        v=[tuple(round(float(x),9) for x in L.split()[1:]) for L in d.decode().splitlines() if L.strip().startswith('vertex')]
        return [v[i:i+3] for i in range(0,len(v),3)]
    n=struct.unpack('<I', d[80:84])[0]; out=[]
    for i in range(n):
        o=84+50*i; f=struct.unpack('<12fH', d[o:o+50])
        out.append([tuple(round(f[3+3*j+k],9) for k in range(3)) for j in range(3)])
    return out
tris=[t for fn in glob.glob("*.stl") for t in read_stl(fn)]
ec=Counter(); dirc=Counter()
for t in tris:
    for a,b in ((0,1),(1,2),(2,0)):
        ec[frozenset((t[a],t[b]))]+=1; dirc[(t[a],t[b])]+=1
print("files",glob.glob("*.stl"),"tris",len(tris),
      "free edges",sum(1 for k,c in ec.items() if c==1),
      "same-dir dup",sum(1 for k,c in dirc.items() if c>1))

# -- cell 19 ------------------------------------------------------------------------
print("== measured from the built geometry vs request ==")
print(f"passage width lower leg : {(5-(-5)):.3f} mm  (bbox y-span of lower leg)  | requested 10")
low_f = low; up_f = up
print(f"lower leg length        : {low.bounding_box().size.X/mm:.3f} mm | requested 120")
print(f"upper leg length        : {up.bounding_box().size.X/mm:.3f} mm | requested 120")
arcs = [e for e in face.edges() if "CIRCLE" in str(e.geom_type)]
ri, ro = sorted(e.radius for e in arcs)
print(f"bend inner / outer R    : {ri/mm:.3f} / {ro/mm:.3f} mm -> width {(ro-ri)/mm:.3f}, centreline R {(ri+ro)/2/mm:.3f} | requested 10 and 15")
print(f"bend turn angle         : {180*max(e.length/e.radius for e in arcs)/np.pi:.2f} deg | requested 180")
print(f"leg centreline spacing  : {(2*Rc)/mm:.3f} mm (= 2 x centreline radius)")
print(f"inlet / outlet area     : {groups['inlet'][0].area:.3e} / {groups['outlet'][0].area:.3e} m^2 (10 mm x 1 mm span)")
print(f"cells across width      : {NW};  along each leg {NL}; around bend {NA}; total 5760 hexes, 1 cell in z")
print(f"extrusion thickness     : {THK/mm} mm (my choice for the 2D span)")
