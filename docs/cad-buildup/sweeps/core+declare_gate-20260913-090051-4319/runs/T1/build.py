"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess
from build123d import *
print(os.getcwd(), os.listdir("."))

W   = 0.010    # passage width
L   = 0.120    # leg length
Rc  = 0.015    # bend centreline radius
Ri, Ro = Rc - W/2, Rc + W/2
X0  = -L
yc  = Rc

l1 = Line((X0, -W/2), (0.0, -W/2))
a_out = CenterArc((0.0, yc), Ro, -90, 180)
l2 = Line((0.0, yc+Ro), (X0, yc+Ro))
outlet_e = Line((X0, yc+Ro), (X0, yc+Ri))
l3 = Line((X0, yc+Ri), (0.0, yc+Ri))
a_in = CenterArc((0.0, yc), Ri, 90, -180)
l4 = Line((0.0, W/2), (X0, W/2))
inlet_e = Line((X0, W/2), (X0, -W/2))
wire = Wire([l1.edge(), a_out.edge(), l2.edge(), outlet_e.edge(), l3.edge(), a_in.edge(), l4.edge(), inlet_e.edge()])
print("closed:", wire.is_closed)
face = Face(wire)
print("area mm^2:", face.area*1e6, " bbox:", face.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from build123d import export_step

fig, ax = plt.subplots(figsize=(8,3))
for e in face.edges():
    p = np.array([[(e@t).X,(e@t).Y] for t in np.linspace(0,1,60)])
    ax.plot(p[:,0]*1000, p[:,1]*1000, '-k')
for nm, e in [("inlet", inlet_e.edge()), ("outlet", outlet_e.edge())]:
    c = e.center(); ax.plot(c.X*1000, c.Y*1000, 'ro'); ax.annotate(nm, (c.X*1000, c.Y*1000))
ax.set_aspect('equal'); ax.grid(True); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
plt.tight_layout()
export_step(face, "ubend.step")
print(os.path.getsize("ubend.step"))
fig

# -- cell 3 -------------------------------------------------------------------------
import gmsh, numpy as np
TH = 0.001          # 2D thickness, one cell
LC = 0.0025         # coarse: 4 cells across the 10 mm passage

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("ubend")
gmsh.model.occ.importShapes("ubend.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
print("surfaces:", surfs)
for d,t in gmsh.model.getEntities(1):
    com = gmsh.model.occ.getCenterOfMass(d,t)
    print(t, np.round(com,4))
gmsh.finalize()

# -- cell 4 -------------------------------------------------------------------------
import gmsh, numpy as np
TH = 0.001
LC = 0.0025

def build_mesh(lc, fname="ubend.msh"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ubend")
    gmsh.model.occ.importShapes("ubend.step")
    gmsh.model.occ.synchronize()
    com_of = {t: np.array(gmsh.model.occ.getCenterOfMass(1,t)) for _,t in gmsh.model.getEntities(1)}
    out = gmsh.model.occ.extrude([(2,1)], 0,0,TH, numElements=[1], recombine=True)
    gmsh.model.occ.synchronize()
    vol = [t for d,t in out if d==3][0]
    gmsh.option.setNumber("Mesh.RecombineAll",1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # classify boundary faces of the volume by construction geometry
    groups = {"inlet":[], "outlet":[], "walls":[], "frontAndBack":[]}
    for d,t in gmsh.model.getBoundary([(3,vol)], oriented=False):
        c = np.array(gmsh.model.occ.getCenterOfMass(2,t))
        if abs(c[2]) < 1e-9 or abs(c[2]-TH) < 1e-9:
            groups["frontAndBack"].append(t)          # the two z-planes of the 2D slab
        else:
            key = min(com_of, key=lambda k: np.linalg.norm(com_of[k][:2]-c[:2]))
            groups["inlet" if key==8 else "outlet" if key==4 else "walls"].append(t)
    for name, tags in groups.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, [vol], name="fluid")
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(fname)
    n_hex = len(gmsh.model.mesh.getElementsByType(5)[0])
    gmsh.finalize()
    return groups, n_hex

groups, n_hex = build_mesh(LC)
print(groups, "hexes:", n_hex)

# -- cell 5 -------------------------------------------------------------------------
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True)
def foam(cls, obj, body):
    return textwrap.dedent(f"""\
    FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}
    """) + body
open("system/controlDict","w").write(foam("dictionary","controlDict","""
application     simpleFoam;
startFrom       startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl    timeStep; writeInterval 100; purgeWrite 0;
writeFormat ascii; writePrecision 6; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(foam("dictionary","fvSchemes","""
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""))
open("system/fvSolution","w").write(foam("dictionary","fvSolution","""
solvers { p { solver GAMG; tolerance 1e-6; relTol 0.01; smoother GaussSeidel; }
          U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } }
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }
"""))
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 7 -------------------------------------------------------------------------
import re, subprocess
def fix_boundary():
    p = "constant/polyMesh/boundary"
    s = open(p).read()
    s = s.replace("""    frontAndBack
    {
        type            patch;
        physicalType    patch;""", """    frontAndBack
    {
        type            empty;
        physicalType    empty;""")
    s = s.replace("""    walls
    {
        type            patch;
        physicalType    patch;""", """    walls
    {
        type            wall;
        physicalType    wall;""")
    open(p,"w").write(s)
fix_boundary()
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 8 -------------------------------------------------------------------------
LC_FINE = 0.0005   # 20 cells across the 10 mm passage
groups, n_hex = build_mesh(LC_FINE)
print(groups, "hexes:", n_hex)

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True); print(r.returncode)
fix_boundary()
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print(open("constant/polyMesh/boundary").read()[-900:])
print(out[out.index("Checking geometry"):][:1600])

# -- cell 10 ------------------------------------------------------------------------
import numpy as np
txt = open("constant/polyMesh/points").read()
body = txt[txt.index("(")+1:txt.rindex(")")]
pts = np.array([[float(v) for v in ln.strip()[1:-1].split()] for ln in body.strip().splitlines() if ln.strip().startswith("(")])
sl = pts[np.isclose(pts[:,2],0.0)]
bend = sl[sl[:,0] > 0.0005]
rb = np.hypot(bend[:,0], bend[:,1]-0.015)
low = sl[np.isclose(sl[:,0],-0.06,atol=2e-4) & (sl[:,1]<0.015)][:,1]
up  = sl[np.isclose(sl[:,0],-0.06,atol=2e-4) & (sl[:,1]>0.015)][:,1]
print("bend inner / outer / centreline R :", rb.min(), rb.max(), 0.5*(rb.min()+rb.max()), "| expected 0.010 0.020 0.015")
print("lower leg width at x=-60mm :", low.max()-low.min(), "expected 0.010; centreline y =", 0.5*(low.max()+low.min()))
print("upper leg width at x=-60mm :", up.max()-up.min(), "expected 0.010; centreline y =", 0.5*(up.max()+up.min()))
print("leg length (open end x to bend tangent x=0):", 0.0-sl[:,0].min(), "expected 0.120")
print("slab thickness z:", np.unique(pts[:,2]))
print("fluid volume / thickness = plan area:", 2.87124e-06/0.001, "expected", 2*0.120*0.010 + np.pi/2*(0.020**2-0.010**2))

# -- cell 11 ------------------------------------------------------------------------
import pyvista as pv
from IPython.display import Image
pv.OFF_SCREEN = True
r = subprocess.run(["foamToVTK","-constant","-noInternal" ], capture_output=True, text=True)
r2 = subprocess.run(["foamToVTK","-constant"], capture_output=True, text=True)
import glob; print(sorted(glob.glob("VTK/*"))[:10])

# -- cell 12 ------------------------------------------------------------------------
import pyvista as pv
from IPython.display import Image
blk = pv.read("VTK/t1_0.vtm")
grid = blk[0] if not isinstance(blk[0], pv.MultiBlock) else blk[0][0]
print(grid)
p = pv.Plotter(off_screen=True, window_size=(1400,500))
p.add_mesh(grid, show_edges=True, color="lightsteelblue", line_width=1)
p.view_xy(); p.enable_parallel_projection()
p.screenshot("mesh.png")
Image("mesh.png")

# -- cell 13 ------------------------------------------------------------------------
p = pv.Plotter(off_screen=True, window_size=(900,900))
p.add_mesh(grid.clip_box([-0.01,0.025,-0.008,0.038,-1,1], invert=False), show_edges=True, color="lightsteelblue")
p.view_xy(); p.enable_parallel_projection(); p.camera.zoom(1.3)
p.screenshot("bend.png")
Image("bend.png")

# -- cell 14 ------------------------------------------------------------------------
# The unstructured quad fill is acceptable but a duct deserves a flow-aligned structured mesh. I'll sp
from build123d import Rectangle, Pos, Compound, export_step, Plane, Face, Wire, Line, CenterArc
# three sub-faces, sharing the two tangent lines at x=0
leg_lo = Pos(-L/2, 0.0) * Rectangle(L, W)                    # x in [-0.12,0], y in [-0.005,0.005]
leg_up = Pos(-L/2, 2*Rc) * Rectangle(L, W)                   # y in [0.025,0.035]
bw = Wire([Line((0,-W/2),(0,0)).edge()])  # placeholder not used
bend_w = Wire([CenterArc((0.0,yc), Ro, -90, 180).edge(),
               Line((0.0, yc+Ro), (0.0, yc+Ri)).edge(),
               CenterArc((0.0,yc), Ri, 90, -180).edge(),
               Line((0.0, W/2), (0.0, -W/2)).edge()])
bend_f = Face(bend_w)
print("areas mm^2:", leg_lo.area*1e6, bend_f.area*1e6, leg_up.area*1e6, "sum", (leg_lo.area+bend_f.area+leg_up.area)*1e6)
export_step(Compound([leg_lo, bend_f, leg_up]), "ubend_parts.step")
print(os.path.getsize("ubend_parts.step"))

# -- cell 15 ------------------------------------------------------------------------
import gmsh, numpy as np
TH = 0.001
DS = 0.0005                      # target cell size 0.5 mm -> 20 cells across the 10 mm passage

def build_struct(ds, fname="ubend.msh"):
    n_w   = int(round(W/ds)) + 1                 # nodes across passage
    n_len = int(round(L/ds)) + 1                 # nodes along each leg
    n_arc = int(round(np.pi*Rc/ds)) + 1          # nodes along the bend (centreline arc length)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ubend")
    gmsh.model.occ.importShapes("ubend_parts.step")
    gmsh.model.occ.synchronize()
    surfs = [t for d,t in gmsh.model.getEntities(2)]
    frag, _ = gmsh.model.occ.fragment([(2,surfs[0])], [(2,t) for t in surfs[1:]])
    gmsh.model.occ.synchronize()
    surfs = [t for d,t in gmsh.model.getEntities(2)]
    com_of = {}
    for _,t in gmsh.model.getEntities(1):
        ln = gmsh.model.occ.getMass(1,t); com_of[t] = (np.array(gmsh.model.occ.getCenterOfMass(1,t)), ln)
        n = n_w if abs(ln-W) < 1e-9 else (n_len if abs(ln-L) < 1e-9 else n_arc)
        gmsh.model.mesh.setTransfiniteCurve(t, n)
    for t in surfs:
        gmsh.model.mesh.setTransfiniteSurface(t); gmsh.model.mesh.setRecombine(2, t)
    out = gmsh.model.occ.extrude([(2,t) for t in surfs], 0,0,TH, numElements=[1], recombine=True)
    gmsh.model.occ.synchronize()
    vols = [t for d,t in out if d==3]
    groups = {"inlet":[], "outlet":[], "walls":[], "frontAndBack":[]}
    for d,t in gmsh.model.getBoundary([(3,v) for v in vols], oriented=False, combined=True):
        c = np.array(gmsh.model.occ.getCenterOfMass(2,abs(t)))
        if abs(c[2]) < 1e-9 or abs(c[2]-TH) < 1e-9:
            groups["frontAndBack"].append(abs(t))
        else:
            key = min(com_of, key=lambda k: np.linalg.norm(com_of[k][0][:2]-c[:2]))
            cc, ln = com_of[key]
            if abs(ln-W) < 1e-9 and abs(cc[0]+L) < 1e-9:
                groups["inlet" if cc[1] < Rc else "outlet"].append(abs(t))
            else:
                groups["walls"].append(abs(t))
    for name, tags in groups.items(): gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, vols, name="fluid")
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.write(fname)
    nh = len(gmsh.model.mesh.getElementsByType(5)[0]); gmsh.finalize()
    return groups, nh, (n_w, n_len, n_arc)

groups, nh, ns = build_struct(DS)
print(groups, "hexes:", nh, "nodes(w,len,arc):", ns)

# -- cell 16 ------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True); print(r.returncode, r.stdout[-400:])
fix_boundary()
r = subprocess.run(["checkMesh"], capture_output=True, text=True); out=r.stdout
print(out[out.index("Mesh stats"):out.index("Checking topology")])
print(out[out.index("Checking geometry"):][:1500])

# -- cell 17 ------------------------------------------------------------------------
b = open("constant/polyMesh/boundary").read()
print(b[b.index("4\n("):])
subprocess.run(["foamToVTK","-constant"], capture_output=True, text=True)
blk = pv.read("VTK/t1_0.vtm"); grid = blk[0]
p = pv.Plotter(off_screen=True, window_size=(1000,900))
p.add_mesh(grid.clip_box([-0.012,0.025,-0.008,0.038,-1,1], invert=False), show_edges=True, color="lightsteelblue")
p.view_xy(); p.enable_parallel_projection(); p.camera.zoom(1.3); p.screenshot("bend2.png")
Image("bend2.png")

# -- cell 18 ------------------------------------------------------------------------
import numpy as np
faces_txt = open("constant/polyMesh/faces").read()
fb = faces_txt[faces_txt.index("(")+1:faces_txt.rindex(")")].strip().splitlines()
fl = [l for l in fb if "(" in l]
def face_nodes(l): return [int(v) for v in l[l.index("(")+1:l.index(")")].split()]
pts_txt = open("constant/polyMesh/points").read()
pb = pts_txt[pts_txt.index("(")+1:pts_txt.rindex(")")]
P = np.array([[float(v) for v in ln.strip()[1:-1].split()] for ln in pb.strip().splitlines() if ln.strip().startswith("(")])
for name, start, n in [("inlet",45326,20), ("outlet",46494,20)]:
    c = np.array([P[face_nodes(fl[i])].mean(0) for i in range(start, start+n)])
    print(name, "centroid", np.round(c.mean(0),5), " y range", round(c[:,1].min(),5), round(c[:,1].max(),5), " x", np.unique(np.round(c[:,0],5)))
