"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import math, numpy as np, build123d as bd
print(bd.__version__)
# --- parameters (all mm, from the request) ---
W = 80.0        # duct section 80 x 80
L_UP, L_CORE, L_DN = 40.0, 20.0, 80.0
FLAT = 8.0      # cell width across flats
TW  = 1.0       # wall thickness
P   = FLAT + TW # lattice pitch across flats = 9
Rlat = P/math.sqrt(3.0)   # circumradius of lattice hexagon
Rcell= FLAT/math.sqrt(3.0)# circumradius of open cell hexagon
print("lattice R", Rlat, "cell R", Rcell)
a = np.array([0.0, P]); b = np.array([1.5*Rlat, P/2])
print("lattice vectors", a, b)
ctrs=[]
for m in range(-12,13):
    for n in range(-12,13):
        c = np.array([W/2,W/2]) + m*a + n*b
        if -Rlat-1 <= c[0] <= W+Rlat+1 and -Rlat-1 <= c[1] <= W+Rlat+1:
            ctrs.append(c)
print(len(ctrs), "candidate cell centres")

# -- cell 2 -------------------------------------------------------------------------
import inspect
print(inspect.signature(bd.RegularPolygon.__init__))

# -- cell 3 -------------------------------------------------------------------------
hex0 = bd.RegularPolygon(radius=Rcell, side_count=6, major_radius=True)
bb = hex0.bounding_box()
print("hex bbox size", bb.size, " area", hex0.area)
sq = bd.Rectangle(W, W).moved(bd.Pos(W/2, W/2))
AMIN = 2.0   # my choice: discard clipped edge cells with < 2 mm^2 open area
faces=[]
for c in ctrs:
    f = (hex0.moved(bd.Pos(*c)) & sq)
    if f.area > AMIN:
        faces.append((c, f))
print(len(faces), "open cells kept; total open area", sum(f.area for _,f in faces),
      "of", W*W, "->", sum(f.area for _,f in faces)/(W*W))
full = [c for c,f in faces if abs(f.area-hex0.area)<1e-6]
print("full cells:", len(full), "partial:", len(faces)-len(full))

# -- cell 4 -------------------------------------------------------------------------
ar = sorted(f.area for _,f in faces)
print([round(x,2) for x in ar[:20]])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,6))
for c,f in faces:
    for e in f.edges():
        p=np.array([[v.X,v.Y] for v in (e @ 0, e @ 1)]); ax.plot(p[:,0],p[:,1],'k-',lw=0.7)
ax.plot([0,W,W,0,0],[0,0,W,W,0],'r-'); ax.set_aspect(1); fig.savefig("cells.png",dpi=110)
print("saved")

# -- cell 5 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("cells.png"))

# -- cell 6 -------------------------------------------------------------------------
L_TOT = L_UP+L_CORE+L_DN
up = bd.Box(W,W,L_UP,  align=(bd.Align.MIN,)*3)
dn = bd.Pos(0,0,L_UP+L_CORE)*bd.Box(W,W,L_DN, align=(bd.Align.MIN,)*3)
chans = [bd.Pos(0,0,L_UP)*bd.extrude(f, L_CORE) for _,f in faces]
fluid = up + dn + chans
print(type(fluid), "solids:", len(fluid.solids()))
print("volume", fluid.volume, "expected", W*W*(L_UP+L_DN)+sum(f.area for _,f in faces)*L_CORE)
print("bbox", fluid.bounding_box())

# -- cell 7 -------------------------------------------------------------------------
groups={"inlet":[], "outlet":[], "ductWalls":[], "honeycombWalls":[]}
tol=1e-6
for f in fluid.faces():
    c=f.center(); n=f.normal_at(c)
    if abs(n.Z)>0.999 and abs(c.Z)<tol:            groups["inlet"].append(f)
    elif abs(n.Z)>0.999 and abs(c.Z-L_TOT)<tol:    groups["outlet"].append(f)
    elif abs(n.X)>0.999 and (abs(c.X)<tol or abs(c.X-W)<tol): groups["ductWalls"].append(f)
    elif abs(n.Y)>0.999 and (abs(c.Y)<tol or abs(c.Y-W)<tol): groups["ductWalls"].append(f)
    else: groups["honeycombWalls"].append(f)
for k,v in groups.items(): print(k, len(v), round(sum(f.area for f in v),2))
print("total faces", len(fluid.faces()), "sum areas", sum(f.area for f in fluid.faces()))
perim_int = sum(sum(e.length for e in f.edges() if not (abs(( e@0).X)<tol and abs((e@1).X)<tol) ) for _,f in faces)
print("check: 2*plate face solid area =", 2*(W*W-sum(f.area for _,f in faces)))

# -- cell 8 -------------------------------------------------------------------------
bd.export_step(fluid, "fluid_mm.step")
import subprocess, os
print(os.listdir("."))
print(subprocess.run(["ls","-la"],capture_output=True,text=True).stdout[-400:])

# -- cell 9 -------------------------------------------------------------------------
import gmsh
def build_msh(lc_core, lc_duct, out):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",1)
    gmsh.model.add("hc")
    gmsh.model.occ.importShapes("fluid_mm.step")
    gmsh.model.occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    g={"inlet":[],"outlet":[],"ductWalls":[],"honeycombWalls":[]}
    t=1e-6
    for dim,tag in surfs:
        x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(dim,tag)
        if abs(z1-z0)<t and abs(z0)<t:            g["inlet"].append(tag)
        elif abs(z1-z0)<t and abs(z0-L_TOT)<t:    g["outlet"].append(tag)
        elif abs(x1-x0)<t and (abs(x0)<t or abs(x0-W)<t): g["ductWalls"].append(tag)
        elif abs(y1-y0)<t and (abs(y0)<t or abs(y0-W)<t): g["ductWalls"].append(tag)
        else: g["honeycombWalls"].append(tag)
    for name,tags in g.items():
        p=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,p,name)
        print(name, len(tags))
    vols=[v[1] for v in gmsh.model.getEntities(3)]
    p=gmsh.model.addPhysicalGroup(3,vols); gmsh.model.setPhysicalName(3,p,"internal")
    # size field: fine inside the honeycomb core, coarser in the plain duct
    f=gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f,"VIn",lc_core); gmsh.model.mesh.field.setNumber(f,"VOut",lc_duct)
    gmsh.model.mesh.field.setNumbers(f,"XMin",[ -1]); # placeholder
    for k,v in dict(XMin=-1,XMax=W+1,YMin=-1,YMax=W+1,ZMin=L_UP-8,ZMax=L_UP+L_CORE+8).items():
        gmsh.model.mesh.field.setNumber(f,k,v)
    gmsh.model.mesh.field.setNumber(f,"Thickness",12)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.write(out)
    ne=len(gmsh.model.mesh.getElementsByType(4)[0]); nn=gmsh.model.mesh.getNodes()[0].size
    gmsh.finalize()
    return ne,nn
print(build_msh(4.0, 6.0, "coarse.msh"))

# -- cell 10 ------------------------------------------------------------------------
import subprocess, os, shutil, textwrap
os.makedirs("system_tmp", exist_ok=True)
os.makedirs("system", exist_ok=True)
# minimal case skeleton for gmshToFoam/checkMesh
os.makedirs("constant", exist_ok=True)
open("system/controlDict","w").write(textwrap.dedent("""\
FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
"""))
os.makedirs("system",exist_ok=True)
shutil.rmtree("system_tmp")
os.rename("system","system2") if False else None
print(os.listdir("."))

# -- cell 11 ------------------------------------------------------------------------
r=subprocess.run(["gmshToFoam","coarse.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-2500:], r.stderr[-1000:])

# -- cell 12 ------------------------------------------------------------------------
r=subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"],capture_output=True,text=True); print(r.returncode)
print(open("constant/polyMesh/boundary").read()[-1200:])

# -- cell 13 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print("\n".join(r.stdout.splitlines()[-40:]))

# -- cell 14 ------------------------------------------------------------------------
print(r.stdout[r.stdout.find("Mesh stats"):])

# -- cell 15 ------------------------------------------------------------------------
print(len(r.stdout)); print(r.stderr[-2000:])

# -- cell 16 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}\n")
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):]); print(r.stderr[-500:])

# -- cell 17 ------------------------------------------------------------------------
print([n for n in dir(bd) if "scale" in n.lower()]); print(inspect.signature(bd.scale)); print(inspect.signature(bd.export_stl))

# -- cell 18 ------------------------------------------------------------------------
for name, fl in groups.items():
    comp = bd.Compound(children=[f for f in fl])
    comp_m = bd.scale(comp, 0.001)
    bd.export_stl(comp_m, f"{name}.stl", tolerance=1e-5, angular_tolerance=0.1)
    print(name, comp_m.bounding_box().size, round(comp_m.area,8))

# -- cell 19 ------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(900,700))
cols={"inlet":"blue","outlet":"green","ductWalls":"lightgray","honeycombWalls":"red"}
for n,c in cols.items():
    m=pv.read(f"{n}.stl"); pl.add_mesh(m, color=c, opacity=0.5 if n=="ductWalls" else 1.0, show_edges=False)
pl.camera_position=[(0.25,-0.2,0.25),(0.04,0.04,0.07),(0,0,1)]
img=pl.screenshot("view.png"); display(Image("view.png"))

# -- cell 20 ------------------------------------------------------------------------
ne,nn = build_msh(2.0, 3.5, "fluid.msh")
print("tets", ne, "nodes", nn)

# -- cell 21 ------------------------------------------------------------------------
import re
r=subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:])
r=subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"],capture_output=True,text=True); print(r.returncode)
b=open("constant/polyMesh/boundary").read()
for p in ("ductWalls","honeycombWalls"):
    b=re.sub(r"(%s\s*\{\s*)type\s+patch;"%p, r"\1type            wall;", b)
open("constant/polyMesh/boundary","w").write(b)
print(b[b.find("4\n("):])

# -- cell 22 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):])

# -- cell 23 ------------------------------------------------------------------------
vec_a = np.array([0.0,P]); vec_b = np.array([1.5*Rlat,P/2])
ctr0=np.array([W/2,W/2])
h_a = hex0.moved(bd.Pos(*ctr0)); h_b = hex0.moved(bd.Pos(*(ctr0+vec_a))); h_c = hex0.moved(bd.Pos(*(ctr0+vec_b)))
bb=h_a.bounding_box()
gap_a = h_b.bounding_box().min.Y - bb.max.Y
nb = vec_b/np.linalg.norm(vec_b)
proj = lambda f,v: (min(np.dot([p.X,p.Y],v) for p in f.vertices()), max(np.dot([p.X,p.Y],v) for p in f.vertices()))
gap_b = proj(h_c,nb)[0]-proj(h_a,nb)[1]
hz=[v.Z for f in groups["honeycombWalls"] for v in f.vertices()]
fb=fluid.bounding_box()
print("across flats   : %.4f mm (asked 8)"%bb.size.Y)
print("wall, y-neighbour     : %.4f mm (asked 1)"%gap_a)
print("wall, 30deg-neighbour : %.4f mm (asked 1)"%gap_b)
print("duct section   : %.3f x %.3f mm (asked 80 x 80)"%(fb.size.X,fb.size.Y))
print("total length   : %.3f mm (40+20+80)"%fb.size.Z)
print("honeycomb z    : %.3f .. %.3f -> %.3f thick (asked 20)"%(min(hz),max(hz),max(hz)-min(hz)))
print("open-area ratio: %.4f  (ideal (8/9)^2=%.4f)"%(sum(f.area for _,f in faces)/W**2,(FLAT/P)**2))
print("cells kept: %d (%d full, %d clipped)"%(len(faces),len(full),len(faces)-len(full)))

# -- cell 24 ------------------------------------------------------------------------
open("case.foam","w").close()
rd=pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.time_values else None
m=rd.read()["internalMesh"]
sl=m.clip(normal="z", origin=(0,0,0.05), invert=False).clip(normal="z", origin=(0,0,0.0501))
pl=pv.Plotter(off_screen=True, window_size=(800,800))
pl.add_mesh(sl, show_edges=True, color="lightsteelblue", line_width=0.4)
pl.camera_position="xy"; pl.screenshot("slice.png"); display(Image("slice.png"))
