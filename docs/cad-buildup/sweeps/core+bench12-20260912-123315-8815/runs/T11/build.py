"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, math, time
from build123d import *

# --- parameters (mm) ---
W      = 80.0      # duct square side
TCORE  = 20.0      # core thickness (flow direction)
AF     = 8.0       # cell width across flats
TW     = 1.0       # wall thickness
LUP    = 40.0      # upstream length
LDN    = 80.0      # downstream length
Z0, Z1 = 0.0, TCORE
ZIN, ZOUT = -LUP, TCORE + LDN

R_cell = AF/math.sqrt(3.0)          # circumradius of the open cell
P_af   = AF + TW                    # centre-to-centre across flats
R_lat  = P_af/math.sqrt(3.0)        # lattice circumradius
dx, dy = 1.5*R_lat, P_af            # column pitch, row pitch

print(f"R_cell={R_cell:.4f}  across-flats={math.sqrt(3)*R_cell:.4f} (want {AF})")
print(f"lattice dx={dx:.4f} dy={dy:.4f}  wall gap across flats={P_af-AF:.4f} (want {TW})")

ncol = int(W/dx)+3; nrow = int(W/dy)+3
centers = [(i*dx, j*dy + (abs(i)%2)*dy/2)
           for i in range(-ncol, ncol+1) for j in range(-nrow, nrow+1)]
print("candidate centres:", len(centers))

# -- cell 2 -------------------------------------------------------------------------
hex_sk = RegularPolygon(R_cell, 6)
bb = hex_sk.bounding_box()
print("hex bbox size:", bb.size, " area:", hex_sk.area, " ideal:", 1.5*math.sqrt(3)*R_cell**2)

# -- cell 3 -------------------------------------------------------------------------
t0=time.time()
INSET = TW/2.0                     # half-wall skin of solid against the duct wall
sq = Rectangle(W-2*INSET, W-2*INSET)
full_area = hex_sk.area
KEEP = 0.20                        # drop boundary slivers smaller than 20% of a full cell
cells = []
for (cx, cy) in centers:
    if abs(cx) > W/2 + R_cell or abs(cy) > W/2 + R_cell:
        continue
    h = Pos(cx, cy) * hex_sk
    if abs(cx)+R_cell < W/2-INSET and abs(cy)+AF/2 < W/2-INSET:
        cells.append((h, full_area))
    else:
        c = h & sq
        if c.area > KEEP*full_area:
            cells.append((c, c.area))
open_sk = sum((c for c,_ in cells), Sketch())
print(f"cells kept: {len(cells)}  full: {sum(1 for _,a in cells if a==full_area)}"
      f"  open area: {open_sk.area:.1f} mm2  porosity: {open_sk.area/W**2:.3f}  ({time.time()-t0:.1f}s)")

# -- cell 4 -------------------------------------------------------------------------
t0=time.time()
duct   = Pos(0,0,(ZIN+ZOUT)/2) * Box(W, W, ZOUT-ZIN)
chan   = extrude(Plane.XY.offset(Z0) * open_sk, TCORE)
slab   = Pos(0,0,(Z0+Z1)/2) * Box(W, W, TCORE)
core   = slab - chan
fluid  = duct - core
print(f"boolean {time.time()-t0:.1f}s")
print("duct vol", duct.volume, "core solid vol", core.volume)
print("fluid vol", fluid.volume, " expected", duct.volume - (W*W*TCORE - open_sk.area*TCORE))
print("fluid bbox", fluid.bounding_box().min, fluid.bounding_box().max)
print("solids:", len(fluid.solids()), " faces:", len(fluid.faces()))

# -- cell 5 -------------------------------------------------------------------------
TOL=1e-6
def classify(bmin, bmax):
    if abs(bmin.Z-ZIN)<TOL and abs(bmax.Z-ZIN)<TOL: return "inlet"
    if abs(bmin.Z-ZOUT)<TOL and abs(bmax.Z-ZOUT)<TOL: return "outlet"
    for v in (bmin.X,bmax.X,bmin.Y,bmax.Y):
        pass
    if (abs(bmin.X-bmax.X)<TOL and abs(abs(bmin.X)-W/2)<TOL) or \
       (abs(bmin.Y-bmax.Y)<TOL and abs(abs(bmin.Y)-W/2)<TOL): return "ductwalls"
    return "honeycombwalls"

groups={}
for f in fluid.faces():
    bb=f.bounding_box(); g=classify(bb.min,bb.max)
    groups.setdefault(g,[]).append(f)
for g,fs in groups.items():
    print(g, len(fs), "faces, area", round(sum(f.area for f in fs),1))
print("inlet area target", W*W, " honeycomb wetted target ~", round(sum(a for _,a in cells)*0+0,1))

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv, numpy as np
pv.start_xvfb() if False else None
pv.global_theme.window_size=[1000,750]
cols={"inlet":"blue","outlet":"red","ductwalls":"lightgray","honeycombwalls":"orange"}
pl=pv.Plotter(off_screen=True)
for g,fs in groups.items():
    verts=[];faces=[]
    for f in fs:
        v,t=f.tessellate(0.3)
        o=len(verts); verts+= [(p.X,p.Y,p.Z) for p in v]
        faces+= [[3,a+o,b+o,c+o] for a,b,c in t]
    pl.add_mesh(pv.PolyData(np.array(verts),np.hstack(faces)),color=cols[g],
                opacity=0.35 if g=="ductwalls" else 1.0,show_edges=False)
pl.camera_position=[(260,-200,220),(0,0,30),(0,0,1)]
pl.screenshot("geom.png"); print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 8 -------------------------------------------------------------------------
export_step(fluid, "fluid_mm.step")
import os; print(os.path.getsize("fluid_mm.step"))

# -- cell 9 -------------------------------------------------------------------------
import gmsh
t0=time.time()
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",1)
gmsh.model.add("hc")
gmsh.model.occ.importShapes("fluid_mm.step")
gmsh.model.occ.synchronize()
vols=gmsh.model.getEntities(3); surfs=gmsh.model.getEntities(2)
print("vols",len(vols),"surfs",len(surfs),"import",round(time.time()-t0,1),"s")
pg={"inlet":[],"outlet":[],"ductwalls":[],"honeycombwalls":[]}
for (d,s) in surfs:
    x0,y0,z0b,x1,y1,z1b = gmsh.model.getBoundingBox(d,s)
    class B: pass
    mn=type("V",(),{"X":x0,"Y":y0,"Z":z0b})(); mx=type("V",(),{"X":x1,"Y":y1,"Z":z1b})()
    pg[classify(mn,mx)].append(s)
for k,v in pg.items(): print(k,len(v))

# -- cell 10 ------------------------------------------------------------------------
for name,ss in pg.items():
    gmsh.model.addPhysicalGroup(2, ss, name=name)
gmsh.model.addPhysicalGroup(3, [vols[0][1]], name="internal")

LC_CORE, LC_BOX = 2.6, 6.0
f1=gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(f1,"VIn",LC_CORE); gmsh.model.mesh.field.setNumber(f1,"VOut",LC_BOX)
gmsh.model.mesh.field.setNumber(f1,"XMin",-W); gmsh.model.mesh.field.setNumber(f1,"XMax",W)
gmsh.model.mesh.field.setNumber(f1,"YMin",-W); gmsh.model.mesh.field.setNumber(f1,"YMax",W)
gmsh.model.mesh.field.setNumber(f1,"ZMin",Z0-8); gmsh.model.mesh.field.setNumber(f1,"ZMax",Z1+8)
gmsh.model.mesh.field.setNumber(f1,"Thickness",12)
gmsh.model.mesh.field.setAsBackgroundMesh(f1)
for o,v in [("Mesh.MeshSizeExtendFromBoundary",0),("Mesh.MeshSizeFromPoints",0),
            ("Mesh.MeshSizeFromCurvature",0),("Mesh.Algorithm",6),("Mesh.Algorithm3D",1),
            ("Mesh.Optimize",1),("Mesh.OptimizeNetgen",0)]:
    gmsh.option.setNumber(o,v)
t0=time.time(); gmsh.model.mesh.generate(3); print("mesh",round(time.time()-t0,1),"s")
gmsh.write("fluid.msh"); print(gmsh.model.mesh.getMaxNodeTag(),"nodes")

# -- cell 11 ------------------------------------------------------------------------
import os, textwrap
gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.write("fluid.msh"); gmsh.finalize()
os.makedirs("system",exist_ok=True); os.makedirs("constant",exist_ok=True)
hdr=lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;\n"
 "writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} SIMPLE{}\n")
r=subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 12 ------------------------------------------------------------------------
r=subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"],capture_output=True,text=True)
print(r.returncode, r.stdout[-400:])
print(open("constant/polyMesh/boundary").read()[-1200:])

# -- cell 13 ------------------------------------------------------------------------
b=open("constant/polyMesh/boundary").read()
for p in ("ductwalls","honeycombwalls"):
    i=b.index(p); j=b.index("type",i); b=b[:j]+b[j:].replace("patch;","wall;",1)
    k=b.index("physicalType",i); b=b[:k]+b[k:].replace("patch;","wall;",1)
open("constant/polyMesh/boundary","w").write(b)
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-2600:])

# -- cell 14 ------------------------------------------------------------------------
# Coarse mesh is topologically right and `checkMesh` is happy. Now refine: 1.6 mm in the core (≈5 cell
LC_CORE2, LC_BOX2 = 1.6, 4.0
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("hc2"); gmsh.model.occ.importShapes("fluid_mm.step"); gmsh.model.occ.synchronize()
pg2={"inlet":[],"outlet":[],"ductwalls":[],"honeycombwalls":[]}
for (d,s) in gmsh.model.getEntities(2):
    x0,y0,z0b,x1,y1,z1b = gmsh.model.getBoundingBox(d,s)
    mn=type("V",(),{"X":x0,"Y":y0,"Z":z0b})(); mx=type("V",(),{"X":x1,"Y":y1,"Z":z1b})()
    pg2[classify(mn,mx)].append(s)
for name,ss in pg2.items(): gmsh.model.addPhysicalGroup(2, ss, name=name)
gmsh.model.addPhysicalGroup(3,[gmsh.model.getEntities(3)[0][1]],name="internal")
f=gmsh.model.mesh.field.add("Box")
for k,v in [("VIn",LC_CORE2),("VOut",LC_BOX2),("XMin",-W),("XMax",W),("YMin",-W),("YMax",W),
            ("ZMin",Z0-6),("ZMax",Z1+6),("Thickness",15)]: gmsh.model.mesh.field.setNumber(f,k,v)
gmsh.model.mesh.field.setAsBackgroundMesh(f)
for o,v in [("Mesh.MeshSizeExtendFromBoundary",0),("Mesh.MeshSizeFromPoints",0),
            ("Mesh.MeshSizeFromCurvature",0),("Mesh.Algorithm",6),("Mesh.Algorithm3D",1),
            ("Mesh.Optimize",1),("Mesh.MshFileVersion",2.2)]: gmsh.option.setNumber(o,v)
t0=time.time(); gmsh.model.mesh.generate(3)
_,tets,_=gmsh.model.mesh.getElements(3); gmsh.write("fluid.msh"); gmsh.finalize()
print("fine mesh", round(time.time()-t0,1),"s  tets:", len(tets[0]))

# -- cell 15 ------------------------------------------------------------------------
subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True)
subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"],capture_output=True,text=True)
b=open("constant/polyMesh/boundary").read()
for p in ("ductwalls","honeycombwalls"):
    i=b.index(p); j=b.index("type",i); b=b[:j]+b[j:].replace("patch;","wall;",1)
    k=b.index("physicalType",i); b=b[:k]+b[k:].replace("patch;","wall;",1)
open("constant/polyMesh/boundary","w").write(b)
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[r.stdout.index("Mesh stats"):][:900])
print(r.stdout[-1400:])

# -- cell 16 ------------------------------------------------------------------------
r=subprocess.run(["foamToVTK","-constant","-ascii","-no-internal","-excludePatches","(inlet outlet)"],capture_output=True,text=True)
print(r.returncode, r.stdout[-300:])
import glob; print(glob.glob("VTK/**/*.vtp",recursive=True)+glob.glob("VTK/**/*.vtk",recursive=True))

# -- cell 17 ------------------------------------------------------------------------
hw=pv.read("VTK/t11_0/boundary/honeycombwalls.vtp"); dw=pv.read("VTK/t11_0/boundary/ductwalls.vtp")
print("honeycombwalls area m2:", hw.area, " CAD:", sum(f.area for f in groups['honeycombwalls'])*1e-6)
print("ductwalls area m2:", dw.area, " CAD:", sum(f.area for f in groups['ductwalls'])*1e-6)
pl=pv.Plotter(off_screen=True)
pl.add_mesh(hw.clip(normal='y',origin=(0,0.0,0)),color="orange",show_edges=True,line_width=0.3)
pl.add_mesh(dw.clip(normal='y',origin=(0,0.0,0)),color="lightgray",opacity=0.5,show_edges=True,line_width=0.2)
pl.camera_position=[(0.26,-0.22,0.18),(0,0,0.02),(0,0,1)]
pl.screenshot("mesh.png"); display(Image("mesh.png"))

# -- cell 18 ------------------------------------------------------------------------
ch_bb = chan.bounding_box(); c0=Pos(*centers[0][:2])*hex_sk
print(f"duct section  : {duct.bounding_box().size.X} x {duct.bounding_box().size.Y} mm  (want 80 x 80)")
print(f"upstream len  : {Z0-ZIN} mm (want 40)   downstream len: {ZOUT-Z1} mm (want 80)")
print(f"core thickness: {ch_bb.size.Z} mm (want 20), channels span z={ch_bb.min.Z}..{ch_bb.max.Z} -> open front to back")
print(f"cell across flats: {math.sqrt(3)*R_cell:.4f} mm (want 8)   wall thickness: {P_af-AF:.4f} mm (want 1)")
print(f"channels: {len(cells)} ({sum(1 for _,a in cells if a==full_area)} full hex, {len(cells)-sum(1 for _,a in cells if a==full_area)} clipped at the duct wall)")
print(f"open area {open_sk.area:.0f} mm2 / 6400 -> porosity {open_sk.area/W**2:.3f}")
