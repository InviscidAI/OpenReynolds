"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run that as a tool call.
import subprocess, os, math
import build123d as bd

os.makedirs("constant/triSurface", exist_ok=True)

R_SHAFT = 0.020        # 40 mm dia sealed length
R_BORE  = 0.020050     # 40.1 mm dia bore -> 0.05 mm radial clearance
L       = 0.060        # 60 mm bore / sealed length
KEY_W   = 0.012        # 12 mm keyway width
KEY_D   = 0.005        # 5 mm keyway depth
KEY_X0  = R_SHAFT - KEY_D   # keyway floor at x = 15 mm

shaft_solid = bd.Cylinder(R_SHAFT, L)
keyway  = bd.Pos(KEY_X0 + (KEY_D + 0.004)/2, 0, 0) * bd.Box(KEY_D + 0.004, KEY_W, L)
shaft_keyed = shaft_solid - keyway
fluid = bd.Cylinder(R_BORE, L) - shaft_keyed

bb = fluid.bounding_box()
print("bbox min/max:", bb.min, bb.max)
print("volume m^3:", fluid.volume)
print("clearance mm:", (R_BORE-R_SHAFT)*1e3, "length mm:", L*1e3)
print("n faces:", len(fluid.faces()))

# -- cell 2 -------------------------------------------------------------------------
# Volume matches annulus + pocket (3.6e-6 pocket + 3.8e-7 film). Now classify the 7 faces by their own
import numpy as np
fl = fluid.faces()
for i,f in enumerate(fl):
    c = f.center()
    n = f.normal_at(c) if f.geom_type == bd.GeomType.PLANE else None
    print(i, f.geom_type, "area=%.3e"%f.area, "ctr=(%.5f,%.5f,%.5f)"%(c.X,c.Y,c.Z),
          "R=%.5f"%getattr(f,'radius',float('nan')) if f.geom_type==bd.GeomType.CYLINDER else ("n=%s"%(tuple(round(v,3) for v in (n.X,n.Y,n.Z)),)))

# -- cell 3 -------------------------------------------------------------------------
import inspect
print(inspect.signature(bd.export_stl))

# -- cell 4 -------------------------------------------------------------------------
from build123d import Compound

def isc(f, R): return f.geom_type==bd.GeomType.CYLINDER and abs(f.radius-R)<1e-6
def isp(f, ax, sign):
    if f.geom_type!=bd.GeomType.PLANE: return False
    n = f.normal_at(f.center()); v=(n.X,n.Y,n.Z)[ax]
    return abs(v-sign)<1e-6

patches = {
 "bore":        [f for f in fl if isc(f, R_BORE)],
 "shaft":       [f for f in fl if isc(f, R_SHAFT)],
 "keywayFloor": [f for f in fl if isp(f,0,-1)],
 "keywayFlanks":[f for f in fl if isp(f,1,1) or isp(f,1,-1)],
 "highPressure":[f for f in fl if isp(f,2,-1)],   # z=-L/2 : 50 mm (oil) end
 "lowPressure": [f for f in fl if isp(f,2, 1)],   # z=+L/2 : 30 mm end
}
tot=0
for k,v in patches.items():
    a=sum(f.area for f in v); tot+=a
    bd.export_stl(Compound(v), f"constant/triSurface/{k}.stl", tolerance=2e-6, angular_tolerance=0.05)
    print(k, len(v), "area=%.4e"%a)
print("sum areas %.4e vs shape %.4e"%(tot, fluid.area))

# -- cell 5 -------------------------------------------------------------------------
import glob, numpy as np, struct
def read_stl(p):
    d=open(p,'rb').read()
    n=struct.unpack('<I', d[80:84])[0]
    tris=np.frombuffer(d[84:84+50*n], dtype=np.uint8).reshape(n,50)
    v=tris[:,12:48].copy().view('<f4').reshape(n,3,3)
    return v.astype(np.float64)
V=np.vstack([read_stl(p) for p in sorted(glob.glob("constant/triSurface/*.stl"))])
print("tris:", len(V))
q=np.round(V.reshape(-1,3)/1e-9).astype(np.int64)
uq,inv=np.unique(q,axis=0,return_inverse=True); inv=inv.reshape(-1,3)
from collections import Counter
ec=Counter(); dc=Counter()
for a,b,c in inv:
    for e in ((a,b),(b,c),(c,a)):
        ec[tuple(sorted(e))]+=1; dc[e]+=1
print("free edges:", sum(1 for k,v in ec.items() if v!=2))
print("same-direction dup:", sum(1 for k,v in dc.items() if v>1))

# -- cell 6 -------------------------------------------------------------------------
# Now the mesh. The cross-section is constant, so I build it in 2D and extrude: a transfinite quad ban
import gmsh, math
THS = math.asin((KEY_W/2)/R_SHAFT)          # 17.46 deg: where flank meets shaft circle
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("seal")
o = gmsh.model.occ
P1 = o.addPoint(R_SHAFT*math.cos(THS),  KEY_W/2, 0)
P2 = o.addPoint(R_SHAFT*math.cos(THS), -KEY_W/2, 0)
Q1 = o.addPoint(R_BORE*math.cos(THS),   R_BORE*math.sin(THS), 0)
Q2 = o.addPoint(R_BORE*math.cos(THS),  -R_BORE*math.sin(THS), 0)
F1 = o.addPoint(KEY_X0,  KEY_W/2, 0); F2 = o.addPoint(KEY_X0, -KEY_W/2, 0)
C  = o.addPoint(0,0,0)
a_shaft = o.addCircleArc(P1, C, P2, center=True)                      # long way? check below
a_borefilm = o.addCircleArc(Q1, C, Q2, center=True)
a_boremouth= o.addCircleArc(Q2, C, Q1, center=True)
l_i1 = o.addLine(P1, Q1); l_i2 = o.addLine(P2, Q2)
l_f1 = o.addLine(P1, F1); l_floor = o.addLine(F1, F2); l_f2 = o.addLine(F2, P2)
o.synchronize()
for t,nm in [(a_shaft,"shaft arc"),(a_borefilm,"bore film arc"),(a_boremouth,"bore mouth arc")]:
    print(nm, "len=%.5f"%(gmsh.model.occ.getMass(1,t)))
print("expected shaft arc len %.5f, mouth %.5f"%( (2*math.pi-2*THS)*R_SHAFT, 2*THS*R_BORE))

# -- cell 7 -------------------------------------------------------------------------
# addCircleArc takes the minor arc, so I split the 325° arcs at θ=π into two 162.5° halves.
gmsh.clear(); gmsh.model.add("seal")
o = gmsh.model.occ
def pt(r,th): return o.addPoint(r*math.cos(th), r*math.sin(th), 0)
C  = o.addPoint(0,0,0)
P1, P2 = pt(R_SHAFT,THS), pt(R_SHAFT,-THS); PM = pt(R_SHAFT, math.pi)
Q1, Q2 = pt(R_BORE,THS),  pt(R_BORE,-THS);  QM = pt(R_BORE, math.pi)
F1 = o.addPoint(KEY_X0,  KEY_W/2, 0); F2 = o.addPoint(KEY_X0, -KEY_W/2, 0)
s1 = o.addCircleArc(P1,C,PM,center=True); s2 = o.addCircleArc(PM,C,P2,center=True)
b1 = o.addCircleArc(Q1,C,QM,center=True); b2 = o.addCircleArc(QM,C,Q2,center=True)
bm = o.addCircleArc(Q2,C,Q1,center=True)          # mouth arc across the keyway
i1 = o.addLine(P1,Q1); i2 = o.addLine(P2,Q2)
f1 = o.addLine(P1,F1); ff = o.addLine(F1,F2); f2 = o.addLine(F2,P2)
lf = o.addCurveLoop([s1,s2,i2,-b2,-b1,-i1])       # film band
lp = o.addCurveLoop([i1,-bm,-i2,-f2,-ff,-f1])     # pocket
Sf = o.addPlaneSurface([lf]); Sp = o.addPlaneSurface([lp])
o.synchronize()
print("film area %.4e (exp 5.683e-06), pocket %.4e (exp 5.696e-05)"%(o.getMass(2,Sf), o.getMass(2,Sp)))
print("sum %.5e vs fluid/L %.5e"%(o.getMass(2,Sf)+o.getMass(2,Sp), fluid.volume/L))

# -- cell 8 -------------------------------------------------------------------------
NTH, NR, NZ = 200, 3, 40          # circumferential / across 0.05mm film / axial
out = o.extrude([(2,Sf),(2,Sp)], 0,0,L, numElements=[NZ], recombine=True)
o.synchronize()
vols = [t for d,t in out if d==3]
# base mesh constraints (film structured, pocket graded triangles)
for c in (s1,s2,b1,b2): gmsh.model.mesh.setTransfiniteCurve(c, NTH+1)
for c in (i1,i2):       gmsh.model.mesh.setTransfiniteCurve(c, NR+1)
gmsh.model.mesh.setTransfiniteSurface(Sf, cornerTags=[P1,P2,Q2,Q1])
gmsh.model.mesh.setRecombine(2, Sf)
fd = gmsh.model.mesh.field.add("Distance"); gmsh.model.mesh.field.setNumbers(fd,"CurvesList",[i1,i2])
ft = gmsh.model.mesh.field.add("Threshold")
for k,v in [("InField",fd),("SizeMin",3e-5),("SizeMax",5e-4),("DistMin",1e-4),("DistMax",1.5e-3)]:
    gmsh.model.mesh.field.setNumber(ft,k,v)
gmsh.model.mesh.field.setAsBackgroundMesh(ft)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0); gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",0)
print("volumes:", vols, "V=%.5e vs %.5e"%(sum(o.getMass(3,v) for v in vols), fluid.volume))

# -- cell 9 -------------------------------------------------------------------------
import numpy as np
groups = {k:[] for k in ["shaft","keywayFloor","keywayFlanks","bore","highPressure","lowPressure"]}
for d,t in gmsh.model.getEntities(2):
    ups = gmsh.model.getAdjacencies(2,t)[0]
    if len(ups)!=1: continue                      # internal interface
    u0,u1,v0,v1 = gmsh.model.getParametrizationBounds(2,t)[0][0],gmsh.model.getParametrizationBounds(2,t)[1][0],gmsh.model.getParametrizationBounds(2,t)[0][1],gmsh.model.getParametrizationBounds(2,t)[1][1]
    p = gmsh.model.getValue(2,t,[(u0+u1)/2,(v0+v1)/2]); r = math.hypot(p[0],p[1])
    if   abs(p[2]-0)<1e-9 and abs(gmsh.model.occ.getCenterOfMass(2,t)[2])<1e-9: g="highPressure"
    elif abs(p[2]-L)<1e-9 and abs(gmsh.model.occ.getCenterOfMass(2,t)[2]-L)<1e-9: g="lowPressure"
    elif abs(r-R_BORE)<1e-7:  g="bore"
    elif abs(r-R_SHAFT)<1e-7: g="shaft"
    elif abs(p[0]-KEY_X0)<1e-7: g="keywayFloor"
    elif abs(abs(p[1])-KEY_W/2)<1e-7: g="keywayFlanks"
    else: raise RuntimeError(("unclassified",t,p))
    groups[g].append(t)
for k,v in groups.items():
    gmsh.model.addPhysicalGroup(2, v, name=k)
    print(k, v, "area=%.4e"%sum(gmsh.model.occ.getMass(2,t) for t in v))
gmsh.model.addPhysicalGroup(3, vols, name="internal")

# -- cell 10 ------------------------------------------------------------------------
# Mesh and solid must sit at the same place: the extrusion runs z=0..L, so I re-export the STLs from t
fluid_p = bd.Pos(0,0,L/2) * fluid
flp = fluid_p.faces()
patches_p = {
 "bore":        [f for f in flp if isc(f, R_BORE)],
 "shaft":       [f for f in flp if isc(f, R_SHAFT)],
 "keywayFloor": [f for f in flp if isp(f,0,-1)],
 "keywayFlanks":[f for f in flp if isp(f,1,1) or isp(f,1,-1)],
 "highPressure":[f for f in flp if isp(f,2,-1)],
 "lowPressure": [f for f in flp if isp(f,2, 1)],
}
for k,v in patches_p.items():
    bd.export_stl(Compound(v), f"constant/triSurface/{k}.stl", tolerance=2e-6, angular_tolerance=0.05)
bb2 = fluid_p.bounding_box(); print("STL z range:", bb2.min.Z, bb2.max.Z)
print("hp z:", patches_p["highPressure"][0].center().Z, " lp z:", patches_p["lowPressure"][0].center().Z)

# -- cell 11 ------------------------------------------------------------------------
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("seal.msh")
ts, _ = gmsh.model.mesh.getElementsByType(5)[0:2] if False else (None,None)
for et,nm in [(4,"tet"),(5,"hex"),(6,"prism")]:
    try: n=len(gmsh.model.mesh.getElementsByType(et)[0])
    except Exception: n=0
    print(nm, n)

# -- cell 12 ------------------------------------------------------------------------
import textwrap, os
os.makedirs("system", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr%"controlDict"+
 "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;\n"
 "writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr%"fvSchemes"+
 "ddtSchemes{default Euler;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr%"fvSolution"+"solvers{} PISO{nCorrectors 1;}\n")
r = subprocess.run(["gmshToFoam","seal.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-3000:])

# -- cell 14 ------------------------------------------------------------------------
# `.*` (defaultFaces) has 40204 faces and is *closed* — the boundary looks duplicated. Let me count wh
from collections import Counter
lines=open("seal.msh").read().split("\n")
i=lines.index("$Elements"); n=int(lines[i+1])
c=Counter(); phys=Counter()
for L_ in lines[i+2:i+2+n]:
    f=L_.split(); c[int(f[1])]+=1; phys[(int(f[1]),int(f[3]))]+=1
print(n, c)
print(phys)
print([ (gmsh.model.getPhysicalName(d,t), gmsh.model.getEntitiesForPhysicalGroup(d,t)) for d,t in gmsh.model.getPhysicalGroups()])

# -- cell 15 ------------------------------------------------------------------------
print(subprocess.run(["grep","-n","-E","type|nFaces|^ *[a-zA-Z]+$","constant/polyMesh/boundary"],capture_output=True,text=True).stdout)

# -- cell 16 ------------------------------------------------------------------------
for p in ["shaft","bore","keywayFlanks","keywayFloor"]:
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/type","-set","wall"],capture_output=True,text=True)
print(subprocess.run(["grep","-n","type","constant/polyMesh/boundary"],capture_output=True,text=True).stdout)

# -- cell 17 ------------------------------------------------------------------------
# Now look at it: the cross-section mesh (zoomed on the film and on the keyway corner) and the 3D patc
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pv.OFF_SCREEN=True
m = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").close()
m = pv.OpenFOAMReader("case.foam"); m.set_active_time_value(0.0); m.disable_all_patch_arrays()
m.enable_patch_array("internalMesh"); grid = m.read()["internalMesh"]
sl = grid.slice(normal=(0,0,1), origin=(0,0,0.03))
pl = pv.Plotter(shape=(1,3), window_size=(1500,520), off_screen=True)
pl.subplot(0,0); pl.add_mesh(sl, show_edges=True, line_width=0.3, color='lightblue'); pl.view_xy(); pl.camera.zoom(1.3)
pl.subplot(0,1); pl.add_mesh(sl, show_edges=True, line_width=0.6, color='lightblue')
pl.view_xy(); pl.camera.focal_point=(0.0191,0.006,0.03); pl.camera.position=(0.0191,0.006,0.04); pl.camera.parallel_projection=True; pl.camera.parallel_scale=0.0012
pl.subplot(0,2); pl.add_mesh(grid.extract_surface(), show_edges=False, color='tan'); pl.view_isometric()
pl.screenshot("view.png"); print("ok")

# -- cell 18 ------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
img = mpimg.imread("view.png")
fig = plt.figure(figsize=(15,5.2)); plt.imshow(img); plt.axis('off'); plt.show()

# -- cell 19 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 20 ------------------------------------------------------------------------
# Numbers from the mesh itself, patch by patch, against the request.
m2 = pv.OpenFOAMReader("case.foam"); m2.set_active_time_value(0.0)
m2.disable_all_patch_arrays()
for p in ["shaft","bore","keywayFloor","keywayFlanks","highPressure","lowPressure"]:
    m2.enable_patch_array("patch/"+p)
b = m2.read()["boundary"]
for p in ["shaft","bore","keywayFloor","keywayFlanks","highPressure","lowPressure"]:
    pts = b[p].points; r = np.hypot(pts[:,0],pts[:,1])
    print("%-13s n=%5d  r[mm] %.4f..%.4f  x[mm] %.3f..%.3f  y[mm] %.3f..%.3f  z[mm] %.2f..%.2f"%(
      p,len(pts),r.min()*1e3,r.max()*1e3,pts[:,0].min()*1e3,pts[:,0].max()*1e3,
      pts[:,1].min()*1e3,pts[:,1].max()*1e3,pts[:,2].min()*1e3,pts[:,2].max()*1e3))
print("\nkeyway width  = %.3f mm (req 12)"%((b["keywayFlanks"].points[:,1].max()-b["keywayFlanks"].points[:,1].min())*1e3))
print("keyway depth  = %.3f mm (req 5)"%((R_SHAFT-b["keywayFloor"].points[:,0].max())*1e3))
print("clearance     = %.4f mm (req 0.05)"%((np.hypot(*b["bore"].points[:,:2].T).min()-np.hypot(*b["shaft"].points[:,:2].T).max())*1e3))
print("sealed length = %.2f mm (req 60)"%((b["lowPressure"].points[:,2].mean()-b["highPressure"].points[:,2].mean())*1e3))
