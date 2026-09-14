"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by laying out the honeycomb cell grid in 2D and checking the hexagon orientation and clip
import build123d as bd, numpy as np
W=8.0; t=1.0; p=W+t
r_cell = W/np.sqrt(3)
h = bd.RegularPolygon(radius=r_cell, side_count=6, major_radius=True)
print(type(h), h.area, h.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Orientation confirmed: 8 mm across flats in y, 9.2376 across corners in x. Now the staggered grid cl
S=80.0
dx=p*np.sqrt(3)/2; dy=p
inset = bd.Rectangle(S-2*t, S-2*t)
frags=[]; full=h.area
for i in range(-8,9):
    for j in range(-8,9):
        cx=i*dx; cy=j*dy+(i%2)*dy/2
        hx = bd.Pos(cx,cy)*h
        if abs(cx)-4.7>S/2 or abs(cy)-4.7>S/2: continue
        f = inset & hx
        if f.area > 0.4*full: frags.append((cx,cy,f))
print(len(frags), "cells;  areas min/max frac:", min(f.area for _,_,f in frags)/full, max(f.area for _,_,f in frags)/full)
print("open area frac of duct:", sum(f.area for _,_,f in frags)/(S*S))

# -- cell 3 -------------------------------------------------------------------------
# 93 cells, smallest fragment 50 % of a full cell, open area 72.9 %. Let me look at the section before
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,6))
for cx,cy,f in frags:
    for w in f.wires():
        pts=np.array([(v.X,v.Y) for v in w.vertices()]+[(w.vertices()[0].X,w.vertices()[0].Y)])
        ax.fill(pts[:,0],pts[:,1],'w',ec='k',lw=0.5)
ax.add_patch(plt.Rectangle((-40,-40),80,80,fill=False,ec='r'))
ax.set_aspect(1); ax.set_xlim(-45,45); ax.set_ylim(-45,45); ax.set_title("honeycomb section (fluid cells white)")
fig.savefig("section.png",dpi=110)
print("ok")

# -- cell 4 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("section.png"))

# -- cell 5 -------------------------------------------------------------------------
# The fill polygons are drawn from unordered vertices; let me redraw from edge geometry to see the rea
fig,ax=plt.subplots(figsize=(6,6))
for cx,cy,f in frags:
    for e in f.edges():
        a,b=e@0, e@1
        ax.plot([a.X,b.X],[a.Y,b.Y],'k-',lw=0.6)
ax.add_patch(plt.Rectangle((-40,-40),80,80,fill=False,ec='r'))
ax.set_aspect(1); ax.set_xlim(-45,45); ax.set_ylim(-45,45)
fig.savefig("section.png",dpi=110); display(Image("section.png"))

# -- cell 6 -------------------------------------------------------------------------
# Now the 3D solid: cells extruded through the full 20 mm, plate = core box minus cells, fluid = duct 
import time
L_up, L_core, L_dn = 40.0, 20.0, 80.0
cells3d = bd.Compound([bd.extrude(bd.Pos(0,0,0)*f, L_core) for _,_,f in frags])
core_box = bd.Pos(0,0,L_core/2)*bd.Box(S,S,L_core)
duct = bd.Pos(0,0,(L_core-L_up+L_dn)/2)*bd.Box(S,S,L_up+L_core+L_dn)
t0=time.time(); plate = core_box - cells3d; print("plate", time.time()-t0, plate.volume)
t0=time.time(); fluid = duct - plate; print("fluid", time.time()-t0, fluid.volume, "expected", 80*80*120+sum(f.area for _,_,f in frags)*20)
print(fluid.bounding_box(), len(fluid.solids()))

# -- cell 7 -------------------------------------------------------------------------
# Faces classified by exact plane test, not by bounding-box side. Let me group them and check areas ag
ZMIN,ZMAX=-L_up, L_core+L_dn
def classify(f):
    bb=f.bounding_box(); tol=1e-7
    flat=lambda a,b: abs(a-b)<tol
    if flat(bb.min.Z,ZMIN) and flat(bb.max.Z,ZMIN): return "inlet"
    if flat(bb.min.Z,ZMAX) and flat(bb.max.Z,ZMAX): return "outlet"
    for c,v in (("X",S/2),("X",-S/2),("Y",S/2),("Y",-S/2)):
        if flat(getattr(bb.min,c),v) and flat(getattr(bb.max,c),v): return "ductwalls"
    return "honeycomb"
groups={}
for f in fluid.faces(): groups.setdefault(classify(f),[]).append(f)
for k,v in groups.items(): print(k, len(v), round(sum(f.area for f in v),1))
print("hand: inlet/outlet 6400, ductwalls", 4*S*(L_up+L_dn), "hc sides", sum(f.area for _,_,f in frags)*0+round(sum(sum(e.length for e in f.edges()) for _,_,f in frags)*20,1), "+ 2 endfaces", round(2*(S*S-sum(f.area for _,_,f in frags)),1))

# -- cell 8 -------------------------------------------------------------------------
# Scaling to metres once, then exporting the STEP and one STL per patch.
import os
sc=0.001
fluid_m = bd.scale(fluid, sc)
def classify_s(f,s=sc):
    bb=f.bounding_box(); tol=1e-9
    flat=lambda a,b: abs(a-b)<tol
    if flat(bb.min.Z,ZMIN*s) and flat(bb.max.Z,ZMIN*s): return "inlet"
    if flat(bb.min.Z,ZMAX*s) and flat(bb.max.Z,ZMAX*s): return "outlet"
    for c,v in (("X",S/2*s),("X",-S/2*s),("Y",S/2*s),("Y",-S/2*s)):
        if flat(getattr(bb.min,c),v) and flat(getattr(bb.max,c),v): return "ductwalls"
    return "honeycomb"
g={}
for f in fluid_m.faces(): g.setdefault(classify_s(f),[]).append(f)
print({k:(len(v),round(sum(x.area for x in v),6)) for k,v in g.items()}, fluid_m.volume)
bd.export_step(fluid_m,"fluid.step")
for k,v in g.items(): bd.export_stl(bd.Compound(v), f"{k}.stl", tolerance=2e-5, angular_tolerance=0.3)
print(sorted(os.listdir(".")))

# -- cell 9 -------------------------------------------------------------------------
# Now the coarse mesh in gmsh: 3 mm in the core, 6 mm in the ducts, physical surfaces from the same cl
import gmsh
def build_mesh(core_h, duct_h, out="fluid.msh", nthreads=8):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",1)
    gmsh.model.add("hc")
    gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
    tol=1e-9; names={}
    for (d,tag) in gmsh.model.getEntities(2):
        x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(d,tag)
        f=lambda a,b: abs(a-b)<tol
        if f(z0,ZMIN*sc) and f(z1,ZMIN*sc): k="inlet"
        elif f(z0,ZMAX*sc) and f(z1,ZMAX*sc): k="outlet"
        elif any(f(a,v) and f(b,v) for a,b,v in [(x0,x1,S/2*sc),(x0,x1,-S/2*sc),(y0,y1,S/2*sc),(y0,y1,-S/2*sc)]): k="ductwalls"
        else: k="honeycomb"
        names.setdefault(k,[]).append(tag)
    for k,tags in names.items():
        gmsh.model.addPhysicalGroup(2,tags,name=k)
    gmsh.model.addPhysicalGroup(3,[v[1] for v in gmsh.model.getEntities(3)],name="fluid")
    fld=gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(fld,"VIn",core_h); gmsh.model.mesh.field.setNumber(fld,"VOut",duct_h)
    gmsh.model.mesh.field.setNumber(fld,"XMin",-1); gmsh.model.mesh.field.setNumber(fld,"XMax",1)
    gmsh.model.mesh.field.setNumber(fld,"YMin",-1); gmsh.model.mesh.field.setNumber(fld,"YMax",1)
    gmsh.model.mesh.field.setNumber(fld,"ZMin",-0.006); gmsh.model.mesh.field.setNumber(fld,"ZMax",0.026)
    gmsh.model.mesh.field.setNumber(fld,"Thickness",0.02)
    gmsh.model.mesh.field.setAsBackgroundMesh(fld)
    for o,v in [("Mesh.MeshSizeFromPoints",0),("Mesh.MeshSizeFromCurvature",0),("Mesh.MeshSizeExtendFromBoundary",0),
                ("Mesh.Algorithm",5),("Mesh.Algorithm3D",10),("General.NumThreads",nthreads),("Mesh.MshFileVersion",2.2)]:
        gmsh.option.setNumber(o,v)
    gmsh.model.mesh.generate(3)
    gmsh.write(out)
    n=len(gmsh.model.mesh.getElementsByType(4)[0]); print("tets",n, {k:len(v) for k,v in names.items()})
    gmsh.finalize(); return n
build_mesh(0.003,0.006)

# -- cell 10 ------------------------------------------------------------------------
# All 554 surfaces landed in one group — the classification test failed, so let me check what coordina
gmsh.initialize(); gmsh.model.add("chk")
gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
print(gmsh.model.getBoundingBox(-1,-1))
ents=gmsh.model.getEntities(2)
print(len(ents), [gmsh.model.getBoundingBox(2,t) for d,t in ents[:3]])
gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# gmsh pads each bounding box by its 1e-7 tolerance. Widening the test to 1e-6 m and silencing the log
import gmsh
def build_mesh(core_h, duct_h, out="fluid.msh", nthreads=8, tol=1e-6):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("hc")
    gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
    names={}
    for (d,tag) in gmsh.model.getEntities(2):
        x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(d,tag)
        f=lambda a,b: abs(a-b)<tol
        if f(z0,ZMIN*sc) and f(z1,ZMIN*sc): k="inlet"
        elif f(z0,ZMAX*sc) and f(z1,ZMAX*sc): k="outlet"
        elif any(f(a,v) and f(b,v) for a,b,v in [(x0,x1,S/2*sc),(x0,x1,-S/2*sc),(y0,y1,S/2*sc),(y0,y1,-S/2*sc)]): k="ductwalls"
        else: k="honeycomb"
        names.setdefault(k,[]).append(tag)
    for k,tags in names.items(): gmsh.model.addPhysicalGroup(2,tags,name=k)
    gmsh.model.addPhysicalGroup(3,[v[1] for v in gmsh.model.getEntities(3)],name="fluid")
    fld=gmsh.model.mesh.field.add("Box")
    for o,v in [("VIn",core_h),("VOut",duct_h),("XMin",-1),("XMax",1),("YMin",-1),("YMax",1),
                ("ZMin",-0.006),("ZMax",0.026),("Thickness",0.02)]:
        gmsh.model.mesh.field.setNumber(fld,o,v)
    gmsh.model.mesh.field.setAsBackgroundMesh(fld)
    for o,v in [("Mesh.MeshSizeFromPoints",0),("Mesh.MeshSizeFromCurvature",0),("Mesh.MeshSizeExtendFromBoundary",0),
                ("Mesh.Algorithm",5),("Mesh.Algorithm3D",10),("General.NumThreads",nthreads),("Mesh.MshFileVersion",2.2)]:
        gmsh.option.setNumber(o,v)
    gmsh.model.mesh.generate(3)
    gmsh.write(out)
    ntet=len(gmsh.model.mesh.getElementsByType(4)[1])//4
    print("tets",ntet,{k:len(v) for k,v in names.items()})
    gmsh.finalize()
build_mesh(0.003,0.006)

# -- cell 12 ------------------------------------------------------------------------
# Groups match the build123d face counts exactly. Now convert to OpenFOAM and check the coarse mesh.
import os, subprocess, textwrap
os.makedirs("system",exist_ok=True); os.makedirs("constant",exist_ok=True)
hdr=lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+textwrap.dedent("""
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} SIMPLE{}\n")
r=subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True); print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2500:])

# -- cell 14 ------------------------------------------------------------------------
# Coarse mesh is topologically clean and the volume matches the CAD exactly. Let me look at it before 
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl=pv.Plotter(off_screen=True,window_size=(900,700))
cols={"honeycomb":"tan","ductwalls":"lightsteelblue","inlet":"green","outlet":"red"}
for k,c in cols.items():
    m=pv.read(f"{k}.stl")
    pl.add_mesh(m,color=c,show_edges=False,opacity=1.0 if k!="ductwalls" else 0.25)
pl.camera_position=[(0.18,-0.15,0.16),(0,0,0.03),(0,0,1)]
pl.screenshot("view.png"); display(Image("view.png"))

# -- cell 15 ------------------------------------------------------------------------
# Shape is right. Now the refined mesh: 1.5 mm in the core (≈5 cells across a channel), 3 mm in the du
import time; t0=time.time()
build_mesh(0.0015,0.003)
print("gmsh secs", round(time.time()-t0,1))

# -- cell 16 ------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh",ignore_errors=True)
r=subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True); print(r.returncode)
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):][:900])
print(r.stdout[r.stdout.find("Checking geometry"):][:1600])

# -- cell 17 ------------------------------------------------------------------------
# Setting the two wall patches to type `wall`, and measuring the stated dimensions back off the built 
import re
b=open("constant/polyMesh/boundary").read()
for p in ["ductwalls","honeycomb"]:
    b=re.sub(r"(\b%s\n\s*\{\s*\n\s*type\s+)patch;"%p, r"\1wall;", b)
open("constant/polyMesh/boundary","w").write(b)
print(re.findall(r"(\w+)\n\s*\{\n\s*type\s+(\w+);", b))
# measurements from the CAD solid (mm)
c0=[f for cx,cy,f in frags if abs(cx)<1e-9 and abs(cy)<1e-9][0]
bb=c0.bounding_box()
print("interior cell across flats (y) mm:", round(bb.max.Y-bb.min.Y,4), "asked 8")
print("cell pitch mm:", dy, "-> wall thickness mm:", round(dy-(bb.max.Y-bb.min.Y),4), "asked 1")
fb=fluid.bounding_box()
print("duct section mm:", fb.max.X-fb.min.X, fb.max.Y-fb.min.Y, "asked 80x80")
print("core thickness mm:", L_core, "upstream", L_up, "downstream", L_dn)
print("cells through full depth? plate volume/(core box - open*20):", round(plate.volume,1), round(S*S*L_core-sum(f.area for _,_,f in frags)*L_core,1))

# -- cell 18 ------------------------------------------------------------------------
# The four STLs were tessellated per group, so shared edges may not match. I'll rewrite them from the 
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.open("fluid.msh")
phys={gmsh.model.getPhysicalName(2,t):t for d,t in gmsh.model.getPhysicalGroups(2)}
ntag,ncrd,_=gmsh.model.mesh.getNodes(); P={int(t):ncrd[3*i:3*i+3] for i,t in enumerate(ntag)}
for name,ptag in phys.items():
    tris=[]
    for ent in gmsh.model.getEntitiesForPhysicalGroup(2,ptag):
        et,_,en=gmsh.model.mesh.getElements(2,ent)
        for t_,nodes in zip(et,en):
            if t_==2: tris += [nodes[i:i+3] for i in range(0,len(nodes),3)]
    with open(f"{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for a,b,c in tris:
            p,q,r_=P[a],P[b],P[c]; n=np.cross(q-p,r_-p); n=n/(np.linalg.norm(n) or 1)
            fh.write("facet normal %g %g %g\n outer loop\n"%tuple(n))
            for v in (p,q,r_): fh.write("  vertex %.9g %.9g %.9g\n"%tuple(v))
            fh.write(" endloop\nendfacet\n")
        fh.write(f"endsolid {name}\n")
    print(name, len(tris))
gmsh.finalize()

# -- cell 19 ------------------------------------------------------------------------
# Let me verify the welded union myself: free edges and winding consistency across all four STLs.
import collections
tris=[]
for name in ["inlet","outlet","ductwalls","honeycomb"]:
    m=pv.read(f"{name}.stl"); f=m.faces.reshape(-1,4)[:,1:]; pts=m.points
    for t_ in f: tris.append([tuple(np.round(pts[i],9)) for i in t_])
de=collections.Counter()
for a,b,c in tris:
    for e in ((a,b),(b,c),(c,a)): de[e]+=1
ue=collections.Counter(tuple(sorted(e)) for e in de.elements())
print("triangles",len(tris))
print("free edges (undirected count!=2):", sum(1 for e,n in ue.items() if n!=2))
print("same-direction duplicated edges:", sum(1 for e,n in de.items() if n>1))

# -- cell 20 ------------------------------------------------------------------------
# Closed and consistently wound. Checking which way the union faces, via signed volume.
V=0.0
for a,b,c in tris:
    a,b,c=map(np.array,(a,b,c)); V+=np.dot(a,np.cross(b,c))/6
print("signed volume", V, "fluid volume", fluid_m.volume, "-> normals point", "outward" if V>0 else "inward")
