"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll build the honeycomb wall solid by punching hexagonal holes through an 80×80×20 mm plate (hex fl
import numpy as np, math
from build123d import *
from build123d import RegularPolygon, Box, Pos, extrude

F = 8.0      # mm, flat-to-flat cell width
t = 1.0      # mm wall
Fi = F - t   # 7 mm opening flat-to-flat
T = 20.0     # mm core thickness
S = 80.0     # mm duct section
up, down = 40.0, 80.0  # mm upstream / downstream

def hex_prism(cx, cy, fi, h):
    p = RegularPolygon(fi/math.sqrt(3), 6)  # radius = flat-to-flat/sqrt(3)
    return Pos(cx, cy, -h/2) * extrude(p, h)

# hex lattice: flat-to-flat pitch F in y (flat-top orientation), sqrt(3)/2*F in x
dx = F*math.sqrt(3)/2
dy = F
pts = []
iy = 0
y = -S/2 + dy
while y < S/2:
    xoff = (dx/2) if (iy % 2) else 0.0
    x = -S/2 + dx + xoff
    while x < S/2:
        pts.append((x, y)); x += dx
    y += dy/2 if False else dy*0.0  # placeholder fixed below
    # proper row pitch: rows spaced sqrt(3)/2 * F ... use staggered grid:
    y += dy*math.sqrt(3)/2 if False else 0  # fallback
    break
print("rework")

# -- cell 2 -------------------------------------------------------------------------
# Cleaning up the lattice math — flat-top hexagons: hole flat-to-flat 7 mm, horizontal pitch √3/2·8, v
import numpy as np, math
from build123d import *
from build123d import RegularPolygon, Box, Pos, extrude, Location

F=8.0; t=1.0; Fi=F-t; T=20.0; S=80.0; up=40.0; down=80.0  # mm

R = Fi/math.sqrt(3)
dx = 1.5*R          # horizontal pitch = sqrt3/2*F
dy = math.sqrt(3)*R # vertical pitch = F
pts=[]
ix=0; x=-S/2+R/2
while x < S/2 - R/2 + 1e-6:
    yoff = dy/2 if ix%2 else 0.0
    y=-S/2+dy/2 + yoff
    while y < S/2 - dy/2 + 1e-6:
        pts.append((x,y)); y += dy
    ix+=1; x += dx

hex_profile = RegularPolygon(R, 6)  # flat-to-flat 2*R*sqrt3/2 = 7mm
template = Pos(0,0,-T/2) * extrude(hex_profile, T)

holes = template
for cx,cy in pts:
    if (cx,cy)==(pts[0][0],pts[0][1]): holes = Pos(cx,cy,0)*template
    else: holes += Pos(cx,cy,0)*template
holes = holes.combine() if hasattr(holes,'combine') else holes
print("cells:", len(pts))
core = Pos(0,0,0)*Box(S,S,T) - holes
print("core volume mm^3:", core.volume, "solid fraction:", core.volume/(S*S*T))

# -- cell 3 -------------------------------------------------------------------------
# Solid fraction too low — I mixed inner and outer radii in the pitch. The pitch must come from the ou
Rout = F/math.sqrt(3); Rin = Fi/math.sqrt(3)
dx = 1.5*Rout; dy = math.sqrt(3)*Rout
pts=[]
ix=0; x=-S/2+Rout
while x <= S/2-Rout+1e-6:
    yoff = dy/2 if ix%2 else 0.0
    y=-S/2+dy/2 if ix%2==0 else -S/2+dy
    # recompute cleanly below
    y = -S/2 + (dy/2 if ix%2 else dy)
    while y <= S/2 - dy/2 + 1e-6:
        pts.append((x,y)); y += dy
    ix+=1; x += dx
cells=[]
for cx,cy in pts:
    cells.append(Pos(cx,cy,-T/2) * extrude(RegularPolygon(Rin,6), T))
from build123d import Compound
holes = Compound(children=cells)
core = Box(S,S,T) - holes
print("cells:",len(pts),"solid frac:", core.volume/(S*S*T))

# -- cell 4 -------------------------------------------------------------------------
# Let me look at the core instead of guessing at fractions:
import matplotlib.pyplot as plt
# top view: draw hole positions and verify geometry analytically
fig,ax=plt.subplots(figsize=(6,6)); ax.set_aspect('equal')
ax.add_patch(plt.Rectangle((-40,-40),80,80,fill=False))
for cx,cy in pts:
    ang=np.linspace(0,2*np.pi,7)+0
    ax.plot(cx+Rin*np.cos(ang), cy+Rin*np.sin(ang),'-')
plt.savefig('layout.png',dpi=100); plt.show()
# analytic: single cell solid fraction
nh = len(pts)
solid_each = (3*math.sqrt(3)/2)*(Rout**2-Rin**2)
print("cell solid area mm^2:", solid_each, "-> fraction if full tiling:", solid_each/(dx*dy))

# -- cell 5 -------------------------------------------------------------------------
# Layout is right. Now the fluid volume: duct 80×80 from z=−40 to z=+100, minus the honeycomb core:
duct = Pos(0,0,(-up+down)/2) * Box(S,S,up+down+T)   # z from -40 to 100
fluid = duct - core
print("fluid volume mm^3:", fluid.volume)
# classify faces (mm)
inlet=[]; outlet=[]; ductwalls=[]; honwalls=[]
for f in fluid.faces():
    c = f.center(); z=c.Z
    if abs(z+up)<1e-6: inlet.append(f)
    elif abs(z-(down+T))<1e-6: outlet.append(f)
    elif abs(abs(c.X)-S/2)<1e-6 or abs(abs(c.Y)-S/2)<1e-6: ductwalls.append(f)
    else: honwalls.append(f)
print("inlet faces:",len(inlet),"outlet:",len(outlet),"ductwalls:",len(ductwalls),"honeycomb faces:",len(honwalls))

# -- cell 6 -------------------------------------------------------------------------
zs = sorted(set(round(f.center().Z,6) for f in fluid.faces()))
print("distinct face-center z:", zs[:6], "...", zs[-6:])
big = [f for f in fluid.faces() if f.area>1500]
print("big faces:", [(round(f.area,1), round(f.center().X,2),round(f.center().Y,2),round(f.center().Z,2)) for f in big])

# -- cell 7 -------------------------------------------------------------------------
# The core and duct were both miscentered — core must span z 0→20, duct −40→+100. Fixing:
core = Pos(0,0,T/2) * (Box(S,S,T) - holes)
duct = Pos(0,0,(down-up)/2) * Box(S,S,up+down+T)  # z=-40..100
fluid = duct - core
faces = {k: [] for k in ("inlet","outlet","ductWalls","honeycombWalls")}
for f in fluid.faces():
    c=f.center()
    if abs(c.Z+up)<0.5: faces["inlet"].append(f)
    elif abs(c.Z-(down+T))<0.5: faces["outlet"].append(f)
    elif abs(c.Z-10)<9.99 and abs(abs(c.X)-S/2)<0.5 or (abs(abs(c.Y)-S/2)<0.5 and abs(c.Z-10)<9.99):
        faces["honeycombWalls"].append(f)  # would double count; decide below
    else: faces["ductWalls"].append(f)
# redo cleanly:
faces = {k: [] for k in ("inlet","outlet","ductWalls","honeycombWalls")}
for f in fluid.faces():
    c=f.center()
    if abs(c.Z+up)<1e-4: faces["inlet"].append(f)
    elif abs(c.Z-(down+T))<1e-4: faces["outlet"].append(f)
    elif abs(c.Z-10)>9.9999: faces["ductWalls"].append(f)
    else: faces["honeycombWalls"].append(f)
for k,v in faces.items(): print(k, len(v), "area mm^2:", round(sum(f.area for f in v),1))

# -- cell 8 -------------------------------------------------------------------------
import math as m
res={k:[] for k in faces}
for f in fluid.faces():
    n=f.normal_at(); bb=f.bounding_box()
    if abs(n.Z)>0.99:
        if bb.min.Z < -up+1e-3: res["inlet"].append(f)
        elif bb.max.Z > down+T-1e-3: res["outlet"].append(f)
        elif n.Z!=0: res["honeycombWalls"].append(f)
    else:
        if bb.min.Z < 0 or bb.max.Z > T: res["ductWalls"].append(f)
        else: res["honeycombWalls"].append(f)
for k,v in res.items(): print(k,len(v),round(sum(f.area for f in v),1))

# -- cell 9 -------------------------------------------------------------------------
zf=[f for f in fluid.faces() if abs(f.normal_at().Z)>0.99]
print([(round(f.area,2), round(f.bounding_box().min.Z,4), round(f.bounding_box().max.Z,4)) for f in zf][:10], "... total", len(zf))

# -- cell 10 ------------------------------------------------------------------------
# Duct center should be 30, not 20. Fixing and finalizing:
duct = Pos(0,0,(T+down-up)/2) * Box(S,S,up+down+T)  # z=-40..100
fluid = duct - core
res={k:[] for k in ("inlet","outlet","ductWalls","honeycombWalls")}
for f in fluid.faces():
    n=f.normal_at(); bb=f.bounding_box()
    if abs(n.Z)>0.99:
        if bb.min.Z < -up+1e-3: res["inlet"].append(f)
        elif bb.min.Z > T+down-1e-3: res["outlet"].append(f)
        else: res["honeycombWalls"].append(f)
    else:
        if bb.min.Z < -1e-6 or bb.max.Z > T+1e-6: res["ductWalls"].append(f)
        else: res["honeycombWalls"].append(f)
for k,v in res.items(): print(k,len(v),round(sum(f.area for f in v),1))
print("volume mm^3:", fluid.volume)

# -- cell 11 ------------------------------------------------------------------------
# Volumes add up. Now scale to metres, export, and mesh with gmsh:
fluid_m = fluid.scale(0.001)
export_step(fluid_m, "fluid.step")
import gmsh
gmsh.initialize()
gmsh.model.add("t11")
ents = gmsh.model.occ.importShapes("fluid.step")
gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3)
surfs = gmsh.model.getEntities(2)
print("volumes:", vols, "surfaces:", len(surfs))

# -- cell 12 ------------------------------------------------------------------------
groups={"inlet":[], "outlet":[], "ductWalls":[], "honeycombWalls":[]}
for (d,t) in surfs:
    cm = gmsh.model.occ.getCenterOfMass(d,t)
    xmin,ymin,zmin,xmax,ymax,zmax = gmsh.model.getBoundingBox(d,t)
    zspan = (zmax-zmin)
    eps=1e-9
    if zspan<1e-7 and zmin < -0.039: groups["inlet"].append(t)
    elif zspan<1e-7 and zmax > 0.099: groups["outlet"].append(t)
    elif zmin > -1e-7 and zmax < 0.0200001: groups["honeycombWalls"].append(t)
    else: groups["ductWalls"].append(t)
for name,ts in groups.items():
    pid = gmsh.model.addPhysicalGroup(2, ts)
    gmsh.model.setPhysicalName(2, pid, name)
    print(name, len(ts))
pid = gmsh.model.addPhysicalGroup(3,[1]); gmsh.model.setPhysicalName(3,pid,"fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0008)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.003)
gmsh.model.mesh.generate(3)
gmsh.write("t11.msh")
print("nodes:", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 13 ------------------------------------------------------------------------
# The mesh ran but patch classification collapsed to one group — getBoundingBox eps. Fixing the group 
gmsh.model.removePhysicalGroups()
groups={"inlet":[], "outlet":[], "ductWalls":[], "honeycombWalls":[]}
for (d,t) in surfs:
    cx,cy,cz = gmsh.model.occ.getCenterOfMass(d,t)
    xmin,ymin,zmin,xmax,ymax,zmax = gmsh.model.getBoundingBox(d,t)
    if zmax-zmin < 1e-4 and cz < -0.035: groups["inlet"].append(t)
    elif zmax-zmin < 1e-4 and cz > 0.095: groups["outlet"].append(t)
    elif zmin > -1e-9 and zmax < 0.020001: groups["honeycombWalls"].append(t)
    else: groups["ductWalls"].append(t)
for name,ts in groups.items():
    pid = gmsh.model.addPhysicalGroup(2, ts)
    gmsh.model.setPhysicalName(2, pid, name)
    print(name, len(ts))
pid = gmsh.model.addPhysicalGroup(3,[1]); gmsh.model.setPhysicalName(3,pid,"fluid")
gmsh.model.mesh.generate(3)
gmsh.write("t11.msh")
import subprocess
r = subprocess.run(["gmshToFoam","t11.msh"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-300:])

# -- cell 14 ------------------------------------------------------------------------
# The gmsh classification misfired — let me see the actual bounding boxes of the imported surfaces:
import collections
cnt = collections.Counter()
for (d,t) in surfs:
    zmin, zmax = gmsh.model.getBoundingBox(d,t)[2], gmsh.model.getBoundingBox(d,t)[5]
    cnt[(round(zmin,3), round(zmax,3))] += 1
for k,v in sorted(cnt.items()): print(k, v)

# -- cell 15 ------------------------------------------------------------------------
# Boxes are clean. Reassigning groups correctly now:
groups={"inlet":[], "outlet":[], "ductWalls":[], "honeycombWalls":[]}
for (d,t) in surfs:
    xmin,ymin,zmin,xmax,ymax,zmax = gmsh.model.getBoundingBox(d,t)
    if zmax<=-0.0399: groups["inlet"].append(t)
    elif zmin>=0.0999: groups["outlet"].append(t)
    elif zmax<=0.0201 and zmin>=-0.021e-3: groups["honeycombWalls"].append(t)
    else: groups["ductWalls"].append(t)
for name,ts in groups.items():
    pid=gmsh.model.addPhysicalGroup(2,ts); gmsh.model.setPhysicalName(2,pid,name)
    print(name,len(ts))
gmsh.model.mesh.generate(3); gmsh.write("t11.msh")
import subprocess, os
os.makedirs("system",exist_ok=True); os.makedirs("constant",exist_ok=True)
open("system/controlDict","w").write("FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}\napplication x; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;")
r=subprocess.run(["gmshToFoam","t11.msh"],capture_output=True,text=True)
print(r.stdout[-400:] or r.stderr[-400:])

# -- cell 16 ------------------------------------------------------------------------
print(r.stdout[r.stdout.find("foamBoundary"):][:300] if "foamBoundary" in r.stdout else r.stdout[-300:])
rc=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(rc.stdout[-1500:])

# -- cell 17 ------------------------------------------------------------------------
tail = rc.stdout[rc.stdout.find("Create mesh"):]
print(tail[-2200:])

# -- cell 18 ------------------------------------------------------------------------
print(subprocess.run(["find",".","-maxdepth","3"],capture_output=True,text=True).stdout)
print(rc.stderr[-800:])

# -- cell 19 ------------------------------------------------------------------------
# gmshToFoam expects MSH v2 — regenerating in that format:
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("t11.msh")
r=subprocess.run(["gmshToFoam","t11.msh"],capture_output=True,text=True)
print(r.stdout[-500:], r.stderr[-200:])
rc=subprocess.run(["checkMesh"],capture_output=True,text=True)
print((rc.stdout+rc.stderr)[-2000:])

# -- cell 20 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\nddtSchemes{}gradSchemes{}divSchemes{}laplacianSchemes{}interpolationSchemes{}snGradSchemes{}")
open("system/fvSolution","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}")
rc=subprocess.run(["checkMesh"],capture_output=True,text=True)
out=rc.stdout+rc.stderr
print(out[out.find("Mesh stats"):out.find("Mesh stats")+1500])

# -- cell 21 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read()[:2500])

# -- cell 22 ------------------------------------------------------------------------
# Duplicate surfaces look likely — the import may have kept standalone faces alongside the solid. Chec
import re
txt=open("t11.msh").read()
els=re.findall(r'\n(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',txt[txt.find("$Elements"):])
import collections
c=collections.Counter()
for e in els:
    typ,ntags,phys,elem = int(e[1]),int(e[2]),int(e[3]),int(e[4])
    if typ==2: c[(phys)] += 1
print("triangle count by physical id:", c)
