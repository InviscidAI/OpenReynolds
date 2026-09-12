"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, numpy as np
from build123d import *

mm = 1e-3
DUCT_L, DUCT_W, DUCT_H = 200*mm, 30*mm, 30*mm
BLK_L, BLK_W, BLK_T = 40*mm, 30*mm, 10*mm
X0 = (DUCT_L - BLK_L)/2.0          # 80 mm, block centred halfway along

fluid = Pos(DUCT_L/2, DUCT_W/2, DUCT_H/2) * Box(DUCT_L, DUCT_W, DUCT_H)
solid = Pos(X0+BLK_L/2, BLK_W/2, -BLK_T/2) * Box(BLK_L, BLK_W, BLK_T)

print("fluid bbox (m):", fluid.bounding_box().min, fluid.bounding_box().max)
print("solid bbox (m):", solid.bounding_box().min, solid.bounding_box().max)
print("solid dims mm:", [round(v/mm,3) for v in (solid.bounding_box().size.X, solid.bounding_box().size.Y, solid.bounding_box().size.Z)])
print("solid volume mm^3:", round(solid.volume/mm**3,1), " expected", 40*30*10)
print("block centre x mm:", round(solid.center().X/mm,3), " duct mid", round(DUCT_L/2/mm,3))
print("interface area mm^2:", round(BLK_L*BLK_W/mm**2,1))
export_step(Compound(children=[fluid, solid]), "duct_block.step")
print("wrote", os.path.abspath("duct_block.step"))

# -- cell 2 -------------------------------------------------------------------------
# Shape measures exactly as asked. Now look at it.
import pyvista as pv
pv.start_xvfb() if False else None
pv.OFF_SCREEN = True
export_stl(fluid, "fluid.stl"); export_stl(solid, "solid.stl")
p = pv.Plotter(window_size=(1000,500))
p.add_mesh(pv.read("fluid.stl"), color="lightblue", opacity=0.35, show_edges=True)
p.add_mesh(pv.read("solid.stl"), color="orange", show_edges=True)
p.add_axes(); p.camera_position = [(0.30,-0.30,0.20),(0.10,0.015,0.0),(0,0,1)]
p.screenshot("look.png")
print("ok")

# -- cell 3 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look.png"))

# -- cell 4 -------------------------------------------------------------------------
# Flush-mounted as intended. Now the conformal blockMesh: x split at 80/120 mm so the fluid floor face
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

XS = [0.0, X0, X0+BLK_L, DUCT_L]      # 0, 80, 120, 200 mm
YS = [0.0, DUCT_W]
ZS = [-BLK_T, 0.0, DUCT_H]            # -10, 0, 30 mm
def V(ix,iy,iz): return iz*8 + iy*4 + ix
verts = [f"({XS[ix]:.6g} {YS[iy]:.6g} {ZS[iz]:.6g})" for iz in range(3) for iy in range(2) for ix in range(4)]

def hexb(ix,iz,n):   # block spanning x[ix..ix+1], full y, z[iz..iz+1]
    a=[V(ix,0,iz),V(ix+1,0,iz),V(ix+1,1,iz),V(ix,1,iz)]
    b=[V(ix,0,iz+1),V(ix+1,0,iz+1),V(ix+1,1,iz+1),V(ix,1,iz+1)]
    return f"    hex ({' '.join(map(str,a+b))}) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)"
NX=[16,8,16]; NY=10; NZF=10; NZS=4
blocks=[hexb(i,1,(NX[i],NY,NZF)) for i in range(3)] + [hexb(1,0,(NX[1],NY,NZS))]

def fx(ix,iz):  return f"({V(ix,0,iz)} {V(ix,0,iz+1)} {V(ix,1,iz+1)} {V(ix,1,iz)})"      # const-x face
def fy(ix,iy,iz):return f"({V(ix,iy,iz)} {V(ix+1,iy,iz)} {V(ix+1,iy,iz+1)} {V(ix,iy,iz+1)})" # const-y
def fz(ix,iz):  return f"({V(ix,0,iz)} {V(ix+1,0,iz)} {V(ix+1,1,iz)} {V(ix,1,iz)})"      # const-z
inlet=[fx(0,1)]; outlet=[fx(3,1)]
duct=[fz(i,2) for i in range(3)] + [fy(i,j,1) for i in range(3) for j in (0,1)] + [fz(0,1), fz(2,1)]
blkw=[fz(1,0)] + [fy(1,j,0) for j in (0,1)] + [fx(1,0), fx(2,0)]
def patch(name,typ,faces): return f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n" + "".join(f"            {f}\n" for f in faces) + "        );\n    }\n"

d = "FoamFile\n{ version 2.0; format ascii; class dictionary; object blockMeshDict; }\n\nscale 1;\n\nvertices\n(\n"
d += "".join(f"    {v}\n" for v in verts) + ");\n\nblocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges ();\n\nboundary\n(\n"
d += patch("inlet","patch",inlet)+patch("outlet","patch",outlet)+patch("ductWalls","wall",duct)+patch("blockWalls","wall",blkw)+");\n\nmergePatchPairs ();\n"
open("system/blockMeshDict","w").write(d)

open("system/controlDict","w").write("FoamFile{version 2.0; format ascii; class dictionary; object controlDict;}\napplication chtMultiRegionFoam;\nstartFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\nwriteControl timeStep; writeInterval 1; writeFormat ascii; timeFormat general; runTimeModifiable true;\n")
open("system/fvSchemes","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSchemes;}\nddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0; format ascii; class dictionary; object fvSolution;}\nsolvers{}\n")
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.stdout[-1500:], r.stderr[-1500:])

# -- cell 5 -------------------------------------------------------------------------
# Now cellZones and the region split; the interface faces are internal, so the split produces conforma
ts = """FoamFile{version 2.0; format ascii; class dictionary; object topoSetDict;}
actions
(
  { name solidCells; type cellSet; action new; source boxToCell; box (0.079 -0.1 -0.011) (0.121 0.1 -0.0001); }
  { name solid; type cellZoneSet; action new; source setToCellZone; set solidCells; }
  { name fluidCells; type cellSet; action new; source boxToCell; box (-0.1 -0.1 0.0001) (0.3 0.1 0.1); }
  { name fluid; type cellZoneSet; action new; source setToCellZone; set fluidCells; }
);
"""
open("system/topoSetDict","w").write(ts)
r=subprocess.run(["topoSet"],capture_output=True,text=True); print(r.stdout[-700:])
r=subprocess.run(["splitMeshRegions","-cellZones","-overwrite"],capture_output=True,text=True)
print(r.stdout[-2000:]); print("ERR",r.stderr[-500:])

# -- cell 6 -------------------------------------------------------------------------
# Interface is 80 faces = 8×10 over the 40×30 mm footprint, as intended. Check both regions.
for reg in ("fluid","solid"):
    r=subprocess.run(["checkMesh","-region",reg],capture_output=True,text=True)
    tail=r.stdout.strip().splitlines()
    print("=== ",reg); print("\n".join(tail[-22:]))

# -- cell 7 -------------------------------------------------------------------------
# Both regions clean and volumes exact (fluid 1.8e‑4 m³ = 200×30×30 mm³; solid 1.2e‑5 m³ = 40×30×10 mm
import shutil
NX=[32,20,32]; NY=15; NZF=15; NZS=5
blocks=[hexb(i,1,(NX[i],NY,NZF)) for i in range(3)] + [hexb(1,0,(NX[1],NY,NZS))]
d = "FoamFile\n{ version 2.0; format ascii; class dictionary; object blockMeshDict; }\n\nscale 1;\n\nvertices\n(\n"
d += "".join(f"    {v}\n" for v in verts) + ");\n\nblocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges ();\n\nboundary\n(\n"
d += patch("inlet","patch",inlet)+patch("outlet","patch",outlet)+patch("ductWalls","wall",duct)+patch("blockWalls","wall",blkw)+");\n\nmergePatchPairs ();\n"
open("system/blockMeshDict","w").write(d)
for p in ("constant/fluid","constant/solid","0"):
    shutil.rmtree(p, ignore_errors=True)
for cmd in (["blockMesh"],["topoSet"],["splitMeshRegions","-cellZones","-overwrite"]):
    r=subprocess.run(cmd,capture_output=True,text=True); print(cmd[0],"rc",r.returncode)
print([l for l in r.stdout.splitlines() if "Faces" in l or "\t" in l][-8:])

# -- cell 8 -------------------------------------------------------------------------
# Verify patch inventory and measure the interface area from the mesh itself, then check both regions.
import re
for reg in ("fluid","solid"):
    b=open(f"constant/{reg}/polyMesh/boundary").read()
    names=re.findall(r"^\s{4}(\w+)$",b,re.M)
    print(reg,"patches:",names)
    r=subprocess.run(["checkMesh","-region",reg,"-writeAllFields"],capture_output=True,text=True)
    print(" ".join([l.strip() for l in r.stdout.splitlines() if "Mesh OK" in l or "cells:" in l or "***" in l or "Max skewness" in l or "non-orthogonality Max" in l]))
r=subprocess.run(["checkMesh","-region","fluid","-allTopology"],capture_output=True,text=True)
print([l for l in r.stdout.splitlines() if "fluid_to_solid" in l])

# -- cell 9 -------------------------------------------------------------------------
# Parsing bug; also the faces file may be compact format. Use a more careful reader.
import numpy as np, re
def read_patch(reg,pname):
    P=open(f"constant/{reg}/polyMesh/points").read(); F=open(f"constant/{reg}/polyMesh/faces").read()
    B=open(f"constant/{reg}/polyMesh/boundary").read()
    pts=np.array([[float(x) for x in m.split()] for m in re.findall(r"\(([^()]*)\)",P.split("(",1)[1])])
    faces=[[int(x) for x in m.split()] for m in re.findall(r"\d+\(([\d\s]+)\)",F)]
    blk=re.search(pname+r"\s*\{(.*?)\}",B,re.S).group(1)
    n=int(re.search(r"nFaces\s+(\d+)",blk).group(1)); s=int(re.search(r"startFace\s+(\d+)",blk).group(1))
    typ=re.search(r"type\s+(\w+)",blk).group(1); fl=faces[s:s+n]
    area=sum(np.linalg.norm(sum(np.cross(pts[f[i]],pts[f[(i+1)%len(f)]]) for i in range(len(f))))/2 for f in fl)
    ids=sorted({i for f in fl for i in f})
    return typ,n,area,np.array(sorted(map(tuple,np.round(pts[ids],9))))
tf,nf,af,pf=read_patch("fluid","fluid_to_solid"); ts,ns,asd,ps=read_patch("solid","solid_to_fluid")
print("fluid_to_solid:",tf,nf,"faces, area mm^2",round(af/mm**2,3))
print("solid_to_fluid:",ts,ns,"faces, area mm^2",round(asd/mm**2,3),"expected",40*30)
print("point sets identical:",pf.shape==ps.shape and np.allclose(pf,ps), pf.shape)
print("interface z(mm):",sorted({round(p[2]/mm,6) for p in pf}),
      " x range mm:",round(pf[:,0].min()/mm,3),round(pf[:,0].max()/mm,3),
      " y range mm:",round(pf[:,1].min()/mm,3),round(pf[:,1].max()/mm,3))

# -- cell 10 ------------------------------------------------------------------------
# Conformal confirmed (identical 336 interface points). Last: verify inlet/outlet and render both regi
for pn in ("inlet","outlet","ductWalls"):
    t,n,a,p=read_patch("fluid",pn)
    print(pn,t,n,"faces area mm^2",round(a/mm**2,2),"x mm",round(p[:,0].min()/mm,2),round(p[:,0].max()/mm,2))
t,n,a,p=read_patch("solid","blockWalls"); print("blockWalls",t,n,"faces area mm^2",round(a/mm**2,2),"expected",2*(40*10)+2*(30*10)+40*30)
subprocess.run(["foamToVTK","-region","fluid","-constant","-excludePatches","(inlet outlet)"],capture_output=True)
subprocess.run(["foamToVTK","-region","solid","-constant"],capture_output=True)
import glob; print(glob.glob("VTK/**/*.vtp",recursive=True)[:6], glob.glob("VTK/*"))

# -- cell 11 ------------------------------------------------------------------------
p = pv.Plotter(window_size=(1100,550)); 
p.add_mesh(pv.read("VTK/fluid/t8_0/boundary/ductWalls.vtp"), color="lightblue", opacity=0.30, show_edges=True, line_width=1)
p.add_mesh(pv.read("VTK/fluid/t8_0/boundary/fluid_to_solid.vtp"), color="red", show_edges=True)
p.add_mesh(pv.read("VTK/solid/t8_0/boundary/blockWalls.vtp"), color="orange", show_edges=True)
p.camera_position=[(0.30,-0.28,0.18),(0.10,0.015,0.0),(0,0,1)]; p.add_axes(); p.screenshot("mesh.png")
display(Image("mesh.png"))
