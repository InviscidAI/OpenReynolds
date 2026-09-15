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
# SI metres
duct_L, duct_W, duct_H = 0.2, 0.03, 0.03
blk_L, blk_W, blk_H = 0.04, 0.03, 0.01
x0 = (duct_L - blk_L)/2.0  # 0.08
solid = bd.Pos(x0, 0, 0) * bd.Box(blk_L, blk_W, blk_H)  # on floor z=0..blk_H
duct = bd.Pos(-duct_L/2, -duct_W/2, -duct_H/2) * bd.Box(duct_L, duct_W, duct_H)
fluid = duct - solid
print("solid bbox:", solid.bounding_box().min, solid.bounding_box().max)
print("fluid vol:", fluid.volume, "solid vol:", solid.volume)
import matplotlib.pyplot as plt
fig = plt.figure(); ax = fig.add_subplot(projection='3d')
from build123d import export_stl
export_stl(fluid, "fluid_visu.stl"); export_stl(solid, "solid_visu.stl")
import pyvista as pv
pv.set_plot_theme("document")
p = pv.Plotter(off_screen=True)
p.add_mesh(pv.read("fluid_visu.stl"), opacity=0.4, color="lightblue")
p.add_mesh(pv.read("solid_visu.stl"), color="brown")
p.show()

# -- cell 2 -------------------------------------------------------------------------
import os, subprocess
from collections import Counter
xs=[0,0.08,0.12,0.2]; Y=0.03; Zs=[0,0.01,0.03]
verts={}; vlist=[]; blocks=[]
def V(x,y,z):
    k=(x,y,z)
    if k not in verts: verts[k]=len(vlist); vlist.append(k)
    return verts[k]
def hexb(x0,x1,y0,y1,z0,z1,nx,ny,nz):
    v=(V(x0,y0,z0),V(x1,y0,z0),V(x1,y1,z0),V(x0,y1,z0),V(x0,y0,z1),V(x1,y0,z1),V(x1,y1,z1),V(x0,y1,z1))
    blocks.append((v,(nx,ny,nz)))
NU,NB,ZS,ZO,NY=53,27,5,10,20
hexb(0,0.08,0,Y,0,0.01,NU,NY,ZS); hexb(0.08,0.12,0,Y,0,0.01,NB,NY,ZS); hexb(0.12,0.2,0,Y,0,0.01,NU,NY,ZS)
hexb(0,0.08,0,Y,0.01,0.03,NU,NY,ZO); hexb(0.08,0.12,0,Y,0.01,0.03,NB,NY,ZO); hexb(0.12,0.2,0,Y,0.01,0.03,NU,NY,ZO)
fd=[(0,1,2,3),(4,7,6,5),(1,5,6,2),(0,4,7,3),(3,2,6,7),(0,1,5,4)]
cnt=Counter(); refs={}
for v,n in blocks:
    for f in fd:
        t=tuple(v[i] for i in f); key=tuple(sorted(t))
        cnt[key]+=1; refs[key]=t
patches={'inlet':[],'outlet':[],'walls':[]}
eps=1e-9
for key,t in refs.items():
    if cnt[key]==2: continue
    P=[vlist[i] for i in t]
    if max(p[0] for p in P)-min(p[0] for p in P)<eps:
        patches['inlet' if P[0][0]==0 else 'outlet'].append(t)
    else: patches['walls'].append(t)
print({k:len(v) for k,v in patches.items()},"verts",len(vlist))
s="FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\nscale 1;\nvertices (\n"
s+="\n".join(f" ({p[0]} {p[1]} {p[2]})" for p in vlist)+"\n);\nblocks (\n"
s+="\n".join(" hex ("+" ".join(map(str,b))+") "+" ".join(map(str,n))+" simpleGrading (1 1 1)" for b,n in blocks)+"\n);\npatches (\n"
for name in ['inlet','outlet','walls']:
    ftype='wall' if name=='walls' else 'patch'
    s+=f" {name} {ftype} (\n"+"\n".join("  ("+" ".join(map(str,f))+")" for f in patches[name])+"\n );\n"
s+=");\nmergePatchPairs ();\n"
os.makedirs("system/constant",exist_ok=True)
open("system/blockMeshDict","w").write(s)
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print((r.stdout+r.stderr)[-800:])

# -- cell 3 -------------------------------------------------------------------------
import subprocess, os
open("system/controlDict","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application chtMultiRegionFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1000; purgeWrite 0;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 8; runTimeModifiable false;
""")
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print((r.stdout+r.stderr)[-400:])

# -- cell 4 -------------------------------------------------------------------------
import subprocess
from collections import Counter
def build_dict():
    global blocks,vlist
    verts={}; vlist=[]; blocks=[]
    def V(x,y,z):
        k=(x,y,z)
        if k not in verts: verts[k]=len(vlist); vlist.append(k)
        return verts[k]
    def hexb(x0,x1,y0,y1,z0,z1,nx,ny,nz):
        v=(V(x0,y0,z0),V(x1,y0,z0),V(x1,y1,z0),V(x0,y1,z0),
           V(x0,y0,z1),V(x1,y0,z1),V(x1,y1,z1),V(x0,y1,z1))
        blocks.append((v,(nx,ny,nz)))
    NU,NB,ZS,ZO,NY=53,27,5,10,20
    hexb(0.0,0.08,0,0.03,0,0.01,NU,NY,ZS); hexb(0.08,0.12,0,0.03,0,0.01,NB,NY,ZS); hexb(0.12,0.2,0,0.03,0,0.01,NU,NY,ZS)
    hexb(0.0,0.08,0,0.03,0.01,0.03,NU,NY,ZO); hexb(0.08,0.12,0,0.03,0.01,0.03,NB,NY,ZO); hexb(0.12,0.2,0,0.03,0.01,0.03,NU,NY,ZO)
    fd=[(0,1,2,3),(4,7,6,5),(1,5,6,2),(0,4,7,3),(3,2,6,7),(0,1,5,4)]
    cnt=Counter(); refs={}
    for v,n in blocks:
        for f in fd:
            t=tuple(v[i] for i in f); key=tuple(sorted(t)); cnt[key]+=1; refs[key]=t
    patches={'inlet':[],'outlet':[],'walls':[]}
    for key,t in refs.items():
        if cnt[key]==2: continue
        P=[vlist[i] for i in t]
        if max(p[0] for p in P)-min(p[0] for p in P)<1e-9:
            patches['inlet' if abs(P[0][0])<1e-9 else 'outlet'].append(t)
        else: patches['walls'].append(t)
    s="FoamFile{ version 2.0; format ascii; class dictionary; object blockMeshDict; }\nscale 1;\nvertices(\n"
    s+="\n".join(f" ({p[0]} {p[1]} {p[2]})" for p in vlist)+"\n);\nblocks(\n"
    s+="\n".join(" hex("+" ".join(map(str,b))+") "+" ".join(map(str,n))+" simpleGrading(1 1 1)" for b,n in blocks)+"\n);\nboundary(\n"
    for name in ['inlet','outlet','walls']:
        ftype='wall' if name=='walls' else 'patch'
        s+=f" {name}{{ type {ftype}; faces(\n"
        s+="\n".join("  ("+" ".join(map(str,f))+")" for f in patches[name]); s+="\n);}\n"
    s+=");\nmergePatchPairs();\n"
    return s
open("system/blockMeshDict","w").write(build_dict())
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print((r.stdout+r.stderr)[-400:])

# -- cell 5 -------------------------------------------------------------------------
import subprocess
s = build_dict()
s = s.replace("vertices(", "vertices (").replace("blocks(", "blocks (").replace("boundary(", "boundary (")
s = s.replace(" hex(", " hex (").replace("faces(", "faces (").replace("mergePatchPairs!", "x")  # no-op guard
# brace entries "{ type patch; faces (...)" are fine
open("system/blockMeshDict","w").write(s)
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print((r.stdout+r.stderr)[-300:])

# -- cell 6 -------------------------------------------------------------------------
import subprocess
s = build_dict()
for a,b in [("vertices(","vertices ("),("blocks(","blocks ("),("boundary(","boundary ("),(" hex("," hex ("),("faces(","faces ("),("simpleGrading(","simpleGrading (")]:
    s=s.replace(a,b)
open("system/blockMeshDict","w").write(s)
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print((r.stdout+r.stderr)[-200:])

# -- cell 7 -------------------------------------------------------------------------
import subprocess
topo = """FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions (
 { name fluid; type cellZone; action new;
   sources (
     boxToCell { action add; box ((0 0 0) (0.2 0.03 0.03)); }
     boxToCell { action remove; box ((0.08 0 0) (0.12 0.03 0.01)); }
   );
 }
 { name solid; type cellZone; action new;
   sources (
     boxToCell { action add; box ((0.08 0 0) (0.12 0.03 0.01)); }
   );
 }
);
"""
open("system/topoSetDict","w").write(topo)
for cmd in ["topoSet","splitMeshRegions"]:
    r=subprocess.run([cmd],capture_output=True,text=True)
    print("\n==",cmd,r.returncode,"\n",(r.stdout+r.stderr)[-500:])

# -- cell 8 -------------------------------------------------------------------------
import subprocess
open("system/topoSetDict","w").write(open("system/topoSetDict").read().replace("type cellZone;","type cellZoneSet;"))
open("system/fvSchemes","w").write("FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n")
open("system/fvSolution","w").write("FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n")
for cmd in ["topoSet","splitMeshRegions"]:
    r=subprocess.run([cmd],capture_output=True,text=True)
    print("\n==",cmd,r.returncode,"\n",(r.stdout+r.stderr)[-700:])

# -- cell 9 -------------------------------------------------------------------------
print(open("system/topoSetDict").read()[:200])

# -- cell 10 ------------------------------------------------------------------------
import subprocess
tt=open("system/topoSetDict").read()
print(repr(tt[:250]))

# -- cell 11 ------------------------------------------------------------------------
import subprocess
open("system/topoSetDict","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions (
 { name fluid; type cellZoneSet; action new;
   sources (
   boxToCell { action add; box ((0 0 0) (0.2 0.03 0.03)); }
   boxToCell { action remove; box ((0.08 0 0) (0.12 0.03 0.01)); }
   );
 }
 { name solid; type cellZoneSet; action new;
   sources ( boxToCell { action add; box ((0.08 0 0) (0.12 0.03 0.01)); } );
 }
);
""")
open("system/fvSchemes","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {} interpolationSchemes {} snGradSchemes {}
""")
open("system/fvSolution","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {}
""")
for cmd in ["topoSet","splitMeshRegions"]:
    r=subprocess.run([cmd],capture_output=True,text=True)
    print("\n==",cmd,r.returncode,"\n",(r.stdout+r.stderr)[-700:])

# -- cell 12 ------------------------------------------------------------------------
import subprocess
open("system/topoSetDict","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions (
 { name fluid; type cellSet; action new;
   source ( boxToCell ); sourceInfo { box ((0 0 0) (0.2 0.03 0.03)); } }
 { name fluid; type cellSet; action remove;
   source ( boxToCell ); sourceInfo { box ((0.08 0 0) (0.12 0.03 0.01)); } }
 { name fluid; type cellZoneSet; action new; source ( cellToCell ); sourceInfo { set fluid; } }
 { name solid; type cellSet; action new;
   source ( boxToCell ); sourceInfo { box ((0.08 0 0) (0.12 0.03 0.01)); } }
 { name solid; type cellZoneSet; action new; source ( cellToCell ); sourceInfo { set solid; } }
);
""")
r=subprocess.run(["topoSet"],capture_output=True,text=True); print(r.returncode,(r.stdout+r.stderr)[-600:])
r2=subprocess.run(["splitMeshRegions","-overwrite"],capture_output=True,text=True)
print("\n",r2.returncode,(r2.stdout+r2.stderr)[-800:])

# -- cell 13 ------------------------------------------------------------------------
import subprocess
t = open("system/topoSetDict").read()
t = t.replace("source ( boxToCell );","source boxToCell;").replace("source ( cellToCell );","source cellToCell;")
open("system/topoSetDict","w").write(t)
r=subprocess.run(["topoSet"],capture_output=True,text=True)
r2=subprocess.run(["splitMeshRegions","-overwrite"],capture_output=True,text=True)
print(r.returncode, r2.returncode,(r2.stdout+r2.stderr)[-500:])

# -- cell 14 ------------------------------------------------------------------------
import subprocess
r=subprocess.run(["topoSet"],capture_output=True,text=True)
print((r.stdout+r.stderr)[-1200:])

# -- cell 15 ------------------------------------------------------------------------
import subprocess
t=open("system/topoSetDict").read()
t=t.replace("box ((0 0 0) (0.2 0.03 0.03));","box (0 0 0) (0.2 0.03 0.03);")
t=t.replace("box ((0.08 0 0) (0.12 0.03 0.01));","box (0.08 0 0) (0.12 0.03 0.01);")
open("system/topoSetDict","w").write(t)
r=subprocess.run(["topoSet"],capture_output=True,text=True)
print(r.returncode,(r.stdout+r.stderr)[-400:])
r2=subprocess.run(["splitMeshRegions","-overwrite"],capture_output=True,text=True)
print(r2.returncode,(r2.stdout+r2.stderr)[-600:])

# -- cell 16 ------------------------------------------------------------------------
import subprocess
t="""FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions (
 { name fluid; type cellSet; action new;
   source boxToCell; sourceInfo { box (0 0 0) (0.08 0.03 0.03); } }
 { name fluid; type cellSet; action add;
   source boxToCell; sourceInfo { box (0.12 0 0) (0.2 0.03 0.03); } }
 { name fluid; type cellSet; action add;
   source boxToCell; sourceInfo { box (0.08 0 0.01) (0.12 0.03 0.03); } }
 { name fluid; type cellZoneSet; action new; source cellToCell; sourceInfo { set fluid; } }
 { name solid; type cellSet; action new;
   source boxToCell; sourceInfo { box (0.08 0 0) (0.12 0.03 0.01); } }
 { name solid; type cellZoneSet; action new; source cellToCell; sourceInfo { set solid; } }
);
"""
open("system/topoSetDict","w").write(t)
r=subprocess.run(["topoSet"],capture_output=True,text=True)
r2=subprocess.run(["splitMeshRegions","-overwrite"],capture_output=True,text=True)
print(r.returncode,(r.stdout+r.stderr)[-200:]); print(r2.returncode,(r2.stdout+r2.stderr)[-700:])
