"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by importing the STEP file and inspecting what it actually contains.
import subprocess
p="/home/qiuzi/.openreynolds-buildup/work/T5-20260913-135927-b3b0/geometry/assembly.step"
print(subprocess.run(["head","-40",p],capture_output=True,text=True).stdout)
print(subprocess.run(["grep","-c","","%s"%p],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
out=subprocess.run(["grep","-n","-E","LENGTH_UNIT|SI_UNIT|MANIFOLD_SOLID|PRODUCT\\(|CLOSED_SHELL",p],capture_output=True,text=True).stdout
print(out[:3000])

# -- cell 3 -------------------------------------------------------------------------
print(subprocess.run(["sed","-n","2405,2435p",p],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
from build123d import *
import time
t=time.time()
asm=import_step(p)
print(type(asm), time.time()-t)
solids=asm.solids()
print(len(solids))
for s in solids:
    print(s.label, s.bounding_box(), s.volume)
print(asm.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
# Unit is declared METRE but coordinates are ~54×46×31 — millimetre magnitude for an LD19 lidar housin
import pyvista as pv, numpy as np
pv.start_xvfb() if False else None
def show(shapes, fname, opacity=0.5):
    pl=pv.Plotter(off_screen=True, window_size=(900,700))
    cols=["red","green","blue","yellow"]
    for i,s in enumerate(shapes):
        v,f=s.tessellate(0.2)
        pts=np.array([[q.X,q.Y,q.Z] for q in v]); faces=np.hstack([[3,*t] for t in f])
        pl.add_mesh(pv.PolyData(pts,faces), color=cols[i%4], opacity=opacity, show_edges=False)
    pl.add_axes(); pl.show_grid()
    pl.camera_position='iso'
    pl.screenshot(fname)
show(solids,"solids.png",0.55)
from IPython.display import Image, display
display(Image("solids.png"))

# -- cell 6 -------------------------------------------------------------------------
union = solids[0]+solids[1]+solids[2]
print(union.volume, len(union.solids()), len(union.shells()))
bb=union.bounding_box(); print(bb)
# section at x=0
sect = union & Box(0.2, 200, 200)
show([sect],"sect_x.png",1.0)
display(Image("sect_x.png"))

# -- cell 7 -------------------------------------------------------------------------
for i,s in enumerate(solids):
    print(i, s.label, "shells:",len(s.shells()), "faces:",len(s.faces()), "vol",s.volume, "bboxvol", s.bounding_box().volume if hasattr(s.bounding_box(),'volume') else None)
def views(shape,fname,cps=('iso','xy','xz','yz')):
    pl=pv.Plotter(off_screen=True, shape=(2,2), window_size=(1000,800))
    v,f=shape.tessellate(0.2); pts=np.array([[q.X,q.Y,q.Z] for q in v]); faces=np.hstack([[3,*t] for t in f])
    m=pv.PolyData(pts,faces)
    for k,cp in enumerate(cps):
        pl.subplot(k//2,k%2); pl.add_mesh(m,color="lightblue",show_edges=False); pl.camera_position=cp; pl.add_text(str(cp))
    pl.screenshot(fname)
views(union,"union.png")
display(Image("union.png"))

# -- cell 8 -------------------------------------------------------------------------
for s in union.solids():
    print(s.volume, s.bounding_box())
box = Box(80,80,60).moved(Location((0,3,15)))
comp = box - union
print("complement solids:", len(comp.solids()), [round(s.volume,1) for s in comp.solids()], "box vol",box.volume)

# -- cell 9 -------------------------------------------------------------------------
c = box - solids[0] - solids[1] - solids[2]
print(len(c.solids()), [round(s.volume,1) for s in c.solids()], 384000-41193.9)
print("valid flags:", solids[0].is_valid, solids[1].is_valid, solids[2].is_valid)

# -- cell 10 ------------------------------------------------------------------------
views(solids[0],"base.png"); display(Image("base.png"))

# -- cell 11 ------------------------------------------------------------------------
# I'll run that inventory cell properly.
radii=[]
for i,s in enumerate(solids):
    for f in s.faces():
        try: r=f.radius
        except Exception: r=None
        if r is not None: radii.append((i,round(r,3),round(f.area,3)))
print(len(radii))
for t in sorted(radii,key=lambda t:t[1])[:30]: print(t)

# -- cell 12 ------------------------------------------------------------------------
# Still defeaturing. Polling again.
print("poll2")

# -- cell 13 ------------------------------------------------------------------------
print("poll")

# -- cell 14 ------------------------------------------------------------------------
print([n for n in ("df","body","body_df","small") if n in dir()])
print("df" in globals(), "body_df" in globals())
