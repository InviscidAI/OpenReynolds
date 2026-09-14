"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by importing the STEP file and finding out what's actually in it.
import subprocess
p = "/home/qiuzi/.openreynolds-buildup/work/T5-20260914-111032-2f8a/geometry/assembly.step"
print(subprocess.run(["head","-40",p],capture_output=True,text=True).stdout)
print(subprocess.run(["grep","-c","","%s"%p],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
print(subprocess.run(["grep","-n","-E","LENGTH_UNIT|SI_UNIT|PRODUCT\\(|MANIFOLD_SOLID|CLOSED_SHELL|ADVANCED_BREP",p],capture_output=True,text=True).stdout[:3000])

# -- cell 3 -------------------------------------------------------------------------
print(subprocess.run(["sed","-n","2405,2440p",p],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
import build123d as bd, time
t=time.time()
asm = bd.import_step(p)
print(type(asm), time.time()-t)
solids = asm.solids()
print(len(solids))
for s in solids:
    print(s.label, s.bounding_box(), s.volume)
print(asm.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
print(subprocess.run(["grep","-n","-m","10","CARTESIAN_POINT",p],capture_output=True,text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
names=["BASE","EMITTER","BODY"]
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,s,n in zip(axs,solids,names):
    for pl,idx,lbl in [(0,(0,2),"XZ")]:
        pass
    # slice through x=0 plane: intersect with a thin section -> use faces projection instead
    verts, tris = s.tessellate(0.3)
    v=np.array([(pt.X,pt.Y,pt.Z) for pt in verts])
    ax.plot(v[:,0],v[:,2],'.',ms=0.5)
    ax.set_title(n); ax.set_aspect('equal')
plt.savefig("xz.png",dpi=90)
plt.close()
print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("xz.png"))

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
def mesh_of(s, tol=0.2):
    v,t = s.tessellate(tol)
    pts=np.array([(q.X,q.Y,q.Z) for q in v]); f=np.array(t)
    return pv.PolyData(pts, np.hstack([np.full((len(f),1),3),f]).ravel())
pl=pv.Plotter(off_screen=True, window_size=(900,700))
cols=["red","green","blue"]
for s,c,n in zip(solids,cols,names): pl.add_mesh(mesh_of(s), color=c, opacity=0.5, label=n)
pl.add_legend(); pl.camera_position='iso'
img=pl.screenshot(return_img=True); plt.figure(figsize=(9,7)); plt.imshow(img); plt.axis('off'); plt.gcf()

# -- cell 9 -------------------------------------------------------------------------
for s,n in zip(solids,names):
    print(n, "shells", len(s.shells()), "faces", len(s.faces()), "vol", round(s.volume,1), "area", round(s.area,1))

# -- cell 10 ------------------------------------------------------------------------
# The previous section drew all three solids in one colour and told me little. Re-emitting with the co
asm_c = bd.Compound(solids)
cols=["red","green","blue"]
pl=pv.Plotter(off_screen=True, window_size=(900,700))
for s,c,n in zip(solids,cols,names):
    piece = s.intersect(bd.Box(120,60,120).moved(bd.Location((0,30,0))))
    for q in piece.solids(): pl.add_mesh(mesh_of(q,0.1), color=c)
pl.camera_position='xz'; img=pl.screenshot(return_img=True)
plt.figure(figsize=(9,7)); plt.imshow(img); plt.axis('off'); plt.gcf()

# -- cell 11 ------------------------------------------------------------------------
u = solids[0].fuse(solids[1], solids[2]).clean()
print(len(u.solids()), [len(s.shells()) for s in u.solids()], u.volume, u.bounding_box())

# -- cell 12 ------------------------------------------------------------------------
for s in u.solids(): print(round(s.volume,1), s.bounding_box())

# -- cell 13 ------------------------------------------------------------------------
for nm,s in [("BASE",solids[0]),("BODY",solids[2])]:
    sl = s.intersect(bd.Box(120,120,0.2).moved(bd.Location((0,0,2.5))))
    for q in sl.solids():
        bb=q.bounding_box(); print(nm, round(q.volume/0.2,1), bb)
    print(nm,"faces at z-slice", len(sl.solids()))

# -- cell 14 ------------------------------------------------------------------------
for z in [0.2,0.8,1.2,1.6,2.0,3.2]:
    sl = solids[0].intersect(bd.Box(120,120,0.05).moved(bd.Location((0,0,z))))
    print(z, [ (round(q.volume/0.05,1)) for q in sl.solids()])
print("BODY z slices:")
for z in [1.6,2.0,5,10,15,19.5]:
    sl = solids[2].intersect(bd.Box(120,120,0.05).moved(bd.Location((0,0,z))))
    print(z, [ (round(q.volume/0.05,1)) for q in sl.solids()])

# -- cell 15 ------------------------------------------------------------------------
bot = [f for f in solids[0].faces() if abs(f.center().Z)<1e-4]
print(len(bot))
for f in bot: print(round(f.area,1), len(f.inner_wires()), f.bounding_box())
