"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll first inspect the STEP header and import it with OCCT/build123d to establish the declared units
from pathlib import Path
import re, build123d as bd
step_path = Path('/home/qiuzi/.openreynolds-buildup/work/T5-20260916-033119-95c3/geometry/assembly.step')
header = step_path.read_text(errors='ignore')[:20000]
unit_lines = [ln.strip() for ln in header.splitlines() if 'SI_UNIT' in ln or 'CONVERSION_BASED_UNIT' in ln]
print('STEP bytes:', step_path.stat().st_size)
print('Unit declarations:', unit_lines[:20])
shape = bd.import_step(step_path)
print('Imported type:', type(shape).__name__)
print('solids:', len(shape.solids()), 'shells:', len(shape.shells()), 'faces:', len(shape.faces()))
print('valid:', shape.is_valid)
print('bbox:', shape.bounding_box().min, shape.bounding_box().max, 'size:', shape.bounding_box().size)
print('volume:', shape.volume)

# -- cell 2 -------------------------------------------------------------------------
# The import is a valid three-solid compound with dimensions consistent with millimetres, but the unit
import subprocess, inspect
print(subprocess.run(['grep','-inE','SI_UNIT|CONVERSION_BASED_UNIT|LENGTH_UNIT',str(step_path)],capture_output=True,text=True).stdout[-4000:])
for i,s in enumerate(shape.solids()):
    bb=s.bounding_box()
    print(f'solid {i}: valid={s.is_valid}, volume={s.volume:.6g}, bbox=({bb.min.X:.3f},{bb.min.Y:.3f},{bb.min.Z:.3f})..({bb.max.X:.3f},{bb.max.Y:.3f},{bb.max.Z:.3f}), faces={len(s.faces())}')
print('tessellate signature:', inspect.signature(shape.tessellate))
verts, tris = shape.tessellate(0.15)
import numpy as np, pyvista as pv
V=np.array([[v.X,v.Y,v.Z] for v in verts]); T=np.array(tris,dtype=int)
mesh=pv.PolyData(V,np.c_[np.full(len(T),3),T])
pl=pv.Plotter(off_screen=True,window_size=(900,700)); pl.add_mesh(mesh,color='lightgray',show_edges=True,edge_color='dimgray'); pl.view_isometric(); pl.add_axes(); pl.show(screenshot='assembly.png')
print('rendered triangles:',len(T))

# -- cell 3 -------------------------------------------------------------------------
# The file explicitly declares metres, but its numerical extents are only 0–54 if read literally; beca
lines=step_path.read_text(errors='ignore').splitlines()
for n in range(2395,2435): print(f'{n+1}: {lines[n]}')
colors=['#d9a441','#5aa0d6','#75b36a']
pl=pv.Plotter(off_screen=True,shape=(1,2),window_size=(1200,600))
for j in range(2):
    pl.subplot(0,j)
    for i,s in enumerate(shape.solids()):
        vv,tt=s.tessellate(0.12); v=np.array([[p.X,p.Y,p.Z] for p in vv]); t=np.array(tt,int)
        pd=pv.PolyData(v,np.c_[np.full(len(t),3),t])
        if j==1: pd=pd.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
        pl.add_mesh(pd,color=colors[i],show_edges=(j==1),opacity=0.9)
    pl.add_axes(); pl.view_isometric()
pl.show(screenshot='solids_cutaway.png')

# -- cell 4 -------------------------------------------------------------------------
# The first overlap probe encountered a null intersection (meaning at least one pair only touches), so
from collections import Counter
solids=list(shape.solids())
for i in range(3):
 for j in range(i+1,3):
  common=solids[i] & solids[j]
  print(f'common {i}-{j}:', 'null/touching only' if common is None else f'volume={common.volume:.9g}, solids={len(common.solids())}, valid={common.is_valid}')
for i,s in enumerate(solids):
 print('\nSOLID',i,'surface types',Counter(str(f.geom_type) for f in s.faces()))
 rows=[]
 for k,f in enumerate(s.faces()):
  c=f.center(); rows.append((f.area,k,str(f.geom_type),c.X,c.Y,c.Z))
 for r in sorted(rows,reverse=True)[:20]: print(' area=%9.3f face=%2d %-15s center=(%7.2f,%7.2f,%6.2f)'%r)

# -- cell 5 -------------------------------------------------------------------------
# The face/volume inventory shows a critical CAD fact: these are three nearly solid enclosure pieces, 
print(subprocess.run(['grep','-inE','PRODUCT\(|PRODUCT_DEFINITION_FORMATION|MANIFOLD_SOLID_BREP',str(step_path)],capture_output=True,text=True).stdout[:5000])
# Sample inside/outside occupancy of each solid on x=0 cross-section using OCCT point classification
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
ys=np.linspace(-25,25,101); zs=np.linspace(0,32,65)
occ=np.zeros((len(zs),len(ys)),int)
for iz,z in enumerate(zs):
 for iy,y in enumerate(ys):
  for i,s in enumerate(solids):
   cl=BRepClass3d_SolidClassifier(s.wrapped,gp_Pnt(0,float(y),float(z)),1e-6)
   if cl.State() in (TopAbs_IN,TopAbs_ON): occ[iz,iy]|=(1<<i)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5)); plt.imshow(occ,origin='lower',extent=[ys[0],ys[-1],zs[0],zs[-1]],aspect='equal',interpolation='nearest',cmap='tab10'); plt.colorbar(label='solid occupancy bitmask'); plt.xlabel('y'); plt.ylabel('z'); plt.title('x=0 section occupancy'); plt.tight_layout(); plt.savefig('section_x0.png',dpi=160); plt.show()
print('occupied fractions by bitmask:',dict(zip(*np.unique(occ,return_counts=True))))
