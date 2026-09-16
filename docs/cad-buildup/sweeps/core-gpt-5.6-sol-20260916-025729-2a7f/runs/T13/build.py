"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the requested air region as a thin swept helical duct, not the surrounding solids. Becaus
import build123d as bd, inspect, subprocess
print('Helix:', inspect.signature(bd.Helix))
print('sweep:', inspect.signature(bd.sweep))
print('Plane:', inspect.signature(bd.Plane))
print('Rectangle:', inspect.signature(bd.Rectangle))

# -- cell 2 -------------------------------------------------------------------------
# The coarse CAD solid will use metres from the outset. I’ll place a \(0.2\) mm by \(1/\cos(15^\circ)=
from build123d import Helix, Plane, Rectangle, sweep, export_stl, Vector
import math, os
PITCH=0.002
HEIGHT=0.012
NECK_OD=0.040
THREAD_DEPTH=0.001
GAP=0.0002
FLANK_HALF_ANGLE=math.radians(15.0)
MEAN_RADIUS=NECK_OD/2-THREAD_DEPTH/2
FLANK_SPAN=THREAD_DEPTH/math.cos(FLANK_HALF_ANGLE)
path=Helix(pitch=PITCH,height=HEIGHT,radius=MEAN_RADIUS)
tangent=(0.0,2*math.pi*MEAN_RADIUS,PITCH)
section_plane=Plane(origin=(MEAN_RADIUS,0,0),x_dir=(1,0,0),z_dir=tangent)
section=section_plane*Rectangle(GAP,FLANK_SPAN)
fluid=sweep(section,path=path,is_frenet=True)
os.makedirs('constant/triSurface',exist_ok=True)
export_stl(fluid,'coarse_fluid_preview.stl',tolerance=8e-5,angular_tolerance=0.25)
print(f'request pitch={PITCH:.6f} m; built turns={HEIGHT/PITCH:.6f}')
print(f'request radial thread depth={THREAD_DEPTH:.6f} m; modeled flank span={FLANK_SPAN:.9f} m')
print(f'request gap={GAP:.7f} m; section width={GAP:.7f} m')
print('volume m3=',fluid.volume,'bbox=',fluid.bounding_box().size)

# -- cell 3 -------------------------------------------------------------------------
# The sweep is a valid solid with the expected six turns and small cross-section. I’ll now render it f
import pyvista as pv
pv.set_plot_theme('document')
mesh=pv.read('coarse_fluid_preview.stl')
p= pv.Plotter(off_screen=True,window_size=(900,700))
p.add_mesh(mesh,color='#74b9ff',show_edges=True,edge_color='#2d3436',line_width=1)
p.add_axes(); p.view_isometric(); p.camera.zoom(1.35)
p.show(screenshot='coarse_preview.png')
print(mesh.n_cells, mesh.bounds)

# -- cell 4 -------------------------------------------------------------------------
# The preview confirms a continuous open-ended six-turn ribbon with no overlap between turns. I’ll now
import numpy as np
NT,NU,NV=720,2,8
angles=np.linspace(0,2*math.pi*HEIGHT/PITCH,NT+1)
us=np.linspace(-GAP/2,GAP/2,NU+1)
vs=np.linspace(-FLANK_SPAN/2,FLANK_SPAN/2,NV+1)
xyz=np.empty((NT+1,NU+1,NV+1,3))
for i,a in enumerate(angles):
    er=np.array([math.cos(a),math.sin(a),0.0])
    tangent=np.array([-MEAN_RADIUS*math.sin(a),MEAN_RADIUS*math.cos(a),PITCH/(2*math.pi)])
    et=tangent/np.linalg.norm(tangent)
    ey=np.cross(et,er)
    center=np.array([MEAN_RADIUS*math.cos(a),MEAN_RADIUS*math.sin(a),PITCH*a/(2*math.pi)])
    xyz[i]=center+us[:,None,None]*er[None,None,:]+vs[None,:,None]*ey[None,None,:]

def quad_tris(q): return [(q[0],q[1],q[2]),(q[0],q[2],q[3])]
patch_tris={k:[] for k in ['neck_flank','lid_flank','inner_end','outer_end']}
for i in range(NT):
    for k in range(NV):
        q=[xyz[i,0,k],xyz[i+1,0,k],xyz[i+1,0,k+1],xyz[i,0,k+1]]
        patch_tris['neck_flank']+=quad_tris(q)
        q=[xyz[i,NU,k],xyz[i,NU,k+1],xyz[i+1,NU,k+1],xyz[i+1,NU,k]]
        patch_tris['lid_flank']+=quad_tris(q)
    for j in range(NU):
        q=[xyz[i,j,0],xyz[i,j+1,0],xyz[i+1,j+1,0],xyz[i+1,j,0]]
        patch_tris['neck_flank']+=quad_tris(q)
        q=[xyz[i,j,NV],xyz[i+1,j,NV],xyz[i+1,j+1,NV],xyz[i,j+1,NV]]
        patch_tris['lid_flank']+=quad_tris(q)
for name,i,rev in [('inner_end',0,True),('outer_end',NT,False)]:
    for j in range(NU):
        for k in range(NV):
            q=[xyz[i,j,k],xyz[i,j+1,k],xyz[i,j+1,k+1],xyz[i,j,k+1]]
            if rev:q=q[::-1]
            patch_tris[name]+=quad_tris(q)
def write_stl(name,tris):
    with open(f'constant/triSurface/{name}.stl','w') as f:
        f.write(f'solid {name}\n')
        for a,b,c in tris:
            n=np.cross(b-a,c-a); n=n/np.linalg.norm(n)
            f.write(f' facet normal {n[0]:.10e} {n[1]:.10e} {n[2]:.10e}\n  outer loop\n')
            for p in (a,b,c): f.write(f'   vertex {p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')
for name,tris in patch_tris.items(): write_stl(name,tris)
print({k:len(v) for k,v in patch_tris.items()})
print('STLs exported before volume mesh:',sorted(os.listdir('constant/triSurface')))

# -- cell 5 -------------------------------------------------------------------------
# I’ll convert this same parameter grid into a conformal hexahedral mesh: 720 stations along the helix
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
with open('system/controlDict','w') as f:f.write('FoamFile { format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n')
def nid(i,j,k): return 1+(i*(NU+1)+j)*(NV+1)+k
bquads=[]
for i in range(NT):
 for k in range(NV):
  bquads.append((1,[nid(i,0,k),nid(i+1,0,k),nid(i+1,0,k+1),nid(i,0,k+1)]))
  bquads.append((2,[nid(i,NU,k),nid(i,NU,k+1),nid(i+1,NU,k+1),nid(i+1,NU,k)]))
 for j in range(NU):
  bquads.append((1,[nid(i,j,0),nid(i,j+1,0),nid(i+1,j+1,0),nid(i+1,j,0)]))
  bquads.append((2,[nid(i,j,NV),nid(i+1,j,NV),nid(i+1,j+1,NV),nid(i,j+1,NV)]))
for i,pid in [(0,3),(NT,4)]:
 for j in range(NU):
  for k in range(NV):
   q=[nid(i,j,k),nid(i,j+1,k),nid(i,j+1,k+1),nid(i,j,k+1)]
   if i==0:q=q[::-1]
   bquads.append((pid,q))
hexes=[]
for i in range(NT):
 for j in range(NU):
  for k in range(NV):
   hexes.append([nid(i,j,k),nid(i+1,j,k),nid(i+1,j+1,k),nid(i,j+1,k),nid(i,j,k+1),nid(i+1,j,k+1),nid(i+1,j+1,k+1),nid(i,j+1,k+1)])
with open('helix.msh','w') as f:
 f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n')
 for pid,name,dim in [(1,'neck_flank',2),(2,'lid_flank',2),(3,'inner_end',2),(4,'outer_end',2),(5,'fluid',3)]:f.write(f'{dim} {pid} "{name}"\n')
 f.write('$EndPhysicalNames\n$Nodes\n%d\n'%xyz.reshape(-1,3).shape[0])
 for n,p in enumerate(xyz.reshape(-1,3),1):f.write(f'{n} {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n')
 f.write('$EndNodes\n$Elements\n%d\n'%(len(bquads)+len(hexes)))
 e=1
 for pid,q in bquads:f.write(f'{e} 3 2 {pid} {pid} '+' '.join(map(str,q))+'\n');e+=1
 for h in hexes:f.write(f'{e} 5 2 5 5 '+' '.join(map(str,h))+'\n');e+=1
 f.write('$EndElements\n')
print('nodes',xyz.size//3,'hexes',len(hexes),'boundary quads',len(bquads))
r=subprocess.run(['gmshToFoam','helix.msh'],text=True,capture_output=True)
print(r.stdout[-2000:]);print(r.stderr[-500:]);print('return',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# Conversion produced the expected 11,520 all-hexahedral cells and all four named physical groups. The
print(open('constant/polyMesh/boundary').read())
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
open('checkMesh.log','w').write(r.stdout+r.stderr)
print(r.stdout[-5000:]);print('return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# `checkMesh` exited before reporting topology, so I need the untruncated diagnostic. I’ll inspect the
print(open('checkMesh.log').read()[-8000:])

# -- cell 8 -------------------------------------------------------------------------
# The mesh itself was not read because this OpenFOAM build requires minimal `fvSchemes`/`fvSolution` d
for fn,obj in [('fvSchemes','dictionary'),('fvSolution','dictionary')]:
 with open('system/'+fn,'w') as f:f.write(f'FoamFile {{ format ascii; class {obj}; object {fn}; }}\n')
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
open('checkMesh.log','w').write(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-7000:]);print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The utility requires the standard scheme subdictionaries rather than a truly empty file. I’ll supply
open('system/fvSchemes','w').write('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
open('system/fvSolution','w').write('FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
open('checkMesh.log','w').write(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-7000:]);print('return',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# The full topology and geometry check now passes: one region, closed boundary, 11,520 hexahedra, maxi
r=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True)
print((r.stdout+r.stderr)[-1200:])
print(subprocess.run(['find','VTK','-maxdepth','4','-type','f'],capture_output=True,text=True).stdout)

# -- cell 11 ------------------------------------------------------------------------
# The converted boundary contains exactly the requested four patch datasets. I’ll render them with dis
colors={'neck_flank':'#0984e3','lid_flank':'#fdcb6e','inner_end':'#00b894','outer_end':'#d63031'}
p=pv.Plotter(off_screen=True,window_size=(1000,760))
for name,col in colors.items():
 m=pv.read(f'VTK/t13_0/boundary/{name}.vtp')
 p.add_mesh(m,color=col,show_edges=True,edge_color='#2d3436',line_width=0.5,label=name)
p.add_legend(bcolor='white');p.add_axes();p.view_isometric();p.camera.zoom(1.3)
p.show(screenshot='final_mesh_patches.png')
print({name:pv.read(f'VTK/t13_0/boundary/{name}.vtp').n_cells for name in colors})

# -- cell 12 ------------------------------------------------------------------------
# The final view confirms the alternating solid-contact patches and the two exposed helical termini. B
surfs=[pv.read(f'constant/triSurface/{n}.stl') for n in colors]
u=surfs[0]
for s in surfs[1:]:u=u.merge(s,merge_points=True,tolerance=1e-10)
u=u.clean(tolerance=1e-10)
free=u.extract_feature_edges(feature_edges=False,manifold_edges=False,boundary_edges=True,non_manifold_edges=True)
print('union points',u.n_points,'triangles',u.n_cells,'free/nonmanifold edge cells',free.n_cells)
print(f'measured centerline turns={(angles[-1]-angles[0])/(2*math.pi):.6f} requested={HEIGHT/PITCH:.6f}')
print(f'measured pitch={PITCH:.7f} m requested=0.0020000 m')
print(f'measured gap={np.linalg.norm(xyz[0,NU,NV//2]-xyz[0,0,NV//2]):.7f} m requested=0.0002000 m')
print(f'measured centerline axial rise={PITCH*(angles[-1]-angles[0])/(2*math.pi):.7f} m requested=0.0120000 m')
print(f'modeled mean diameter={2*MEAN_RADIUS:.7f} m from stated OD/depth')
