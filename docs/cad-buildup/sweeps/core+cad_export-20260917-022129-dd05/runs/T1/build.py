"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll interpret each 120 mm leg length as the straight tangent-to-open-end length. The planar centrel
import build123d as bd, inspect
print('ThreePointArc', inspect.signature(bd.ThreePointArc))
print('Wire', inspect.signature(bd.Wire))
print('Face', inspect.signature(bd.Face))
print('Box', inspect.signature(bd.Box))
print('extrude', inspect.signature(bd.extrude))

# -- cell 2 -------------------------------------------------------------------------
# The boundary will be a single closed wire: outer radius 20 mm, inner radius 10 mm, joined to the two
from build123d import Edge, ThreePointArc, Wire, Face, extrude
import pyvista as pv
L=0.120; R=0.015; W=0.010; T=0.001
p0=(0,-W/2,0); p1=(L,-W/2,0); p2=(L+R+W/2,R,0); p3=(L,2*R+W/2,0)
p4=(0,2*R+W/2,0); p5=(0,2*R-W/2,0); p6=(L,2*R-W/2,0)
p7=(L+R-W/2,R,0); p8=(L,W/2,0); p9=(0,W/2,0)
edges=[Edge.make_line(p0,p1),ThreePointArc(p1,p2,p3),Edge.make_line(p3,p4),Edge.make_line(p4,p5),Edge.make_line(p5,p6),ThreePointArc(p6,p7,p8),Edge.make_line(p8,p9),Edge.make_line(p9,p0)]
section=Face(Wire(edges)); fluid=extrude(section,amount=T)
print('requested width / measured radial & straight width:',W, section.area/(2*L+3.141592653589793*R))
print('requested straight length:',L,' measured:',p1[0]-p0[0])
print('requested centreline radius:',R,' constructed:',(W/2+(R-W/2)))
print('bounds:',fluid.bounding_box().min,fluid.bounding_box().max,' volume:',fluid.volume)
bd.export_stl(fluid,'coarse_geometry.stl',tolerance=2.5e-4)
pl=pv.Plotter(off_screen=True,window_size=(900,320)); pl.add_mesh(pv.read('coarse_geometry.stl'),show_edges=True,color='lightblue'); pl.view_xy(); pl.camera.zoom(1.25); pl.show(screenshot='coarse_geometry.png')

# -- cell 3 -------------------------------------------------------------------------
# The preview confirms the intended U-passage; the diagonal lines are only STL triangulation, not geom
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
faces=list(fluid.faces())
def nearest_face(pt): return min(faces,key=lambda f:f.distance_to(pt))
inlet=nearest_face((0,0,T/2)); outlet=nearest_face((0,2*R,T/2))
frontback=[f for f in faces if abs(abs(f.normal_at().Z)-1)<1e-8]
walls=[f for f in faces if f not in [inlet,outlet] and f not in frontback]
print('faces:',len(faces),'inlet area',inlet.area,'outlet area',outlet.area,'front/back',len(frontback),'walls',len(walls))
export_patches(fluid,{'inlet':[inlet],'outlet':[outlet],'frontAndBack':frontback,'walls':walls},tolerance=2.5e-5)

# -- cell 4 -------------------------------------------------------------------------
# For the coarse mesh I’ll use a conformal multi-block topology: 60 cells along each straight, 24 arou
import os, math, subprocess
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
verts=[]; vid={}
def V(x,y,z):
 key=(round(x,12),round(y,12),round(z,12))
 if key not in vid: vid[key]=len(verts); verts.append(key)
 return vid[key]
def prism(q): return tuple(V(x,y,0) for x,y in q)+tuple(V(x,y,T) for x,y in q)
blocks=[]; arcs=[]
lower=prism([(0,-W/2),(L,-W/2),(L,W/2),(0,W/2)]); blocks.append((lower,(60,5,1)))
N=12; bend=[]
for k in range(N):
 a0=-math.pi/2+k*math.pi/N; a1=-math.pi/2+(k+1)*math.pi/N; am=(a0+a1)/2
 q=[(L+(R+W/2)*math.cos(a0),R+(R+W/2)*math.sin(a0)),(L+(R+W/2)*math.cos(a1),R+(R+W/2)*math.sin(a1)),(L+(R-W/2)*math.cos(a1),R+(R-W/2)*math.sin(a1)),(L+(R-W/2)*math.cos(a0),R+(R-W/2)*math.sin(a0))]
 h=prism(q); bend.append(h); blocks.append((h,(2,5,1)))
 for zoff in (0,4):
  arcs += [(h[0+zoff],h[1+zoff],(L+(R+W/2)*math.cos(am),R+(R+W/2)*math.sin(am),zoff//4*T)),(h[3+zoff],h[2+zoff],(L+(R-W/2)*math.cos(am),R+(R-W/2)*math.sin(am),zoff//4*T))]
upper=prism([(0,2*R-W/2),(L,2*R-W/2),(L,2*R+W/2),(0,2*R+W/2)]); blocks.append((upper,(60,5,1)))
inlet=[(lower[0],lower[4],lower[7],lower[3])]; outlet=[(upper[3],upper[7],upper[4],upper[0])]
wallfaces=[(lower[0],lower[1],lower[5],lower[4]),(lower[3],lower[7],lower[6],lower[2]),(upper[0],upper[1],upper[5],upper[4]),(upper[3],upper[7],upper[6],upper[2])]
for h in bend: wallfaces += [(h[0],h[1],h[5],h[4]),(h[3],h[7],h[6],h[2])]
fb=[]
for h,c in blocks: fb += [(h[0],h[3],h[2],h[1]),(h[4],h[5],h[6],h[7])]
def fs(lst): return '\n'.join('        ('+' '.join(map(str,f))+')' for f in lst)
d='''FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices\n(\n%s\n);\nblocks\n(\n%s\n);\nedges\n(\n%s\n);\nboundary\n(\n inlet { type patch; faces (%s); }\n outlet { type patch; faces (%s); }\n walls { type wall; faces (%s); }\n frontAndBack { type empty; faces (%s); }\n);\n'''%( '\n'.join('    (%.12g %.12g %.12g)'%v for v in verts), '\n'.join('    hex ('+' '.join(map(str,h))+') ('+' '.join(map(str,c))+') simpleGrading (1 1 1)' for h,c in blocks), '\n'.join('    arc %d %d (%.12g %.12g %.12g)'%(a,b,*p) for a,b,p in arcs), fs(inlet),fs(outlet),fs(wallfaces),fs(fb))
open('system/blockMeshDict','w').write(d)
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-2500:]); print(r.stderr); print('return',r.returncode)

# -- cell 5 -------------------------------------------------------------------------
# `blockMesh` stopped before reading the mesh dictionary because an OpenFOAM case requires `system/con
open('system/controlDict','w').write('''FoamFile { format ascii; class dictionary; object controlDict; }
application checkMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-3000:]); print(r.stderr); print('return',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# The coarse mesh built successfully with 720 cells and the intended patch names. I’ll now inspect it 
open('t1.foam','w').close()
reader=pv.OpenFOAMReader('t1.foam'); reader.set_active_time_value(0.0); data=reader.read()
print(data.keys())
mesh=data['internalMesh']
print('mesh bounds:',mesh.bounds,'cells:',mesh.n_cells,'points:',mesh.n_points)
pl=pv.Plotter(off_screen=True,window_size=(1000,350)); pl.add_mesh(mesh,show_edges=True,color='white',line_width=1); pl.view_xy(); pl.camera.zoom(1.2); pl.show(screenshot='coarse_mesh.png')

# -- cell 7 -------------------------------------------------------------------------
# The topology is conformal and follows the exact bend smoothly. For the final solve mesh I’ll refine 
d_final=d.replace('(60 5 1)','(240 20 1)').replace('(2 5 1)','(8 20 1)')
open('system/blockMeshDict','w').write(d_final)
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-1800:]); print(r.stderr); print('return',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The final 11,520-cell mesh is built. I’ll run the binding bare `checkMesh` now and inspect its topol
r=subprocess.run(['checkMesh'],text=True,capture_output=True); open('checkMesh.log','w').write(r.stdout+r.stderr); print(r.stdout[-5000:]); print(r.stderr); print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# This OpenFOAM build requires the standard numerical-scheme dictionaries even for `checkMesh`; the me
open('system/fvSchemes','w').write('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
open('system/fvSolution','w').write('''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}\n''')
r=subprocess.run(['checkMesh'],text=True,capture_output=True); open('checkMesh.log','w').write(r.stdout+r.stderr); print(r.stdout[-5000:]); print(r.stderr); print('return',r.returncode)
