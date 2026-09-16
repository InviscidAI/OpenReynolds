"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I will model the fluid volume with the plenum centred on the engine valley: length along \(x\), bank
import build123d as bd, inspect, os, math, numpy as np
print('build123d', bd.__version__)
print('sweep:', inspect.signature(bd.sweep))
print('Plane:', inspect.signature(bd.Plane))
print('Polyline:', inspect.signature(bd.Polyline))
print('CenterArc:', inspect.signature(bd.CenterArc))

# -- cell 2 -------------------------------------------------------------------------
# The first construction exposed that `CenterArc` is created in the global \(xy\)-plane, so those edge
print('RadiusArc:', inspect.signature(bd.RadiusArc))

# -- cell 3 -------------------------------------------------------------------------
# I’ll now rebuild the coarse CAD volume with the corrected radius arcs and print the overall bounds a
from build123d import Box, Circle, Plane, Pos, Vector, Line, RadiusArc, Wire, sweep
L, W, H = 0.300, 0.120, 0.080
D_THROTTLE, D_RUNNER, R_BEND = 0.070, 0.038, 0.060
STRAIGHT = 0.030
X_STATIONS = [-0.1125, -0.0375, 0.0375, 0.1125]
plenum = Pos(-L/2, -W/2, 0) * Box(L, W, H)
runners, runner_paths = [], []
for side in (-1, 1):
    y0 = side*W/2
    for x in X_STATIONS:
        p0 = Vector(x, y0-side*0.002, H/2)
        p1 = Vector(x, y0+side*STRAIGHT, H/2)
        p2 = Vector(x, y0+side*(STRAIGHT+R_BEND), H/2-R_BEND)
        line = Line(p0, p1)
        arc = RadiusArc(p1, p2, R_BEND)
        path = Wire([line, arc])
        section = Plane(origin=p0, x_dir=(1,0,0), z_dir=(0,side,0)) * Circle(D_RUNNER/2)
        runners.append(sweep(section, path, is_frenet=True))
        runner_paths.append(path)
fluid = plenum
for tube in runners:
    fluid = fluid + tube
fluid = fluid.clean()
print('solids', len(fluid.solids()), 'faces', len(fluid.faces()), 'volume m3', round(fluid.volume,9))
print('bounds min/max:', fluid.bounding_box().min, fluid.bounding_box().max)
print('station pitch m:', np.diff(X_STATIONS))

# -- cell 4 -------------------------------------------------------------------------
# The coarse topology is wrong: eight disconnected solids and bounds smaller than the requested plenum
print('Box', inspect.signature(bd.Box)); print('Circle', inspect.signature(bd.Circle)); print('plenum raw bounds', plenum.bounding_box().min, plenum.bounding_box().max, 'vol', plenum.volume)
for i,(p,t) in enumerate(zip(runner_paths,runners)):
    print(i, 'path', p.bounding_box().min, p.bounding_box().max, 'tube', t.bounding_box().min, t.bounding_box().max, 'vol', t.volume)

# -- cell 5 -------------------------------------------------------------------------
# The diagnostics found two concrete issues: `Box` defaults to centred alignment, and a spatial `Radiu
from build123d import Box, Circle, Plane, Pos, Vector, Line, CenterArc, Wire, sweep
L, W, H = 0.300, 0.120, 0.080
D_THROTTLE, D_RUNNER, R_BEND = 0.070, 0.038, 0.060
STRAIGHT = 0.030
X_STATIONS = [-0.1125, -0.0375, 0.0375, 0.1125]
plenum = Pos(0, 0, H/2) * Box(L, W, H)
runners, runner_paths = [], []
for side in (-1, 1):
    y0 = side*W/2
    for x in X_STATIONS:
        p0 = Vector(x, y0-side*0.002, H/2)
        p1 = Vector(x, y0+side*STRAIGHT, H/2)
        c = Vector(x, y0+side*STRAIGHT, H/2-R_BEND)
        line = Line(p0, p1)
        arc_plane = Plane(origin=c, x_dir=(0,side,0), z_dir=(1,0,0))
        arc = arc_plane * CenterArc((0,0), R_BEND, side*90, -side*90)
        path = Wire([line, arc])
        section = Plane(origin=p0, x_dir=(1,0,0), z_dir=(0,side,0)) * Circle(D_RUNNER/2)
        runners.append(sweep(section, path, is_frenet=True))
        runner_paths.append(path)
fluid = plenum
for tube in runners:
    fluid = fluid + tube
fluid = fluid.clean()
print('solids', len(fluid.solids()), 'faces', len(fluid.faces()), 'volume m3', round(fluid.volume,9))
print('bounds min/max:', fluid.bounding_box().min, fluid.bounding_box().max)
for i,p in enumerate(runner_paths): print(i, p.bounding_box().min, p.bounding_box().max)

# -- cell 6 -------------------------------------------------------------------------
# The corrected fluid is one solid with the expected volume scale and path bounds. The lowest \(z=-20\
from build123d import Mesher, export_step
os.makedirs('constant/triSurface', exist_ok=True)
export_step(fluid, 'intakeFluid.step')
coarse = Mesher()
coarse.add_shape(fluid)
coarse.write('coarseFluid.stl')
import pyvista as pv
pv.OFF_SCREEN = True
m = pv.read('coarseFluid.stl')
p = pv.Plotter(window_size=(900,650), off_screen=True)
p.add_mesh(m, color='lightsteelblue', show_edges=True, edge_color='navy', opacity=0.9)
p.add_axes(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot='coarse_geometry.png')
print('coarse STL:', m.n_cells, 'triangles; bounds:', np.array(m.bounds).reshape(3,2))

# -- cell 7 -------------------------------------------------------------------------
# The rendered coarse solid shows the requested arrangement: four runners on each side, connected clea
print([n for n in dir(bd.Face) if 'split' in n.lower() or 'project' in n.lower()])
for n in ['split_by_perimeter','project_to_shape']:
    if hasattr(bd.Face,n): print(n, inspect.signature(getattr(bd.Face,n)))

# -- cell 8 -------------------------------------------------------------------------
# I will classify the already-created boundary by analytic face type and known construction membership
for i,f in enumerate(fluid.faces()):
    c=f.center(); gt=f.geom_type
    try: n=f.normal_at()
    except: n='-'
    print(i, gt, 'A',f'{f.area:.8f}', 'C',tuple(round(v,5) for v in c), 'N', n if isinstance(n,str) else tuple(round(v,3) for v in n))

# -- cell 9 -------------------------------------------------------------------------
# I’ll re-emit the split cell self-contained, including the inlet circle definition, so the final repl
from build123d import Keep
end_face = fluid.faces()[0]
inlet_plane = Plane(origin=(-L/2,0,H/2), x_dir=(0,1,0), z_dir=(-1,0,0))
inlet_disk_seed = inlet_plane * Circle(D_THROTTLE/2)
throttle_inlet = end_face.split_by_perimeter(inlet_disk_seed.wire(), keep=Keep.INSIDE)
inlet_end_wall = end_face.split_by_perimeter(inlet_disk_seed.wire(), keep=Keep.OUTSIDE)
print('throttle area', throttle_inlet.area, 'target', math.pi*(D_THROTTLE/2)**2)
print('end-wall area', inlet_end_wall.area, 'partition sum/original', throttle_inlet.area+inlet_end_wall.area, end_face.area)

# -- cell 10 ------------------------------------------------------------------------
# I’ll move directly to a coarse conformal mesh. I’ll import the STEP volume into Gmsh, imprint the ze
import gmsh
gmsh.initialize(); gmsh.model.add('intake')
vols=gmsh.model.occ.importShapes('intakeFluid.step')
# Disk normal is x; addDisk local normal z rotated to x using zAxis and xAxis.
disk=gmsh.model.occ.addDisk(-L/2,0,H/2,D_THROTTLE/2,D_THROTTLE/2,zAxis=[1,0,0],xAxis=[0,1,0])
out,_=gmsh.model.occ.fragment(vols, [(2,disk)], removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()
print('entities after imprint:', {d:len(gmsh.model.getEntities(d)) for d in (2,3)}, 'vol out', out)
for s in gmsh.model.getBoundary(gmsh.model.getEntities(3), oriented=False):
    tag=s[1]; typ=gmsh.model.getType(2,tag); com=gmsh.model.occ.getCenterOfMass(2,tag); area=gmsh.model.occ.getMass(2,tag)
    print(tag,typ,tuple(round(v,5) for v in com),round(area,8))
gmsh.write('coarse_imprinted.brep'); gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# Reloading the BREP renumbered its entities, so hard-coded surface tags were invalid. I’ll classify t
import gmsh, os, subprocess, numpy as np, math
try: gmsh.finalize()
except: pass
gmsh.initialize(); gmsh.model.add('intakeMesh'); gmsh.model.occ.importShapes('coarse_imprinted.brep'); gmsh.model.occ.synchronize()
throttle=[]; plenum=[]; runner=[]; left=[]; right=[]
for _,s in gmsh.model.getEntities(2):
    typ=gmsh.model.getType(2,s); a=gmsh.model.occ.getMass(2,s); c=gmsh.model.occ.getCenterOfMass(2,s)
    if typ in ('Cylinder','Torus'): runner.append(s)
    elif abs(a-math.pi*(D_RUNNER/2)**2)<1e-6: (left if c[1]<0 else right).append((c[0],s))
    elif abs(a-math.pi*(D_THROTTLE/2)**2)<1e-6: throttle.append(s)
    else: plenum.append(s)
left.sort(); right.sort()
patches={'throttle_inlet':throttle,'plenum_walls':plenum,'runner_walls':runner}
for i,(_,s) in enumerate(left,1): patches[f'port_left_{i}']=[s]
for i,(_,s) in enumerate(right,1): patches[f'port_right_{i}']=[s]
for name,tags in patches.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
voltag=gmsh.model.getEntities(3)[0][1]; vg=gmsh.model.addPhysicalGroup(3,[voltag]); gmsh.model.setPhysicalName(3,vg,'air')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.005); gmsh.option.setNumber('Mesh.MeshSizeMax',0.012)
curves=gmsh.model.getBoundary([(2,t) for t in runner+ [s for _,s in left+right]],combined=True,recursive=True)
gmsh.model.mesh.setSize([q for q in curves if q[0]==0],0.007)
gmsh.model.mesh.generate(3); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0); gmsh.write('intake.msh')
nodeTags,nodeXYZ,_=gmsh.model.mesh.getNodes(); xyz=dict(zip(nodeTags,np.asarray(nodeXYZ).reshape(-1,3)))
os.makedirs('constant/triSurface',exist_ok=True)
for name,surfs in patches.items():
    with open(f'constant/triSurface/{name}.stl','w') as f:
        f.write(f'solid {name}\n')
        for s in surfs:
            ets,_,enodes=gmsh.model.mesh.getElements(2,s)
            for et,conn in zip(ets,enodes):
                nn=gmsh.model.mesh.getElementProperties(et)[3]
                for row in np.asarray(conn).reshape(-1,nn):
                    a,b,c=(xyz[int(q)] for q in row[:3]); n=np.cross(b-a,c-a); ln=np.linalg.norm(n); n=n/ln if ln else n
                    f.write(' facet normal %.9g %.9g %.9g\n  outer loop\n'%tuple(n))
                    for v in (a,b,c): f.write('   vertex %.9g %.9g %.9g\n'%tuple(v))
                    f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')
gmsh.finalize()
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write("FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
r=subprocess.run(['gmshToFoam','intake.msh'],text=True,capture_output=True)
print(r.stdout[-1200:]); print('return',r.returncode,'patch sizes',{k:len(v) for k,v in patches.items()})

# -- cell 12 ------------------------------------------------------------------------
# The coarse volume mesh now exists: 15,587 tetrahedra, all eleven requested named boundaries, and one
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
print(r.stdout[-5000:]); print('return',r.returncode)
print(open('constant/polyMesh/boundary').read())

# -- cell 13 ------------------------------------------------------------------------
# `checkMesh` exited before reporting mesh statistics, despite the boundary file showing that all unde
r=subprocess.run(['checkMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(r.stdout); print('return',r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The mesh itself was not rejected; OpenFOAM stopped because this newly created mesh-only case lacks m
foam_header=lambda cls,obj: f"FoamFile\n{{\n version 2.0;\n format ascii;\n class {cls};\n object {obj};\n}}\n"
open('system/fvSchemes','w').write(foam_header('dictionary','fvSchemes')+'ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
open('system/fvSolution','w').write(foam_header('dictionary','fvSolution')+'solvers {}\n')
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(r.stdout[-6500:]); print('return',r.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The standard `checkMesh` completed its topology and geometry sections, but the combined cell failed 
r=subprocess.run(['checkMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print('\n'.join(r.stdout.splitlines()[-35:])); print('checkMesh return',r.returncode)
stls=[pv.read('constant/triSurface/'+f) for f in sorted(os.listdir('constant/triSurface')) if f.endswith('.stl')]
surf=stls[0].merge(stls[1:])
p=pv.Plotter(window_size=(900,650),off_screen=True); p.add_mesh(surf,color='lightcyan',show_edges=True,edge_color='steelblue'); p.add_axes(); p.view_isometric(); p.camera.zoom(1.2); p.show(screenshot='coarse_mesh.png')
print('rendered boundary triangles',surf.n_cells,'bounds',surf.bounds)

# -- cell 16 ------------------------------------------------------------------------
# The measurement cell encountered a harmless namespace collision: the later surface-classification li
plenum_check = Pos(0,0,H/2) * Box(L,W,H); plbb=plenum_check.bounding_box()
port_faces=[f for f in fluid.faces() if f.geom_type==bd.GeomType.PLANE and abs(f.area-math.pi*(D_RUNNER/2)**2)<1e-8]
port_area=sum(f.area for f in port_faces)/len(port_faces)
print(f'Plenum CAD: {(plbb.max.X-plbb.min.X)*1000:.3f} x {(plbb.max.Y-plbb.min.Y)*1000:.3f} x {(plbb.max.Z-plbb.min.Z)*1000:.3f} mm; requested 300 x 120 x 80 mm')
print(f'Throttle diameter from split-face area: {2*math.sqrt(throttle_inlet.area/math.pi)*1000:.3f} mm; requested 70 mm')
print(f'Runner diameter from 8 port areas: {2*math.sqrt(port_area/math.pi)*1000:.3f} mm; requested 38 mm')
print(f'Bend centreline radius: {R_BEND*1000:.3f} mm; turn: 90.000 deg; requested 60 mm / 90 deg')
print(f'Port centreline offset from plenum wall: {(STRAIGHT+R_BEND)*1000:.3f} mm; requested 90 mm')
print('Longitudinal centres mm:',[x*1000 for x in X_STATIONS],'; pitches mm:',list(np.diff(X_STATIONS)*1000))
print('Assumptions: 30 mm straight lead before bend; equal 37.5 mm end margins giving 75 mm pitch.')
print('Outputs:',len([f for f in os.listdir('constant/triSurface') if f.endswith('.stl')]),'STLs; 15587 cells; 1 region; 11 patches.')
