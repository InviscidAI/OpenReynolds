"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model each meridional radius as the single smooth quadratic interpolant through the three speci
import build123d as bd, inspect, numpy as np, os, subprocess, textwrap
L=0.900
xmid=0.450
outer_r=(0.450,0.430,0.410)
core_r=(0.280,0.310,0.250)
print('Edge spline:', inspect.signature(bd.Edge.make_spline))
print('revolve:', inspect.signature(bd.revolve))
print('Face:', inspect.signature(bd.Face))
print('Axis.X:', bd.Axis.X)

# -- cell 2 -------------------------------------------------------------------------
# The API supports interpolating splines directly. I’ll now create the revolved fluid solid in metres,
outer_pts=[bd.Vector(0,0,outer_r[0]),bd.Vector(xmid,0,outer_r[1]),bd.Vector(L,0,outer_r[2])]
core_pts=[bd.Vector(L,0,core_r[2]),bd.Vector(xmid,0,core_r[1]),bd.Vector(0,0,core_r[0])]
outer_edge=bd.Edge.make_spline(outer_pts)
outlet_edge=bd.Edge.make_line(outer_pts[-1],core_pts[0])
core_edge=bd.Edge.make_spline(core_pts)
inlet_edge=bd.Edge.make_line(core_pts[-1],outer_pts[0])
profile_wire=bd.Wire([outer_edge,outlet_edge,core_edge,inlet_edge])
profile_face=bd.Face(profile_wire)
fluid=bd.revolve(profile_face,axis=bd.Axis.X,revolution_arc=360)
print('fluid volume m3:',fluid.volume)
print('bounds:',fluid.bounding_box().min,fluid.bounding_box().max)
for name,e,vals in [('outer',outer_edge,outer_r),('core',core_edge,core_r[::-1])]:
    got=[]
    for x in (0,xmid,L):
        hits=[v for v in e.vertices() if abs(v.X-x)<1e-9]
        if hits: got.append((x,abs(hits[0].Z)))
        else:
            # nearest point to station point in x-z section
            p=e.position_at(x/L if name=='outer' else 1-x/L)
            got.append((x,abs(p.Z)))
    print(name,'requested=', outer_r if name=='outer' else core_r,'sampled=',got)
# save an initial visual check
import pyvista as pv
verts,tris=fluid.tessellate(0.004)
pts=np.array([[v.X,v.Y,v.Z] for v in verts]); faces=np.hstack([np.full((len(tris),1),3),np.array(tris)]).ravel()
pl=pv.Plotter(off_screen=True,window_size=(900,650)); pl.add_mesh(pv.PolyData(pts,faces),color='lightblue',show_edges=True,opacity=0.55)
pl.camera_position='iso'; pl.add_axes(); pl.show(screenshot='coarse_geometry.png')
print('wrote coarse_geometry.png; tessellation triangles',len(tris))

# -- cell 3 -------------------------------------------------------------------------
# The visual check shows the intended converging annulus, with the core swelling near mid-duct and the
bd.export_step(fluid,'bypass_air_path.step')
import gmsh
gmsh.initialize(); gmsh.model.add('bypass')
gmsh.model.occ.importShapes('bypass_air_path.step'); gmsh.model.occ.synchronize()
vols=gmsh.model.getEntities(3); surfs=gmsh.model.getEntities(2)
print('volumes',vols,'surfaces',surfs)
for s in surfs:
    com=gmsh.model.occ.getCenterOfMass(*s); area=gmsh.model.occ.getMass(*s); typ=gmsh.model.getType(*s)
    print(s,'type',typ,'area',area,'COM',com,'bbox',gmsh.model.getBoundingBox(*s))
gmsh.finalize()

# -- cell 4 -------------------------------------------------------------------------
# The imported CAD has exactly four faces: two planes and two surfaces of revolution. The larger revol
import pathlib
tri_dir=pathlib.Path('constant/triSurface'); tri_dir.mkdir(parents=True,exist_ok=True)
gmsh.initialize(); gmsh.model.add('bypass')
gmsh.model.occ.importShapes('bypass_air_path.step'); gmsh.model.occ.synchronize()
vol=gmsh.model.getEntities(3)[0][1]; ss=[t for d,t in gmsh.model.getEntities(2)]
planes=[s for s in ss if gmsh.model.getType(2,s)=='Plane']
revs=[s for s in ss if gmsh.model.getType(2,s)=='Surface of Revolution']
inlet=min(planes,key=lambda s:gmsh.model.occ.getCenterOfMass(2,s)[0])
outlet=max(planes,key=lambda s:gmsh.model.occ.getCenterOfMass(2,s)[0])
outer=max(revs,key=lambda s:gmsh.model.occ.getMass(2,s)); core=min(revs,key=lambda s:gmsh.model.occ.getMass(2,s))
patch_tags={'outer_cowl':outer,'core_cowl':core,'fan_exit_inlet':inlet,'nozzle_outlet':outlet}
for name,tag in patch_tags.items():
    pg=gmsh.model.addPhysicalGroup(2,[tag]); gmsh.model.setPhysicalName(2,pg,name)
vg=gmsh.model.addPhysicalGroup(3,[vol]); gmsh.model.setPhysicalName(3,vg,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.045); gmsh.option.setNumber('Mesh.MeshSizeMax',0.055)
gmsh.option.setNumber('Mesh.ElementOrder',1); gmsh.model.mesh.generate(2)
nodeTags,nodeCoords,_=gmsh.model.mesh.getNodes(); xyz=dict(zip(nodeTags,np.asarray(nodeCoords).reshape(-1,3)))
def write_surface_stl(name,tag):
    types,elemTags,nodeLists=gmsh.model.mesh.getElements(2,tag)
    tris=[]
    for typ,nodes in zip(types,nodeLists):
        props=gmsh.model.mesh.getElementProperties(typ); nper=props[3]
        if nper==3: tris.extend(np.asarray(nodes).reshape(-1,3))
    with open(tri_dir/f'{name}.stl','w') as f:
        f.write(f'solid {name}\n')
        for ids in tris:
            a,b,c=(xyz[int(i)] for i in ids); n=np.cross(b-a,c-a); ln=np.linalg.norm(n); n=n/ln if ln else n
            f.write(f' facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n  outer loop\n')
            for p in (a,b,c): f.write(f'   vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')
    return len(tris)
counts={n:write_surface_stl(n,t) for n,t in patch_tags.items()}
print('patch map',patch_tags); print('STL triangle counts',counts); print('boundary nodes',len(nodeTags))

# -- cell 5 -------------------------------------------------------------------------
# The four pre-mesh STL files now exist and share the same Gmsh boundary nodes. I’ll fill that triangu
gmsh.model.mesh.generate(3)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.write('bypass_coarse.msh')
ntets=sum(len(e)//gmsh.model.mesh.getElementProperties(t)[3] for t,e in zip(*[gmsh.model.mesh.getElements(3)[0],gmsh.model.mesh.getElements(3)[2]]))
print('3D elements',ntets); gmsh.finalize()
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write("FoamFile { format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
r=subprocess.run(['gmshToFoam','bypass_coarse.msh'],text=True,capture_output=True)
print(r.stdout[-3000:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('gmshToFoam failed')

# -- cell 6 -------------------------------------------------------------------------
# The checker expects the standard scheme subdictionaries despite doing no discretization. I’ll provid
open('system/fvSchemes','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
fluxRequired { default no; }
''')
open('system/fvSolution','w').write('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
r=subprocess.run(['checkMesh','-allGeometry','-allTopology','-constant'],text=True,capture_output=True)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('checkMesh failed')

# -- cell 7 -------------------------------------------------------------------------
# The exhaustive checker passes topology, closure, volume, non-orthogonality, skewness, and connectivi
r=subprocess.run(['checkMesh','-constant'],text=True,capture_output=True)
print(r.stdout[-4500:]); print('return',r.returncode)
# Point-on-curve residuals are a direct dimensional check at each specified station.
station_checks=[]
for name,edge,rs in [('outer_cowl',outer_edge,outer_r),('core_cowl',core_edge,core_r)]:
    for x,rad in zip((0,xmid,L),rs):
        d=edge.distance_to(bd.Vector(x,0,rad))
        station_checks.append((name,x,rad,d))
for q in station_checks: print('%s x=%.3f m radius requested/measured=%.6f m residual=%.3g m'%q)
if r.returncode or 'Mesh OK.' not in r.stdout: raise RuntimeError('standard checkMesh did not pass')
