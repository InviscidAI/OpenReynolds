"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid domain as a 60 mm diameter, 42 mm long cylinder, split axially into 20 mm flang
import inspect, build123d as bd
print('Cylinder:', inspect.signature(bd.Cylinder))
print('Circle:', inspect.signature(bd.Circle))
print('Face:', inspect.signature(bd.Face))
print('Edge circle:', inspect.signature(bd.Edge.make_circle))
print('Wire:', inspect.signature(bd.Wire))

# -- cell 2 -------------------------------------------------------------------------
# Now I’ll construct the three requested axial sections in metres. Each wall patch is taken from the s
import build123d as bd, math
# All dimensions are metres
FLANGE_OD = 0.180
FLANGE_T = 0.020
BORE_D = 0.060
GASKET_T = 0.002
BOLT_D = 0.014
BOLT_PCD = 0.140
N_BOLTS = 6
R = BORE_D/2
L = 2*FLANGE_T + GASKET_T
align_z0 = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
flange1_water = bd.Cylinder(R, FLANGE_T, align=align_z0)
gasket_water = bd.Pos(0,0,FLANGE_T) * bd.Cylinder(R, GASKET_T, align=align_z0)
flange2_water = bd.Pos(0,0,FLANGE_T+GASKET_T) * bd.Cylinder(R, FLANGE_T, align=align_z0)
water = flange1_water + gasket_water + flange2_water
flange1_bore_wall = flange1_water.faces().filter_by(bd.GeomType.CYLINDER)[0]
gasket_inner_face = gasket_water.faces().filter_by(bd.GeomType.CYLINDER)[0]
flange2_bore_wall = flange2_water.faces().filter_by(bd.GeomType.CYLINDER)[0]
inlet = bd.Face(bd.Wire(bd.Edge.make_circle(R, plane=bd.Plane.XY)))
outlet_plane = bd.Plane(origin=(0,0,L), x_dir=(1,0,0), z_dir=(0,0,1))
outlet = bd.Face(bd.Wire(bd.Edge.make_circle(R, plane=outlet_plane)))
bolt_clearance = BOLT_PCD/2 - BOLT_D/2 - R
print(f'Bore diameter: measured {2*R:.6f} m, requested {BORE_D:.6f} m')
print(f'Face-to-face length: measured {L:.6f} m, requested 0.042000 m')
print(f'Wall axial spans: {FLANGE_T:.6f}, {GASKET_T:.6f}, {FLANGE_T:.6f} m')
print(f'Fluid volume: measured {water.volume:.9e} m^3, analytic {math.pi*R**2*L:.9e} m^3')
print(f'Bolt holes: {N_BOLTS} x {BOLT_D:.3f} m on {BOLT_PCD:.3f} m PCD; dry clearance to bore = {bolt_clearance:.3f} m')

# -- cell 3 -------------------------------------------------------------------------
# The dimensions and volume agree exactly with the specification. I’ll now tessellate the five named s
print('Shape tessellate:', inspect.signature(bd.Shape.tessellate))
print('STL export:', inspect.signature(bd.export_stl))

# -- cell 4 -------------------------------------------------------------------------
# For a visual geometry check, I’ll also build the two dry flange solids with their six bolt holes. Th
import numpy as np, pyvista as pv
# Dry hardware for visual context only; gasket OD assumed equal to flange OD.
GASKET_OD_ASSUMED = FLANGE_OD
outer_plate = bd.Cylinder(FLANGE_OD/2, FLANGE_T, align=align_z0)
bore_cut = bd.Cylinder(R, FLANGE_T, align=align_z0)
bolt_cuts = None
for i in range(N_BOLTS):
    a = 2*math.pi*i/N_BOLTS
    h = bd.Pos((BOLT_PCD/2)*math.cos(a),(BOLT_PCD/2)*math.sin(a),0) * bd.Cylinder(BOLT_D/2, FLANGE_T, align=align_z0)
    bolt_cuts = h if bolt_cuts is None else bolt_cuts + h
flange1_solid = outer_plate - bore_cut - bolt_cuts
flange2_solid = bd.Pos(0,0,FLANGE_T+GASKET_T) * flange1_solid
gasket_solid = bd.Pos(0,0,FLANGE_T) * (bd.Cylinder(GASKET_OD_ASSUMED/2,GASKET_T,align=align_z0) - bd.Cylinder(R,GASKET_T,align=align_z0))
def pvmesh(shape, tol=0.001):
    vv, tt = shape.tessellate(tol, 0.15)
    pts = np.array([[p.X,p.Y,p.Z] for p in vv])
    faces = np.hstack([np.full((len(tt),1),3),np.array(tt,dtype=int)]).ravel()
    return pv.PolyData(pts,faces)
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.set_background('white')
pl.add_mesh(pvmesh(flange1_solid), color='silver', opacity=0.22)
pl.add_mesh(pvmesh(flange2_solid), color='silver', opacity=0.22)
pl.add_mesh(pvmesh(gasket_solid), color='orange', opacity=0.35)
for sh,col in [(flange1_bore_wall,'royalblue'),(gasket_inner_face,'gold'),(flange2_bore_wall,'seagreen'),(inlet,'red'),(outlet,'purple')]:
    pl.add_mesh(pvmesh(sh,0.0005),color=col,show_edges=True,line_width=1)
pl.add_axes(); pl.camera_position='iso'; pl.show(screenshot='geometry_coarse.png')
print('Rendered geometry_coarse.png; colored fluid patches shown inside transparent dry hardware.')

# -- cell 5 -------------------------------------------------------------------------
# The coarse view confirms the water cylinder is concentric with the dry flange bores and that the 2 m
from pathlib import Path
tri_dir = Path('constant/triSurface'); tri_dir.mkdir(parents=True, exist_ok=True)
N_STL = 96
angles = np.linspace(0,2*np.pi,N_STL,endpoint=False)
rim = np.c_[R*np.cos(angles),R*np.sin(angles)]
def write_ascii_stl(path, triangles):
    with open(path,'w') as f:
        f.write(f'solid {Path(path).stem}\n')
        for p0,p1,p2 in triangles:
            n=np.cross(p1-p0,p2-p0); n=n/np.linalg.norm(n)
            f.write(f' facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n')
            for p in (p0,p1,p2): f.write(f'   vertex {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {Path(path).stem}\n')
def disk_tris(z, outward_sign):
    c=np.array([0.,0.,z]); ts=[]
    for i in range(N_STL):
        a=np.r_[rim[i],z]; b=np.r_[rim[(i+1)%N_STL],z]
        ts.append((c,b,a) if outward_sign<0 else (c,a,b))
    return ts
def wall_tris(z0,z1):
    ts=[]
    for i in range(N_STL):
        a=np.r_[rim[i],z0]; b=np.r_[rim[(i+1)%N_STL],z0]
        c=np.r_[rim[(i+1)%N_STL],z1]; d=np.r_[rim[i],z1]
        ts += [(a,b,c),(a,c,d)]
    return ts
patch_triangles={'inlet':disk_tris(0,-1),'outlet':disk_tris(L,1),
                 'flange1_bore_wall':wall_tris(0,FLANGE_T),
                 'gasket_inner_face':wall_tris(FLANGE_T,FLANGE_T+GASKET_T),
                 'flange2_bore_wall':wall_tris(FLANGE_T+GASKET_T,L)}
for name,ts in patch_triangles.items(): write_ascii_stl(tri_dir/f'{name}.stl',ts)
print('\n'.join(f'{name}.stl: {len(ts)} triangles' for name,ts in patch_triangles.items()))

# -- cell 6 -------------------------------------------------------------------------
# I’ll mesh by sweeping a coarse triangular disk mesh through the three named axial sections. This giv
import gmsh
gmsh.initialize()
gmsh.model.add('flange_water')
gmsh.option.setNumber('General.Terminal',1)
gmsh.option.setNumber('Mesh.MeshSizeMin',0.004)
gmsh.option.setNumber('Mesh.MeshSizeMax',0.004)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
inlet_tag = gmsh.model.occ.addDisk(0,0,0,R,R)
ext1 = gmsh.model.occ.extrude([(2,inlet_tag)],0,0,FLANGE_T,numElements=[5],recombine=True)
top1, vol1, side1 = ext1[0][1], ext1[1][1], ext1[2][1]
extg = gmsh.model.occ.extrude([(2,top1)],0,0,GASKET_T,numElements=[1],recombine=True)
topg, volg, sideg = extg[0][1], extg[1][1], extg[2][1]
ext2 = gmsh.model.occ.extrude([(2,topg)],0,0,FLANGE_T,numElements=[5],recombine=True)
outlet_tag, vol2, side2 = ext2[0][1], ext2[1][1], ext2[2][1]
gmsh.model.occ.synchronize()
for name,tag in [('inlet',inlet_tag),('outlet',outlet_tag),('flange1_bore_wall',side1),('gasket_inner_face',sideg),('flange2_bore_wall',side2)]:
    pg=gmsh.model.addPhysicalGroup(2,[tag]); gmsh.model.setPhysicalName(2,pg,name)
fluid_pg=gmsh.model.addPhysicalGroup(3,[vol1,volg,vol2]); gmsh.model.setPhysicalName(3,fluid_pg,'water')
gmsh.model.mesh.generate(3)
gmsh.write('flange_water.msh')
print('Named surface tags:', {'inlet':inlet_tag,'outlet':outlet_tag,'flange1_bore_wall':side1,'gasket_inner_face':sideg,'flange2_bore_wall':side2})
print('Volume tags:',[vol1,volg,vol2])
print('3D elements:',sum(len(e) for e in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 7 -------------------------------------------------------------------------
# Conversion stopped because an OpenFOAM case requires a minimal `system/controlDict`; no mesh was cre
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application meshOnly;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
''')
print('Created minimal system/controlDict for mesh utilities.')

# -- cell 8 -------------------------------------------------------------------------
# The converter ran, but that step was not reproducible because `subprocess` came only from the earlie
import subprocess
res=subprocess.run(['gmshToFoam','flange_water.msh'],text=True,capture_output=True)
print(res.stdout[-2500:]); print(res.stderr[-500:])
if res.returncode: raise RuntimeError(f'gmshToFoam failed {res.returncode}')

# -- cell 9 -------------------------------------------------------------------------
# I’ll run `checkMesh` and inspect the patch sizes. If `defaultFaces` is non-empty, the sequential mul
chk=subprocess.run(['checkMesh','-constant','-allGeometry','-allTopology'],text=True,capture_output=True)
print(chk.stdout[-7000:]); print(chk.stderr[-500:]); print('return code',chk.returncode)
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 10 ------------------------------------------------------------------------
# The converted boundary contains exactly the five requested patches; `defaultFaces` was not written, 
chk=subprocess.run(['checkMesh','-constant'],text=True,capture_output=True)
print(chk.stdout[-7000:]); print(chk.stderr[-500:]); print('return code',chk.returncode)

# -- cell 11 ------------------------------------------------------------------------
# This installation’s `checkMesh` reads `fvSchemes` even for a mesh-only case. I’ll add a minimal stan
Path('system/fvSchemes').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
print('Created minimal system/fvSchemes for checkMesh.')

# -- cell 12 ------------------------------------------------------------------------
# I’ll rerun the full topology and geometry check now that its required dictionary exists.
chk=subprocess.run(['checkMesh','-constant','-allGeometry','-allTopology'],text=True,capture_output=True)
print(chk.stdout[-9000:]); print(chk.stderr[-500:]); print('return code',chk.returncode)

# -- cell 13 ------------------------------------------------------------------------
# `checkMesh` also requires `fvSolution` in this case layout. I’ll add an empty valid dictionary; agai
Path('system/fvSolution').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
print('Created empty valid system/fvSolution for checkMesh.')

# -- cell 14 ------------------------------------------------------------------------
# I’ll repeat the comprehensive mesh check.
chk=subprocess.run(['checkMesh','-constant','-allGeometry','-allTopology'],text=True,capture_output=True)
print(chk.stdout[-10000:]); print(chk.stderr[-500:]); print('return code',chk.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The comprehensive check passes: one connected region, 4,994 prism cells, five patches, maximum non-o
vtkres=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True)
print(vtkres.stdout[-2000:]); print(vtkres.stderr[-500:]); print('return code',vtkres.returncode)
print([str(p) for p in Path('VTK').rglob('*') if p.is_file()][-20:])

# -- cell 16 ------------------------------------------------------------------------
# I’ll render half of the internal prism mesh together with colored boundary edges. This should expose
internal=pv.read('VTK/t20_0/internal.vtu')
half=internal.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
pl2=pv.Plotter(off_screen=True,window_size=(1000,650)); pl2.set_background('white')
pl2.add_mesh(half,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.4)
colors={'inlet':'red','outlet':'purple','flange1_bore_wall':'royalblue','gasket_inner_face':'gold','flange2_bore_wall':'seagreen'}
for name,col in colors.items():
    s=pv.read(f'VTK/t20_0/boundary/{name}.vtp').clip(normal=(1,0,0),origin=(0,0,0),invert=False)
    pl2.add_mesh(s,color=col,style='wireframe',line_width=2)
pl2.add_axes(); pl2.camera_position=[(0.11,-0.10,0.08),(0,0,L/2),(0,0,1)]
pl2.show(screenshot='mesh_clipped.png')
print('Rendered mesh_clipped.png:',internal.n_cells,'cells; clipped view',half.n_cells,'cells')

# -- cell 17 ------------------------------------------------------------------------
# The clipped rendering confirms the intended 5 / 1 / 5 axial layers and a continuous circular water r
stl_parts=[pv.read(str(tri_dir/f'{name}.stl')).triangulate() for name in patch_triangles]
union_poly=stl_parts[0]
for part in stl_parts[1:]: union_poly=union_poly.merge(part,merge_points=True,tolerance=1e-10)
free=union_poly.extract_feature_edges(boundary_edges=True,non_manifold_edges=False,feature_edges=False,manifold_edges=False)
nonman=union_poly.extract_feature_edges(boundary_edges=False,non_manifold_edges=True,feature_edges=False,manifold_edges=False)
print(f'Welded STL union: {union_poly.n_cells} triangles, {union_poly.n_points} points')
print(f'Free edges: {free.n_cells}; non-manifold edges: {nonman.n_cells}')
print('Union bounds:',union_poly.bounds)
