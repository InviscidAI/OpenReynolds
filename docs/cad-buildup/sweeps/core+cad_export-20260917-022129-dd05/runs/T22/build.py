"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I will model the fluid directly in metres as the 12 mm annular gap between the two discs, then subtr
import inspect, sys, subprocess
import build123d as bd
sys.path.insert(0, '.reference')
from cad_export import export_patches
print('build123d', bd.__version__ if hasattr(bd,'__version__') else 'installed')
print('Cylinder', inspect.signature(bd.Cylinder))
print('Box', inspect.signature(bd.Box))
print('export_patches', inspect.signature(export_patches))

# -- cell 2 -------------------------------------------------------------------------
# The vane width is interpreted as a constant 8 mm tangential width (straight radial plates), not an 8
from build123d import Cylinder, Box, Pos, Rot, Align, export_stl
import math, os
RO, RI, GAP, VANE_W, N_VANES = 0.160, 0.095, 0.012, 0.008, 36
annulus = Cylinder(RO, GAP, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(RI, GAP, align=(Align.CENTER, Align.CENTER, Align.MIN))
vanes = None
for i in range(N_VANES):
    tool0 = Pos((RI+RO)/2, 0, 0) * Box(RO-RI+0.001, VANE_W, GAP, align=(Align.CENTER, Align.CENTER, Align.MIN))
    tool = Rot(0, 0, i*360/N_VANES) * tool0
    vanes = tool if vanes is None else vanes + tool
fluid = annulus - vanes
expected_vol = (math.pi*(RO**2-RI**2) - N_VANES*VANE_W*(RO-RI))*GAP
print(f'solids/passages = {len(fluid.solids())} (requested {N_VANES})')
print(f'CAD volume = {fluid.volume:.9e} m^3; planar estimate = {expected_vol:.9e} m^3; ratio={fluid.volume/expected_vol:.6f}')
print('bounds m:', fluid.bounding_box().min, fluid.bounding_box().max, 'faces:', len(fluid.faces()))
os.makedirs('preview', exist_ok=True)
export_stl(fluid, 'preview/fluid_coarse.stl', tolerance=0.001, angular_tolerance=0.2)
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
p=pv.Plotter(off_screen=True, window_size=(900,700)); m=pv.read('preview/fluid_coarse.stl')
p.add_mesh(m, color='deepskyblue', show_edges=True, edge_color='navy'); p.view_isometric(); p.camera.zoom(1.25)
p.add_axes(); p.show(screenshot='preview/fluid_coarse.png')
print('preview written')

# -- cell 3 -------------------------------------------------------------------------
# The preview confirms 36 open radial passages with no unintended bridges. The slight 0.05 mm reductio
from collections import Counter, defaultdict
info=[]
for f in fluid.faces():
    gt=str(f.geom_type)
    c=f.center()
    n=f.normal_at()
    info.append((gt, f.area, (c.X,c.Y,c.Z), (n.X,n.Y,n.Z), f))
print(Counter(x[0] for x in info))
for gt in sorted(set(x[0] for x in info)):
    rows=[x for x in info if x[0]==gt]
    print('\n',gt,'count',len(rows),'area sum',sum(x[1] for x in rows))
    for x in rows[:3]: print(' area, center, normal=',x[:4])

# -- cell 4 -------------------------------------------------------------------------
# Each passage has exactly six faces: two friction planes, two vane sidewalls, one inner cylindrical i
cf=[f for f in fluid.faces() if str(f.geom_type)=='GeomType.CYLINDER'][0]
print('face has radius:', hasattr(cf,'radius'), getattr(cf,'radius',None))
print('wrapped methods radius-like:', [a for a in dir(cf) if 'radius' in a.lower()])
# classify planar faces by outward normal
lower=[]; upper=[]; vane_faces=[]; cyl_faces=[]
for f in fluid.faces():
    if str(f.geom_type)=='GeomType.PLANE':
        nz=f.normal_at().Z
        (upper if nz>0.9 else lower if nz<-0.9 else vane_faces).append(f)
    else: cyl_faces.append(f)
print('planes lower/upper/vanes:',len(lower),len(upper),len(vane_faces),'cylinders',len(cyl_faces))

# -- cell 5 -------------------------------------------------------------------------
# The installed kernel exposes the exact cylindrical radius, so inlet/outlet classification is unambig
from build123d import export_step
inner_inlet=[f for f in cyl_faces if abs(f.radius-RI)<1e-8]
outer_outlet=[f for f in cyl_faces if abs(f.radius-RO)<1e-8]
# actual z-plane separation
zlo=sum(f.center().Z for f in lower)/len(lower); zhi=sum(f.center().Z for f in upper)/len(upper)
# faces bounding the vane centered on +X have centers near +X and planes y=+-W/2
v0=[f for f in vane_faces if f.center().X>0.12 and abs(f.center().Y)<0.006]
yplanes=sorted([f.center().Y for f in v0])
measured_w=max(yplanes)-min(yplanes)
print(f'inner radius measured {inner_inlet[0].radius:.6f} m; requested {RI:.6f} m')
print(f'outer radius measured {outer_outlet[0].radius:.6f} m; requested {RO:.6f} m')
print(f'face gap measured {zhi-zlo:.6f} m; requested {GAP:.6f} m')
print(f'0-deg vane width measured {measured_w:.6f} m; requested {VANE_W:.6f} m')
print('patch face counts:', {'friction_face_lower':len(lower),'friction_face_upper':len(upper),'vanes':len(vane_faces),'inner_inlet':len(inner_inlet),'outer_outlet':len(outer_outlet)})
export_step(fluid, 'fluid_passages.step')
patch_result=export_patches(fluid, {'friction_face_lower':lower,'friction_face_upper':upper,'vanes':vane_faces,'inner_inlet':inner_inlet,'outer_outlet':outer_outlet}, tolerance=2.5e-4, angular_tolerance=0.1)
print(patch_result)

# -- cell 6 -------------------------------------------------------------------------
# The patch export is watertight and conformal: 1,008 triangles, zero open, flipped, or non-manifold e
import gmsh, inspect
gmsh.initialize()
gmsh.model.add('brake_disc_air')
dimtags=gmsh.model.occ.importShapes('fluid_passages.step')
gmsh.model.occ.synchronize()
voltags=[t for d,t in gmsh.model.getEntities(3)]
surftags=[t for d,t in gmsh.model.getEntities(2)]
print('imported volumes',len(voltags),'surfaces',len(surftags),Counter(gmsh.model.getType(2,t) for t in surftags))
for t in surftags[:6]:
    typ=gmsh.model.getType(2,t); uv=gmsh.model.getParametrizationBounds(2,t)
    u=sum(uv[0][::2])/len(uv[0][::2]) if False else None
    print(t,typ,'bounds',uv,'mass',gmsh.model.occ.getMass(2,t))
print('getNormal sig',inspect.signature(gmsh.model.getNormal))
print('getPrincipalCurvatures sig',inspect.signature(gmsh.model.getPrincipalCurvatures))
gmsh.finalize()

# -- cell 7 -------------------------------------------------------------------------
# The imported topology matches the CAD exactly: 36 volumes and 216 boundary surfaces. I’ll use a deli
import gmsh, math, os
gmsh.initialize()
gmsh.model.add('brake_disc_air')
gmsh.model.occ.importShapes('fluid_passages.step')
gmsh.model.occ.synchronize()
voltags=[t for d,t in gmsh.model.getEntities(3)]
groups={'friction_face_lower':[],'friction_face_upper':[],'vanes':[],'inner_inlet':[],'outer_outlet':[]}
for _,t in gmsh.model.getEntities(2):
    typ=gmsh.model.getType(2,t)
    uvmin,uvmax=gmsh.model.getParametrizationBounds(2,t)
    uv=[0.5*(uvmin[i]+uvmax[i]) for i in range(len(uvmin))]
    if typ=='Plane':
        n=gmsh.model.getNormal(t,uv)
        if n[2] > 0.9: groups['friction_face_upper'].append(t)
        elif n[2] < -0.9: groups['friction_face_lower'].append(t)
        else: groups['vanes'].append(t)
    elif typ=='Cylinder':
        cmax1,cmax2,_,_=gmsh.model.getPrincipalCurvatures(t,uv)
        k=max(abs(float(cmax1[0])),abs(float(cmax2[0])))
        radius=1.0/k
        groups['inner_inlet' if abs(radius-RI)<abs(radius-RO) else 'outer_outlet'].append(t)
    else: raise RuntimeError(f'unexpected surface type {typ}')
print('mesh patch surface counts:',{k:len(v) for k,v in groups.items()})
for name,tags in groups.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
vg=gmsh.model.addPhysicalGroup(3,voltags); gmsh.model.setPhysicalName(3,vg,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0035); gmsh.option.setNumber('Mesh.MeshSizeMax',0.0035)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.option.setNumber('Mesh.ElementOrder',1); gmsh.option.setNumber('Mesh.Optimize',1)
gmsh.model.mesh.generate(3)
gmsh.write('brake_disc_air.msh')
print('nodes',len(gmsh.model.mesh.getNodes()[0]),'3D elements',sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 8 -------------------------------------------------------------------------
# Conversion stopped only because OpenFOAM utilities require a minimal `system/controlDict` even for p
from pathlib import Path
Path('system').mkdir(exist_ok=True)
foam_header='''FoamFile\n{\n    format ascii;\n    class dictionary;\n    object %s;\n}\n'''
Path('system/controlDict').write_text(foam_header%'controlDict' + '''application checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n''')
Path('system/fvSchemes').write_text(foam_header%'fvSchemes' + '''ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n''')
Path('system/fvSolution').write_text(foam_header%'fvSolution' + '''solvers {}\n''')
print('wrote utility dictionaries:',*[str(p) for p in Path('system').iterdir()])

# -- cell 9 -------------------------------------------------------------------------
# With the utility dictionaries present, I’m rerunning only the format conversion and reading back the
r=subprocess.run(['gmshToFoam','brake_disc_air.msh'],text=True,capture_output=True)
print(r.stdout[-3500:]); print(r.stderr[-500:])
if r.returncode: raise RuntimeError(f'gmshToFoam failed {r.returncode}')
print('\n--- boundary patch names/types ---')
text=Path('constant/polyMesh/boundary').read_text()
print(text)

# -- cell 10 ------------------------------------------------------------------------
# All five names survived conversion, with 29,378 initially undefined tetrahedral faces subsequently p
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout); print(r.stderr)
print('return code',r.returncode)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
