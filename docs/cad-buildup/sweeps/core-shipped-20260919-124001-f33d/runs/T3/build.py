"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll interpret the requested 2D valve as a 6 mm-wide straight fluid channel plus four repeated 6 mm-
import build123d as bd, inspect, subprocess, pathlib
print('build123d', bd.__version__)
for name in ['Line','Bezier','Wire','Face','Plane','extrude','export_step']:
    obj=getattr(bd,name)
    try: print(name, inspect.signature(obj))
    except Exception: print(name, type(obj))
print(subprocess.run(['grep','-n','sweep','.reference/b123d_api.md'],capture_output=True,text=True).stdout[:2000])

# -- cell 2 -------------------------------------------------------------------------
# The installed API supports direct Bézier wires and sweeps, but for a planar channel I need a 2D offs
for objname in ['offset','make_face','Rectangle','fillet']:
    obj=getattr(bd,objname,None)
    print('\n',objname,obj)
    if obj:
        try: print(inspect.signature(obj))
        except Exception as e: print(e)
for clsname in ['Wire','Edge']:
    cls=getattr(bd,clsname)
    print(clsname,'offset_2d',inspect.signature(cls.offset_2d), 'make_wire' if hasattr(cls,'make_wire') else '')

# -- cell 3 -------------------------------------------------------------------------
# The first trial exposed an API issue before anything was written: the offset call returned boundary 
# Requested dimensions and explicit first-pass assumptions, in mm for CAD construction.
CHANNEL_W = 6.0
LENGTH = 120.0
THICKNESS = 1.0
N_LOOPS = 4
STAGE_PITCH = 24.0
LOOP_SPAN = 20.0
LOOP_RISE = 12.0
FIRST_BRANCH_X = 12.0
main_2d = bd.Pos(LENGTH/2, 0) * bd.Rectangle(LENGTH, CHANNEL_W)
x0, x1 = FIRST_BRANCH_X, FIRST_BRANCH_X + LOOP_SPAN
stage_path = bd.Wire(bd.Bezier((x0,0),(x0+4,0),(x1-5,LOOP_RISE),(x1,0)))
stage_boundary = stage_path.offset_2d(CHANNEL_W/2, side=bd.Side.BOTH, closed=True)
print(type(stage_boundary), 'closed:', stage_boundary.is_closed, 'edges:', len(stage_boundary.edges()))
stage_strip = bd.Face(stage_boundary)
print(type(stage_strip), 'area mm2:', stage_strip.area, 'valid:', stage_strip.is_valid)
stage_2d = main_2d + stage_strip
print('union area mm2:', stage_2d.area, 'faces:', len(stage_2d.faces()), 'valid:', stage_2d.is_valid)
print('bounds mm:', stage_2d.bounding_box().min, stage_2d.bounding_box().max)

# -- cell 4 -------------------------------------------------------------------------
# The test union is valid, but measuring its bounds showed that a Bézier control ordinate of 12 mm pro
import numpy as np, pyvista as pv
# Corrected repeated Tesla stages; all dimensions are mm at this CAD stage.
LOOP_RISE = 12.0
BEZIER_CONTROL_Y = 2.25 * LOOP_RISE
loop_paths = []
loop_strips = []
for i in range(N_LOOPS):
    xa = FIRST_BRANCH_X + i*STAGE_PITCH
    xb = xa + LOOP_SPAN
    path = bd.Wire(bd.Bezier((xa,0),(xa+4,0),(xb-5,BEZIER_CONTROL_Y),(xb,0)))
    strip = bd.Face(path.offset_2d(CHANNEL_W/2, side=bd.Side.BOTH, closed=True))
    loop_paths.append(path)
    loop_strips.append(strip)
fluid_2d = main_2d
for strip in loop_strips:
    fluid_2d = fluid_2d + strip
fluid_mm = bd.extrude(fluid_2d, amount=THICKNESS)
# Numerical measurement of path excursions and direct solid extents.
rises = []
for path in loop_paths:
    ys = [path.position_at(t).Y for t in np.linspace(0,1,1001)]
    rises.append(max(ys))
bb = fluid_mm.bounding_box()
print(f'overall length measured {bb.size.X:.6f} mm; requested about {LENGTH:.6f} mm')
print(f'inlet channel width measured {bb.size.Y-(max(rises)+CHANNEL_W/2):.6f} mm by end-face span; requested {CHANNEL_W:.6f} mm')
print('loop centerline rises measured mm:', [round(v,6) for v in rises], '; assumed target', LOOP_RISE)
print(f'extrusion thickness measured {bb.size.Z:.6f} mm; assumed {THICKNESS:.6f} mm')
print('fluid solid volume mm3:', round(fluid_mm.volume,6), 'valid:', fluid_mm.is_valid)
bd.export_step(fluid_mm, 'tesla_valve_fluid.step')
bd.export_stl(fluid_mm, 'tesla_valve_preview.stl', tolerance=0.08, angular_tolerance=0.15)
mesh_preview = pv.read('tesla_valve_preview.stl')
pl = pv.Plotter(off_screen=True, window_size=(1200,420))
pl.add_mesh(mesh_preview, color='#58a6ff', show_edges=True, edge_color='#204060')
pl.view_xy(); pl.camera.zoom(1.2); pl.add_axes(); pl.show(screenshot='geometry_preview.png')
print('preview:', mesh_preview.n_cells, 'triangles; bounds mm:', mesh_preview.bounds)

# -- cell 5 -------------------------------------------------------------------------
# The preview confirms four distinct bypass arches and four enclosed teardrop islands. The earlier pri
import sys
sys.path.insert(0, '.reference')
from cad_export import export_patches
print('scale signature:', inspect.signature(bd.Shape.scale))
fluid = fluid_mm.scale(0.001)
faces = list(fluid.faces())
for j,f in enumerate(faces):
    c=f.center(); n=f.normal_at()
    print(j, 'area', f.area, 'center', tuple(round(v,7) for v in c), 'normal', tuple(round(v,4) for v in n))

# -- cell 6 -------------------------------------------------------------------------
# The scaled solid is valid. Its end faces are unambiguous: each has area \(6\times10^{-6}\,\mathrm{m^
tol_n = 1e-8
inlet_faces = [f for f in faces if f.normal_at().X < -1+tol_n]
outlet_faces = [f for f in faces if f.normal_at().X > 1-tol_n]
front_back_faces = [f for f in faces if abs(f.normal_at().Z) > 1-tol_n]
used = {id(f) for f in inlet_faces + outlet_faces + front_back_faces}
wall_faces = [f for f in faces if id(f) not in used]
print('patch face counts:', len(inlet_faces), len(outlet_faces), len(wall_faces), len(front_back_faces))
print(f'inlet width measured {(sum(f.area for f in inlet_faces)/0.001)*1000:.6f} mm; requested {CHANNEL_W:.6f} mm')
export_patches(fluid, {'inlet': inlet_faces, 'outlet': outlet_faces,
                       'walls': wall_faces, 'frontAndBack': front_back_faces},
               tolerance=2.5e-5)
bd.export_step(fluid, 'tesla_valve_fluid_m.step', unit=bd.Unit.M)
print('metre-space CAD extent:', fluid.bounding_box().size)

# -- cell 7 -------------------------------------------------------------------------
# For the actual planar mesh I’ll use Gmsh’s OpenCASCADE importer on the exact scaled 2D fluid face, g
import gmsh
fluid_face = fluid_2d.scale(0.001)
bd.export_step(fluid_face, 'tesla_valve_planar_m.step', unit=bd.Unit.M)
gmsh.initialize()
gmsh.model.add('teslaValve2D')
imported = gmsh.model.occ.importShapes('tesla_valve_planar_m.step', highestDimOnly=True)
gmsh.model.occ.synchronize()
base_surfaces = gmsh.model.getEntities(2)
print('imported:', imported, 'base surfaces:', base_surfaces)
base_curves = gmsh.model.getBoundary(base_surfaces, oriented=False, recursive=False)
print('boundary curves:', len(base_curves))
extruded = gmsh.model.occ.extrude(base_surfaces, 0,0,THICKNESS*0.001,
                                  numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
print('extruded output first/last:', extruded[:5], extruded[-5:], 'count', len(extruded))
print('entities after extrusion:', {d:len(gmsh.model.getEntities(d)) for d in range(4)})

# -- cell 8 -------------------------------------------------------------------------
# The first planar STEP was written as a sketch compound, which Gmsh correctly reported as containing 
planar_face = fluid_2d.faces()[0].scale(0.001)
bd.export_step(planar_face, 'tesla_valve_planar_face_m.step', unit=bd.Unit.M)
gmsh.clear(); gmsh.model.add('teslaValve2D')
imported = gmsh.model.occ.importShapes('tesla_valve_planar_face_m.step', highestDimOnly=True)
gmsh.model.occ.synchronize()
base_surfaces = gmsh.model.getEntities(2)
base_curves = gmsh.model.getBoundary(base_surfaces, oriented=False, recursive=False)
print('imported:', imported, 'surfaces:', base_surfaces, 'boundary curves:', len(base_curves))
extruded = gmsh.model.occ.extrude(base_surfaces, 0,0,THICKNESS*0.001,
                                  numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
print('extruded count', len(extruded), 'first:', extruded[:6], 'last:', extruded[-6:])
print('entities:', {d:len(gmsh.model.getEntities(d)) for d in range(4)})

# -- cell 9 -------------------------------------------------------------------------
# Gmsh recognizes the STEP label as a face but `highestDimOnly=True` is discarding this non-solid enti
gmsh.clear(); gmsh.model.add('teslaValve2D')
imported = gmsh.model.occ.importShapes('tesla_valve_planar_face_m.step', highestDimOnly=False)
gmsh.model.occ.synchronize()
print('imported dimtags:', imported)
print('entities:', {d:gmsh.model.getEntities(d) for d in range(4)})

# -- cell 10 ------------------------------------------------------------------------
# The STEP writer/importer combination is not transferring a standalone 2D face despite the label. Ope
print('export_brep:', inspect.signature(bd.export_brep))
bd.export_brep(planar_face, 'tesla_valve_planar_face_m.brep')
gmsh.clear(); gmsh.model.add('teslaValve2D')
imported = gmsh.model.occ.importShapes('tesla_valve_planar_face_m.brep', highestDimOnly=False)
gmsh.model.occ.synchronize()
print('imported dimtags:', imported[:10], 'count', len(imported))
print('entity counts:', {d:len(gmsh.model.getEntities(d)) for d in range(4)})

# -- cell 11 ------------------------------------------------------------------------
# The lossless BREP import succeeds: one planar face with 196 boundary curves. I’ll now extrude it one
# Extrude the exact imported planar face into one prism layer and assign physical groups.
base_surfaces = gmsh.model.getEntities(2)
extruded = gmsh.model.occ.extrude(base_surfaces, 0,0,THICKNESS*0.001,
                                  numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
top = [tag for dim,tag in extruded if dim==2][0]
volume = [tag for dim,tag in extruded if dim==3][0]
bottom = base_surfaces[0][1]
lateral = [tag for dim,tag in extruded if dim==2 and tag != top]
inlet_s = []; outlet_s = []; wall_s = []
for tag in lateral:
    cx,cy,cz = gmsh.model.occ.getCenterOfMass(2,tag)
    area = gmsh.model.occ.getMass(2,tag)
    if abs(cx) < 1e-10:
        inlet_s.append(tag)
    elif abs(cx-LENGTH*0.001) < 1e-10:
        outlet_s.append(tag)
    else:
        wall_s.append(tag)
print('surface counts inlet/outlet/walls/frontback:', len(inlet_s),len(outlet_s),len(wall_s),2)
print('opening areas m2:', [gmsh.model.occ.getMass(2,t) for t in inlet_s], [gmsh.model.occ.getMass(2,t) for t in outlet_s])
for tags,name in [(inlet_s,'inlet'),(outlet_s,'outlet'),(wall_s,'walls'),([bottom,top],'frontAndBack')]:
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
pgv=gmsh.model.addPhysicalGroup(3,[volume]); gmsh.model.setPhysicalName(3,pgv,'fluid')
# Coarse first-pass mesh, around 1 mm in-plane and exactly one element through thickness.
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.00075)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.0012)
gmsh.option.setNumber('Mesh.Algorithm', 6)
gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
gmsh.model.mesh.generate(3)
gmsh.write('tesla_coarse.msh')
print('gmsh nodes:', len(gmsh.model.mesh.getNodes()[0]))
gmsh.finalize()

# -- cell 12 ------------------------------------------------------------------------
# The coarse Gmsh mesh is now on disk (5,646 nodes), but you are right that it is not yet converted in
import os, re, subprocess, pathlib
pathlib.Path('system').mkdir(exist_ok=True)
pathlib.Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n''')
r = subprocess.run(['gmshToFoam','tesla_coarse.msh'], text=True, capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-1000:]); r.check_returncode()
bpath=pathlib.Path('constant/polyMesh/boundary')
btxt=bpath.read_text()
print('boundary before edit:\n', btxt[:2500])
# gmshToFoam creates physical names as patches; only convert the planar pair to empty.
pat = r'(frontAndBack\s*\{[^}]*?type\s+)patch(\s*;)'
btxt2,nsub = re.subn(pat, r'\1empty\2', btxt, flags=re.S)
assert nsub==1, f'frontAndBack edit count {nsub}'
bpath.write_text(btxt2)
chk=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('--- checkMesh ---\n',chk.stdout[-5000:]); print(chk.stderr[-1000:])
# Render actual OpenFOAM volume mesh from its boundary/edges.
foam=pathlib.Path('tesla.foam'); foam.write_text('')
grid=pv.POpenFOAMReader(str(foam.resolve())).read()
# Reader output is multiblock; combine visible blocks.
print('OpenFOAM blocks:', grid)
pl=pv.Plotter(off_screen=True, window_size=(1200,420))
pl.add_mesh(grid, color='#69b3e7', show_edges=True, edge_color='#24445c')
pl.view_xy(); pl.camera.zoom(1.15); pl.show(screenshot='openfoam_coarse_mesh.png')
print('checkMesh return code:',chk.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The OpenFOAM mesh now exists and the rendered mesh shows all four passages and islands. It contains 
pathlib.Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }\n''')
pathlib.Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }\n''')
chk=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(chk.stdout); print(chk.stderr); print('return code',chk.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The installed `checkMesh` also instantiates standard discretization subdictionaries. I’ll provide th
pathlib.Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
pathlib.Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers { }
''')
chk=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(chk.stdout); print(chk.stderr); print('return code',chk.returncode)
