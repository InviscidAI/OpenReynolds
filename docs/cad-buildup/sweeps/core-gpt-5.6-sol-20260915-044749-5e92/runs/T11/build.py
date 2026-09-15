"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the 8 mm cell width as the **clear opening across flats**, with 1 mm solid web between ne
import math, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from build123d import *

MM = 1e-3
DUCT_W = 80*MM
CORE_T = 20*MM
UPSTREAM = 40*MM
DOWNSTREAM = 80*MM
CELL_AF = 8*MM          # requested clear opening across flats
WEB = 1*MM              # requested wall thickness between openings
PITCH_AF = CELL_AF + WEB
HEX_R = CELL_AF/math.sqrt(3)
PERIMETER_MIN = WEB      # assumption: retain complete cells with >=1 mm edge frame
Z0, Z1, Z2, Z3 = 0.0, UPSTREAM, UPSTREAM+CORE_T, UPSTREAM+CORE_T+DOWNSTREAM

# Pointy hexagons: neighboring rows stagger by half the 9 mm flat-normal pitch.
dx = math.sqrt(3)/2 * PITCH_AF
centers = []
for i in range(-20,21):
    x = i*dx
    for j in range(-20,21):
        y = (j + 0.5*(i & 1))*PITCH_AF
        verts = [(x+HEX_R*math.cos(k*math.pi/3), y+HEX_R*math.sin(k*math.pi/3)) for k in range(6)]
        if max(abs(px) for px,py in verts) <= DUCT_W/2-PERIMETER_MIN+1e-12 and max(abs(py) for px,py in verts) <= DUCT_W/2-PERIMETER_MIN+1e-12:
            centers.append((x,y))

fig, ax = plt.subplots(figsize=(6,6))
ax.add_patch(plt.Rectangle((-DUCT_W/2/MM,-DUCT_W/2/MM),DUCT_W/MM,DUCT_W/MM,fill=False,lw=2,color='black'))
for x,y in centers:
    xy=np.array([((x+HEX_R*math.cos(k*math.pi/3))/MM,(y+HEX_R*math.sin(k*math.pi/3))/MM) for k in range(6)])
    ax.add_patch(plt.Polygon(xy,fill=False,lw=.7,color='tab:blue'))
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title(f'Centered honeycomb layout: {len(centers)} complete cells')
ax.set_xlim(-43,43); ax.set_ylim(-43,43); ax.grid(alpha=.2)
plt.show()
print(f'cells={len(centers)}, clear_AF={CELL_AF/MM:.3f} mm (requested 8), web={PITCH_AF/MM-CELL_AF/MM:.3f} mm (requested 1)')
print(f'z stations [mm]: inlet={Z0/MM:.1f}, core front={Z1/MM:.1f}, core back={Z2/MM:.1f}, outlet={Z3/MM:.1f}')

# -- cell 2 -------------------------------------------------------------------------
# The coarse layout is centered and leaves a continuous perimeter band; it contains 67 full cells. Nex
import inspect
print('Polygon:', inspect.signature(Polygon))
print('extrude:', inspect.signature(extrude))
print('Box:', inspect.signature(Box))
print('export_step:', inspect.signature(export_step))

# -- cell 3 -------------------------------------------------------------------------
# The installed API confirms direct algebra-mode primitives and extrusion. I’ll now make the exact B-r
# Exact build123d algebra-mode geometry, all coordinates in metres.
duct_solid = Pos(0,0,(Z0+Z3)/2) * Box(DUCT_W, DUCT_W, Z3-Z0)
plate_blank = Pos(0,0,(Z1+Z2)/2) * Box(DUCT_W, DUCT_W, CORE_T)
opening_prisms = []
for x,y in centers:
    poly = Polygon(*[(x+HEX_R*math.cos(k*math.pi/3), y+HEX_R*math.sin(k*math.pi/3)) for k in range(6)])
    opening_prisms.append(Pos(0,0,Z1) * extrude(poly, amount=CORE_T))
openings = Compound(opening_prisms)
honeycomb_solid = plate_blank - openings
fluid = duct_solid - honeycomb_solid

bb = fluid.bounding_box()
expected_open_area = len(centers)*(math.sqrt(3)/2)*CELL_AF**2
expected_volume = DUCT_W**2*(UPSTREAM+DOWNSTREAM) + expected_open_area*CORE_T
print(f'fluid solids={len(fluid.solids())}, faces={len(fluid.faces())}, cells={len(centers)}')
print(f'bounds x=[{bb.min.X/MM:.3f},{bb.max.X/MM:.3f}] mm, y=[{bb.min.Y/MM:.3f},{bb.max.Y/MM:.3f}] mm, z=[{bb.min.Z/MM:.3f},{bb.max.Z/MM:.3f}] mm')
print(f'fluid volume={fluid.volume:.12g} m^3; analytic={expected_volume:.12g} m^3; rel.err={(fluid.volume/expected_volume-1):.3e}')
print(f'core open area={expected_open_area/MM**2:.3f} mm^2 of {DUCT_W**2/MM**2:.1f} mm^2 ({100*expected_open_area/DUCT_W**2:.2f}%)')

# -- cell 4 -------------------------------------------------------------------------
# The B-rep is one connected fluid solid, has the requested 80 × 80 × 140 mm extent, and its volume ma
print('export_stl:', inspect.signature(export_stl))
print('export_brep:', inspect.signature(export_brep))

# -- cell 5 -------------------------------------------------------------------------
# I’ll save the exact fluid as a unitless OpenCASCADE BREP whose numerical coordinates are already met
import pyvista as pv
export_brep(fluid, 'fluid.brep')
export_stl(fluid, 'fluid_preview.stl', tolerance=0.00015, angular_tolerance=0.12)
surf = pv.read('fluid_preview.stl')
p = pv.Plotter(shape=(1,2), off_screen=True, window_size=(1200,520))
p.subplot(0,0); p.add_mesh(surf, color='lightblue', opacity=0.38, show_edges=True, edge_color='gray'); p.view_isometric(); p.add_axes(); p.add_text('Oblique fluid boundary',font_size=11)
p.subplot(0,1); p.add_mesh(surf, color='lightblue', opacity=0.50, show_edges=True, edge_color='gray'); p.view_xy(); p.camera.position=(0,0,-0.35); p.camera.focal_point=(0,0,Z1); p.camera.up=(0,1,0); p.add_axes(); p.add_text('Looking downstream through inlet',font_size=11)
p.show()

# -- cell 6 -------------------------------------------------------------------------
# The render confirms a continuous upstream plenum, 67 through-openings, and a longer downstream plenu
patch_faces = {'inlet':[], 'outlet':[], 'ductWalls':[], 'honeycombWalls':[]}
tol = 1e-9
for f in fluid.faces():
    vv = [v.center() for v in f.vertices()]
    xs, ys, zs = [p.X for p in vv], [p.Y for p in vv], [p.Z for p in vv]
    if all(abs(z-Z0)<tol for z in zs):
        patch_faces['inlet'].append(f)
    elif all(abs(z-Z3)<tol for z in zs):
        patch_faces['outlet'].append(f)
    elif (all(abs(x-DUCT_W/2)<tol for x in xs) or all(abs(x+DUCT_W/2)<tol for x in xs) or
          all(abs(y-DUCT_W/2)<tol for y in ys) or all(abs(y+DUCT_W/2)<tol for y in ys)):
        patch_faces['ductWalls'].append(f)
    else:
        patch_faces['honeycombWalls'].append(f)

Path('constant/triSurface').mkdir(parents=True, exist_ok=True)
for name, faces in patch_faces.items():
    shp = Compound(faces)
    export_stl(shp, f'constant/triSurface/{name}.stl', tolerance=0.00012, angular_tolerance=0.1)
    print(f'{name}: faces={len(faces)}, area={sum(f.area for f in faces)/MM**2:.3f} mm^2')
print('assigned faces:',sum(map(len,patch_faces.values())),'of',len(fluid.faces()))

# -- cell 7 -------------------------------------------------------------------------
# All 414 boundary faces are assigned exactly once, with inlet and outlet areas each 6400 mm². I’ll im
import gmsh
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 1)
gmsh.model.add('honeycombAir')
gmsh.model.occ.importShapes('fluid.brep', highestDimOnly=True)
gmsh.model.occ.synchronize()
vol_tags=[t for d,t in gmsh.model.getEntities(3)]
surf_tags=[t for d,t in gmsh.model.getEntities(2)]
groups={'inlet':[],'outlet':[],'ductWalls':[],'honeycombWalls':[]}
for s in surf_tags:
    cx,cy,cz=gmsh.model.occ.getCenterOfMass(2,s)
    typ=gmsh.model.getType(2,s)
    if typ=='Plane' and abs(cz-Z0)<tol: groups['inlet'].append(s)
    elif typ=='Plane' and abs(cz-Z3)<tol: groups['outlet'].append(s)
    elif typ=='Plane' and (abs(abs(cx)-DUCT_W/2)<tol or abs(abs(cy)-DUCT_W/2)<tol): groups['ductWalls'].append(s)
    else: groups['honeycombWalls'].append(s)
print('gmsh volumes',vol_tags,'surfaces',len(surf_tags),'groups',{k:len(v) for k,v in groups.items()})
for k,v in groups.items(): print(k,'area mm2',sum(gmsh.model.occ.getMass(2,s) for s in v)/MM**2)
gmsh.finalize()

# -- cell 8 -------------------------------------------------------------------------
# The BREP translation preserves one volume, all 414 surfaces, and every patch area exactly. I’ll now 
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 1)
gmsh.model.add('honeycombAir')
gmsh.model.occ.importShapes('fluid.brep', highestDimOnly=True)
gmsh.model.occ.synchronize()
vol_tags=[t for d,t in gmsh.model.getEntities(3)]
surf_tags=[t for d,t in gmsh.model.getEntities(2)]
groups={'inlet':[],'outlet':[],'ductWalls':[],'honeycombWalls':[]}
for s in surf_tags:
    cx,cy,cz=gmsh.model.occ.getCenterOfMass(2,s); typ=gmsh.model.getType(2,s)
    if typ=='Plane' and abs(cz-Z0)<tol: groups['inlet'].append(s)
    elif typ=='Plane' and abs(cz-Z3)<tol: groups['outlet'].append(s)
    elif typ=='Plane' and (abs(abs(cx)-DUCT_W/2)<tol or abs(abs(cy)-DUCT_W/2)<tol): groups['ductWalls'].append(s)
    else: groups['honeycombWalls'].append(s)
for tag,(name,ents) in enumerate(groups.items(),start=1):
    pg=gmsh.model.addPhysicalGroup(2,ents,tag); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(3,vol_tags,10); gmsh.model.setPhysicalName(3,pg,'fluid')
all_pts=gmsh.model.getEntities(0); gmsh.model.mesh.setSize(all_pts,0.0028)
hb_pts=gmsh.model.getBoundary([(2,s) for s in groups['honeycombWalls']],combined=False,oriented=False,recursive=True)
hb_pts=list(set((d,t) for d,t in hb_pts if d==0)); gmsh.model.mesh.setSize(hb_pts,0.0014)
gmsh.option.setNumber('Mesh.MeshSizeMin',0.0010); gmsh.option.setNumber('Mesh.MeshSizeMax',0.0032)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',1)
gmsh.option.setNumber('Mesh.Algorithm3D',10)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.write('honeycomb.msh')
node_count=len(gmsh.model.mesh.getNodes()[0]); tet_count=len(gmsh.model.mesh.getElementsByType(4)[0])
print(f'coarse gmsh mesh: nodes={node_count}, linear_tets={tet_count}')
gmsh.finalize()

# -- cell 9 -------------------------------------------------------------------------
# The coarse mesh generated successfully. It is relatively dense (about 149k nodes) because every 1 mm
import subprocess, textwrap, os
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text(textwrap.dedent('''\
FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
'''))
r=subprocess.run(['gmshToFoam','honeycomb.msh'],text=True,capture_output=True)
print('return',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-35:]))
print(Path('constant/polyMesh/boundary').read_text()[:1800] if Path('constant/polyMesh/boundary').exists() else 'no boundary')

# -- cell 10 ------------------------------------------------------------------------
# Conversion produced 779,380 tetrahedra and exactly the four requested named patches. The “undefined 
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('return',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-80:]))

# -- cell 11 ------------------------------------------------------------------------
# The mesh itself was not read: the optional `-allTopology` mode requests an `fvSchemes` dictionary th
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('return',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-75:]))

# -- cell 12 ------------------------------------------------------------------------
# This OpenFOAM build requires the standard empty numerical dictionaries even for plain `checkMesh`. I
Path('system/fvSchemes').write_text(textwrap.dedent('''\
FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''))
Path('system/fvSolution').write_text(textwrap.dedent('''\
FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
'''))
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('return',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-85:]))

# -- cell 13 ------------------------------------------------------------------------
# `checkMesh` now passes: one connected region, closed combined boundary, 779,380 tetrahedra, maximum 
r=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True)
print('foamToVTK return',r.returncode, '\n', '\n'.join((r.stdout+r.stderr).splitlines()[-12:]))
print([str(p) for p in Path('VTK').rglob('*')][:20])

# -- cell 14 ------------------------------------------------------------------------
# The OpenFOAM-to-VTK export also finds all four patches. I’ll render those patches by name and overla
ductpv=pv.read('VTK/t11_0/boundary/ductWalls.vtp'); honeypv=pv.read('VTK/t11_0/boundary/honeycombWalls.vtp')
inpv=pv.read('VTK/t11_0/boundary/inlet.vtp'); outpv=pv.read('VTK/t11_0/boundary/outlet.vtp')
internal=pv.read('VTK/t11_0/internal.vtu')
cut=internal.slice(normal=(1,0,0),origin=(0,0,0.07))
p=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1250,520))
p.subplot(0,0); p.add_mesh(ductpv,color='silver',opacity=.18); p.add_mesh(honeypv,color='orange',opacity=.85); p.add_mesh(inpv,color='royalblue'); p.add_mesh(outpv,color='crimson'); p.view_isometric(); p.add_axes(); p.add_text('Named OpenFOAM patches',font_size=11)
p.subplot(0,1); p.add_mesh(cut,color='white',show_edges=True,edge_color='navy',line_width=.3); p.camera.position=(0.30,0,0.07); p.camera.focal_point=(0,0,0.07); p.camera.up=(0,0,1); p.add_axes(); p.add_text('Volume mesh slice at x = 0',font_size=11)
p.show()
print('slice cells',cut.n_cells,'boundary faces',{n:m.n_cells for n,m in [('ductWalls',ductpv),('honeycombWalls',honeypv),('inlet',inpv),('outlet',outpv)]})

# -- cell 15 ------------------------------------------------------------------------
# The final image confirms the four patch identities and shows the center-plane tetrahedra passing thr
import shutil
for f in ['fluid_preview.stl','honeycomb.msh']:
    Path(f).unlink(missing_ok=True)
shutil.rmtree('VTK',ignore_errors=True)
Path('GEOMETRY_MESH_REPORT.txt').write_text(textwrap.dedent(f'''\
Honeycomb flow-straightener air mesh
Units: metres in CAD and OpenFOAM mesh

Requested and measured geometry
- Duct section: requested 80 x 80 mm; measured 80.000 x 80.000 mm
- Upstream length: requested 40 mm; measured 40.000 mm
- Core thickness: requested 20 mm; measured 20.000 mm
- Downstream length: requested 80 mm; measured 80.000 mm
- Clear hex cell width across flats: requested 8 mm; constructed 8.000 mm
- Wall/web thickness between adjacent cell flats: requested 1 mm; constructed 1.000 mm
- Complete through-cells: {len(centers)}
- Core open area: {expected_open_area/MM**2:.3f} mm^2 ({100*expected_open_area/DUCT_W**2:.2f}%)
- Fluid volume: {fluid.volume:.12g} m^3

Assumption
- "8 mm cell width" is interpreted as clear opening across flats. A centered pointy-hex lattice
  uses 9 mm flat-normal pitch. Only complete cells with at least a 1 mm perimeter frame are kept;
  edge cells are not clipped open at the square duct wall.

Mesh
- 779380 linear tetrahedra, 148624 points, one connected region
- patches: inlet, outlet, ductWalls, honeycombWalls
- checkMesh: Mesh OK
- max aspect ratio 7.22894; max non-orthogonality 67.7888 deg; max skewness 0.93181
'''))
print(Path('GEOMETRY_MESH_REPORT.txt').read_text())
print('polyMesh size MB',sum(p.stat().st_size for p in Path('constant/polyMesh').rglob('*') if p.is_file())/1e6)
