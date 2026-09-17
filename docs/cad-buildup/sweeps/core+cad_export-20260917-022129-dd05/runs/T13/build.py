"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll represent the leak path as a 0.2 mm-thick helical ribbon following one 30° trapezoidal thread f
import build123d as bd, inspect
PITCH=0.002; THREAD_DEPTH=0.001; GAP=0.0002; THREADED_HEIGHT=0.012
NECK_OD=0.040; CREST_R=NECK_OD/2; ROOT_R=CREST_R-THREAD_DEPTH
FLANK_INCLUDED_ANGLE=30.0  # assumption: standard metric trapezoidal profile
print('turns =', THREADED_HEIGHT/PITCH)
print('Helix:', inspect.signature(bd.Helix))
print('sweep:', inspect.signature(bd.sweep))
print('Plane:', inspect.signature(bd.Plane))

# -- cell 2 -------------------------------------------------------------------------
# The API supports a direct solid sweep. I’ll now make a very coarse CAD ribbon in metres, sweep a rec
import numpy as np, matplotlib.pyplot as plt, pyvista as pv
alpha=np.deg2rad(FLANK_INCLUDED_ANGLE/2)
R_PATH=(ROOT_R+CREST_R)/2
helix=bd.Helix(PITCH, THREADED_HEIGHT, R_PATH)
tangent=bd.Vector(0, 2*np.pi*R_PATH, PITCH).normalized()
radial=bd.Vector(1,0,0)
section_plane=bd.Plane(origin=(R_PATH,0,0), x_dir=radial, z_dir=tangent)
profile=section_plane * bd.Rectangle(THREAD_DEPTH/np.cos(alpha), GAP)
fluid=bd.sweep(profile, helix, is_frenet=True)
print('requested turns:',THREADED_HEIGHT/PITCH,' measured helix length [mm]:',helix.length*1000)
print('requested radial depth [mm]:',THREAD_DEPTH*1000,' requested gap [mm]:',GAP*1000)
print('fluid volume [mm3]:',fluid.volume*1e9,' faces:',len(fluid.faces()))
print('bounds [mm]:',tuple(round(v*1000,3) for v in (*fluid.bounding_box().min,*fluid.bounding_box().max)))
bd.export_stl(fluid,'coarse_preview.stl',tolerance=5e-5,angular_tolerance=0.2)
m=pv.read('coarse_preview.stl'); p=pv.Plotter(off_screen=True,window_size=(900,700)); p.add_mesh(m,color='lightsteelblue',show_edges=True); p.view_isometric(); p.show(screenshot='coarse_preview.png')
from IPython.display import display, Image
display(Image('coarse_preview.png'))

# -- cell 3 -------------------------------------------------------------------------
# The preview shows the intended single continuous six-turn path and no overlap between turns. The swe
for i,f in enumerate(fluid.faces()):
    c=f.center(); print(i, 'area_mm2=',round(f.area*1e6,4),'center_mm=',tuple(round(v*1000,4) for v in c),'geom=',f.geom_type)
print('expected end area mm2:',THREAD_DEPTH/np.cos(alpha)*GAP*1e6)
print('expected broad flank area mm2 ~',helix.length*THREAD_DEPTH/np.cos(alpha)*1e6)
print('expected edge-strip area mm2 ~',helix.length*GAP*1e6)

# -- cell 4 -------------------------------------------------------------------------
# I’ll re-emit the export step self-contained so the reproducible script imports the exporter and recr
import os, sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
faces=list(fluid.faces())
edge_faces=sorted([f for f in faces if 1e-5 < f.area < 3e-4], key=lambda f:f.center().X)
broad_faces=sorted([f for f in faces if f.area > 3e-4], key=lambda f:f.center().Z)
cap_faces=sorted([f for f in faces if f.area < 1e-5], key=lambda f:f.center().Z)
neck_faces=[broad_faces[0],edge_faces[0]]
lid_faces=[broad_faces[1],edge_faces[1]]
inner_end=[cap_faces[0]]; outer_end=[cap_faces[1]]
os.makedirs('constant/triSurface',exist_ok=True)
report=export_patches(fluid,{'neck_flank':neck_faces,'lid_flank':lid_faces,'inner_end':inner_end,'outer_end':outer_end},tolerance=2.5e-5)
bd.export_step(fluid,'helical_gap.step')
print('patch areas [mm2]:', {k:round(sum(f.area for f in v)*1e6,4) for k,v in {'neck_flank':neck_faces,'lid_flank':lid_faces,'inner_end':inner_end,'outer_end':outer_end}.items()})

# -- cell 5 -------------------------------------------------------------------------
# The conformal export is watertight (0 open, flipped, or non-manifold edges). I’ll mesh the exact STE
import gmsh
gmsh.initialize(); gmsh.model.add('helical_gap')
vols=gmsh.model.occ.importShapes('helical_gap.step'); gmsh.model.occ.synchronize()
surfs=gmsh.model.getEntities(2)
for dim,tag in surfs:
    a=gmsh.model.occ.getMass(dim,tag); c=gmsh.model.occ.getCenterOfMass(dim,tag)
    print(tag,'area_mm2',round(a*1e6,4),'center_mm',tuple(round(x*1000,4) for x in c))
print('volumes:',vols,'volume_mm3',gmsh.model.occ.getMass(3,vols[0][1])*1e9)
gmsh.finalize()

# -- cell 6 -------------------------------------------------------------------------
# STEP translation preserved all six areas and the 152.19 mm³ volume. I’ll now generate a coarse tetra
import gmsh, subprocess, os
gmsh.initialize(); gmsh.model.add('helical_gap')
gmsh.model.occ.importShapes('helical_gap.step'); gmsh.model.occ.synchronize()
groups={'neck_flank':[1,4],'lid_flank':[2,3],'inner_end':[5],'outer_end':[6]}
for name,tags in groups.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
vg=gmsh.model.addPhysicalGroup(3,[1]); gmsh.model.setPhysicalName(3,vg,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',8e-5); gmsh.option.setNumber('Mesh.MeshSizeMax',2e-4)
gmsh.option.setNumber('Mesh.Algorithm',6); gmsh.option.setNumber('Mesh.Algorithm3D',1)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.model.mesh.generate(3); gmsh.write('helical_gap.msh')
print('gmsh nodes',len(gmsh.model.mesh.getNodes()[0]),'3D elements',sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()
os.makedirs('system',exist_ok=True)
r=subprocess.run(['gmshToFoam','helical_gap.msh'],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The coarse Gmsh mesh contains about 139k tetrahedra; conversion stopped only because an empty case l
from pathlib import Path
Path('system/controlDict').write_text('''FoamFile\n{\n    format ascii;\n    class dictionary;\n    object controlDict;\n}\napplication checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n''')
r=subprocess.run(['gmshToFoam','helical_gap.msh'],text=True,capture_output=True)
print('gmshToFoam return',r.returncode); print(r.stdout[-1800:]); print(r.stderr[-500:])
c=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('checkMesh return',c.returncode); print(c.stdout[-3500:]); print(c.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
# Conversion succeeded and preserved all four patch names. `checkMesh` in this OpenFOAM build also exp
Path('system/fvSchemes').write_text('''FoamFile\n{ format ascii; class dictionary; object fvSchemes; }\n''')
Path('system/fvSolution').write_text('''FoamFile\n{ format ascii; class dictionary; object fvSolution; }\n''')
c=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('checkMesh return',c.returncode); print(c.stdout[-5000:]); print(c.stderr[-500:])
print('\nBoundary file tail:\n',Path('constant/polyMesh/boundary').read_text()[-1800:])

# -- cell 9 -------------------------------------------------------------------------
# The boundary file has exactly the four requested patches and no `defaultFaces`; their face counts ar
Path('system/fvSchemes').write_text('''FoamFile\n{ format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n''')
Path('system/fvSolution').write_text('''FoamFile\n{ format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
c=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('checkMesh return',c.returncode); print(c.stdout[-6000:]); print(c.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
# Topology, patching, volume, non-orthogonality, and skewness all pass, but 38 sliver tetrahedra trigg
import gmsh
gmsh.initialize(); gmsh.model.add('helical_gap')
gmsh.model.occ.importShapes('helical_gap.step'); gmsh.model.occ.synchronize()
for name,tags in {'neck_flank':[1,4],'lid_flank':[2,3],'inner_end':[5],'outer_end':[6]}.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
vg=gmsh.model.addPhysicalGroup(3,[1]); gmsh.model.setPhysicalName(3,vg,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',8e-5); gmsh.option.setNumber('Mesh.MeshSizeMax',2e-4)
gmsh.option.setNumber('Mesh.Algorithm',6); gmsh.option.setNumber('Mesh.Algorithm3D',10)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize('Netgen')
gmsh.write('helical_gap.msh')
print('optimized nodes',len(gmsh.model.mesh.getNodes()[0]),'tets',sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
# The optimized mesh has 184,287 tetrahedra after Netgen removed all illegal elements. I’ll reconvert 
import shutil
shutil.rmtree('constant/polyMesh',ignore_errors=True)
r=subprocess.run(['gmshToFoam','helical_gap.msh'],text=True,capture_output=True)
print('convert return',r.returncode, r.stdout[-900:])
c=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('checkMesh return',c.returncode); print(c.stdout[-5000:]); print(c.stderr[-400:])

# -- cell 12 ------------------------------------------------------------------------
# The optimized OpenFOAM mesh now passes bare `checkMesh`: 184,287 cells, one region, maximum aspect r
import pyvista as pv
Path('t13.foam').touch()
reader=pv.OpenFOAMReader('t13.foam'); reader.set_active_time_value(reader.time_values[-1] if reader.time_values else 0); data=reader.read()
print('OpenFOAM blocks:',data.keys())
internal=data['internalMesh']; surf=internal.extract_surface()
p=pv.Plotter(off_screen=True,window_size=(900,700)); p.add_mesh(surf,color='lightcyan',show_edges=True,line_width=0.3); p.view_isometric(); p.show(screenshot='final_mesh.png')
from IPython.display import display,Image
display(Image('final_mesh.png'))
