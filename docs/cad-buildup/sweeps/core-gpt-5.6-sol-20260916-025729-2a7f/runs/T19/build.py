"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the sealed oil domain as the 60 mm axial extrusion of the bore cross-section minus the ke
import build123d as bd
import numpy as np
import matplotlib.pyplot as plt
# All dimensions are metres.
L=0.060; R_BORE=0.02005; R_SHAFT=0.020; KEY_W=0.012; KEY_DEPTH=0.005
# Algebra-mode CAD: shaft keyway extends from the nominal crown down to y=15 mm.
outer = bd.Cylinder(R_BORE, L)
shaft_blank = bd.Cylinder(R_SHAFT, L)
key_cut = bd.Pos(-KEY_W/2, R_SHAFT-KEY_DEPTH, 0) * bd.Box(KEY_W, KEY_DEPTH+0.001, L)
shaft_keyed = shaft_blank - key_cut
fluid = outer - shaft_keyed
print(f"sealed length: {L:.6f} m (requested 0.060000 m)")
print(f"bore diameter: {2*R_BORE:.6f} m (requested 0.040100 m)")
print(f"shaft diameter: {2*R_SHAFT:.6f} m (requested 0.040000 m)")
print(f"radial clearance: {R_BORE-R_SHAFT:.8f} m (requested 0.00005000 m)")
print(f"keyway width: {KEY_W:.6f} m; radial crown-to-floor depth: {KEY_DEPTH:.6f} m")
print(f"fluid volume: {fluid.volume:.10e} m^3; faces: {len(fluid.faces())}; solids: {len(fluid.solids())}")
# Cross-section plot, full and clearance close-up.
th=np.linspace(0,2*np.pi,1000); yi=np.sqrt(R_SHAFT**2-(KEY_W/2)**2)
fig,ax=plt.subplots(1,2,figsize=(11,4.5))
for a in ax:
    a.plot(R_BORE*np.cos(th)*1e3,R_BORE*np.sin(th)*1e3,'b-',lw=2,label='bore')
    # keyed shaft boundary
    tt=np.linspace(np.arcsin(yi/R_SHAFT),2*np.pi+np.arcsin(yi/R_SHAFT),1200)
    keep=~((np.sin(tt)>0)&(np.abs(R_SHAFT*np.cos(tt))<KEY_W/2))
    a.plot((R_SHAFT*np.cos(tt)[keep])*1e3,(R_SHAFT*np.sin(tt)[keep])*1e3,'k-',lw=1.5)
    a.plot(np.array([-6,-6,6,6]),np.array([yi*1e3,15,15,yi*1e3]),'k-',lw=1.5)
    a.set_aspect('equal'); a.grid(True); a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]')
ax[0].set_xlim(-22,22); ax[0].set_ylim(-22,22); ax[0].set_title('Oil cross-section')
ax[1].set_xlim(17.5,20.2); ax[1].set_ylim(-4,4); ax[1].set_title('0.05 mm clearance detail')
plt.tight_layout(); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The coarse inspection confirms a single connected leakage domain and visibly resolves the narrow ann
faces=list(fluid.faces())
for i,f in enumerate(faces):
    c=f.center(); n=f.normal_at();
    print(i, type(f.geom_adaptor()).__name__, f"area={f.area:.10e}", f"center=({c.X:.6g},{c.Y:.6g},{c.Z:.6g})", f"normal=({n.X:.3g},{n.Y:.3g},{n.Z:.3g})")

# -- cell 3 -------------------------------------------------------------------------
# The face audit exposed a CAD alignment issue: the initial inspection sketch was correct, but build12
import inspect
print(inspect.signature(bd.Box)); print(inspect.signature(bd.Cylinder)); print(bd.Align.MIN)

# -- cell 4 -------------------------------------------------------------------------
# I’ll rebuild with x/y centred and z starting at 0 for both cylinders, while the key cutter uses mini
A=bd.Align
outer = bd.Cylinder(R_BORE,L,align=(A.CENTER,A.CENTER,A.MIN))
shaft_blank = bd.Cylinder(R_SHAFT,L,align=(A.CENTER,A.CENTER,A.MIN))
key_cut = bd.Pos(-KEY_W/2,R_SHAFT-KEY_DEPTH,0)*bd.Box(KEY_W,KEY_DEPTH+0.001,L,align=(A.MIN,A.MIN,A.MIN))
shaft_keyed=shaft_blank-key_cut
fluid=outer-shaft_keyed
faces=list(fluid.faces())
print(f"volume={fluid.volume:.10e} m3, connected solids={len(fluid.solids())}, faces={len(faces)}")
for i,f in enumerate(faces):
 c=f.center(); n=f.normal_at(); print(i,type(f.geom_adaptor()).__name__,f"A={f.area:.9e}",f"c=({c.X:.6g},{c.Y:.6g},{c.Z:.6g})",f"n=({n.X:.2g},{n.Y:.2g},{n.Z:.2g})")

# -- cell 5 -------------------------------------------------------------------------
# The corrected CAD now has exactly seven faces: bore, shaft, floor, two flanks, and two ends. I’ll ex
print(inspect.signature(bd.export_stl))

# -- cell 6 -------------------------------------------------------------------------
# Patch selection will use surface class and intrinsic face normals: cylindrical bore, trimmed cylindr
from pathlib import Path
tri=Path('constant/triSurface'); tri.mkdir(parents=True,exist_ok=True)
bore=[f for f in faces if type(f.geom_adaptor()).__name__=='Geom_CylindricalSurface']
shaft=[f for f in faces if type(f.geom_adaptor()).__name__=='Geom_RectangularTrimmedSurface']
planes=[f for f in faces if type(f.geom_adaptor()).__name__=='Geom_Plane']
high=[f for f in planes if f.normal_at().Z < -0.9]  # z=0, 50 mm shaft side
low=[f for f in planes if f.normal_at().Z > 0.9]    # z=L, 30 mm shaft side
floor=[f for f in planes if f.normal_at().Y < -0.9]
flanks=[f for f in planes if abs(f.normal_at().X) > 0.9]
patch_faces={'shaft':shaft,'keywayFloor':floor,'keywayFlanks':flanks,'bore':bore,'highPressure':high,'lowPressure':low}
for name,fs in patch_faces.items():
    shape=fs[0] if len(fs)==1 else bd.Compound(fs)
    ok=bd.export_stl(shape,tri/f'{name}.stl',tolerance=5e-6,angular_tolerance=np.deg2rad(1),ascii_format=True)
    print(name,'faces',len(fs),'CAD area',sum(f.area for f in fs),'export',ok)
print('STLs:',sorted(p.name for p in tri.glob('*.stl')))

# -- cell 7 -------------------------------------------------------------------------
# The preview is geometrically correct, but the replay checker rightly notes that the imports only occ
import gmsh, pyvista as pv
gmsh.initialize(); gmsh.model.add('oilSection')
o=gmsh.model.occ.addDisk(0,0,0,R_BORE,R_BORE); s=gmsh.model.occ.addDisk(0,0,0,R_SHAFT,R_SHAFT)
k=gmsh.model.occ.addRectangle(-KEY_W/2,R_SHAFT-KEY_DEPTH,0,KEY_W,KEY_DEPTH+0.001)
shaft2,_=gmsh.model.occ.cut([(2,s)],[(2,k)]); fluid2,_=gmsh.model.occ.cut([(2,o)],shaft2)
gmsh.model.occ.synchronize(); gmsh.option.setNumber('Mesh.MeshSizeMin',2.5e-5); gmsh.option.setNumber('Mesh.MeshSizeMax',5e-4); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',20)
gmsh.model.mesh.generate(2); gmsh.write('section.vtk'); gmsh.finalize()
sec=pv.read('section.vtk'); print('2D cells:',sec.n_cells,'points:',sec.n_points,'bounds:',tuple(round(v,7) for v in sec.bounds))
pl=pv.Plotter(off_screen=True,window_size=(800,700)); pl.add_mesh(sec,show_edges=True,color='lightblue',line_width=0.4); pl.view_xy(); pl.camera.zoom(1.2); pl.show(screenshot='sectionMesh.png')

# -- cell 8 -------------------------------------------------------------------------
# The measured extrusion entities are unambiguous: one 0.00755867 m² outer surface is the bore; the tw
import gmsh
gmsh.initialize(); gmsh.model.add('oilLeak')
o=gmsh.model.occ.addDisk(0,0,0,R_BORE,R_BORE); s=gmsh.model.occ.addDisk(0,0,0,R_SHAFT,R_SHAFT); k=gmsh.model.occ.addRectangle(-KEY_W/2,R_SHAFT-KEY_DEPTH,0,KEY_W,KEY_DEPTH+0.001)
shaft2,_=gmsh.model.occ.cut([(2,s)],[(2,k)]); fluid2,_=gmsh.model.occ.cut([(2,o)],shaft2); base=fluid2[0][1]
gmsh.model.occ.synchronize(); ext=gmsh.model.occ.extrude([(2,base)],0,0,L,numElements=[60],recombine=True); gmsh.model.occ.synchronize(); vol=[t for d,t in ext if d==3][0]; top=ext[0][1]
all_surfs=[t for d,t in gmsh.model.getBoundary([(3,vol)],oriented=False) if d==2]; side=[t for t in all_surfs if t not in (base,top)]; areas={t:gmsh.model.occ.getMass(2,t) for t in all_surfs}; planes=[t for t in side if gmsh.model.getType(2,t)=='Plane']; swept=[t for t in side if t not in planes]
bore_g=[max(swept,key=lambda t:areas[t])]; shaft_g=[t for t in swept if t not in bore_g]; floor_g=[max(planes,key=lambda t:areas[t])]; flanks_g=[t for t in planes if t not in floor_g]
groups={'shaft':shaft_g,'keywayFloor':floor_g,'keywayFlanks':flanks_g,'bore':bore_g,'highPressure':[base],'lowPressure':[top]}
for pid,(name,tags) in enumerate(groups.items(),1):
 gmsh.model.addPhysicalGroup(2,tags,pid); gmsh.model.setPhysicalName(2,pid,name); print(name,'area',sum(areas[t] for t in tags))
gmsh.model.addPhysicalGroup(3,[vol],100); gmsh.model.setPhysicalName(3,100,'oil'); gmsh.option.setNumber('Mesh.MeshSizeMin',2.5e-5); gmsh.option.setNumber('Mesh.MeshSizeMax',5e-4); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',20); gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3); gmsh.write('oilLeak.msh'); gmsh.write('oilLeak.vtk'); print('nodes',len(gmsh.model.mesh.getNodes()[0]),'3D cells',sum(len(a) for a in gmsh.model.mesh.getElements(3)[1])); gmsh.finalize()

# -- cell 9 -------------------------------------------------------------------------
# `gmshToFoam` requires a minimal OpenFOAM `controlDict` even for conversion; the geometry and Gmsh me
import subprocess
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile\n{\n version 2.0; format ascii; class dictionary; object controlDict;\n}\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
r=subprocess.run(['gmshToFoam','oilLeak.msh'],text=True,capture_output=True); print(r.stdout[-1800:]); print(r.stderr[-500:]); r.check_returncode()
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 10 ------------------------------------------------------------------------
# Conversion produced exactly the six requested OpenFOAM patches and 61,920 prism cells. I’ll now run 
c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); print(c.stdout[-7000:]); print(c.stderr[-1000:]); print('return',c.returncode)

# -- cell 11 ------------------------------------------------------------------------
# This OpenFOAM build also loads `fvSchemes`/`fvSolution` when `checkMesh -allGeometry` is requested, 
Path('system/fvSchemes').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n')
c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); print(c.stdout[-7000:]); print(c.stderr[-800:]); print('return',c.returncode)

# -- cell 12 ------------------------------------------------------------------------
# The geometry check expects the standard scheme subdictionaries, not merely an empty file. I’ll provi
Path('system/fvSchemes').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); print(c.stdout[-7000:]); print(c.stderr[-500:]); print('return',c.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The first 3D check is topologically sound but fails one quality criterion: 912 prisms have determina
gmsh.initialize(); gmsh.model.add('oilLeakRefined')
o=gmsh.model.occ.addDisk(0,0,0,R_BORE,R_BORE); s=gmsh.model.occ.addDisk(0,0,0,R_SHAFT,R_SHAFT); k=gmsh.model.occ.addRectangle(-KEY_W/2,R_SHAFT-KEY_DEPTH,0,KEY_W,KEY_DEPTH+0.001)
shaft2,_=gmsh.model.occ.cut([(2,s)],[(2,k)]); fluid2,_=gmsh.model.occ.cut([(2,o)],shaft2); base=fluid2[0][1]; gmsh.model.occ.synchronize(); ext=gmsh.model.occ.extrude([(2,base)],0,0,L,numElements=[240],recombine=True); gmsh.model.occ.synchronize(); vol=[t for d,t in ext if d==3][0]; top=ext[0][1]
surfs=[t for d,t in gmsh.model.getBoundary([(3,vol)],oriented=False) if d==2]; side=[t for t in surfs if t not in (base,top)]; areas={t:gmsh.model.occ.getMass(2,t) for t in surfs}; planes=[t for t in side if gmsh.model.getType(2,t)=='Plane']; swept=[t for t in side if t not in planes]; bore_g=[max(swept,key=lambda t:areas[t])]; shaft_g=[t for t in swept if t not in bore_g]; floor_g=[max(planes,key=lambda t:areas[t])]; flanks_g=[t for t in planes if t not in floor_g]
for pid,(name,tags) in enumerate({'shaft':shaft_g,'keywayFloor':floor_g,'keywayFlanks':flanks_g,'bore':bore_g,'highPressure':[base],'lowPressure':[top]}.items(),1): gmsh.model.addPhysicalGroup(2,tags,pid); gmsh.model.setPhysicalName(2,pid,name)
gmsh.model.addPhysicalGroup(3,[vol],100); gmsh.model.setPhysicalName(3,100,'oil'); gmsh.option.setNumber('Mesh.MeshSizeMin',2.5e-5); gmsh.option.setNumber('Mesh.MeshSizeMax',5e-4); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',20); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.model.mesh.generate(3); gmsh.write('oilLeak.msh'); gmsh.write('oilLeak.vtk'); print('refined nodes',len(gmsh.model.mesh.getNodes()[0]),'cells',sum(len(a) for a in gmsh.model.mesh.getElements(3)[1])); gmsh.finalize()

# -- cell 14 ------------------------------------------------------------------------
# The refined mesh contains 247,680 prisms. I’ll reconvert it over the coarse OpenFOAM mesh and rerun 
r=subprocess.run(['gmshToFoam','oilLeak.msh'],text=True,capture_output=True); print(r.stdout[-900:]); r.check_returncode(); c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); print(c.stdout[-4200:]); print('return',c.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The refined mesh passes all checks: one region, six patches, determinant minimum 0.00297, aspect rat
import pyvista as pv
grid=pv.read('oilLeak.vtk'); cut=grid.clip(normal=(1,0,0),origin=(0,0,0)); surf=cut.extract_surface()
print('VTK cells',grid.n_cells,'cutaway surface cells',surf.n_cells,'bounds',grid.bounds)
pl=pv.Plotter(off_screen=True,window_size=(1000,650)); pl.add_mesh(surf,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.15); pl.view_isometric(); pl.camera.zoom(1.35); pl.show(screenshot='finalMeshCutaway.png')
