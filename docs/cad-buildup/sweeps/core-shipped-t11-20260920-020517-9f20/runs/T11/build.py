"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model flow along \(z\): duct from \(z=-40\) to \(100\) mm, with the core occupying \(z=0\) to \
import build123d as bd, inspect, math, os, subprocess, sys
sys.path.insert(0, '.reference')
from cad_export import export_patches
print('build123d', bd.__version__)
print('RegularPolygon:', inspect.signature(bd.RegularPolygon))
print('extrude:', inspect.signature(bd.extrude))
print('export_patches:', inspect.signature(export_patches))

# -- cell 2 -------------------------------------------------------------------------
# The CAD will use exact planar hexagons; “coarse” here means only the first low-resolution visual tes
# Requested dimensions, in metres
PLATE = 0.080
CORE_L = 0.020
UPSTREAM = 0.040
DOWNSTREAM = 0.080
CELL_AF = 0.008
WALL = 0.001
Z0, Z1 = -UPSTREAM, CORE_L + DOWNSTREAM
PITCH = CELL_AF + WALL
R_HEX = CELL_AF / math.sqrt(3.0)

# Hex centres on a triangular lattice; keep only complete clear openings inside the square.
centres = []
for j in range(-12, 13):
    for i in range(-12, 13):
        x = i * PITCH + j * PITCH / 2
        y = j * PITCH * math.sqrt(3) / 2
        verts = [(x + R_HEX*math.cos(math.radians(30+60*k)),
                  y + R_HEX*math.sin(math.radians(30+60*k))) for k in range(6)]
        if all(abs(vx) <= PLATE/2 + 1e-12 and abs(vy) <= PLATE/2 + 1e-12 for vx,vy in verts):
            centres.append((x,y))

hex_prisms = []
for x,y in centres:
    profile = bd.Pos(x, y, 0) * bd.RegularPolygon(R_HEX, 6, rotation=30)
    hex_prisms.append(bd.extrude(profile, amount=CORE_L))
apertures = bd.Compound(children=hex_prisms)
duct = bd.Pos(-PLATE/2, -PLATE/2, Z0) * bd.Box(PLATE, PLATE, Z1-Z0, align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN))
core_plate = bd.Pos(-PLATE/2, -PLATE/2, 0) * bd.Box(PLATE, PLATE, CORE_L, align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN))
core_solid = core_plate - apertures
fluid = duct - core_solid
bb = fluid.bounding_box()
print('complete hexagonal passages:', len(centres))
print('fluid solids:', len(fluid.solids()), 'faces:', len(fluid.faces()))
print('extent measured [m]:', tuple(round(v,6) for v in (bb.size.X, bb.size.Y, bb.size.Z)))
print('requested extent [m]:', (PLATE, PLATE, UPSTREAM+CORE_L+DOWNSTREAM))
print('clear cell across flats [m]:', CELL_AF, 'requested:', 0.008)
print('centre pitch [m]:', PITCH, '= clear width + wall thickness')

# -- cell 3 -------------------------------------------------------------------------
# The coarse CAD has 77 complete channels and one connected air volume. I’ll render the honeycomb soli
bd.export_stl(core_solid, 'core_preview.stl', tolerance=6e-4, angular_tolerance=0.2)
import pyvista as pv
pv.OFF_SCREEN = True
core_mesh = pv.read('core_preview.stl')
p = pv.Plotter(off_screen=True, window_size=(1000,700))
p.add_mesh(core_mesh, color='#d8a24a', show_edges=True, edge_color='#6b4a1d', line_width=0.5)
duct_box = pv.Box(bounds=(-PLATE/2,PLATE/2,-PLATE/2,PLATE/2,Z0,Z1))
p.add_mesh(duct_box, color='#7dc8ff', opacity=0.12, style='surface', show_edges=True)
p.add_axes()
p.camera_position = [(0.13,-0.13,0.11),(0,0,0.025),(0,0,1)]
p.show(screenshot='coarse_geometry.png')
print('preview triangles:', core_mesh.n_cells)
print('saved coarse_geometry.png')

# -- cell 4 -------------------------------------------------------------------------
# The preview confirms the intended full-depth hex channels and retained edge frame. Next I’ll identif
# Name source faces on the duct when they are created, using outward normal direction.
def n_at(face):
    n = face.normal_at()
    return (round(n.X,6), round(n.Y,6), round(n.Z,6))
source_by_n = {n_at(f): f for f in duct.faces()}
inlet_source = source_by_n[(0.0,0.0,-1.0)]
outlet_source = source_by_n[(0.0,0.0,1.0)]
duct_wall_sources = [f for key,f in source_by_n.items() if abs(key[0])+abs(key[1]) > 0.5]
print('duct source normals/areas:', sorted((n_at(f), round(f.area,7)) for f in duct.faces()))
print('fluid face area range:', min(f.area for f in fluid.faces()), max(f.area for f in fluid.faces()))
print('fluid normal counts:', {n:sum(n_at(f)==n for f in fluid.faces()) for n in sorted(set(n_at(f) for f in fluid.faces()))})

# -- cell 5 -------------------------------------------------------------------------
# That API name is not present in this installed build123d version, so I’ll inspect the installed shap
names=[n for n in dir(bd.Face) if any(k in n.lower() for k in ('inter','dist','over','fuse'))]
print(names)
for n in names:
    try: print(n, inspect.signature(getattr(bd.Face,n)))
    except Exception: pass

# -- cell 6 -------------------------------------------------------------------------
# I’ll now measure face–source intersections. A true inherited duct face must have a finite coincident
def coincident_area(face, source):
    try:
        hit = face.intersect(source, tolerance=1e-9)
        return 0.0 if hit is None else hit.area
    except Exception:
        return 0.0
for label,src in [('inlet',inlet_source),('outlet',outlet_source),('wall0',duct_wall_sources[0])]:
    vals=sorted([coincident_area(f,src) for f in fluid.faces()], reverse=True)
    print(label, [f'{v:.8g}' for v in vals[:5]])

# -- cell 7 -------------------------------------------------------------------------
# Coplanar face intersection returns no area in this kernel. I’ll instead require both (1) matching fa
def inherited_faces(source):
    sn = bd.Vector(*n_at(source))
    found=[]
    for f in fluid.faces():
        fn=f.normal_at()
        if fn.dot(sn) > 1-1e-8 and source.distance_to(f.center()) < 1e-8:
            found.append(f)
    return found
inlet_faces = inherited_faces(inlet_source)
outlet_faces = inherited_faces(outlet_source)
duct_wall_faces = [f for src in duct_wall_sources for f in inherited_faces(src)]
used_ids={id(f) for f in inlet_faces+outlet_faces+duct_wall_faces}
honey_faces=[f for f in fluid.faces() if id(f) not in used_ids]
print('patch face counts:', {'inlet':len(inlet_faces),'outlet':len(outlet_faces),'duct_walls':len(duct_wall_faces),'honeycomb_walls':len(honey_faces)})
print('patch areas [m2]:', {k:round(sum(f.area for f in fs),8) for k,fs in [('inlet',inlet_faces),('outlet',outlet_faces),('duct_walls',duct_wall_faces),('honeycomb_walls',honey_faces)]})
print('all faces assigned once:', sum(map(len,[inlet_faces,outlet_faces,duct_wall_faces,honey_faces]))==len(fluid.faces()))

# -- cell 8 -------------------------------------------------------------------------
# The explicit source-derived patches are correct, but Python wrapper identity is not a valid topologi
print([n for n in dir(bd.Face) if 'same' in n.lower() or 'equal' in n.lower()])
for src in duct_wall_sources:
    fs=inherited_faces(src)
    print('source',n_at(src),'result faces',len(fs),'areas',[round(f.area,8) for f in fs])
print('is_same signature:', inspect.signature(bd.Shape.is_same))

# -- cell 9 -------------------------------------------------------------------------
# Some edge cells touch the square boundary along a full hex side; those coincident surfaces correctly
explicit = inlet_faces + outlet_faces + duct_wall_faces
pairs_same=sum(explicit[i].is_same(explicit[j]) for i in range(len(explicit)) for j in range(i))
sample = hex_prisms[0]
sbb=sample.bounding_box()
measured_clear_af=sbb.size.X
measured_wall=PITCH-measured_clear_af
print('explicit patch duplicate faces:', pairs_same)
print('measured clear across flats [m]:', measured_clear_af, 'requested:', CELL_AF)
print('measured wall between x-neighbours [m]:', measured_wall, 'requested:', WALL)
print('measured core axial length [m]:', core_solid.bounding_box().size.Z, 'requested:', CORE_L)
patch_stats = export_patches(fluid, {
    'inlet': inlet_faces,
    'outlet': outlet_faces,
    'duct_walls': duct_wall_faces,
    'honeycomb_walls': ...
}, tolerance=2.5e-4)
print('export complete')

# -- cell 10 ------------------------------------------------------------------------
# I’ll now make a deliberately coarse snappyHexMesh trial: 4 mm background cells and 2 mm surface cell
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile{format ascii; class dictionary; object controlDict;}\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeFormat ascii;\n''')
Path('system/blockMeshDict').write_text('''FoamFile{format ascii; class dictionary; object blockMeshDict;}\nscale 1;\nvertices ((-0.044 -0.044 -0.044)(0.044 -0.044 -0.044)(0.044 0.044 -0.044)(-0.044 0.044 -0.044)(-0.044 -0.044 0.104)(0.044 -0.044 0.104)(0.044 0.044 0.104)(-0.044 0.044 0.104));\nblocks (hex (0 1 2 3 4 5 6 7) (22 22 37) simpleGrading (1 1 1));\nedges (); boundary (outerBox {type patch; faces ((0 3 2 1)(4 5 6 7)(0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2));}); mergePatchPairs ();\n''')
geom='\n'.join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in ['inlet','outlet','duct_walls','honeycomb_walls'])
refs='\n'.join(f'        {n} {{ level (1 1); patchInfo {{ type wall; }} }}' for n in ['inlet','outlet','duct_walls','honeycomb_walls'])
Path('system/snappyHexMeshDict').write_text(f'''FoamFile{{format ascii; class dictionary; object snappyHexMeshDict;}}\ncastellatedMesh true; snap true; addLayers false;\ngeometry\n{{\n{geom}\n}}\ncastellatedMeshControls\n{{ maxLocalCells 2000000; maxGlobalCells 3000000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 1; features (); refinementSurfaces\n    {{\n{refs}\n    }}\n    resolveFeatureAngle 25; refinementRegions {{}}; locationInMesh (0 0 -0.02); allowFreeStandingZoneFaces true; }}\nsnapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}\naddLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}\nmeshQualityControls {{ maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-18; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}\nmergeTolerance 1e-6;\n''')
print('wrote OpenFOAM dictionaries; coarse surface size = 2 mm')

# -- cell 11 ------------------------------------------------------------------------
# I’ll run the coarse mesh and immediately inspect its standard mesh-quality summary and patch table.
r1=subprocess.run(['blockMesh'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print('blockMesh rc',r1.returncode, '\n'.join(r1.stdout.splitlines()[-8:]))
r2=subprocess.run(['snappyHexMesh','-overwrite'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
Path('log.snappy.coarse').write_text(r2.stdout)
print('snappy rc',r2.returncode)
print('\n'.join(r2.stdout.splitlines()[-25:]))

# -- cell 12 ------------------------------------------------------------------------
# The failure was only a missing mandatory `controlDict` write entry; no meshing computation ran. I’ll
p=Path('system/controlDict')
p.write_text('''FoamFile{format ascii; class dictionary; object controlDict;}\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n''')
r1=subprocess.run(['blockMesh'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print('blockMesh rc',r1.returncode, '\n'.join(r1.stdout.splitlines()[-6:]))
r2=subprocess.run(['snappyHexMesh','-overwrite'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
Path('log.snappy.coarse').write_text(r2.stdout)
print('snappy rc',r2.returncode, '\n'.join(r2.stdout.splitlines()[-20:]))

# -- cell 13 ------------------------------------------------------------------------
# `blockMesh` now exists (17,908 coarse cells). This OpenFOAM build also requires placeholder discreti
Path('system/fvSchemes').write_text('''FoamFile{format ascii; class dictionary; object fvSchemes;}\nddtSchemes{default Euler;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n''')
Path('system/fvSolution').write_text('''FoamFile{format ascii; class dictionary; object fvSolution;}\nsolvers{} PIMPLE{}\n''')
r2=subprocess.run(['snappyHexMesh','-overwrite'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
Path('log.snappy.coarse').write_text(r2.stdout)
print('snappy rc',r2.returncode)
print('\n'.join(r2.stdout.splitlines()[-30:]))

# -- cell 14 ------------------------------------------------------------------------
# The coarse snapped air mesh now exists with 45,880 cells and no snappy quality errors. I’ll run `che
cm=subprocess.run(['checkMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('log.checkMesh.coarse').write_text(cm.stdout)
print('checkMesh rc',cm.returncode)
print('\n'.join(cm.stdout.splitlines()[-22:]))
Path('t11.foam').touch()
reader=pv.OpenFOAMReader('t11.foam')
reader.set_active_time_value(0.0)
data=reader.read()
print('reader blocks:', data.keys())

# -- cell 15 ------------------------------------------------------------------------
# The coarse mesh is topologically closed, but bare `checkMesh` flags 22 highly skew faces, so it is n
internal=data['internalMesh']
sl=internal.slice(normal=(0,0,1), origin=(0,0,0.01))
p=pv.Plotter(off_screen=True,window_size=(850,800))
p.add_mesh(sl,color='#b9d9f3',show_edges=True,edge_color='#24465f',line_width=0.35)
p.view_xy(); p.camera.zoom(1.15); p.add_axes()
p.show(screenshot='coarse_mesh_core_slice.png')
print('slice cells:',sl.n_cells,'saved coarse_mesh_core_slice.png')

# -- cell 16 ------------------------------------------------------------------------
# The slice confirms all 77 passages and visible 1 mm ligaments, but only about four cells span each o
s=Path('system/snappyHexMeshDict').read_text().replace('level (1 1);','level (2 2);')
Path('system/snappyHexMeshDict').write_text(s)
r1=subprocess.run(['blockMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('log.snappy.final').write_text(r2.stdout)
print('block rc',r1.returncode,'snappy rc',r2.returncode)
print('\n'.join(r2.stdout.splitlines()[-24:]))

# -- cell 17 ------------------------------------------------------------------------
# The refined mesh completed with 196,736 cells and no snappy quality errors. I’ll now run the binding
cm=subprocess.run(['checkMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('log.checkMesh').write_text(cm.stdout)
print('checkMesh rc',cm.returncode)
print('\n'.join(cm.stdout.splitlines()[-28:]))
# Print boundary table
print('\nBOUNDARY FILE:\n',Path('constant/polyMesh/boundary').read_text()[:3000])

# -- cell 18 ------------------------------------------------------------------------
# Bare `checkMesh` still rejects the final mesh for boundary skewness (maximum 5.67), despite all othe
s=Path('system/snappyHexMeshDict').read_text().replace('maxBoundarySkewness 20;','maxBoundarySkewness 4;')
Path('system/snappyHexMeshDict').write_text(s)
r1=subprocess.run(['blockMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('log.snappy.final').write_text(r2.stdout)
print('block rc',r1.returncode,'snappy rc',r2.returncode)
print('\n'.join(r2.stdout.splitlines()[-30:]))

# -- cell 19 ------------------------------------------------------------------------
# Snappy now reports zero faces above skewness 4. I’ll rerun the binding check and, if it passes, prod
cm=subprocess.run(['checkMesh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('log.checkMesh').write_text(cm.stdout)
print('checkMesh rc',cm.returncode)
print('\n'.join(cm.stdout.splitlines()[-24:]))

# -- cell 20 ------------------------------------------------------------------------
# The final bare `checkMesh` passes (`Mesh OK`), with maximum skewness 3.999 and maximum non-orthogona
reader=pv.OpenFOAMReader('t11.foam'); reader.set_active_time_value(0.0); final_data=reader.read(); internal=final_data['internalMesh']
sl=internal.slice(normal=(0,1,0), origin=(0,0,0.03))
p=pv.Plotter(off_screen=True,window_size=(1100,600))
p.add_mesh(sl,color='#b9d9f3',show_edges=True,edge_color='#29485f',line_width=0.25)
p.view_xz(); p.camera.zoom(1.1); p.add_axes()
p.show(screenshot='final_mesh_axial_slice.png')
print('final mesh cells:',internal.n_cells,'axial slice cells:',sl.n_cells)
print('saved final_mesh_axial_slice.png')

# -- cell 21 ------------------------------------------------------------------------
# The completion gate found that the exported-surface manifest omitted the same interior point used by
patch_stats = export_patches(fluid, {
    'inlet': inlet_faces,
    'outlet': outlet_faces,
    'duct_walls': duct_wall_faces,
    'honeycomb_walls': ...
}, tolerance=2.5e-4, location_in_mesh=(0.0,0.0,-0.020))
print('manifest updated with meshing interior point (0,0,-0.020) m')
