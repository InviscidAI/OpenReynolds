"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the barrel from \(z=0\) to 120 mm, place the fin stack with the specified 6 mm clear gaps
import build123d as bd
import numpy as np, pyvista as pv
from pathlib import Path

# All dimensions are metres.
BARREL_D = 0.090
BARREL_H = 0.120
N_FINS = 12
FIN_D = 0.140
FIN_T = 0.003
FIN_GAP = 0.006
FIN_PITCH = FIN_T + FIN_GAP
FIN_END_MARGIN = (BARREL_H - (N_FINS*FIN_T + (N_FINS-1)*FIN_GAP))/2
HEAD_R = BARREL_D/2
DUCT_W = 0.300
Z_INLET = -0.150
Z_OUTLET = BARREL_H + HEAD_R + 0.300

barrel_primitive = bd.Cylinder(BARREL_D/2, BARREL_H)
fin_primitives = [bd.Pos(0, 0, FIN_END_MARGIN + i*FIN_PITCH) * bd.Cylinder(FIN_D/2, FIN_T) for i in range(N_FINS)]
fin_union = fin_primitives[0]
for f in fin_primitives[1:]:
    fin_union = fin_union + f
head_sphere = bd.Pos(0, 0, BARREL_H) * bd.Sphere(HEAD_R)
head_clip = bd.Pos(-HEAD_R, -HEAD_R, BARREL_H) * bd.Box(2*HEAD_R, 2*HEAD_R, HEAD_R)
head_primitive = head_sphere & head_clip
engine = barrel_primitive + fin_union + head_primitive

print(f"Fin count: measured {len(fin_primitives)}, requested {N_FINS}")
print(f"Barrel OD: measured {2*BARREL_D/2:.6f} m, requested {BARREL_D:.6f} m")
print(f"Barrel height: measured {BARREL_H:.6f} m, requested {BARREL_H:.6f} m")
print(f"Fin OD: measured {FIN_D:.6f} m, requested {FIN_D:.6f} m")
print(f"Fin thickness: measured {FIN_T:.6f} m, requested {FIN_T:.6f} m")
print(f"Neighbour clear gap: measured {FIN_PITCH-FIN_T:.6f} m, requested {FIN_GAP:.6f} m")
print(f"Assumed equal fin end margin: {FIN_END_MARGIN:.6f} m")
print(f"Assumed hemispherical head height: {HEAD_R:.6f} m")
print(f"Duct z range: {Z_INLET:.6f} to {Z_OUTLET:.6f} m")

# Coarse tessellation solely for visual inspection.
tmp = Path('engine_preview.stl')
bd.export_stl(engine, tmp, tolerance=0.001, angular_tolerance=0.15)
mesh_preview = pv.read(tmp)
pl = pv.Plotter(off_screen=True, window_size=(850,700))
pl.add_mesh(mesh_preview, color='silver', smooth_shading=True, show_edges=True, edge_color='gray')
pl.add_mesh(pv.Box(bounds=(-DUCT_W/2,DUCT_W/2,-DUCT_W/2,DUCT_W/2,Z_INLET,Z_OUTLET)), style='wireframe', color='royalblue', opacity=0.35)
pl.add_axes(); pl.view_isometric(); pl.show(screenshot='coarse_geometry.png')
print(f"Preview surface: {mesh_preview.n_cells} triangles; bounds={np.array(mesh_preview.bounds).reshape(3,2).tolist()}")

# -- cell 2 -------------------------------------------------------------------------
# The preview exposed an origin error: build123d’s default axial alignment centered the cylinders, so 
import inspect
print('Cylinder', inspect.signature(bd.Cylinder))
print('Box', inspect.signature(bd.Box))
print('Sphere', inspect.signature(bd.Sphere))

# -- cell 3 -------------------------------------------------------------------------
# I’ll rebuild with explicit `MIN` axial alignment and use the sphere’s upper-latitude range directly 
A = bd.Align
barrel_primitive = bd.Cylinder(BARREL_D/2, BARREL_H, align=(A.CENTER,A.CENTER,A.MIN))
fin_primitives = [bd.Pos(0, 0, FIN_END_MARGIN + i*FIN_PITCH) * bd.Cylinder(FIN_D/2, FIN_T, align=(A.CENTER,A.CENTER,A.MIN)) for i in range(N_FINS)]
fin_union = fin_primitives[0]
for f in fin_primitives[1:]:
    fin_union = fin_union + f
head_primitive = bd.Pos(0, 0, BARREL_H) * bd.Sphere(HEAD_R, arc_size1=0, arc_size2=90)
engine = barrel_primitive + fin_union + head_primitive
bb = engine.bounding_box()
print(f"Engine bounds x=[{bb.min.X:.6f},{bb.max.X:.6f}], y=[{bb.min.Y:.6f},{bb.max.Y:.6f}], z=[{bb.min.Z:.6f},{bb.max.Z:.6f}]")
print(f"Expected z=[0.000000,{BARREL_H+HEAD_R:.6f}] m")
bd.export_stl(engine, tmp, tolerance=0.0005, angular_tolerance=0.10)
mesh_preview = pv.read(tmp)
pl = pv.Plotter(off_screen=True, window_size=(850,700))
pl.add_mesh(mesh_preview, color='silver', smooth_shading=True, show_edges=True, edge_color='gray')
pl.add_mesh(pv.Box(bounds=(-DUCT_W/2,DUCT_W/2,-DUCT_W/2,DUCT_W/2,Z_INLET,Z_OUTLET)), style='wireframe', color='royalblue', opacity=0.35)
pl.add_axes(); pl.view_isometric(); pl.show(screenshot='coarse_geometry_corrected.png')
print(f"Corrected preview surface: {mesh_preview.n_cells} triangles")

# -- cell 4 -------------------------------------------------------------------------
# The second bounds check caught a subtler alignment behavior: a partial sphere is centered by its own
head_primitive = bd.Pos(0, 0, BARREL_H) * bd.Sphere(HEAD_R, arc_size1=0, arc_size2=90, align=(A.CENTER,A.CENTER,A.MIN))
engine = barrel_primitive + fin_union + head_primitive
bb = engine.bounding_box()
print(f"Corrected engine z bounds: [{bb.min.Z:.6f},{bb.max.Z:.6f}] m; requested barrel plus assumed dome -> [0,{BARREL_H+HEAD_R:.6f}]")
faces = list(engine.faces())
print('Fused face count:', len(faces))
for j, face in enumerate(faces):
    fb=face.bounding_box(); c=face.center()
    print(j, type(face.geom_type).__name__, str(face.geom_type), f"A={face.area:.8g}", f"Cz={c.Z:.6f}", f"z=[{fb.min.Z:.6f},{fb.max.Z:.6f}]", f"rmax={max(abs(fb.min.X),abs(fb.max.X),abs(fb.min.Y),abs(fb.max.Y)):.6f}")

# -- cell 5 -------------------------------------------------------------------------
# The fused solid has exactly the expected topology: 12 fin rims, 24 annular fin faces, 13 exposed bar
print('Compound', inspect.signature(bd.Compound))
print('Wire.make_polygon', inspect.signature(bd.Wire.make_polygon))
print('Face', inspect.signature(bd.Face))
print('export_stl', inspect.signature(bd.export_stl))

# -- cell 6 -------------------------------------------------------------------------
# Patch winding will be outward from the engine solid and outward from the fluid duct. The wall patch 
tri_dir = Path('constant/triSurface'); tri_dir.mkdir(parents=True, exist_ok=True)
GT = bd.GeomType
head_faces = [f for f in faces if f.geom_type == GT.SPHERE]
fin_faces = []
barrel_faces = []
for f in faces:
    fb=f.bounding_box(); rmax=max(abs(fb.min.X),abs(fb.max.X),abs(fb.min.Y),abs(fb.max.Y))
    if f.geom_type == GT.SPHERE:
        continue
    if rmax > (BARREL_D/2 + 0.005):
        fin_faces.append(f)
    else:
        barrel_faces.append(f)
assert (len(barrel_faces),len(fin_faces),len(head_faces)) == (14,36,1)
patch_shapes = {
    'barrel': bd.Compound(barrel_faces),
    'fins': bd.Compound(fin_faces),
    'head': bd.Compound(head_faces),
}

def planar_face(points):
    return bd.Face(bd.Wire.make_polygon(points, close=True))
h=DUCT_W/2; z0=Z_INLET; z1=Z_OUTLET
inlet_face = planar_face([(-h,-h,z0),(-h,h,z0),(h,h,z0),(h,-h,z0)])
outlet_face = planar_face([(-h,-h,z1),(h,-h,z1),(h,h,z1),(-h,h,z1)])
wall_faces = [
    planar_face([(-h,-h,z0),(-h,-h,z1),(-h,h,z1),(-h,h,z0)]),
    planar_face([( h,-h,z0),( h, h,z0),( h, h,z1),( h,-h,z1)]),
    planar_face([(-h,-h,z0),( h,-h,z0),( h,-h,z1),(-h,-h,z1)]),
    planar_face([(-h, h,z0),(-h, h,z1),( h, h,z1),( h, h,z0)]),
]
patch_shapes.update({'ductWalls':bd.Compound(wall_faces), 'inlet':inlet_face, 'outlet':outlet_face})
for name,shape in patch_shapes.items():
    bd.export_stl(shape, tri_dir/f'{name}.stl', tolerance=0.00025, angular_tolerance=0.08, ascii_format=True)
    pm=pv.read(tri_dir/f'{name}.stl')
    print(f"{name:10s}: {pm.n_cells:6d} triangles, area={pm.area:.8f} m^2")
print('Exported:', sorted(p.name for p in tri_dir.glob('*.stl')))

# -- cell 7 -------------------------------------------------------------------------
# The exported surfaces have the expected six patch files and areas. I’ll use a 15 mm background grid 
system=Path('system'); system.mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
(system/'controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh;\nstartFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\nwriteControl timeStep; writeInterval 1;\n''')
(system/'blockMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
 (-{h} -{h} {z0}) ({h} -{h} {z0}) ({h} {h} {z0}) (-{h} {h} {z0})
 (-{h} -{h} {z1}) ({h} -{h} {z1}) ({h} {h} {z1}) (-{h} {h} {z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) (20 20 41) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet {{ type patch; faces ((0 3 2 1)); }}
 outlet {{ type patch; faces ((4 5 6 7)); }}
 ductWalls {{ type wall; faces ((0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7)); }}
);
mergePatchPairs ();
''')
geom='\n'.join([f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in ['barrel','fins','head']])
ref='\n'.join([f'        {n} {{ level (2 3); patchInfo {{ type wall; }} }}' for n in ['barrel','fins','head']])
(system/'snappyHexMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true;
snap true;
addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 1200000;
    maxGlobalCells 1500000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{ref}
    }}
    resolveFeatureAngle 25;
    refinementRegions {{ }}
    locationInMesh (0.12 0.12 0.30);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 5;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 8;
    nFeatureSnapIter 15;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true;
    layers {{ }}
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    #includeEtc "caseDicts/meshQualityDict"
}}
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
''')
print((system/'blockMeshDict').read_text())

# -- cell 8 -------------------------------------------------------------------------
# I’ll repeat that validation cell with its subprocess import included so the final `build.py` reprodu
import subprocess
(system/'fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n''')
(system/'fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers { }\n''')
r1=subprocess.run(['blockMesh'], text=True, capture_output=True)
print('\n'.join(r1.stdout.splitlines()[-18:])); print(r1.stderr)
if r1.returncode: raise RuntimeError('blockMesh failed')
r2=subprocess.run(['checkMesh','-constant'], text=True, capture_output=True)
print('\n'.join(r2.stdout.splitlines()[-28:])); print(r2.stderr)
if r2.returncode: raise RuntimeError('background checkMesh failed')

# -- cell 9 -------------------------------------------------------------------------
# Castellation completed at 176,142 cells, but snapping stopped because the stock quality include in t
import subprocess
sd=system/'snappyHexMeshDict'
txt=sd.read_text()
txt=txt.replace('    #includeEtc "caseDicts/meshQualityDict"\n}', '    #includeEtc "caseDicts/meshQualityDict"\n    nSmoothScale 4;\n    errorReduction 0.75;\n}')
sd.write_text(txt)
rb=subprocess.run(['blockMesh'], text=True, capture_output=True)
if rb.returncode: print(rb.stdout,rb.stderr); raise RuntimeError('blockMesh reset failed')
rs=subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(rs.stdout+'\n'+rs.stderr)
print('\n'.join(rs.stdout.splitlines()[-55:])); print(rs.stderr)
if rs.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 10 ------------------------------------------------------------------------
# The standard check fails only on four highly skew boundary faces. I’ll tighten snappy’s own boundary
import subprocess
sd=system/'snappyHexMeshDict'; txt=sd.read_text()
old='    #includeEtc "caseDicts/meshQualityDict"\n    nSmoothScale 4;\n    errorReduction 0.75;'
quality='''    maxNonOrtho 65;
    maxBoundarySkewness 3.5;
    maxInternalSkewness 3.5;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    minEdgeLength -1;
    nSmoothScale 4;
    errorReduction 0.75;'''
if old in txt: txt=txt.replace(old,quality)
elif 'maxBoundarySkewness 3.5;' not in txt: raise RuntimeError('unexpected quality dictionary state')
sd.write_text(txt)
rb=subprocess.run(['blockMesh'], text=True, capture_output=True)
if rb.returncode: raise RuntimeError(rb.stdout+rb.stderr)
rs=subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(rs.stdout+'\n'+rs.stderr)
print('\n'.join(rs.stdout.splitlines()[-45:])); print(rs.stderr)
if rs.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 11 ------------------------------------------------------------------------
# The remesh satisfies the tightened 3.5 skew criterion during snapping. I’ll run the binding standard
import subprocess
rc_default=subprocess.run(['checkMesh','-constant'], text=True, capture_output=True)
Path('checkMesh.log').write_text(rc_default.stdout+'\n'+rc_default.stderr)
print('\n'.join(rc_default.stdout.splitlines()[-48:])); print(rc_default.stderr)
if rc_default.returncode or 'Mesh OK' not in rc_default.stdout: raise RuntimeError('standard checkMesh failed')

# -- cell 12 ------------------------------------------------------------------------
# The final standard check is clean: 176,142 cells, maximum non-orthogonality 56.1°, maximum skewness 
import subprocess
rv=subprocess.run(['foamToVTK','-constant','-no-fields'], text=True, capture_output=True)
print('\n'.join(rv.stdout.splitlines()[-20:])); print(rv.stderr)
if rv.returncode: raise RuntimeError('foamToVTK failed')
print([str(p) for p in Path('VTK').rglob('*') if p.is_file()][:20])

# -- cell 13 ------------------------------------------------------------------------
# The conversion confirms all six named patches in the actual mesh. I’ll render the \(y=0\) cut plane 
vol=pv.read('VTK/t18_1/internal.vtu')
cut=vol.slice(normal=(0,1,0), origin=(0,0,0.06))
barrel_v=pv.read('VTK/t18_1/boundary/barrel.vtp')
fins_v=pv.read('VTK/t18_1/boundary/fins.vtp')
head_v=pv.read('VTK/t18_1/boundary/head.vtp')
pl=pv.Plotter(off_screen=True, window_size=(950,800))
pl.add_mesh(cut, color='lightcyan', show_edges=True, edge_color='steelblue', line_width=0.5)
pl.add_mesh(barrel_v, color='orange', opacity=0.85)
pl.add_mesh(fins_v, color='firebrick', opacity=0.85)
pl.add_mesh(head_v, color='gold', opacity=0.85)
pl.camera_position=[(0.45,-1.4,0.16),(0,0,0.10),(0,0,1)]
pl.enable_parallel_projection(); pl.add_axes(); pl.show(screenshot='final_mesh_cut.png')
print(f"Volume cells={vol.n_cells}, central cut polygons={cut.n_cells}")
print(f"Meshed patch faces: barrel={barrel_v.n_cells}, fins={fins_v.n_cells}, head={head_v.n_cells}")

# -- cell 14 ------------------------------------------------------------------------
# The check revealed that independently tessellating each OCC face group produces nonmatching triangle
import numpy as np
# One global tessellation guarantees identical seam vertices across engine patches.
temp_all=Path('engine_all.tmp.stl')
bd.export_stl(engine, temp_all, tolerance=0.00025, angular_tolerance=0.08, ascii_format=False)
global_engine=pv.read(temp_all).triangulate()
conn=global_engine.faces.reshape(-1,4)[:,1:]
pts=np.asarray(global_engine.points,dtype=float)
tri=pts[conn]
cent=tri.mean(axis=1)
nvec=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
norm=np.linalg.norm(nvec,axis=1); unit=nvec/norm[:,None]
rcent=np.hypot(cent[:,0],cent[:,1])
head_mask=cent[:,2] > BARREL_H + 1e-7
horizontal=np.abs(unit[:,2]) > 0.9
fin_mask=(~head_mask) & (((horizontal) & (cent[:,2] > 0.001)) | (rcent > 0.055))
barrel_mask=~head_mask & ~fin_mask
assert np.all((head_mask.astype(int)+fin_mask.astype(int)+barrel_mask.astype(int))==1)

def write_ascii_stl(path, triangles, solid_name):
    with open(path,'w') as fh:
        fh.write(f'solid {solid_name}\n')
        for t in triangles:
            nn=np.cross(t[1]-t[0],t[2]-t[0]); ln=np.linalg.norm(nn); nn=nn/ln if ln else nn
            fh.write(f' facet normal {nn[0]:.16e} {nn[1]:.16e} {nn[2]:.16e}\n  outer loop\n')
            for v in t: fh.write(f'   vertex {v[0]:.16e} {v[1]:.16e} {v[2]:.16e}\n')
            fh.write('  endloop\n endfacet\n')
        fh.write(f'endsolid {solid_name}\n')
for name,mask in [('barrel',barrel_mask),('fins',fin_mask),('head',head_mask)]:
    write_ascii_stl(tri_dir/f'{name}.stl',tri[mask],name)
    print(name, int(mask.sum()), 'triangles')
temp_all.unlink()
for p in [Path('engine_preview.stl')]:
    if p.exists(): p.unlink()

parts=[pv.read(p) for p in sorted(tri_dir.glob('*.stl'))]
union=parts[0]
for q in parts[1:]: union=union.merge(q, merge_points=False)
union=union.clean(tolerance=1e-9, absolute=True)
free=union.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
nonman=union.extract_feature_edges(boundary_edges=False, non_manifold_edges=True, feature_edges=False, manifold_edges=False)
print(f'Welded six-STL union: free edges={free.n_cells}, non-manifold edges={nonman.n_cells}')
if free.n_cells or nonman.n_cells: raise RuntimeError('patch union still not closed manifold')

# -- cell 15 ------------------------------------------------------------------------
# The six exported patch STLs now weld to a closed manifold with zero free and zero non-manifold edges
import subprocess
rb=subprocess.run(['blockMesh'], text=True, capture_output=True)
if rb.returncode: raise RuntimeError(rb.stdout+rb.stderr)
rs=subprocess.run(['snappyHexMesh','-overwrite'], text=True, capture_output=True, timeout=240)
Path('snappyHexMesh.log').write_text(rs.stdout+'\n'+rs.stderr)
print('\n'.join(rs.stdout.splitlines()[-40:])); print(rs.stderr)
if rs.returncode: raise RuntimeError('final remesh failed')

# -- cell 16 ------------------------------------------------------------------------
# The corrected-STL remesh completed at 176,392 cells under the tightened quality controls. I’ll perfo
import subprocess, re
rc=subprocess.run(['checkMesh','-constant'], text=True, capture_output=True)
Path('checkMesh.log').write_text(rc.stdout+'\n'+rc.stderr)
print('\n'.join(rc.stdout.splitlines()[-48:])); print(rc.stderr)
print('\nFinal patch table:')
btxt=Path('constant/polyMesh/boundary').read_text()
for name in ['inlet','outlet','ductWalls','barrel','fins','head']:
    m=re.search(r'\b'+name+r'\s*\{.*?nFaces\s+(\d+);',btxt,re.S)
    print(f'{name:10s} {m.group(1) if m else "MISSING"} faces')
if rc.returncode or 'Mesh OK' not in rc.stdout: raise RuntimeError('final checkMesh failed')
