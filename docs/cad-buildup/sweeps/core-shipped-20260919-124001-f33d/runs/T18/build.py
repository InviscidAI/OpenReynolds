"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The installed build123d does not expose a module-level `fuse`; I’ll use algebraic `+` unions instead
import build123d as bd
import numpy as np, os, subprocess, textwrap, json, math
# All dimensions are metres.
BARREL_R = 0.045
BARREL_H = 0.120
FIN_R = 0.070
FIN_T = 0.003
FIN_GAP = 0.006
N_FINS = 12
FIN_SPAN = N_FINS*FIN_T + (N_FINS-1)*FIN_GAP
FIN_Z0 = (BARREL_H-FIN_SPAN)/2
HEAD_R = BARREL_R
DUCT_HALF = 0.150
Z_IN = -0.150
Z_HEAD_TOP = BARREL_H + HEAD_R
Z_OUT = Z_HEAD_TOP + 0.300
barrel = bd.Cylinder(BARREL_R, BARREL_H)
fins = [bd.Pos(0,0,FIN_Z0+i*(FIN_T+FIN_GAP))*bd.Cylinder(FIN_R, FIN_T) for i in range(N_FINS)]
fin_solid = bd.Compound(fins)
head_sphere = bd.Pos(0,0,BARREL_H)*bd.Sphere(HEAD_R)
head_clip = bd.Pos(-HEAD_R,-HEAD_R,BARREL_H)*bd.Box(2*HEAD_R,2*HEAD_R,HEAD_R)
head = head_sphere & head_clip
engine = barrel + head
for fin_i in fins:
    engine = engine + fin_i
duct = bd.Pos(-DUCT_HALF,-DUCT_HALF,Z_IN)*bd.Box(2*DUCT_HALF,2*DUCT_HALF,Z_OUT-Z_IN)
fluid = duct - engine
print(f'Fin count requested/modelled: {N_FINS}/{len(fins)}')
print(f'Fin OD requested/modelled: 0.140/{2*FIN_R:.3f} m')
print(f'Fin thickness requested/modelled: 0.003/{FIN_T:.3f} m')
print(f'Neighbour gap requested/modelled: 0.006/{FIN_GAP:.3f} m')
print(f'Barrel OD x height requested/modelled: 0.090 x 0.120/{2*BARREL_R:.3f} x {BARREL_H:.3f} m')
print(f'Duct cross-section requested/modelled: 0.300 x 0.300/{2*DUCT_HALF:.3f} x {2*DUCT_HALF:.3f} m')
print(f'z extent: [{Z_IN:.3f},{Z_OUT:.3f}] m; dome apex {Z_HEAD_TOP:.3f} m')
print('Fluid volume:', round(fluid.volume,9), 'm^3; engine volume:', round(engine.volume,9), 'm^3; fluid faces:', len(fluid.faces()))

# -- cell 2 -------------------------------------------------------------------------
# The numeric dimensions are consistent. I’ll now tessellate only a temporary engine preview and rende
bd.export_stl(engine, 'engine_preview.stl', tolerance=2.5e-4, angular_tolerance=0.15)
import pyvista as pv
pv.OFF_SCREEN = True
surf = pv.read('engine_preview.stl')
p = pv.Plotter(window_size=(850,650), off_screen=True)
p.add_mesh(surf, color='#b8b8b8', smooth_shading=True, show_edges=True, edge_color='#555555')
p.add_axes()
p.view_isometric()
p.camera.zoom(1.35)
p.show(screenshot='engine_preview.png')
print('Preview triangles:', surf.n_cells, 'bounds:', tuple(round(v,4) for v in surf.bounds))

# -- cell 3 -------------------------------------------------------------------------
# The preview exposed an error: `Box` is center-aligned by default, so the dome clipping slab ended at
import inspect
print(inspect.signature(bd.Box))
head_clip = bd.Pos(0,0,BARREL_H)*bd.Box(2*HEAD_R,2*HEAD_R,HEAD_R, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
head = head_sphere & head_clip
engine = barrel + head
for fin_i in fins:
    engine = engine + fin_i
duct = bd.Pos(0,0,Z_IN)*bd.Box(2*DUCT_HALF,2*DUCT_HALF,Z_OUT-Z_IN, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
fluid = duct - engine
ebb = engine.bounding_box(); dbb = duct.bounding_box()
print('Engine bounds:', ebb.min, ebb.max)
print('Duct bounds:', dbb.min, dbb.max)
print(f'Dome height requested assumption/measured: {HEAD_R:.3f}/{ebb.max.Z-BARREL_H:.3f} m')
print('Fluid faces:',len(fluid.faces()),'volume:',round(fluid.volume,9))

# -- cell 4 -------------------------------------------------------------------------
# The same check shows cylinders are also center-aligned by default: the barrel currently spans −60 to
barrel = bd.Cylinder(BARREL_R, BARREL_H, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
fins = [bd.Pos(0,0,FIN_Z0+i*(FIN_T+FIN_GAP))*bd.Cylinder(FIN_R, FIN_T, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)) for i in range(N_FINS)]
fin_solid = bd.Compound(fins)
head_sphere = bd.Pos(0,0,BARREL_H)*bd.Sphere(HEAD_R)
head_clip = bd.Pos(0,0,BARREL_H)*bd.Box(2*HEAD_R,2*HEAD_R,HEAD_R, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
head = head_sphere & head_clip
engine = barrel + head
for fin_i in fins:
    engine = engine + fin_i
duct = bd.Pos(0,0,Z_IN)*bd.Box(2*DUCT_HALF,2*DUCT_HALF,Z_OUT-Z_IN, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
fluid = duct - engine
ebb=engine.bounding_box(); dbb=duct.bounding_box()
fin_bounds=[(f.bounding_box().min.Z,f.bounding_box().max.Z) for f in fins]
print('Barrel z requested/measured: [0,0.120]/',barrel.bounding_box().min.Z,barrel.bounding_box().max.Z)
print('First/last fin z:',tuple(round(x,4) for x in fin_bounds[0]),tuple(round(x,4) for x in fin_bounds[-1]))
print('Fin stack requested/measured span:',FIN_SPAN,round(fin_bounds[-1][1]-fin_bounds[0][0],6))
print('Engine bounds:',ebb.min,ebb.max)
print('Duct bounds:',dbb.min,dbb.max)
print('Fluid volume:',round(fluid.volume,9),'faces:',len(fluid.faces()))

# -- cell 5 -------------------------------------------------------------------------
# The corrected extents now match the intended geometry. I’ll regenerate the preview and include the d
bd.export_stl(engine, 'engine_preview.stl', tolerance=2.5e-4, angular_tolerance=0.15)
bd.export_stl(duct, 'duct_preview.stl', tolerance=5e-4, angular_tolerance=0.2)
eng_mesh=pv.read('engine_preview.stl'); duct_mesh=pv.read('duct_preview.stl')
p=pv.Plotter(window_size=(800,700),off_screen=True)
p.add_mesh(eng_mesh,color='#c0a070',smooth_shading=True)
p.add_mesh(duct_mesh,color='#6aaed6',style='wireframe',opacity=0.25,line_width=2)
p.add_axes(); p.view_isometric(); p.camera.zoom(1.15)
p.show(screenshot='geometry_preview.png')
print('Engine preview bounds:',tuple(round(v,3) for v in eng_mesh.bounds))
print('Duct preview bounds:',tuple(round(v,3) for v in duct_mesh.bounds))

# -- cell 6 -------------------------------------------------------------------------
# The placement preview is correct. I’ll inspect the fluid boundary faces by analytic surface type, ar
from collections import Counter, defaultdict
rows=[]
for i,f in enumerate(fluid.faces()):
    c=f.center(); gt=str(f.geom_type); n=f.normal_at(c)
    rows.append((i,gt,round(f.area,7),(round(c.X,4),round(c.Y,4),round(c.Z,4)),(round(n.X,2),round(n.Y,2),round(n.Z,2))))
print(Counter(r[1] for r in rows))
for r in rows:
    print(r)

# -- cell 7 -------------------------------------------------------------------------
# The analytic inventory is unambiguous: 24 annular fin faces plus 12 outer fin cylinders; 13 exposed 
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
all_faces=list(fluid.faces())
inlet=[]; outlet=[]; duct_walls=[]; barrel_faces=[]; fin_faces=[]; head_faces=[]
for f in all_faces:
    c=f.center(); gt=f.geom_type
    if gt==bd.GeomType.SPHERE:
        head_faces.append(f)
    elif gt==bd.GeomType.PLANE and f.area>0.05:
        n=f.normal_at(c)
        if n.Z < -0.9: inlet.append(f)
        elif n.Z > 0.9: outlet.append(f)
        else: duct_walls.append(f)
    elif gt==bd.GeomType.PLANE:
        if f.area>0.008: fin_faces.append(f)
        else: barrel_faces.append(f)
    elif gt==bd.GeomType.CYLINDER:
        if abs(c.X) > 0.06 or abs(c.Y) > 0.06: fin_faces.append(f)
        else: barrel_faces.append(f)
patches={'inlet':inlet,'outlet':outlet,'duct_walls':duct_walls,'barrel':barrel_faces,'fins':fin_faces,'head':head_faces}
assigned=[f for fs in patches.values() for f in fs]
assert len(assigned)==len(all_faces)==len({id(f.wrapped) for f in assigned})
print({k:(len(v),round(sum(f.area for f in v),7)) for k,v in patches.items()})
os.makedirs('constant/triSurface',exist_ok=True)
export_patches(fluid,patches,tolerance=2.5e-4,angular_tolerance=0.15,directory='constant/triSurface')

# -- cell 8 -------------------------------------------------------------------------
# The exported union is closed and consistently wound. For the coarse mesh I’ll use a 15 mm background
os.makedirs('system',exist_ok=True)
os.makedirs('constant',exist_ok=True)
open('system/controlDict','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
open('system/blockMeshDict','w').write(f'''FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices ((-{DUCT_HALF} -{DUCT_HALF} {Z_IN}) ({DUCT_HALF} -{DUCT_HALF} {Z_IN}) ({DUCT_HALF} {DUCT_HALF} {Z_IN}) (-{DUCT_HALF} {DUCT_HALF} {Z_IN}) (-{DUCT_HALF} -{DUCT_HALF} {Z_OUT}) ({DUCT_HALF} -{DUCT_HALF} {Z_OUT}) ({DUCT_HALF} {DUCT_HALF} {Z_OUT}) (-{DUCT_HALF} {DUCT_HALF} {Z_OUT}));
blocks (hex (0 1 2 3 4 5 6 7) (20 20 41) simpleGrading (1 1 1));
edges ();
boundary (
 inlet {{ type patch; faces ((0 3 2 1)); }}
 outlet {{ type patch; faces ((4 5 6 7)); }}
 duct_walls {{ type wall; faces ((0 1 5 4)(1 2 6 5)(2 3 7 6)(3 0 4 7)); }}
);
mergePatchPairs ();
''')
open('system/surfaceFeatureExtractDict','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
barrel.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }
fins.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }
head.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }
''')
open('system/snappyHexMeshDict','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry {
 barrel.stl { type triSurfaceMesh; name barrel; }
 fins.stl { type triSurfaceMesh; name fins; }
 head.stl { type triSurfaceMesh; name head; }
 engineBox { type searchableBox; min (-0.075 -0.075 -0.005); max (0.075 0.075 0.170); }
}
castellatedMeshControls {
 maxLocalCells 500000; maxGlobalCells 1000000; minRefinementCells 0; nCellsBetweenLevels 2;
 features ({ file "barrel.eMesh"; level 2; } { file "fins.eMesh"; level 3; } { file "head.eMesh"; level 2; });
 refinementSurfaces {
  barrel { level (2 2); patchInfo { type wall; } }
  fins { level (3 3); patchInfo { type wall; } }
  head { level (2 2); patchInfo { type wall; } }
 }
 resolveFeatureAngle 30;
 refinementRegions { engineBox { mode inside; levels ((1e15 2)); } }
 locationInMesh (0.10 0 0.30); allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #include "meshQualityDict" }
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
''')
open('system/meshQualityDict','w').write('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }''')
print('Wrote OpenFOAM dictionaries; background cells:',20*20*41)

# -- cell 9 -------------------------------------------------------------------------
# The dictionaries are ready. I’ll generate the background mesh and edge features first, then inspect 
r1=subprocess.run(['blockMesh'],capture_output=True,text=True)
print('blockMesh rc',r1.returncode); print('\n'.join((r1.stdout+r1.stderr).splitlines()[-18:]))
r2=subprocess.run(['surfaceFeatureExtract'],capture_output=True,text=True)
print('surfaceFeatureExtract rc',r2.returncode); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-18:]))
assert r1.returncode==0 and r2.returncode==0

# -- cell 10 ------------------------------------------------------------------------
# snappyHexMesh requires minimal `fvSchemes` and `fvSolution` files even though no flow solution is be
open('system/fvSchemes','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
open('system/fvSolution','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { }
''')
print('Added minimal meshing support dictionaries')

# -- cell 11 ------------------------------------------------------------------------
# The required support dictionaries are now present; I’ll rerun snappyHexMesh and capture its complete
sn=subprocess.run(['snappyHexMesh','-overwrite'],capture_output=True,text=True)
open('log.snappyHexMesh','w').write(sn.stdout+sn.stderr)
print('snappy rc',sn.returncode)
print('\n'.join((sn.stdout+sn.stderr).splitlines()[-40:]))
assert sn.returncode==0

# -- cell 12 ------------------------------------------------------------------------
# The coarse mesh completed with 195,756 cells and no snappy quality violations. I’ll run the binding 
cm=subprocess.run(['checkMesh'],capture_output=True,text=True)
open('log.checkMesh','w').write(cm.stdout+cm.stderr)
print('checkMesh rc',cm.returncode)
print('\n'.join((cm.stdout+cm.stderr).splitlines()[-45:]))
print('\nBoundary file patch summary:')
btxt=open('constant/polyMesh/boundary').read()
for name in ['barrel','fins','head','duct_walls','inlet','outlet']:
    pos=btxt.find(name)
    print(name, btxt[pos:btxt.find('}',pos)+1].replace('\n',' ') if pos>=0 else 'MISSING')
assert cm.returncode==0 and 'Mesh OK' in cm.stdout

# -- cell 13 ------------------------------------------------------------------------
# `checkMesh` passes and all six requested boundary names are populated. I’ll make a field-free VTK co
vtk=subprocess.run(['foamToVTK','-constant','-noFields'],capture_output=True,text=True)
print('foamToVTK rc',vtk.returncode); print('\n'.join((vtk.stdout+vtk.stderr).splitlines()[-12:]))
for root,dirs,files in os.walk('VTK'):
    for fn in files:
        if fn.endswith(('.vtu','.vtk','.vtp')): print(os.path.join(root,fn))

# -- cell 14 ------------------------------------------------------------------------
# This OpenFOAM release uses a different field-suppression option. I’ll query the local utility help a
h=subprocess.run(['foamToVTK','-help'],capture_output=True,text=True).stdout
print('\n'.join(line for line in h.splitlines() if 'field' in line.lower() or 'constant' in line.lower()))

# -- cell 15 ------------------------------------------------------------------------
# I’ll convert using the supported `-no-fields` option and list the generated datasets before renderin
vtk=subprocess.run(['foamToVTK','-constant','-no-fields'],capture_output=True,text=True)
print('foamToVTK rc',vtk.returncode); print('\n'.join((vtk.stdout+vtk.stderr).splitlines()[-12:]))
vtk_files=[]
for root,dirs,files in os.walk('VTK'):
    for fn in files:
        if fn.endswith(('.vtu','.vtk','.vtp')): vtk_files.append(os.path.join(root,fn))
print('\n'.join(vtk_files[:20])); assert vtk.returncode==0

# -- cell 16 ------------------------------------------------------------------------
# The final mesh is the `t18_1` dataset. I’ll render its `x=0` slice with cell edges and overlay the n
vol=pv.read('VTK/t18_1/internal.vtu')
sl=vol.slice(normal=(1,0,0),origin=(0,0,0.06))
p=pv.Plotter(window_size=(900,700),off_screen=True)
p.add_mesh(sl,color='white',show_edges=True,edge_color='#4676a9',line_width=0.5)
for nm,col in [('barrel','#555555'),('fins','#d28b26'),('head','#9a6b3a')]:
    m=pv.read(f'VTK/t18_1/boundary/{nm}.vtp')
    p.add_mesh(m.slice(normal=(1,0,0),origin=(0,0,0.06)),color=col,line_width=4)
p.view_yz(); p.camera.zoom(1.45); p.add_axes()
p.show(screenshot='mesh_central_slice.png')
print('Slice cells:',sl.n_cells,'slice bounds:',tuple(round(v,4) for v in sl.bounds))

# -- cell 17 ------------------------------------------------------------------------
# The central slice shows a continuous external-air region, a clean excluded engine volume, and resolv
import shutil
for path in ['VTK','engine_preview.stl','duct_preview.stl']:
    if os.path.isdir(path): shutil.rmtree(path)
    elif os.path.exists(path): os.remove(path)
print('Remaining triSurface files:',sorted(os.listdir('constant/triSurface')))
print('Mesh points/faces/cells from checkMesh log:')
for line in open('log.checkMesh'):
    if any(k in line for k in ['points:','faces:','cells:']): print(line.strip())

# -- cell 18 ------------------------------------------------------------------------
# The completion gate found that the export manifest did not record the fluid-side point, although sna
import inspect
print(inspect.signature(export_patches))
print(export_patches.__doc__)

# -- cell 19 ------------------------------------------------------------------------
# I’ll write the fluid-side point into the manifest by rerunning the same one-call patch export. The p
mesh_point=(0.10,0.0,0.30)
export_patches(fluid,patches,tolerance=2.5e-4,angular_tolerance=0.15,directory='constant/triSurface',location_in_mesh=mesh_point)
manifest=json.load(open('constant/triSurface/patches.json'))
print('Manifest location_in_mesh:',manifest['location_in_mesh'])
print('snappy location line:',next(line.strip() for line in open('system/snappyHexMeshDict') if 'locationInMesh' in line))
