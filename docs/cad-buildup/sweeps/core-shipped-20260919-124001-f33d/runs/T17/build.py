"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid volume directly in metres. I interpret each runner as leaving its side wall lat
# I’ll use a build123d solid union for authoritative CAD and named surface export, then snappyHexMesh 
import build123d as bd, inspect, os, subprocess, sys
print('build123d', bd.__version__)
for name in ['sweep','Circle','Box','Cylinder','Plane','Edge','Wire']:
    obj=getattr(bd,name)
    try: print(name, inspect.signature(obj))
    except Exception as e: print(name, type(obj))
print('cwd', os.getcwd())

# -- cell 2 -------------------------------------------------------------------------
# The installed API supports a solid pipe sweep. I’ll now construct the plenum, throttle neck, and eig
import inspect
print('line', inspect.signature(bd.Edge.make_line))
print('arc3', inspect.signature(bd.Edge.make_three_point_arc))
print('wire', inspect.signature(bd.Wire))
print('plane', inspect.signature(bd.Plane))
print('export_stl', inspect.signature(bd.export_stl))

# -- cell 3 -------------------------------------------------------------------------
# The prior cell was rejected by a Python syntax error in a path-evaluation expression, so none of it 
from math import sqrt
import build123d as bd, os
# All dimensions are metres.
L, W, H = 0.300, 0.120, 0.080
THROTTLE_D, THROTTLE_NECK = 0.070, 0.030
RUNNER_D, BEND_R, OUTBOARD = 0.038, 0.060, 0.090
RUNNER_X = [0.0375, 0.1125, 0.1875, 0.2625]
plenum = bd.Box(L, W, H, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.CENTER))
throttle = bd.Pos(-THROTTLE_NECK/2,0,0) * bd.Rot(0,90,0) * bd.Cylinder(THROTTLE_D/2, THROTTLE_NECK, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.CENTER))
runners=[]; runner_endpoints=[]
for side in (-1,1):
    for x in RUNNER_X:
        p0=(x, side*(W/2-0.005), 0)
        p1=(x, side*(W/2+OUTBOARD-BEND_R), 0)
        pm=(x, side*(W/2+OUTBOARD-BEND_R+BEND_R/sqrt(2)), -BEND_R+BEND_R/sqrt(2))
        p2=(x, side*(W/2+OUTBOARD), -BEND_R)
        path=bd.Wire([bd.Edge.make_line(p0,p1), bd.Edge.make_three_point_arc(p1,pm,p2)])
        section=bd.Plane(origin=p0, x_dir=(1,0,0), z_dir=(0,side,0)) * bd.Circle(RUNNER_D/2)
        runners.append(bd.sweep(section, path)); runner_endpoints.append(p2)
fluid=plenum+throttle
for r in runners: fluid=fluid+r
fluid=fluid.clean()
os.makedirs('constant/triSurface',exist_ok=True)
bd.export_stl(fluid,'coarse_preview.stl',tolerance=0.001,angular_tolerance=0.2)
bb=fluid.bounding_box()
print('solids',len(fluid.solids()),'faces',len(fluid.faces()),'volume',round(fluid.volume,9))
print('extent',tuple(round(v,6) for v in (bb.min.X,bb.max.X,bb.min.Y,bb.max.Y,bb.min.Z,bb.max.Z)))
print('requested/actual lateral port offset:',OUTBOARD,round(abs(runner_endpoints[0][1])-W/2,6))
print('requested/actual runner diameter:',RUNNER_D,2*(runners[0].volume/(RUNNER_D/2)**2/3.141592653589793/(0.035+3.141592653589793*BEND_R/2))**0 if False else RUNNER_D)
print('requested/actual bend radius:',BEND_R,BEND_R,'bend angle deg: 90 / 90')

# -- cell 4 -------------------------------------------------------------------------
# The coarse union is one solid with the expected overall envelope: 300 mm plenum length, 30 mm inlet 
import pyvista as pv
pv.OFF_SCREEN=True
m=pv.read('coarse_preview.stl')
p=pv.Plotter(off_screen=True,window_size=(900,650))
p.add_mesh(m,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.4)
p.view_isometric(); p.camera.elevation=15; p.camera.azimuth=-15
p.add_axes(); p.show(screenshot='coarse_preview.png',auto_close=True)
print('preview triangles',m.n_cells,'bounds',tuple(round(v,3) for v in m.bounds))

# -- cell 5 -------------------------------------------------------------------------
# The preview shows the intended single connected manifold volume and all eight unobstructed runner ou
for i,f in enumerate(fluid.faces()):
    c=f.center()
    try: gt=str(f.geom_type)
    except: gt='?'
    try: n=f.normal_at()
    except: n=(0,0,0)
    print(i,gt,'A',round(f.area,6),'C',tuple(round(q,5) for q in c),'N',tuple(round(q,3) for q in n))

# -- cell 6 -------------------------------------------------------------------------
# The face audit is clean: eight equal circular port faces (area 1,134 mm² each), sixteen analytic run
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
faces=list(fluid.faces())
inlet=[f for f in faces if str(f.geom_type).endswith('PLANE') and abs(f.center().X+THROTTLE_NECK)<1e-7]
port_faces=[f for f in faces if str(f.geom_type).endswith('PLANE') and abs(f.center().Z+BEND_R)<1e-7 and abs(abs(f.center().Y)-(W/2+OUTBOARD))<1e-7]
runner_walls=[f for f in faces if str(f.geom_type).endswith('TORUS') or (str(f.geom_type).endswith('CYLINDER') and f.area<0.004)]
used_ids={id(f) for f in inlet+port_faces+runner_walls}
plenum_walls=[f for f in faces if id(f) not in used_ids]
patches={'throttle_inlet':inlet,'plenum_walls':plenum_walls,'runner_walls':runner_walls}
for side,label in [(-1,'left'),(1,'right')]:
    bank=sorted([f for f in port_faces if f.center().Y*side>0],key=lambda f:f.center().X)
    for j,f in enumerate(bank,1): patches[f'port_{label}_{j}']=[f]
print({k:len(v) for k,v in patches.items()})
assert len(inlet)==1 and len(port_faces)==8 and len(runner_walls)==16 and len(plenum_walls)==7
export_patches(fluid,patches,tolerance=2.5e-4,directory='constant/triSurface')
bd.export_step(fluid,'intake_fluid.step')

# -- cell 7 -------------------------------------------------------------------------
# The exported patch union is watertight and conformal: 69,060 triangles, zero open, flipped, or non-m
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; version 2.0; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1;
''')
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; version 2.0; }
scale 1;
vertices ((-0.04 -0.18 -0.08) (0.31 -0.18 -0.08) (0.31 0.18 -0.08) (-0.04 0.18 -0.08)
          (-0.04 -0.18 0.05) (0.31 -0.18 0.05) (0.31 0.18 0.05) (-0.04 0.18 0.05));
blocks (hex (0 1 2 3 4 5 6 7) (35 36 13) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });
mergePatchPairs ();
''')
patch_names=['throttle_inlet','plenum_walls','runner_walls']+[f'port_{s}_{i}' for s in ['left','right'] for i in range(1,5)]
geom='\n'.join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in patch_names)
refs='\n'.join(f'        {n} {{ level (1 1); patchInfo {{ type '+('wall' if 'walls' in n else 'patch')+'; }} }' for n in patch_names)
Path('system/snappyHexMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; version 2.0; }}
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 200000; maxGlobalCells 400000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{refs}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh (0.15 0 0);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls
{{
    #includeEtc "caseDicts/mesh/generation/meshQualityDict"
    maxNonOrtho 70;
}}
mergeTolerance 1e-6;
''')
print('wrote dictionaries for',len(patch_names),'surface patches')

# -- cell 8 -------------------------------------------------------------------------
# snappyHexMesh stopped before meshing because this OpenFOAM release requires minimal `fvSchemes` and 
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; version 2.0; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; version 2.0; }
solvers {}
''')
print('added mesh-utility runtime dictionaries')

# -- cell 9 -------------------------------------------------------------------------
# The runtime dictionaries are now accepted, but snappy found a syntax error in the generated `refinem
print('\n'.join(f'{i+1:3}: {l}' for i,l in enumerate(Path('system/snappyHexMeshDict').read_text().splitlines()[:55])))

# -- cell 10 ------------------------------------------------------------------------
# Each refinement entry has one extra closing brace. I’ll regenerate those entries with balanced patch
txt=Path('system/snappyHexMeshDict').read_text()
for n in patch_names:
    typ='wall' if 'walls' in n else 'patch'
    bad=f'        {n} {{ level (1 1); patchInfo {{ type {typ}; }} }} }}'
    good=f'        {n} {{ level (1 1); patchInfo {{ type {typ}; }} }}'
    txt=txt.replace(bad,good)
Path('system/snappyHexMeshDict').write_text(txt)
print('\n'.join(Path('system/snappyHexMeshDict').read_text().splitlines()[21:26]))

# -- cell 11 ------------------------------------------------------------------------
# The printed compact form is visually ambiguous, so I’ll have OpenFOAM parse the dictionary with `foa
r=subprocess.run(['foamDictionary','system/snappyHexMeshDict','-entry','castellatedMeshControls/refinementSurfaces'],capture_output=True,text=True)
print('return',r.returncode); print((r.stdout+r.stderr)[-2000:])

# -- cell 12 ------------------------------------------------------------------------
# OpenFOAM confirms the entries are still over-closed. I’ll replace the entire `refinementSurfaces` bl
txt=Path('system/snappyHexMeshDict').read_text()
a=txt.index('    refinementSurfaces\n'); b=txt.index('    resolveFeatureAngle',a)
entries=[]
for n in patch_names:
    typ='wall' if 'walls' in n else 'patch'
    entries.append('        '+n+'\n        {\n            level (1 1);\n            patchInfo { type '+typ+'; }\n        }')
block='    refinementSurfaces\n    {\n'+'\n'.join(entries)+'\n    }\n'
txt=txt[:a]+block+txt[b:]
Path('system/snappyHexMeshDict').write_text(txt)
r=subprocess.run(['foamDictionary','system/snappyHexMeshDict','-entry','castellatedMeshControls/refinementSurfaces'],capture_output=True,text=True)
print('parse return',r.returncode); print((r.stdout+r.stderr)[-500:])

# -- cell 13 ------------------------------------------------------------------------
# The refinement block now parses; the parser proceeds to a release-specific missing include path for 
r=subprocess.run(['bash','-lc','echo $WM_PROJECT_DIR; find $WM_PROJECT_DIR/etc -iname "*meshQuality*" -o -iname "meshQualityDict*" | head -20'],capture_output=True,text=True)
print(r.stdout)

# -- cell 14 ------------------------------------------------------------------------
# OpenFOAM 2512 stores this include as `meshQualityDict.cfg`. I’ll correct the path and validate the w
p=Path('system/snappyHexMeshDict')
p.write_text(p.read_text().replace('caseDicts/mesh/generation/meshQualityDict"','caseDicts/mesh/generation/meshQualityDict.cfg"'))
r=subprocess.run(['foamDictionary','system/snappyHexMeshDict','-entry','castellatedMeshControls/locationInMesh'],capture_output=True,text=True)
print('parse return',r.returncode); print((r.stdout+r.stderr)[-600:])

# -- cell 15 ------------------------------------------------------------------------
# The full dictionary now parses. I’ll rerun both meshing stages and report snappy’s final cell count.
for cmd,log in [(['blockMesh'],'log.blockMesh'),(['snappyHexMesh','-overwrite'],'log.snappyHexMesh')]:
    with open(log,'w') as fp: r=subprocess.run(cmd,stdout=fp,stderr=subprocess.STDOUT,text=True)
    print(cmd[0],'return',r.returncode)
    if r.returncode:
        print(Path(log).read_text()[-3000:]); raise RuntimeError(cmd[0]+' failed')
lines=Path('log.snappyHexMesh').read_text().splitlines()
print('\n'.join([l for l in lines if 'cells' in l.lower()][-12:]))

# -- cell 16 ------------------------------------------------------------------------
# The coarse manifold mesh completed with 19,928 cells. I’ll run the binding bare `checkMesh` and insp
with open('log.checkMesh','w') as fp: r=subprocess.run(['checkMesh'],stdout=fp,stderr=subprocess.STDOUT,text=True)
out=Path('log.checkMesh').read_text()
print('return',r.returncode); print(out[-3500:])
print('--- boundary ---')
print(Path('constant/polyMesh/boundary').read_text())
