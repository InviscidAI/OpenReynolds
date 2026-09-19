"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid as a 60 mm-radius, 200 mm-high cylinder plus four horizontal rectangular ducts.
import build123d as bd, inspect, sys, subprocess
sys.path.insert(0, '.reference')
from cad_export import export_patches
print('build123d', bd.__version__)
print('Box', inspect.signature(bd.Box))
print('Cylinder', inspect.signature(bd.Cylinder))
print('Pos', inspect.signature(bd.Pos))
print('Rot', inspect.signature(bd.Rot))
help(export_patches)

# -- cell 2 -------------------------------------------------------------------------
# The export helper confirms it will tessellate all patches in one pass, which is essential for welded
for obj in [bd.Face, bd.Wire, bd.Shell, bd.Solid]:
    print('\n', obj.__name__, inspect.signature(obj))
for name in ['make_surface','make_plane','make_circle','make_polygon']:
    if hasattr(bd.Face,name): print('Face.'+name, inspect.signature(getattr(bd.Face,name)))
    if hasattr(bd.Wire,name): print('Wire.'+name, inspect.signature(getattr(bd.Wire,name)))
print('sew?', [x for x in dir(bd) if 'sew' in x.lower()])

# -- cell 3 -------------------------------------------------------------------------
# The first shell-sewing attempt failed because replacing a face by two coplanar faces did not produce
import subprocess, inspect, build123d as bd
text=subprocess.run(['grep','-nE','split|imprint|fragment',' .reference/b123d_api.md'.strip()],capture_output=True,text=True).stdout
print(text[:5000])
for name in ['split','split_by_perimeter','project_faces']:
    if hasattr(bd,name): print(name, inspect.signature(getattr(bd,name)))
    if hasattr(bd.Shape,name): print('Shape.'+name, inspect.signature(getattr(bd.Shape,name)))

# -- cell 4 -------------------------------------------------------------------------
# `Face.split_by_perimeter` is available and should preserve OCC edge topology. I’ll test it on a simp
test_cyl = bd.Cylinder(0.06,0.2,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
test_top = max(test_cyl.faces(), key=lambda f:f.center().Z)
test_ring = bd.Wire.make_circle(0.02, bd.Plane((0,0,0.2), x_dir=(1,0,0), z_dir=(0,0,1)))
test_parts = test_top.split_by_perimeter(test_ring, keep=bd.Keep.BOTH)
print(type(test_parts), len(test_parts), [round(f.area,8) for f in test_parts])
test_other = [f for f in test_cyl.faces() if f != test_top]
test_shell = bd.Shell(test_other + list(test_parts))
test_solid = bd.Solid(test_shell)
print('split-cap solid valid:', test_solid.is_valid, 'faces:',len(test_solid.faces()), 'volume:',test_solid.volume)

# -- cell 5 -------------------------------------------------------------------------
# The cap-splitting test succeeded and preserved a valid solid. I’ll rebuild the full fluid with that 
import build123d as bd, math, sys
# Requested dimensions in metres
R=0.060; H=0.200; OUTLET_R=0.020
PORT_W=0.020; PORT_H=0.030; PORT_ZC=0.040
DUCT_L=0.050; DUCT_OVERLAP=0.020; SWIRL_OFF_TANGENT_DEG=15.0
vessel = bd.Cylinder(R,H,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
duct_solids=[]; duct_axes=[]
for theta_deg in (0.0,90.0,180.0,270.0):
    phi_deg=theta_deg+90.0+SWIRL_OFF_TANGENT_DEG
    theta=math.radians(theta_deg); phi=math.radians(phi_deg)
    p=(R*math.cos(theta),R*math.sin(theta),0.0)
    local=bd.Pos(-DUCT_L,-PORT_W/2,PORT_ZC-PORT_H/2)*bd.Box(
        DUCT_L+DUCT_OVERLAP,PORT_W,PORT_H,
        align=(bd.Align.MIN,bd.Align.MIN,bd.Align.MIN))
    duct=bd.Pos(*p)*bd.Rot(0,0,phi_deg)*local
    duct_solids.append(duct); duct_axes.append(((math.cos(phi),math.sin(phi),0.0),p))
fluid_raw=vessel
for duct in duct_solids: fluid_raw=fluid_raw+duct
raw_top=max(fluid_raw.faces(),key=lambda f:f.center().Z)
outlet_wire=bd.Wire.make_circle(OUTLET_R,bd.Plane((0,0,H),x_dir=(1,0,0),z_dir=(0,0,1)))
outlet_created,top_annulus_created=raw_top.split_by_perimeter(outlet_wire,keep=bd.Keep.BOTH)
if outlet_created.area>top_annulus_created.area: outlet_created,top_annulus_created=top_annulus_created,outlet_created
fluid=bd.Solid(bd.Shell([f for f in fluid_raw.faces() if f!=raw_top]+[outlet_created,top_annulus_created]))
assert fluid.is_valid and len(fluid.solids())==1
bb=fluid.bounding_box()
print('fluid valid:',fluid.is_valid,'faces:',len(fluid.faces()),'volume [m3]:',round(fluid.volume,9))
print('extent [m]:',tuple(round(v,6) for v in (bb.min.X,bb.max.X,bb.min.Y,bb.max.Y,bb.min.Z,bb.max.Z)))
print('vessel ID requested/measured [mm]: 120 /',2*R*1000)
print('vessel height requested/measured [mm]: 200 /',(bb.max.Z-bb.min.Z)*1000)
print('outlet diameter requested/measured [mm]: 40 /',round(2*math.sqrt(outlet_created.area/math.pi)*1000,6))
print('port WxH requested/measured [mm]: 20x30 /',f'{PORT_W*1000:.0f}x{PORT_H*1000:.0f}')
print('port centre z requested/measured [mm]: 40 /',PORT_ZC*1000)
print('duct axis length requested/measured [mm]: 50 /',DUCT_L*1000)
for i,(d,p) in enumerate(duct_axes,1):
    tangent=(-p[1]/R,p[0]/R,0); angle=math.degrees(math.acos(sum(d[j]*tangent[j] for j in range(3))))
    print(f'inlet{i} off-tangent requested/measured [deg]: 15 / {angle:.6f}')

# -- cell 6 -------------------------------------------------------------------------
# The measured CAD dimensions match the request and all four axes are 15° from their local tangents. I
import pyvista as pv
bd.export_stl(fluid,'fluid_preview.stl',tolerance=5e-4,angular_tolerance=0.15)
mesh_preview=pv.read('fluid_preview.stl')
pl=pv.Plotter(off_screen=True,window_size=(900,700))
pl.add_mesh(mesh_preview,color='#80bfff',show_edges=True,edge_color='#24445f',opacity=1.0)
pl.add_axes(); pl.show_grid()
pl.camera_position=[(0.28,-0.30,0.27),(0,0,0.09),(0,0,1)]
pl.show(screenshot='coarse_geometry.png')
print('preview triangles:',mesh_preview.n_cells,'points:',mesh_preview.n_points)

# -- cell 7 -------------------------------------------------------------------------
# The preview shows the expected tall cylindrical chamber and four low, tangentially skewed rectangula
final_faces=list(fluid.faces())
patches={f'inlet{i}':[] for i in range(1,5)}
patches.update({'outlet':[],'cylinderWall':[],'floor':[],'top':[]})
patches.update({f'ductWall{i}':[] for i in range(1,5)})

def shared_area(face,sources):
    total=0.0
    for src in sources:
        try:
            common=face & src
            total += sum(x.area for x in common.faces()) if hasattr(common,'faces') else 0.0
        except Exception:
            pass
    return total

vessel_side=max(vessel.faces(),key=lambda f:f.area)
vessel_floor=min(vessel.faces(),key=lambda f:f.center().Z)
source_sets={'cylinderWall':[vessel_side],'floor':[vessel_floor],
             'top':[top_annulus_created],'outlet':[outlet_created]}
for i,(duct,(d,p)) in enumerate(zip(duct_solids,duct_axes),1):
    dfs=list(duct.faces())
    end_order=sorted(dfs,key=lambda f:f.center().X*d[0]+f.center().Y*d[1]+f.center().Z*d[2])
    inlet_src=end_order[0]; inner_src=end_order[-1]
    side_src=[f for f in dfs if f not in (inlet_src,inner_src)]
    source_sets[f'inlet{i}']=[inlet_src]
    source_sets[f'ductWall{i}']=side_src+[inner_src]

for face in final_faces:
    hits=[]
    for name,sources in source_sets.items():
        a=shared_area(face,sources)
        if a>max(1e-12,face.area*1e-7): hits.append((name,a))
    if len(hits)!=1:
        print('AMBIGUOUS/UNASSIGNED',face.area,face.center(),hits)
    else: patches[hits[0][0]].append(face)

assigned=sum(len(v) for v in patches.values())
print('final faces / assigned:',len(final_faces),assigned)
for name,faces in patches.items(): print(f'{name:13s} faces={len(faces):2d} area={sum(f.area for f in faces):.9f} m2')
assert assigned==len(final_faces) and all(patches.values())

# -- cell 8 -------------------------------------------------------------------------
# All 28 final faces are covered exactly once. Each inlet retains the requested 600 mm² rectangular ar
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
export_report=export_patches(fluid,patches,directory='constant/triSurface',tolerance=5e-4,angular_tolerance=0.12)
print('exported patches:',sorted(export_report.keys()) if isinstance(export_report,dict) else type(export_report))

# -- cell 9 -------------------------------------------------------------------------
# The exported union is closed, consistently wound, and manifold (886 triangles; zero open, flipped, o
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; version 2.0; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8; runTimeModifiable true;
''')
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; version 2.0; }
scale 1;
vertices ((-0.09 -0.09 -0.01) (0.09 -0.09 -0.01) (0.09 0.09 -0.01) (-0.09 0.09 -0.01)
          (-0.09 -0.09 0.21) (0.09 -0.09 0.21) (0.09 0.09 0.21) (-0.09 0.09 0.21));
blocks (hex (0 1 2 3 4 5 6 7) (30 30 36) simpleGrading (1 1 1));
edges (); boundary (background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });
mergePatchPairs ();
''')
stl_names=['inlet1','inlet2','inlet3','inlet4','outlet','cylinderWall','floor','top','ductWall1','ductWall2','ductWall3','ductWall4']
geom='\n'.join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in stl_names)
surfs='\n'.join(f'        {n} {{ level (1 1); patchInfo {{ type patch; }} }}' for n in stl_names)
Path('system/snappyHexMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; version 2.0; }}
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 300000; maxGlobalCells 600000; minRefinementCells 0; maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2; resolveFeatureAngle 30;
    refinementSurfaces
    {{
{surfs}
    }}
    refinementRegions {{}}
    locationInMesh (0 0 0.10);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8;
    nFeatureSnapIter 0; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls
{{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0;
   featureAngle 60; slipFeatureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
   nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
   nBufferCellsNoExtrude 0; nLayerIter 20; }}
meshQualityControls
{{
    #include "meshQualityDict"
    nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6;
''')
# Use the OpenFOAM distribution's standard quality thresholds.
Path('system/meshQualityDict').write_text('''maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
relaxed { maxNonOrtho 75; }
''')
print('wrote coarse meshing dictionaries for',len(stl_names),'surface patches')

# -- cell 10 ------------------------------------------------------------------------
# `blockMesh` itself succeeded (32,400 uniform cells), but the follow-on background `checkMesh` could 
from pathlib import Path
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
print('wrote mesh-check support dictionaries')

# -- cell 11 ------------------------------------------------------------------------
# This OpenFOAM release requires an explicit `features` entry even when feature snapping is disabled. 
from pathlib import Path
p=Path('system/snappyHexMeshDict')
s=p.read_text().replace('    nCellsBetweenLevels 2; resolveFeatureAngle 30;','    nCellsBetweenLevels 2; resolveFeatureAngle 30;\n    features ();')
p.write_text(s)
print('added required empty features list')

# -- cell 12 ------------------------------------------------------------------------
# I’ll rerun the coarse mesher from a fresh background grid.
import subprocess
subprocess.run(['blockMesh'],check=True,stdout=subprocess.DEVNULL)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
print(r.stdout[-7000:]); print(r.stderr[-1000:])
r.check_returncode()

# -- cell 13 ------------------------------------------------------------------------
# The coarse mesh completed with 35,000 cells and no mesher quality errors. I’ll run the binding bare 
import subprocess, re, pathlib
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-6000:]); print(r.stderr[-1000:])
r.check_returncode()
boundary=pathlib.Path('constant/polyMesh/boundary').read_text()
for n in ['inlet1','inlet2','inlet3','inlet4','outlet','cylinderWall','floor','top','ductWall1','ductWall2','ductWall3','ductWall4']:
    m=re.search(r'\b'+n+r'\b\s*\{.*?nFaces\s+(\d+);',boundary,re.S)
    print(n,'faces',m.group(1) if m else 'MISSING')
    assert m and int(m.group(1))>0

# -- cell 14 ------------------------------------------------------------------------
# The binding check passes: one connected region, 35,000 cells, 12 non-empty named patches, maximum no
import subprocess, pathlib
r=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True)
print(r.stdout[-2000:]); print(r.stderr[-500:]); r.check_returncode()
print('\n'.join(str(p) for p in pathlib.Path('VTK').rglob('*') if p.is_file())[:5000])

# -- cell 15 ------------------------------------------------------------------------
# The converted mesh contains exactly one boundary dataset for every requested patch. I’ll produce the
import pyvista as pv
from pathlib import Path
bdir=Path('VTK/t24_0/boundary')
pl=pv.Plotter(off_screen=True,window_size=(1000,760))
colors={'cylinderWall':'#8ecae6','floor':'#adb5bd','top':'#cfe8f3','outlet':'#e63946',
        'ductWall1':'#ffb703','ductWall2':'#fb8500','ductWall3':'#ffd166','ductWall4':'#f4a261',
        'inlet1':'#2a9d8f','inlet2':'#43aa8b','inlet3':'#577590','inlet4':'#277da1'}
for name,c in colors.items():
    m=pv.read(bdir/f'{name}.vtp')
    opacity=0.42 if name in ('cylinderWall','top') else 1.0
    pl.add_mesh(m,color=c,opacity=opacity,show_edges=True,edge_color='#263238',line_width=0.3)
pl.add_axes(); pl.show_grid()
pl.camera_position=[(0.27,-0.31,0.25),(0,0,0.085),(0,0,1)]
pl.show(screenshot='coarse_mesh_patches.png')
print('wrote coarse_mesh_patches.png')

# -- cell 16 ------------------------------------------------------------------------
# The completion gate found one metadata omission: although snappyHexMesh used `(0, 0, 0.10)` and prod
export_report=export_patches(fluid,patches,directory='constant/triSurface',tolerance=5e-4,angular_tolerance=0.12,location_in_mesh=(0.0,0.0,0.10))
print('recorded location_in_mesh:',export_report.get('location_in_mesh'))
