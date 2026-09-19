"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water volume—not the dry metal—as a Ø60 mm, 42 mm-long cylinder. Its sidewall will be
import build123d as bd, inspect, subprocess, os, sys
print('build123d', bd.__version__)
for name in ['Solid','Face','Shell','Wire','Edge','Cylinder','revolve','split']:
    obj=getattr(bd,name,None); print(name, inspect.signature(obj) if callable(obj) else obj)
print(subprocess.run(['grep','-n','make_surface\|make_shell\|make_solid','.reference/b123d_api.md'],capture_output=True,text=True).stdout[:3000])

# -- cell 2 -------------------------------------------------------------------------
# A stitched boundary made from three cylindrical side faces plus two end disks will preserve the requ
print([n for n in dir(bd.Solid) if 'make' in n.lower() or 'fix' in n.lower()])
print([n for n in dir(bd.Shell) if 'make' in n.lower() or 'fix' in n.lower()])
print(inspect.signature(bd.Shell)); print(inspect.getsource(bd.Shell.__init__)[:1800])

# -- cell 3 -------------------------------------------------------------------------
# The first stitching attempt failed because independently generated circular seams were not accepted 
R0=.03; z0=0; z1=.02; z2=.022; z3=.042
pts=[(0,0,z0),(R0,0,z0),(R0,0,z1),(R0,0,z2),(R0,0,z3),(0,0,z3),(0,0,z0)]
edges=[bd.Edge.make_line(pts[i],pts[i+1]) for i in range(len(pts)-1)]
profile=bd.Face(bd.Wire(edges))
testfluid=bd.revolve(profile,axis=bd.Axis.Z,clean=False)
print(type(testfluid),testfluid.is_valid,len(testfluid.solids()),len(testfluid.faces()),testfluid.volume)
for i,f in enumerate(testfluid.faces()): print(i,round(f.area,8),tuple(round(v,5) for v in f.center().to_tuple()))

# -- cell 4 -------------------------------------------------------------------------
# The revolved prototype is valid and has exactly five boundary faces, with the three requested axial 
# Definitive geometry (SI units).
import build123d as bd, math, os, subprocess, sys
FLANGE_OD=0.180; FLANGE_T=0.020; BORE_D=0.060; GASKET_T=0.002
BOLT_D=0.014; BOLT_PCD=0.140; N_BOLTS=6
R=BORE_D/2; TOTAL_L=2*FLANGE_T+GASKET_T
profile_points=[(0,0,0),(R,0,0),(R,0,FLANGE_T),(R,0,FLANGE_T+GASKET_T),
                (R,0,TOTAL_L),(0,0,TOTAL_L),(0,0,0)]
profile_edges=[bd.Edge.make_line(profile_points[i],profile_points[i+1]) for i in range(6)]
fluid=bd.revolve(bd.Face(bd.Wire(profile_edges)),axis=bd.Axis.Z,clean=False)
# Revolving the ordered profile produces cap, side1, gasket-side, side2, cap.
inlet=[fluid.faces()[0]]; flange1_bore=[fluid.faces()[1]]; gasket_inner=[fluid.faces()[2]]
flange2_bore=[fluid.faces()[3]]; outlet=[fluid.faces()[4]]
# Dry reference solids, excluded from the fluid mesh.
def flange_ring(z):
    outer=bd.Cylinder(FLANGE_OD/2,FLANGE_T,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Location((0,0,z)))
    cuts=bd.Cylinder(R,FLANGE_T,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Location((0,0,z)))
    for i in range(N_BOLTS):
        a=2*math.pi*i/N_BOLTS; xy=(BOLT_PCD/2*math.cos(a),BOLT_PCD/2*math.sin(a),z)
        cuts=cuts+bd.Cylinder(BOLT_D/2,FLANGE_T,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Location(xy))
    return outer-cuts
flange1=flange_ring(0); flange2=flange_ring(FLANGE_T+GASKET_T)
gasket=bd.Cylinder(FLANGE_OD/2,GASKET_T,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Location((0,0,FLANGE_T)))-bd.Cylinder(R,GASKET_T,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Location((0,0,FLANGE_T)))
bd.export_step(fluid,'fluid_volume.step')
bd.export_stl(fluid,'fluid_preview.stl',tolerance=0.0004,angular_tolerance=0.12)
bd.export_stl(flange1+flange2,'flanges_preview.stl',tolerance=0.001,angular_tolerance=0.15)
bd.export_stl(gasket,'gasket_preview.stl',tolerance=0.001,angular_tolerance=0.15)
print('valid/solids/faces:',fluid.is_valid,len(fluid.solids()),len(fluid.faces()))
print(f'length measured/requested: {fluid.bounding_box().size.Z:.6f} / {TOTAL_L:.6f} m')
print(f'bore diameter measured/requested: {fluid.bounding_box().size.X:.6f} / {BORE_D:.6f} m')
print(f'fluid volume measured/analytic: {fluid.volume:.12g} / {math.pi*R**2*TOTAL_L:.12g} m3')
print('wall axial lengths measured/requested:',[round(f.area/(2*math.pi*R),6) for f in [flange1_bore[0],gasket_inner[0],flange2_bore[0]]], '/', [FLANGE_T,GASKET_T,FLANGE_T])
print(f'flange OD reference measured/requested: {flange1.bounding_box().size.X:.6f} / {FLANGE_OD:.6f} m')

# -- cell 5 -------------------------------------------------------------------------
# The definitive fluid volume measures Ø60 × 42 mm, with wall bands 20/2/20 mm as requested. I’ll insp
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
p=pv.Plotter(off_screen=True,window_size=(1000,700))
p.add_mesh(pv.read('flanges_preview.stl'),color='lightgray',style='wireframe',line_width=1.2,opacity=0.35,label='dry flanges')
p.add_mesh(pv.read('gasket_preview.stl'),color='orange',opacity=0.30,label='dry gasket')
p.add_mesh(pv.read('fluid_preview.stl'),color='dodgerblue',opacity=0.85,label='water volume')
p.add_axes(); p.add_legend(); p.view_isometric(); p.camera.zoom(1.45)
p.show(screenshot='coarse_geometry.png')
print('wrote coarse_geometry.png')

# -- cell 6 -------------------------------------------------------------------------
# The visual check shows the central water column correctly passing through both dry flanges and the g
sys.path.insert(0,'.reference')
from cad_export import export_patches
os.makedirs('constant/triSurface',exist_ok=True)
export_patches(fluid,{'inlet':inlet,'outlet':outlet,'flange1_bore':flange1_bore,
                      'gasket_inner':gasket_inner,'flange2_bore':flange2_bore},
               tolerance=2.5e-4,directory='constant/triSurface')

# -- cell 7 -------------------------------------------------------------------------
# The exported union is watertight and consistently wound (0 open, flipped, or non-manifold edges). I’
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
def write(path,text):
    with open(path,'w') as f:f.write(text)
foam='FoamFile\n{ version 2.0; format ascii; class dictionary; object %s; }\n'
write('system/controlDict',(foam%'controlDict')+'''application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
runTimeModifiable true;''')
write('system/blockMeshDict',(foam%'blockMeshDict')+'''convertToMeters 1;
vertices ((-0.035 -0.035 -0.005) (0.035 -0.035 -0.005) (0.035 0.035 -0.005) (-0.035 0.035 -0.005)
          (-0.035 -0.035 0.047) (0.035 -0.035 0.047) (0.035 0.035 0.047) (-0.035 0.035 0.047));
blocks (hex (0 1 2 3 4 5 6 7) (14 14 11) simpleGrading (1 1 1));
edges (); boundary (background {type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}); mergePatchPairs ();''')
patches=['inlet','outlet','flange1_bore','gasket_inner','flange2_bore']
geom='\n'.join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in patches)
refs='\n'.join(f'    {n} {{ level ({"2 3" if n=="gasket_inner" else "1 2"}); patchInfo {{ type wall; }} }}' for n in patches)
write('system/snappyHexMeshDict',(foam%'snappyHexMeshDict')+f'''castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls {{ maxLocalCells 200000; maxGlobalCells 400000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2; features (); refinementSurfaces {{
{refs}
 }} resolveFeatureAngle 30; refinementRegions {{}} locationInMesh (0 0 0.010); allowFreeStandingZoneFaces true; }}
snapControls {{ nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #include "meshQualityDict" }}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;''')
write('system/meshQualityDict','''maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-1800:]); print('return',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The 2,156-cell background grid is ready. I’ll run the coarse surface-conforming mesh and inspect its
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('snappyHexMesh.log','w').write(r.stdout+r.stderr)
print(r.stdout[-5000:]); print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# `snappyHexMesh` stopped immediately while reading its setup, before modifying the mesh. I’ll inspect
print(open('snappyHexMesh.log').read()[-5000:])

# -- cell 10 ------------------------------------------------------------------------
# The mesh dictionaries are valid; OpenFOAM 2512 additionally requires placeholder `fvSchemes` and `fv
write('system/fvSchemes',(foam%'fvSchemes')+'ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
write('system/fvSolution',(foam%'fvSolution')+'solvers {}\n')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('snappyHexMesh.log','w').write(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-2000:]); print('return',r.returncode)

# -- cell 11 ------------------------------------------------------------------------
# The coarse conformal mesh completed with 8,132 cells and no mesher quality errors. Before validation
refs='\n'.join(f'    {n} {{ level ({"2 3" if n=="gasket_inner" else "1 2"}); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in patches)
snappy=open('system/snappyHexMeshDict').read()
start=snappy.index('refinementSurfaces {')+len('refinementSurfaces {')
end=snappy.index('\n } resolveFeatureAngle',start)
snappy=snappy[:start]+'\n'+refs+snappy[end:]
write('system/snappyHexMeshDict',snappy)
r1=subprocess.run(['blockMesh'],text=True,capture_output=True)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('snappyHexMesh.log','w').write(r2.stdout+r2.stderr)
print('blockMesh/snappy returns:',r1.returncode,r2.returncode)
print('\n'.join([line for line in r2.stdout.splitlines() if 'Snapped mesh' in line or 'Finished meshing' in line]))

# -- cell 12 ------------------------------------------------------------------------
# The final regenerated mesh retains 8,132 cells with the proper patch classes. I’ll now run the bindi
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
open('checkMesh.log','w').write(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)
print('\nBoundary file patch summary:')
b=open('constant/polyMesh/boundary').read()
for n in patches:
    i=b.find('\n    '+n); print(b[i:i+180].replace('\n',' '))

# -- cell 13 ------------------------------------------------------------------------
# `checkMesh` passes, but it exposed an unacceptable patch-coverage issue: the 2 mm gasket band was me
snappy=open('system/snappyHexMeshDict').read()
needle='  flange2_bore.stl { type triSurfaceMesh; name flange2_bore; }\n'
snappy=snappy.replace(needle,needle+'  gasketRefine { type searchableBox; min (-0.032 -0.032 0.018); max (0.032 0.032 0.024); }\n')
snappy=snappy.replace('refinementRegions {}','refinementRegions { gasketRefine { mode inside; levels ((1E15 3)); } }')
write('system/snappyHexMeshDict',snappy)
r1=subprocess.run(['blockMesh'],text=True,capture_output=True)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('snappyHexMesh.log','w').write(r2.stdout+r2.stderr)
print('returns',r1.returncode,r2.returncode)
for line in r2.stdout.splitlines():
    if 'Snapped mesh' in line or 'Finished meshing' in line or ('\t3\t' in line): print(line)

# -- cell 14 ------------------------------------------------------------------------
# The local band refinement produced 108,128 cells and completed without mesher errors. I’ll verify th
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
open('checkMesh.log','w').write(r.stdout+r.stderr)
print(r.stdout[-6500:]); print('return',r.returncode)
b=open('constant/polyMesh/boundary').read(); print('requested patch presence/counts:')
import re
for n in patches:
    m=re.search(r'\b'+re.escape(n)+r'\s*\{.*?nFaces\s+(\d+);',b,re.S)
    print(n, int(m.group(1)) if m else 'MISSING')

# -- cell 15 ------------------------------------------------------------------------
# All five patches now survive, and `checkMesh` reports **Mesh OK** for the 108,128-cell, single-regio
open('t20.foam','a').close()
reader=pv.OpenFOAMReader('t20.foam'); mesh=reader.read()
print(mesh)
print('blocks:',mesh.keys() if hasattr(mesh,'keys') else 'n/a')

# -- cell 16 ------------------------------------------------------------------------
# The OpenFOAM reader confirms the final mesh bounds are exactly the intended Ø60 mm by 42 mm flow dom
internal=mesh['internalMesh']; boundary=mesh['boundary']
print('boundary blocks:',boundary.keys())
p=pv.Plotter(off_screen=True,window_size=(1050,700))
cut=internal.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
p.add_mesh(cut,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.25,opacity=0.65,label='cells (cutaway)')
colors={'inlet':'royalblue','outlet':'crimson','flange1_bore':'silver','gasket_inner':'orange','flange2_bore':'dimgray'}
for n,c in colors.items(): p.add_mesh(boundary[n],color=c,opacity=0.95,label=n)
p.add_axes(); p.add_legend(bcolor='white',face='rectangle'); p.view_isometric(); p.camera.zoom(1.35)
p.show(screenshot='final_mesh.png')
print('wrote final_mesh.png')

# -- cell 17 ------------------------------------------------------------------------
# The completion gate found that the mesh is valid but the STL manifest does not record the interior s
import inspect
print(inspect.signature(export_patches))

# -- cell 18 ------------------------------------------------------------------------
# The exporter supports the required metadata. I’ll record the same verified interior point `(0, 0, 0.
export_patches(fluid,{'inlet':inlet,'outlet':outlet,'flange1_bore':flange1_bore,
                      'gasket_inner':gasket_inner,'flange2_bore':flange2_bore},
               tolerance=2.5e-4,directory='constant/triSurface',location_in_mesh=(0,0,0.010))
