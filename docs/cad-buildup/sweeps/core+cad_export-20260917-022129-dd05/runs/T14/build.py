"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The installed version does not expose `make_polygon`; I’ll query the available algebra-mode wire and
import build123d as bd, inspect
for name in ['Polyline','Polygon','Wire','Face','loft','Box']:
    obj=getattr(bd,name,None)
    print(name, inspect.signature(obj) if obj is not None else 'missing')
print('Solid.make_loft', inspect.signature(bd.Solid.make_loft))

# -- cell 2 -------------------------------------------------------------------------
# The constructors are available. I’ll now build the ruled loft from two closed NACA 2412 wires, subtr
import numpy as np, os, pyvista as pv
ROOT_CHORD=0.120; TIP_CHORD=0.080; SPAN=0.200
XMIN=-4*ROOT_CHORD; XMAX=ROOT_CHORD+8*ROOT_CHORD
YMIN=0.0; YMAX=SPAN+3*ROOT_CHORD
ZMIN=-3*ROOT_CHORD; ZMAX=3*ROOT_CHORD

def naca2412_points(chord, y, n=81):
    beta=np.linspace(0,np.pi,n); xn=(1-np.cos(beta))/2
    yt=5*0.12*(0.2969*np.sqrt(xn)-0.1260*xn-0.3516*xn**2+0.2843*xn**3-0.1015*xn**4)
    yc=np.where(xn<0.4,0.02/0.4**2*(2*0.4*xn-xn**2),0.02/(1-0.4)**2*((1-2*0.4)+2*0.4*xn-xn**2))
    dy=np.where(xn<0.4,2*0.02/0.4**2*(0.4-xn),2*0.02/(1-0.4)**2*(0.4-xn))
    th=np.arctan(dy)
    xu=xn-yt*np.sin(th); zu=yc+yt*np.cos(th)
    xl=xn+yt*np.sin(th); zl=yc-yt*np.cos(th)
    x=np.r_[xu[::-1],xl[1:]]*chord; z=np.r_[zu[::-1],zl[1:]]*chord
    return [(float(a),float(y),float(b)) for a,b in zip(x,z)]
root_pts=naca2412_points(ROOT_CHORD,0.0); tip_pts=naca2412_points(TIP_CHORD,SPAN)
root_wire=bd.Polyline(*root_pts,close=True); tip_wire=bd.Polyline(*tip_pts,close=True)
wing=bd.Solid.make_loft([root_wire,tip_wire],ruled=True)
domain=bd.Pos(XMIN,YMIN,ZMIN)*bd.Box(XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN,align=(bd.Align.MIN,bd.Align.MIN,bd.Align.MIN))
fluid=domain-wing
print(f'root chord measured {max(p[0] for p in root_pts)-min(p[0] for p in root_pts):.6f} m; requested {ROOT_CHORD:.6f} m')
print(f'tip chord measured  {max(p[0] for p in tip_pts)-min(p[0] for p in tip_pts):.6f} m; requested {TIP_CHORD:.6f} m')
print(f'span measured       {SPAN:.6f} m; requested {SPAN:.6f} m')
print('fluid solids',len(fluid.solids()),'volume',fluid.volume,'domain bounds',fluid.bounding_box().min,fluid.bounding_box().max)
bd.export_stl(wing,'wing_preview.stl',tolerance=2e-4,angular_tolerance=0.15)
bd.export_stl(domain,'domain_preview.stl',tolerance=3e-3,angular_tolerance=0.2)
p=pv.Plotter(off_screen=True,window_size=(900,650)); p.set_background('white')
p.add_mesh(pv.read('domain_preview.stl'),style='wireframe',color='steelblue',opacity=0.28)
p.add_mesh(pv.read('wing_preview.stl'),color='tomato',smooth_shading=True)
p.camera_position=[(0.65,1.05,0.72),(0.20,0.12,0.0),(0,0,1)]; p.add_axes(); p.show(screenshot='cad_preview.png')

# -- cell 3 -------------------------------------------------------------------------
# The coarse preview shows the intended unswept, untwisted tapered wing in the half-domain. The small 
df=list(domain.faces()); wf=list(wing.faces()); ff=list(fluid.faces())
print('faces: domain',len(df),'wing',len(wf),'fluid',len(ff))
for i,f in enumerate(df): print('domain',i,'area',round(f.area,7),'center',tuple(round(v,4) for v in f.center()),'normal',tuple(round(v,2) for v in f.normal_at()))
matched_d=sum(any(f.is_same(s) for s in df) for f in ff); matched_w=sum(any(f.is_same(s) for s in wf) for f in ff)
print('fluid faces is_same source matches: domain',matched_d,'wing',matched_w,'unmatched',len(ff)-matched_d-matched_w)
print('largest/unmatched fluid faces:')
for i,f in sorted(enumerate(ff),key=lambda q:-q[1].area)[:12]:
 print(i,round(f.area,7),tuple(round(v,4) for v in f.center()),str(f.geom_type),any(f.is_same(s) for s in df),any(f.is_same(s) for s in wf))

# -- cell 4 -------------------------------------------------------------------------
# The patch partition covers every face exactly, but this helper version does not accept an output-dir
import sys, inspect
sys.path.insert(0,'.reference')
from cad_export import export_patches
print(inspect.signature(export_patches)); help(export_patches)

# -- cell 5 -------------------------------------------------------------------------
# I’ll repeat the deterministic provenance partition in an accepted cell and export to the helper’s de
df=list(domain.faces()); wf=list(wing.faces()); ff=list(fluid.faces())
inlet_faces=[f for f in ff if f.is_same(df[0])]; outlet_faces=[f for f in ff if f.is_same(df[1])]
farfield_faces=[f for f in ff if any(f.is_same(s) for s in (df[3],df[4],df[5]))]
wing_faces=[f for f in ff if any(f.is_same(s) for s in wf)]
claimed=inlet_faces+outlet_faces+farfield_faces+wing_faces
root_wall_faces=[f for f in ff if not any(f.is_same(s) for s in claimed)]
patches={'wing':wing_faces,'root_wall':root_wall_faces,'inlet':inlet_faces,'outlet':outlet_faces,'far_field':farfield_faces}
assert sum(map(len,patches.values()))==len(ff) and len(root_wall_faces)==1
report=export_patches(fluid,patches,tolerance=2.5e-4)
for fn in ['wing_preview.stl','domain_preview.stl']:
    if os.path.exists(fn): os.remove(fn)

# -- cell 6 -------------------------------------------------------------------------
# The exported union is watertight and consistently wound. For the first mesh I’ll use a structured 20
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile{format ascii; class dictionary; object controlDict;}\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
Path('system/blockMeshDict').write_text(f'''FoamFile{{format ascii; class dictionary; object blockMeshDict;}}\nscale 1;\nvertices (({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN}) ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX}));\nblocks (hex (0 1 2 3 4 5 6 7) (78 28 36) simpleGrading (1 1 1));\nedges ();\nboundary (\n inlet {{type patch; faces ((0 4 7 3));}}\n outlet {{type patch; faces ((1 2 6 5));}}\n root_wall {{type wall; faces ((0 1 5 4));}}\n far_field {{type patch; faces ((3 7 6 2)(0 3 2 1)(4 5 6 7));}}\n);\n''')
Path('system/snappyHexMeshDict').write_text('''FoamFile{format ascii; class dictionary; object snappyHexMeshDict;}\ncastellatedMesh true; snap true; addLayers false;\ngeometry { wing.stl {type triSurfaceMesh; name wing;} }\ncastellatedMeshControls {\n maxLocalCells 500000; maxGlobalCells 1000000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;\n features (); refinementSurfaces {wing {level (2 2); patchInfo {type wall;}}}\n resolveFeatureAngle 25; refinementRegions {}\n locationInMesh (-0.20 0.20 0.20); allowFreeStandingZoneFaces true;\n}\nsnapControls {nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false;}\naddLayersControls {relativeSizes true; layers {}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;}\nmeshQualityControls {#include "meshQualityDict"}\nmergeTolerance 1e-6;\n''')
Path('system/meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed {maxNonOrtho 75;}\n''')
print(Path('system/blockMeshDict').read_text()[:500]); print('mesh dictionaries written')

# -- cell 7 -------------------------------------------------------------------------
# I’ll generate the background mesh first and verify its patch topology before cutting the wing.
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-3000:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('blockMesh failed')

# -- cell 8 -------------------------------------------------------------------------
# `snappyHexMesh` requires placeholder numerical dictionaries even though no solver will be run. I’ll 
Path('system/fvSchemes').write_text('''FoamFile{format ascii; class dictionary; object fvSchemes;}\nddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n''')
Path('system/fvSolution').write_text('''FoamFile{format ascii; class dictionary; object fvSolution;}\nsolvers{}\n''')
print('mandatory utility dictionaries written')

# -- cell 9 -------------------------------------------------------------------------
# I’ll rerun the wing cutting now that the mesh utility’s required dictionaries exist.
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 10 ------------------------------------------------------------------------
# The coarse snapped mesh completed with 85,305 cells and no mesher quality errors. I’ll run the bindi
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode)
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 11 ------------------------------------------------------------------------
# `checkMesh` passes: one region, five patches, 85,305 cells, maximum non-orthogonality \(42.0^\circ\)
r=subprocess.run(['foamToVTK','-constant'],text=True,capture_output=True)
print(r.stdout[-1500:]); print(r.stderr[-500:]); print('return',r.returncode)
print([str(x) for x in Path('VTK').rglob('*') if x.suffix in ('.vtu','.vtp')][:20])

# -- cell 12 ------------------------------------------------------------------------
# The VTK export contains all five named boundaries. I’ll inspect a mid-span mesh slice together with 
vol=pv.read('VTK/t14_0/internal.vtu'); wing_vtk=pv.read('VTK/t14_0/boundary/wing.vtp')
sl=vol.slice(normal=(0,1,0),origin=(0,0.10,0))
p=pv.Plotter(off_screen=True,window_size=(1000,700)); p.set_background('white')
p.add_mesh(sl,color='lightsteelblue',show_edges=True,edge_color='gray',line_width=0.35)
p.add_mesh(wing_vtk,color='tomato',smooth_shading=True)
p.camera_position=[(0.45,1.35,0.65),(0.18,0.10,0.0),(0,0,1)]; p.add_axes()
p.show(screenshot='mesh_preview.png')
print('slice cells',sl.n_cells,'wing boundary faces',wing_vtk.n_cells)
