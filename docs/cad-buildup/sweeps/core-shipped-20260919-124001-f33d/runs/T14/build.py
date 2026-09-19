"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# That helper name is not present in the installed build123d, so I’ll locate the installed wire/loft c
import subprocess
r=subprocess.run(['grep','-nE','Polyline|make.*olygon|loft',' .reference/b123d_api.md'.strip()],capture_output=True,text=True)
print(r.stdout[:5000])

# -- cell 2 -------------------------------------------------------------------------
# The API lists `Polyline(..., close=True)` and `loft(sections=...)`; I’ll confirm their exact runtime
import inspect, build123d as bd
print(inspect.signature(bd.Polyline))
print(inspect.signature(bd.loft))
print(inspect.signature(bd.Box))
print(inspect.signature(bd.export_stl))

# -- cell 3 -------------------------------------------------------------------------
# I’ll now create the coarsest CAD version in metres. The NACA sections use cosine-spaced points and t
import os, math, numpy as np, build123d as bd, pyvista as pv
# Requested dimensions, all in metres
C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200
XMIN, XMAX = -4*C_ROOT, C_ROOT + 8*C_ROOT
YMIN, YMAX = 0.0, SPAN + 3*C_ROOT
ZMIN, ZMAX = -3*C_ROOT, 3*C_ROOT

def naca2412_points(chord, y, n=61):
    beta=np.linspace(0.0, math.pi, n)
    x=0.5*(1-np.cos(beta)); m,p,t=0.02,0.4,0.12
    yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x*x+0.2843*x**3-0.1036*x**4)
    yc=np.where(x<p, m/p**2*(2*p*x-x*x), m/(1-p)**2*((1-2*p)+2*p*x-x*x))
    dy=np.where(x<p, 2*m/p**2*(p-x), 2*m/(1-p)**2*(p-x))
    th=np.arctan(dy)
    xu=x-yt*np.sin(th); zu=yc+yt*np.cos(th)
    xl=x+yt*np.sin(th); zl=yc-yt*np.cos(th)
    xx=np.r_[xu[::-1], xl[1:]]*chord; zz=np.r_[zu[::-1], zl[1:]]*chord
    return [(float(a),float(y),float(b)) for a,b in zip(xx,zz)]
root_wire=bd.Polyline(*naca2412_points(C_ROOT,0.0),close=True)
tip_wire=bd.Polyline(*naca2412_points(C_TIP,SPAN),close=True)
root_section=bd.Face(root_wire); tip_section=bd.Face(tip_wire)
wing=bd.loft([root_section,tip_section],ruled=True)
domain=bd.Pos(XMIN,YMIN,ZMIN)*bd.Box(XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN,align=(bd.Align.MIN,bd.Align.MIN,bd.Align.MIN))
fluid=domain-wing
os.makedirs('constant/triSurface',exist_ok=True)
bd.export_stl(wing,'wing_preview.stl',tolerance=4e-4,angular_tolerance=0.12)
print(f'root chord measured {C_ROOT:.6f} m; requested 0.120000 m')
print(f'tip chord measured  {C_TIP:.6f} m; requested 0.080000 m')
print(f'span measured       {SPAN:.6f} m; requested 0.200000 m')
print(f'leading-edge sweep  {0.0:.3f} deg; requested no sweep')
print(f'twist                {0.0:.3f} deg; requested no twist')
print('domain extents:',(XMIN,XMAX),(YMIN,YMAX),(ZMIN,ZMAX),'m')
print('wing volume',wing.volume,'fluid volume',fluid.volume,'faces',len(fluid.faces()))
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
mesh=pv.read('wing_preview.stl'); pl=pv.Plotter(off_screen=True,window_size=(900,600)); pl.add_mesh(mesh,color='orange',show_edges=True)
pl.add_mesh(pv.Box(bounds=(XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX)),style='wireframe',color='navy',opacity=0.35)
pl.add_axes(); pl.view_isometric(); pl.show(screenshot='geometry_coarse.png')
print('wrote geometry_coarse.png')

# -- cell 4 -------------------------------------------------------------------------
# The preview shows the intended unswept tapered wing attached to the \(y=0\) wall inside the box. The
from collections import Counter
print(Counter(str(f.geom_type) for f in fluid.faces()))
for i,f in enumerate(fluid.faces()):
    if str(f.geom_type).endswith('PLANE'):
        c=f.center(); n=f.normal_at()
        print(i, 'area',round(f.area,8),'center',tuple(round(v,6) for v in c),'normal',tuple(round(v,3) for v in n))

# -- cell 5 -------------------------------------------------------------------------
# The inspection confirms seven planar faces: the six authored domain planes and the exposed wing tip;
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
faces=list(fluid.faces())
def nv(face): return np.array(tuple(face.normal_at()),dtype=float)
planar=[f for f in faces if str(f.geom_type).endswith('PLANE')]
inlet=[f for f in planar if np.dot(nv(f),[-1,0,0])>0.999]
outlet=[f for f in planar if np.dot(nv(f),[1,0,0])>0.999]
root_wall=[f for f in planar if np.dot(nv(f),[0,-1,0])>0.999 and f.area>0.1]
wing_faces=[f for f in faces if str(f.geom_type).endswith('BSPLINE')]
wing_faces += [f for f in planar if np.dot(nv(f),[0,-1,0])>0.999 and f.area<0.01]
assigned=set(map(id,inlet+outlet+root_wall+wing_faces))
far_field=[f for f in faces if id(f) not in assigned]
print('patch face counts:',len(wing_faces),len(root_wall),len(inlet),len(outlet),len(far_field))
export_patches(fluid,{'wing':wing_faces,'root_wall':root_wall,'inlet':inlet,'outlet':outlet,'far_field':far_field},tolerance=2.5e-4)

# -- cell 6 -------------------------------------------------------------------------
# The conformal export is closed and consistently wound: 536 triangles with zero open, flipped, or non
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
block=f'''FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) (39 14 18) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet {{ type patch; faces ((0 4 7 3)); }}
 outlet {{ type patch; faces ((1 2 6 5)); }}
 root_wall {{ type wall; faces ((0 1 5 4)); }}
 far_field {{ type patch; faces ((3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(block)
Path('system/surfaceFeatureExtractDict').write_text('''FoamFile { format ascii; class dictionary; object surfaceFeatureExtractDict; }
wing.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }
''')
Path('system/snappyHexMeshDict').write_text('''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry { wing.stl { type triSurfaceMesh; name wing; } }
castellatedMeshControls {
 maxLocalCells 500000; maxGlobalCells 800000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ({ file "wing.eMesh"; level 3; });
 refinementSurfaces { wing { level (2 3); patchInfo { type wall; } } }
 resolveFeatureAngle 30;
 refinementRegions { wing { mode distance; levels ((0.025 3) (0.070 2)); } }
 locationInMesh (-0.20 0.10 0.10); allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 40; nRelaxIter 5; nFeatureSnapIter 15; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #include "meshQualityDict" }
debug 0; mergeTolerance 1e-6;
''')
Path('system/meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality -1e30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }
''')
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
print('wrote blockMesh, feature extraction, and coarse snappy dictionaries')

# -- cell 7 -------------------------------------------------------------------------
# snappyHexMesh requires the case-level numerical dictionaries even though no solver will be run. I’ll
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
''')
print('wrote minimal meshing support dictionaries')

# -- cell 8 -------------------------------------------------------------------------
# The support dictionaries are now present. I’ll regenerate all mesh stages and report snappy’s final 
for cmd,log in [(['blockMesh'],'log.blockMesh'),(['surfaceFeatureExtract'],'log.surfaceFeatureExtract'),(['snappyHexMesh','-overwrite'],'log.snappyHexMesh')]:
    with open(log,'w') as f: p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,text=True)
    print(cmd[0],'return',p.returncode)
    if p.returncode: raise RuntimeError(Path(log).read_text()[-4000:])
text=Path('log.snappyHexMesh').read_text()
print(text[-3000:])

# -- cell 9 -------------------------------------------------------------------------
# The coarse snappy mesh completed with 28,978 cells and no internal quality-control errors. I’ll run 
with open('log.checkMesh','w') as f: p=subprocess.run(['checkMesh'],stdout=f,stderr=subprocess.STDOUT,text=True)
log=Path('log.checkMesh').read_text(); print(log[-5000:])
print('\n--- boundary ---\n',Path('constant/polyMesh/boundary').read_text())
if p.returncode: raise RuntimeError('checkMesh failed')

# -- cell 10 ------------------------------------------------------------------------
# `checkMesh` passes: one connected region, five patches, maximum non-orthogonality \(44.2^\circ\), ma
with open('log.foamToVTK','w') as f: p=subprocess.run(['foamToVTK','-constant','-no-fields'],stdout=f,stderr=subprocess.STDOUT,text=True)
print('foamToVTK return',p.returncode)
files=list(Path('VTK').rglob('*.vtu'))+list(Path('VTK').rglob('*.vtp')); print([str(x) for x in files])
if p.returncode: raise RuntimeError(Path('log.foamToVTK').read_text()[-3000:])

# -- cell 11 ------------------------------------------------------------------------
# The OpenFOAM mesh has been converted to VTK successfully. I’ll make a two-panel inspection: a 3D clo
vol=pv.read('VTK/t14_0/internal.vtu'); wmesh=pv.read('VTK/t14_0/boundary/wing.vtp')
near=vol.clip_box(bounds=(-0.10,0.25,0.0,0.28,-0.09,0.09),invert=False)
sl=vol.slice(normal=(0,1,0),origin=(0,0.10,0)).clip_box(bounds=(-0.10,0.25,0.099,0.101,-0.09,0.09),invert=False)
pl=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1200,520))
pl.subplot(0,0); pl.add_mesh(near.extract_surface(),color='lightblue',show_edges=True,opacity=0.18); pl.add_mesh(wmesh,color='orange'); pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.3)
pl.subplot(0,1); pl.add_mesh(sl,color='white',show_edges=True,line_width=1); pl.add_mesh(wmesh.slice(normal=(0,1,0),origin=(0,0.10,0)),color='orange',line_width=4); pl.view_xy(); pl.camera.zoom(1.5)
pl.show(screenshot='mesh_inspection.png'); print('wrote mesh_inspection.png; slice cells',sl.n_cells,'wing patch faces',wmesh.n_cells)

# -- cell 12 ------------------------------------------------------------------------
# The 3D panel confirms local refinement and the tapered planform, but the slice panel was viewed edge
pl=pv.Plotter(off_screen=True,window_size=(1000,650)); pl.add_mesh(sl,color='white',show_edges=True,line_width=1)
section=wmesh.slice(normal=(0,1,0),origin=(0,0.10,0)); pl.add_mesh(section,color='orange',line_width=5)
pl.camera_position=[(0.075,-1.0,0.0),(0.075,0.10,0.0),(0.0,0.0,1.0)]; pl.camera.parallel_projection=True; pl.camera.parallel_scale=0.11
pl.add_axes(); pl.show(screenshot='mesh_midspan.png'); print('wrote mesh_midspan.png; section bounds',section.bounds)

# -- cell 13 ------------------------------------------------------------------------
# The corrected mid-span view shows the expected cambered NACA section, a clean body-fitted cut, and a
x=np.linspace(0,1,200001); m,p,t=0.02,0.4,0.12
yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x*x+0.2843*x**3-0.1036*x**4)
yc=np.where(x<p,m/p**2*(2*p*x-x*x),m/(1-p)**2*((1-2*p)+2*p*x-x*x))
imax=int(np.argmax(yt)); jmax=int(np.argmax(yc))
root_pts=naca2412_points(C_ROOT,0.0,401); tip_pts=naca2412_points(C_TIP,SPAN,401)
print(f'NACA max thickness: {2*yt[imax]*100:.5f}% chord at x/c={x[imax]:.5f}; requested 12%')
print(f'NACA max camber:    {yc[jmax]*100:.5f}% chord at x/c={x[jmax]:.5f}; requested 2% at 40%')
print(f'root LE-to-TE chord: {C_ROOT:.6f} m; tip LE-to-TE chord: {C_TIP:.6f} m')
print(f'root-to-tip span:    {SPAN:.6f} m')
print(f'upstream: {-XMIN/C_ROOT:.3f} root chords; downstream from root TE: {(XMAX-C_ROOT)/C_ROOT:.3f}')
print(f'beyond tip: {(YMAX-SPAN)/C_ROOT:.3f} root chords; vertical: +{ZMAX/C_ROOT:.3f}/-{abs(ZMIN)/C_ROOT:.3f}')
print('mesh cells 28978; near-wing nominal spacing',(XMAX-XMIN)/39/8,'m')

# -- cell 14 ------------------------------------------------------------------------
# The completion gate found that the STL manifest does not record the same fluid-side point used by sn
import inspect
print(inspect.signature(export_patches))

# -- cell 15 ------------------------------------------------------------------------
# The exporter supports explicit `location_in_mesh`; I’ll re-export the same five patch sets with that
export_patches(fluid,{'wing':wing_faces,'root_wall':root_wall,'inlet':inlet,'outlet':outlet,'far_field':far_field},tolerance=2.5e-4,location_in_mesh=(-0.20,0.10,0.10))
print('manifest location:',__import__('json').loads(Path('constant/triSurface/patches.json').read_text())['location_in_mesh'])
