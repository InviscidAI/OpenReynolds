"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the wing with the chord along +x, thickness along y, and span along +z. “No sweep” is tak
import os, inspect, subprocess, textwrap, numpy as np
os.makedirs('constant/triSurface', exist_ok=True)
import build123d as bd
print('build123d', bd.__version__)
print('loft:', inspect.signature(bd.loft))
print('Spline:', inspect.signature(bd.Spline))
print('Wire:', inspect.signature(bd.Wire))
print('export_stl:', inspect.signature(bd.export_stl))

# -- cell 2 -------------------------------------------------------------------------
# The algebraic `loft` wrapper flattened the two bare wires as one section list, so that cell was not 
import os, numpy as np, matplotlib.pyplot as plt
import build123d as bd
C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200
XMIN, XMAX = -4*C_ROOT, C_ROOT + 8*C_ROOT
YMIN, YMAX, ZMIN, ZMAX = -3*C_ROOT, 3*C_ROOT, 0.0, SPAN + 3*C_ROOT
def naca2412_points(chord,z,n=81):
    beta=np.linspace(0,np.pi,n); x=(1-np.cos(beta))/2; m,p,t=.02,.4,.12
    yt=5*t*(.2969*np.sqrt(x)-.1260*x-.3516*x*x+.2843*x**3-.1036*x**4)
    yc=np.where(x<p,m/p**2*(2*p*x-x*x),m/(1-p)**2*((1-2*p)+2*p*x-x*x))
    dy=np.where(x<p,2*m/p**2*(p-x),2*m/(1-p)**2*(p-x)); a=np.arctan(dy)
    up=np.c_[(x-yt*np.sin(a))*chord,(yc+yt*np.cos(a))*chord]
    lo=np.c_[(x+yt*np.sin(a))*chord,(yc-yt*np.cos(a))*chord]
    xy=np.vstack([up[::-1],lo[1:]])
    return [(float(X),float(Y),float(z)) for X,Y in xy]
root_pts=naca2412_points(C_ROOT,0); tip_pts=naca2412_points(C_TIP,SPAN)
root_wire=bd.Wire.make_polygon(root_pts,close=True); tip_wire=bd.Wire.make_polygon(tip_pts,close=True)
wing=bd.Solid.make_loft([root_wire,tip_wire],ruled=False)
domain=bd.Box(XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN).moved(bd.Location((XMIN,YMIN,ZMIN)))
fluid=domain-wing
r=np.array(root_pts); q=np.array(tip_pts); bb=wing.bounding_box()
print(f'root chord: {r[:,0].max()-r[:,0].min():.6f} m / requested 0.120000')
print(f'tip chord:  {q[:,0].max()-q[:,0].min():.6f} m / requested 0.080000')
print(f'span:       {bb.max.Z-bb.min.Z:.6f} m / requested 0.200000')
print(f'LE offset:  {q[:,0].min()-r[:,0].min():.3e} m / no sweep')
print(f'root tmax:  {r[:,1].max()-r[:,1].min():.6f} m / NACA 12% nominal 0.014400')
print(f'domain: x {XMIN:.3f}..{XMAX:.3f}, y {YMIN:.3f}..{YMAX:.3f}, z {ZMIN:.3f}..{ZMAX:.3f} m')
fig=plt.figure(figsize=(8,5)); ax=fig.add_subplot(111,projection='3d')
for i in range(0,len(r),4): ax.plot([r[i,0],q[i,0]],[r[i,2],q[i,2]],[r[i,1],q[i,1]],'b-',alpha=.3)
ax.plot(r[:,0],r[:,2],r[:,1],'k'); ax.plot(q[:,0],q[:,2],q[:,1],'k')
ax.set(xlabel='x',ylabel='span z',zlabel='thickness y',title='Coarse tapered NACA 2412 wing'); ax.set_box_aspect((.12,.2,.05)); plt.tight_layout(); plt.show()

# -- cell 3 -------------------------------------------------------------------------
# The preview and measurements confirm the intended half-wing. I’ll now create each boundary patch as 
import inspect, build123d as bd
print('Face.make_plane',inspect.signature(bd.Face.make_plane))
print('Face.make_rect',inspect.signature(bd.Face.make_rect))
print('Compound',inspect.signature(bd.Compound))
print('Shape.tessellate',inspect.signature(bd.Shape.tessellate))

# -- cell 4 -------------------------------------------------------------------------
# I’ll export the six requested patch files now. The root wall is constructed as a planar face with th
import inspect
print('Face constructor',inspect.signature(bd.Face))
print('Plane constructor',inspect.signature(bd.Plane))

# -- cell 5 -------------------------------------------------------------------------
# The constructor signatures confirm planar polygon faces can be used directly. I’ll now perform the f
# Explicitly authored domain boundary faces
os.makedirs('constant/triSurface',exist_ok=True)
def polyface(points): return bd.Face(bd.Wire.make_polygon(points,close=True))
inlet_face=polyface([(XMIN,YMIN,ZMIN),(XMIN,YMAX,ZMIN),(XMIN,YMAX,ZMAX),(XMIN,YMIN,ZMAX)])
outlet_face=polyface([(XMAX,YMAX,ZMIN),(XMAX,YMIN,ZMIN),(XMAX,YMIN,ZMAX),(XMAX,YMAX,ZMAX)])
yminus_face=polyface([(XMIN,YMIN,ZMIN),(XMIN,YMIN,ZMAX),(XMAX,YMIN,ZMAX),(XMAX,YMIN,ZMIN)])
yplus_face=polyface([(XMAX,YMAX,ZMIN),(XMAX,YMAX,ZMAX),(XMIN,YMAX,ZMAX),(XMIN,YMAX,ZMIN)])
top_face=polyface([(XMIN,YMIN,ZMAX),(XMIN,YMAX,ZMAX),(XMAX,YMAX,ZMAX),(XMAX,YMIN,ZMAX)])
root_outer=polyface([(XMIN,YMIN,0),(XMAX,YMIN,0),(XMAX,YMAX,0),(XMIN,YMAX,0)])
root_profile=bd.Face(root_wire)
root_wall=root_outer-root_profile
wing_faces=[f for f in wing.faces() if not all(abs(v.Z)<1e-9 for v in f.vertices())]
wing_patch=bd.Compound(wing_faces); far_patch=bd.Compound([yminus_face,yplus_face,top_face])
patch_shapes={'wing':wing_patch,'rootWall':root_wall,'inlet':inlet_face,'outlet':outlet_face,'farField':far_patch}
for name,shape in patch_shapes.items():
    path=f'constant/triSurface/{name}.stl'; bd.export_stl(shape,path,tolerance=0.0002,angular_tolerance=0.08,ascii_format=True)
    print(name, 'faces=',len(shape.faces()),'area=',f'{shape.area:.6f}','file=',os.path.getsize(path))
# CAD fluid volume sanity
print('fluid solids:',len(fluid.solids()),'volume:',f'{fluid.volume:.8f} m3','expected box less wing:',f'{domain.volume-wing.volume:.8f}')

# -- cell 6 -------------------------------------------------------------------------
# The exported surface areas are sensible, but the coincident CAD subtraction at z=0 gives an unreliab
import os, textwrap
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
open('system/controlDict','w').write('''FoamFile{format ascii; class dictionary; object controlDict;}\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
open('system/blockMeshDict','w').write(f'''FoamFile{{format ascii; class dictionary; object blockMeshDict;}}
scale 1;
vertices (({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN}) ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX}));
blocks (hex (0 1 2 3 4 5 6 7) (52 24 20) simpleGrading (1 1 1));
edges ();
boundary (
 inlet{{type patch; faces((0 4 7 3));}}
 outlet{{type patch; faces((1 2 6 5));}}
 rootWall{{type wall; faces((0 3 2 1));}}
 farField{{type patch; faces((0 1 5 4)(3 7 6 2)(4 5 6 7));}}
);
mergePatchPairs();
''')
open('system/surfaceFeatureExtractDict','w').write('''FoamFile{format ascii; class dictionary; object surfaceFeatureExtractDict;}\nwing.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }\n''')
open('system/snappyHexMeshDict','w').write('''FoamFile{format ascii; class dictionary; object snappyHexMeshDict;}
castellatedMesh true; snap true; addLayers false;
geometry { wing.stl { type triSurfaceMesh; name wing; } }
castellatedMeshControls {
 maxLocalCells 300000; maxGlobalCells 500000; minRefinementCells 0; nCellsBetweenLevels 2;
 features (); refinementSurfaces { wing { level (2 2); patchInfo { type wall; } } }
 resolveFeatureAngle 30; refinementRegions { wing { mode distance; levels ((0.050 2)); } }
 locationInMesh (0.50 0.20 0.30); allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
debug 0; mergeTolerance 1e-6;
''')
print(open('system/blockMeshDict').read())

# -- cell 7 -------------------------------------------------------------------------
# `blockMesh` rejected the compact patch syntax in the dictionary; no mesh was produced. I’ll rewrite 
open('system/blockMeshDict','w').write(f'''FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) (52 24 20) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet {{ type patch; faces ( (0 4 7 3) ); }}
 outlet {{ type patch; faces ( (1 2 6 5) ); }}
 rootWall {{ type wall; faces ( (0 3 2 1) ); }}
 farField {{ type patch; faces ( (0 1 5 4) (3 7 6 2) (4 5 6 7) ); }}
);
mergePatchPairs ();
''')
for cmd in [['blockMesh'],['surfaceFeatureExtract']]:
    r=subprocess.run(cmd,text=True,capture_output=True); print('###',cmd[0],r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-12:])); assert r.returncode==0

# -- cell 8 -------------------------------------------------------------------------
# `snappyHexMesh` requires placeholder finite-volume dictionaries even though no solver will be run. I
open('system/fvSchemes','w').write('''FoamFile{format ascii; class dictionary; object fvSchemes;}\nddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n''')
open('system/fvSolution','w').write('''FoamFile{format ascii; class dictionary; object fvSolution;}\nsolvers{} potentialFlow{} relaxationFactors{}\n''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
open('snappyHexMesh.log','w').write(r.stdout+r.stderr); print('return',r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-30:])); assert r.returncode==0

# -- cell 9 -------------------------------------------------------------------------
# Snapping completed with 37,319 cells and no faces violating the configured quality limits. I’ll now 
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
open('checkMesh.log','w').write(r.stdout+r.stderr)
print('return',r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-55:]))
print('\nBOUNDARY FILE:\n',open('constant/polyMesh/boundary').read())

# -- cell 10 ------------------------------------------------------------------------
# The exhaustive geometric diagnostics flag 659 concave cut cells, although all standard quality measu
r=subprocess.run(['foamToVTK','-constant','-no-internal','-surfaceFields'],text=True,capture_output=True)
print('foamToVTK',r.returncode,'\n'.join((r.stdout+r.stderr).splitlines()[-8:]))
import pyvista as pv
pv.OFF_SCREEN=True
# locate generated boundary vtps
import glob
files=glob.glob('VTK/**/*.*',recursive=True); print([f for f in files if f.endswith(('.vtp','.vtk'))][-12:])

# -- cell 11 ------------------------------------------------------------------------
# The boundary export contains exactly the five requested patches. I’ll render two views: the full dom
p=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1200,520))
base='VTK/t14_0/boundary/'
colors={'inlet':'royalblue','outlet':'tomato','farField':'lightgray','rootWall':'tan','wing':'gold'}
for j in range(2):
 p.subplot(0,j)
 for name in colors:
  mesh=pv.read(base+name+'.vtp'); op=.12 if name in ['farField','inlet','outlet'] else (.35 if name=='rootWall' else 1)
  p.add_mesh(mesh,color=colors[name],opacity=op,show_edges=(name=='wing'),line_width=.3)
 p.add_axes(); p.set_background('white')
 if j==0: p.view_isometric(); p.camera.zoom(1.15); p.add_text('Full external domain',color='black')
 else:
  p.camera_position=[(.26,-.26,.22),(.055,0,.1),(0,0,1)]; p.camera.zoom(2.5); p.add_text('Wing and root-wall cut',color='black')
p.show(screenshot='mesh_preview.png')

# -- cell 12 ------------------------------------------------------------------------
# The render confirms the wall-mounted half-wing and external box. The close-up camera is nearly tange
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
open('checkMesh.standard.log','w').write(r.stdout+r.stderr)
print('return',r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-35:]))
