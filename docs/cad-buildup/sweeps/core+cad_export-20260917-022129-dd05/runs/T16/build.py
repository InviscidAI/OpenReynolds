"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the full 360° annular fluid domain in metres. I’ll use piecewise half-cosine axial blends
import inspect, build123d as bd
print('build123d', getattr(bd, '__version__', 'installed'))
for obj in [bd.Edge.make_spline, bd.Wire, bd.Face, bd.revolve]:
    print(getattr(obj, '__qualname__', str(obj)), inspect.signature(obj))

# -- cell 2 -------------------------------------------------------------------------
# The available API supports direct spline edges and revolution. I’ll now construct the meridional ann
import numpy as np
from build123d import Edge, Wire, Face, Axis, Vector, revolve
L=0.900; XM=0.450
RO=(0.450,0.430,0.410); RI=(0.280,0.310,0.250)
def cosine_profile(r0,rm,r1,n_half=24):
    x1=np.linspace(0,XM,n_half+1); x2=np.linspace(XM,L,n_half+1)[1:]
    s1=x1/XM; s2=(x2-XM)/(L-XM)
    y1=r0+(rm-r0)*(1-np.cos(np.pi*s1))/2
    y2=rm+(r1-rm)*(1-np.cos(np.pi*s2))/2
    return np.r_[x1,x2],np.r_[y1,y2]
xo,ro=cosine_profile(*RO); xi,ri=cosine_profile(*RI)
outer_edge=Edge.make_spline([Vector(float(x),float(r),0) for x,r in zip(xo,ro)])
outlet_edge=Edge.make_line((L,RO[2],0),(L,RI[2],0))
core_edge=Edge.make_spline([Vector(float(x),float(r),0) for x,r in zip(xi[::-1],ri[::-1])])
inlet_edge=Edge.make_line((0,RI[0],0),(0,RO[0],0))
profile_wire=Wire([outer_edge,outlet_edge,core_edge,inlet_edge])
profile_face=Face(profile_wire)
fluid=revolve(profile_face,axis=Axis.X)
print('solid volume',fluid.volume,'expected scale ~',np.pi*L*((sum(RO)/3)**2-(sum(RI)/3)**2))
print('bbox',fluid.bounding_box().min,fluid.bounding_box().max)
print('topology solids/faces/edges',len(fluid.solids()),len(fluid.faces()),len(fluid.edges()))
for i,f in enumerate(fluid.faces()): print(i,round(f.area,6),f.geom_type)

# -- cell 3 -------------------------------------------------------------------------
# The revolved result is a valid single solid with exactly four boundary faces and the intended 0–0.9 
import matplotlib.pyplot as plt
from build123d import export_stl
export_stl(fluid,'coarse_geometry_preview.stl',tolerance=0.004,angular_tolerance=0.25)
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(xo,ro,'b.-',label='outer cowl inner surface')
ax.plot(xi,ri,'r.-',label='core cowl outer surface')
ax.fill_between(xo,np.interp(xo,xi,ri),ro,color='skyblue',alpha=.35,label='bypass air')
for x in (0,XM,L): ax.axvline(x,color='k',lw=.5,ls='--')
ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('radius [m]'); ax.legend(loc='center right'); ax.grid(True)
plt.tight_layout(); plt.show()
for name,e,targets in [('outer',outer_edge,RO),('core',core_edge,RI)]:
    vals=[]
    for x in (0,XM,L):
        pts=e.intersect(Axis((x,0,0),(0,1,0)))
        vals.append(abs(pts[0].Y) if pts else float('nan'))
    print(name,'measured radii [m]',vals,'requested',targets,'errors',np.array(vals)-np.array(targets))

# -- cell 4 -------------------------------------------------------------------------
# The preview confirms the intended converging annulus and the core-cowl mid-duct bulge. All six reque
print('Edge revolve?',hasattr(outer_edge,'revolve'))
import inspect
if hasattr(outer_edge,'revolve'): print(inspect.signature(outer_edge.revolve))
for nm,e in [('outer_cowl',outer_edge),('nozzle_outlet',outlet_edge),('core_cowl',core_edge),('fan_exit_inlet',inlet_edge)]:
    try:
        s=e.revolve(Axis.X,360)
        print(nm,type(s).__name__,len(s.faces()),s.area)
    except Exception as ex: print(nm,'FAILED',type(ex).__name__,ex)

# -- cell 5 -------------------------------------------------------------------------
# Edges do not expose a method-form revolution in this installed build123d release. I’ll use the modul
generated={}
for nm,e in [('outer_cowl',outer_edge),('nozzle_outlet',outlet_edge),('core_cowl',core_edge),('fan_exit_inlet',inlet_edge)]:
    g=revolve(e,axis=Axis.X)
    generated[nm]=g
    print(nm,type(g).__name__,len(g.faces()),round(g.area,9),g.geom_type)

# -- cell 6 -------------------------------------------------------------------------
# The mapping was unique and correct, but the guard tolerance was 2.0×10⁻⁶ m² while the sampled-spline
import os, sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
def swept_area(r0,rm,r1):
    x,r=cosine_profile(r0,rm,r1,4000)
    return float(2*np.pi*np.trapezoid(r*np.sqrt(1+np.gradient(r,x)**2),x))
expected_areas={'outer_cowl':swept_area(*RO),'core_cowl':swept_area(*RI),
                'fan_exit_inlet':np.pi*(RO[0]**2-RI[0]**2),
                'nozzle_outlet':np.pi*(RO[2]**2-RI[2]**2)}
unused=list(fluid.faces()); patch_faces={}
for name,a in expected_areas.items():
    f=min(unused,key=lambda q:abs(q.area-a)); patch_faces[name]=[f]; unused.remove(f)
    print(name,'face area',f.area,'expected from named edge',a,'delta',f.area-a)
assert not unused and all(abs(patch_faces[n][0].area-a)<5e-6 for n,a in expected_areas.items())
os.makedirs('constant/triSurface',exist_ok=True)
export_report=export_patches(fluid,patch_faces,out_dir='constant/triSurface',tolerance=2.5e-4)

# -- cell 7 -------------------------------------------------------------------------
# The four STLs are now exported from one tessellation; their welded union has 13,500 triangles, zero 
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; version 2.0; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 10; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
''')
Path('system/blockMeshDict').write_text('''FoamFile { format ascii; class dictionary; object blockMeshDict; version 2.0; }
convertToMeters 1;
vertices ((-0.06 -0.48 -0.48) (0.96 -0.48 -0.48) (0.96 0.48 -0.48) (-0.06 0.48 -0.48)
          (-0.06 -0.48 0.48) (0.96 -0.48 0.48) (0.96 0.48 0.48) (-0.06 0.48 0.48));
blocks (hex (0 1 2 3 4 5 6 7) (34 32 32) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
''')
Path('system/snappyHexMeshDict').write_text('''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; version 2.0; }
castellatedMesh true; snap true; addLayers false;
geometry {
 outer_cowl.stl { type triSurfaceMesh; name outer_cowl; }
 core_cowl.stl { type triSurfaceMesh; name core_cowl; }
 fan_exit_inlet.stl { type triSurfaceMesh; name fan_exit_inlet; }
 nozzle_outlet.stl { type triSurfaceMesh; name nozzle_outlet; }
}
castellatedMeshControls {
 maxLocalCells 300000; maxGlobalCells 500000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2; resolveFeatureAngle 30;
 refinementSurfaces {
  outer_cowl { level (1 1); patchInfo { type wall; } }
  core_cowl { level (1 1); patchInfo { type wall; } }
  fan_exit_inlet { level (1 1); patchInfo { type patch; } }
  nozzle_outlet { level (1 1); patchInfo { type patch; } }
 }
 refinementRegions {};
 locationInMesh (0.20 0.35 0.0); allowFreeStandingZoneFaces true;
}
snapControls {
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls {
 maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
 minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}
debug 0; mergeTolerance 1e-6;
''')
print('wrote',*[str(p) for p in Path('system').iterdir()])

# -- cell 8 -------------------------------------------------------------------------
# The meshing dictionaries are in place. I’ll build and check the coarse background mesh first, before
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('blockMesh failed')

# -- cell 9 -------------------------------------------------------------------------
# `snappyHexMesh` stopped before modifying the background mesh because OpenFOAM 2512 requires placehol
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; version 2.0; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; version 2.0; }
solvers {}
''')
print('meshing-only fv dictionaries written')

# -- cell 10 ------------------------------------------------------------------------
# OpenFOAM 2512 also requires an explicit empty `features` list when feature files are not used. I’ll 
p=Path('system/snappyHexMeshDict'); txt=p.read_text(); txt=txt.replace(' resolveFeatureAngle 30;',' resolveFeatureAngle 30; features ();'); p.write_text(txt)
print('added explicit features ()')

# -- cell 11 ------------------------------------------------------------------------
# I’ll rerun `snappyHexMesh` with the completed 2512 dictionary.
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('log.snappyHexMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 12 ------------------------------------------------------------------------
# The coarse snapped mesh completed with 52,834 cells and no quality errors under the mesher’s configu
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)
if r.returncode: raise RuntimeError('checkMesh failed')

# -- cell 13 ------------------------------------------------------------------------
# The mesh itself is valid; only the visualization utility rejected an outdated `-noFields` spelling. 
r=subprocess.run(['foamToVTK','-help'],text=True,capture_output=True)
print('\n'.join([s for s in (r.stdout+r.stderr).splitlines() if 'field' in s.lower() or 'mesh' in s.lower() or 'usage' in s.lower()][:40]))

# -- cell 14 ------------------------------------------------------------------------
# The installed option is `-no-fields`. I’ll now render the cutaway from the exported volume mesh.
r=subprocess.run(['foamToVTK','-no-fields'],text=True,capture_output=True)
print(r.stdout[-1200:],r.stderr[-300:]); assert r.returncode==0
import glob, pyvista as pv
files=glob.glob('VTK/**/internal.vtu',recursive=True)+glob.glob('VTK/**/internalMesh.vtu',recursive=True)
print('volume files',files)
grid=pv.read(files[-1]); cut=grid.clip(normal=(0,0,1),origin=(0,0,0),invert=False)
p=pv.Plotter(off_screen=True,window_size=(1000,650))
p.add_mesh(cut,show_edges=True,color='lightsteelblue',edge_color='gray',line_width=0.25)
p.camera_position=[(1.45,-1.25,0.85),(0.43,0,0),(0,0,1)]
p.add_axes(); p.show(screenshot='mesh_cutaway.png')
print('rendered cells',cut.n_cells,'of',grid.n_cells)

# -- cell 15 ------------------------------------------------------------------------
# The cutaway shows a single continuous annular volume, with the inner core excluded and both axial an
def radius_at(edge,x):
    pts=edge.intersect(Axis((float(x),0,0),(0,1,0)))
    if not pts: raise ValueError(f'no spline intersection at x={x}')
    return abs(pts[0].Y)
def station_slopes(edge,h=1e-4):
    return ((radius_at(edge,h)-radius_at(edge,0))/h,
            (radius_at(edge,XM)-radius_at(edge,XM-h))/h,
            (radius_at(edge,XM+h)-radius_at(edge,XM))/h,
            (radius_at(edge,L)-radius_at(edge,L-h))/h)
for name,e in [('outer_cowl',outer_edge),('core_cowl',core_edge)]:
    print(name,'dr/dx at inlet, mid-left, mid-right, outlet =',station_slopes(e),'target 0')
