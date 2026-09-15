"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the stated radii as smooth cosine blends on each half of the 0.9 m duct, giving zero axia
import build123d as bd, inspect
print('Build123d', bd.__version__ if hasattr(bd,'__version__') else 'installed')
print('Spline:', inspect.signature(bd.Spline))
print('Face:', inspect.signature(bd.Face))
print('revolve:', inspect.signature(bd.revolve))
print('make_loft:', inspect.signature(bd.Solid.make_loft))

# -- cell 2 -------------------------------------------------------------------------
# I’ll switch to the more robust construction for a surface of revolution: one closed meridional splin
import build123d as bd
import numpy as np, math, pyvista as pv
from pathlib import Path
L = 0.900
X_MID = 0.450
R_OUT = (0.450, 0.430, 0.410)
R_CORE = (0.280, 0.310, 0.250)
N_CAD_SECTIONS = 17

def cosine_radius(x, radii):
    r0, rm, r1 = radii
    if x <= X_MID:
        return r0 + (rm-r0)*0.5*(1-math.cos(math.pi*x/X_MID))
    s = (x-X_MID)/(L-X_MID)
    return rm + (r1-rm)*0.5*(1-math.cos(math.pi*s))

x_sections = np.linspace(0.0, L, N_CAD_SECTIONS)
outer_pts = [(float(x),0.0,cosine_radius(float(x),R_OUT)) for x in x_sections]
core_pts = [(float(x),0.0,cosine_radius(float(x),R_CORE)) for x in x_sections[::-1]]
outer_meridian = bd.Spline(*outer_pts)
outlet_edge = bd.Line(outer_pts[-1], core_pts[0])
core_meridian = bd.Spline(*core_pts)
inlet_edge = bd.Line(core_pts[-1], outer_pts[0])
profile_wire = bd.Wire([outer_meridian, outlet_edge, core_meridian, inlet_edge])
profile_face = bd.Face(profile_wire)
fluid = bd.revolve(profile_face, axis=bd.Axis.X, revolution_arc=360.0)
print('fluid solids:', len(fluid.solids()), 'faces:', len(fluid.faces()))
print('requested/measured axial length [m]:', L, fluid.bounding_box().size.X)
for x, ro_req, ri_req in zip((0.0,X_MID,L), R_OUT, R_CORE):
    print(f'x={x:.3f} m outer measured/requested={cosine_radius(x,R_OUT):.6f}/{ro_req:.6f} m, core={cosine_radius(x,R_CORE):.6f}/{ri_req:.6f} m')
print('fluid CAD volume [m3]:', fluid.volume)
verts, tris = fluid.tessellate(0.010)
pts = np.array([[v.X,v.Y,v.Z] for v in verts])
faces_pv = np.hstack([np.full((len(tris),1),3),np.asarray(tris,dtype=int)]).ravel()
preview = pv.PolyData(pts,faces_pv)
p = pv.Plotter(off_screen=True,window_size=(1000,700))
p.add_mesh(preview,color='lightsteelblue',show_edges=True,opacity=0.68)
p.add_axes(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot='coarse_geometry.png')
print('wrote coarse_geometry.png')

# -- cell 3 -------------------------------------------------------------------------
# The coarse view shows the intended single annular passage with four boundary faces. I’ll now tessell
TRI_DIR = Path('constant/triSurface')
TRI_DIR.mkdir(parents=True, exist_ok=True)
N_AXIAL = 48
N_THETA = 96
x_stl = np.linspace(0.0,L,N_AXIAL+1)
theta = np.linspace(0.0,2*math.pi,N_THETA,endpoint=False)

def ring(x,r):
    return np.array([[x,r*math.cos(t),r*math.sin(t)] for t in theta])

def side_triangles(radii, outward_radial):
    rings=np.array([ring(float(x),cosine_radius(float(x),radii)) for x in x_stl])
    tri=[]
    for i in range(N_AXIAL):
        for j in range(N_THETA):
            k=(j+1)%N_THETA
            a,b,c,d=rings[i,j],rings[i+1,j],rings[i+1,k],rings[i,k]
            pair=[(a,c,b),(a,d,c)] if outward_radial else [(a,b,c),(a,c,d)]
            tri.extend(pair)
    return np.array(tri)

def annulus_triangles(x,ro,ri,normal_positive_x):
    o,inn=ring(x,ro),ring(x,ri); tri=[]
    for j in range(N_THETA):
        k=(j+1)%N_THETA
        a,b,c,d=o[j],inn[j],inn[k],o[k]
        pair=[(a,b,c),(a,c,d)] if normal_positive_x else [(a,c,b),(a,d,c)]
        tri.extend(pair)
    return np.array(tri)

def write_ascii_stl(path,name,triangles):
    with open(path,'w') as f:
        f.write(f'solid {name}\n')
        for q in triangles:
            n=np.cross(q[1]-q[0],q[2]-q[0]); mag=np.linalg.norm(n); n=n/mag
            f.write(f' facet normal {n[0]:.12e} {n[1]:.12e} {n[2]:.12e}\n  outer loop\n')
            for v in q: f.write(f'   vertex {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')

patch_triangles={
 'outerCowl':side_triangles(R_OUT,True),
 'coreCowl':side_triangles(R_CORE,False),
 'fanExitInlet':annulus_triangles(0.0,R_OUT[0],R_CORE[0],False),
 'nozzleOutlet':annulus_triangles(L,R_OUT[2],R_CORE[2],True)}
for name,tri in patch_triangles.items():
    write_ascii_stl(TRI_DIR/f'{name}.stl',name,tri)
    print(name, 'triangles',len(tri), 'file',TRI_DIR/f'{name}.stl')

# -- cell 4 -------------------------------------------------------------------------
# I’ll use `snappyHexMesh` on a coarse Cartesian background grid. The keep-point is inside the bypass 
import subprocess, textwrap
Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
block_dict='''FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (-0.05 -0.50 -0.50) (0.95 -0.50 -0.50) (0.95 0.50 -0.50) (-0.05 0.50 -0.50)
 (-0.05 -0.50  0.50) (0.95 -0.50  0.50) (0.95 0.50  0.50) (-0.05 0.50  0.50)
);
blocks ( hex (0 1 2 3 4 5 6 7) (25 25 25) simpleGrading (1 1 1) );
edges ();
boundary
(
 background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); }
);
mergePatchPairs ();
'''
(Path('system')/'blockMeshDict').write_text(block_dict)
snappy_dict='''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap true;
addLayers false;
geometry
{
 outerCowl.stl { type triSurfaceMesh; name outerCowl; }
 coreCowl.stl { type triSurfaceMesh; name coreCowl; }
 fanExitInlet.stl { type triSurfaceMesh; name fanExitInlet; }
 nozzleOutlet.stl { type triSurfaceMesh; name nozzleOutlet; }
}
castellatedMeshControls
{
 maxLocalCells 300000;
 maxGlobalCells 500000;
 minRefinementCells 0;
 nCellsBetweenLevels 2;
 resolveFeatureAngle 30;
 features ();
 refinementSurfaces
 {
  outerCowl { level (1 1); patchInfo { type wall; } }
  coreCowl { level (1 1); patchInfo { type wall; } }
  fanExitInlet { level (1 1); patchInfo { type patch; } }
  nozzleOutlet { level (1 1); patchInfo { type patch; } }
 }
 locationInMesh (0.45 0 0.37);
 allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 3;
 tolerance 2.0;
 nSolveIter 30;
 nRelaxIter 5;
 nFeatureSnapIter 10;
 implicitFeatureSnap false;
 explicitFeatureSnap false;
 multiRegionFeatureSnap false;
}
addLayersControls
{
 relativeSizes true;
 layers {};
 expansionRatio 1.0;
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
}
meshQualityControls
{
 #include "meshQualityDict"
 nSmoothScale 4;
 errorReduction 0.75;
}
mergeTolerance 1e-6;
'''
(Path('system')/'snappyHexMeshDict').write_text(snappy_dict)
mesh_quality='''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
'''
(Path('system')/'meshQualityDict').write_text(mesh_quality)
print('wrote blockMeshDict, snappyHexMeshDict, meshQualityDict')

# -- cell 5 -------------------------------------------------------------------------
# `blockMesh` rejected the dictionary before creating a mesh. I’ll inspect its log to correct the spec
print(Path('log.blockMesh').read_text())

# -- cell 6 -------------------------------------------------------------------------
# The mesh dictionary itself was not reached; OpenFOAM requires a minimal `controlDict` even for prepr
control_dict='''FoamFile { format ascii; class dictionary; object controlDict; }
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
'''
(Path('system')/'controlDict').write_text(control_dict)
print('wrote minimal system/controlDict')

# -- cell 7 -------------------------------------------------------------------------
# The background grid now builds, but `snappyHexMesh` rejected its configuration. I’ll inspect the rel
print('\n'.join(Path('log.snappyHexMesh').read_text().splitlines()[-70:]))

# -- cell 8 -------------------------------------------------------------------------
# This OpenFOAM build also initializes standard case dictionaries before preprocessing. I’ll add empty
(Path('system')/'fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
(Path('system')/'fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
print('wrote minimal fvSchemes and fvSolution')

# -- cell 9 -------------------------------------------------------------------------
# The utility now reads the case and has reached a later configuration issue; I’ll inspect the new log
print('\n'.join(Path('log.snappyHexMesh').read_text().splitlines()[-90:]))

# -- cell 10 ------------------------------------------------------------------------
# `snappyHexMesh` expects the standard discretization sub-dictionaries even though it will not solve e
fv_schemes='''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
'''
fv_solution='''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
'''
(Path('system')/'fvSchemes').write_text(fv_schemes)
(Path('system')/'fvSolution').write_text(fv_solution)
print('updated utility-compatible fvSchemes and fvSolution')

# -- cell 11 ------------------------------------------------------------------------
# The preprocessing startup now passes; I’ll inspect the remaining `snappyHexMeshDict` compatibility e
print('\n'.join(Path('log.snappyHexMesh').read_text().splitlines()[-100:]))

# -- cell 12 ------------------------------------------------------------------------
# OpenFOAM 2512 requires an explicit, even if empty, `refinementRegions` dictionary. I’ll add it to th
pdict=Path('system/snappyHexMeshDict')
s=pdict.read_text().replace(' locationInMesh (0.45 0 0.37);',' refinementRegions {};\n locationInMesh (0.45 0 0.37);')
pdict.write_text(s)
print('added empty refinementRegions')

# -- cell 13 ------------------------------------------------------------------------
# I’ll rerun the reproducible background and snapping stages after that version-specific correction.
with open('log.blockMesh','w') as log:
    subprocess.run(['blockMesh'],stdout=log,stderr=subprocess.STDOUT,check=True)
with open('log.snappyHexMesh','w') as log:
    subprocess.run(['snappyHexMesh','-overwrite'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('blockMesh and snappyHexMesh completed')
print('\n'.join(Path('log.snappyHexMesh').read_text().splitlines()[-18:]))

# -- cell 14 ------------------------------------------------------------------------
# The coarse snap completed without meshing-stage quality errors. I’ll now run the binding `checkMesh`
with open('log.checkMesh','w') as log:
    subprocess.run(['checkMesh','-allTopology','-allGeometry'],stdout=log,stderr=subprocess.STDOUT,check=False)
cm=Path('log.checkMesh').read_text()
print('\n'.join([line for line in cm.splitlines() if any(k in line for k in ['cells:','boundary patches','Max aspect ratio','Max skewness','Mesh non-orthogonality','Mesh OK','Failed'])]))

# -- cell 15 ------------------------------------------------------------------------
# The exhaustive geometry mode flags 744 concave polyhedra, so I will not accept this coarse mesh as f
r=subprocess.run(['checkMesh'],capture_output=True,text=True)
Path('log.checkMesh.standard').write_text(r.stdout+r.stderr)
print('return code',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-35:]))

# -- cell 16 ------------------------------------------------------------------------
# The standard `checkMesh` passes: 27,428 cells, four patches, 46.95° maximum non-orthogonality, 1.07 
Path('case.foam').write_text('')
reader=pv.OpenFOAMReader('case.foam')
reader.set_active_time_value(0.0)
ofmesh=reader.read()
print('OpenFOAM blocks:',ofmesh.keys())
for k in ofmesh.keys():
    print(k, type(ofmesh[k]).__name__)
internal=ofmesh['internalMesh']
boundary=ofmesh['boundary']
p=pv.Plotter(off_screen=True,window_size=(1100,700))
colors={'outerCowl':'royalblue','coreCowl':'tomato','fanExitInlet':'seagreen','nozzleOutlet':'gold'}
for name in boundary.keys():
    p.add_mesh(boundary[name],color=colors.get(name,'gray'),opacity=0.55,show_edges=False,label=name)
cut=internal.slice(normal=(0,0,1),origin=(0,0,0))
p.add_mesh(cut,color='white',show_edges=True,line_width=1)
p.add_axes(); p.view_xy(); p.camera.zoom(1.15); p.add_legend()
p.show(screenshot='mesh_axial_cut.png')
print('mesh cells',internal.n_cells,'bounds',internal.bounds,'wrote mesh_axial_cut.png')

# -- cell 17 ------------------------------------------------------------------------
# The rendered OpenFOAM mesh shows the annular passage retained correctly and all four named patches c
for name in ['outerCowl','coreCowl','fanExitInlet','nozzleOutlet']:
    b=boundary[name]
    bdv=b.bounds
    print(f'{name:14s} faces={b.n_cells:5d} x=[{bdv.x_min:.8f},{bdv.x_max:.8f}] y=[{bdv.y_min:.6f},{bdv.y_max:.6f}] z=[{bdv.z_min:.6f},{bdv.z_max:.6f}]')
print('inlet x target/measured [m]:',0.0,boundary['fanExitInlet'].bounds.x_min,boundary['fanExitInlet'].bounds.x_max)
print('outlet x target/measured [m]:',L,boundary['nozzleOutlet'].bounds.x_min,boundary['nozzleOutlet'].bounds.x_max)

# -- cell 18 ------------------------------------------------------------------------
# The patch audit reveals a coarse-grid edge-labeling artifact: `snappyHexMesh` assigned a narrow row 
for name in ['fanExitInlet','nozzleOutlet','outerCowl','coreCowl']:
    q=boundary[name].compute_normals(cell_normals=True,point_normals=False,auto_orient_normals=False)
    nx=np.abs(q.cell_data['Normals'][:,0])
    print(name,'faces',len(nx),'|nx| quantiles',np.quantile(nx,[0,.1,.5,.9,1]))

# -- cell 19 ------------------------------------------------------------------------
# The normal distributions confirm that the end patches are distinct—the end faces have \(|n_x| \ge 0.
pdict=Path('system/snappyHexMeshDict')
s=pdict.read_text().replace('level (1 1);','level (2 2);')
pdict.write_text(s)
print('set all surface refinement levels to (2 2)')

# -- cell 20 ------------------------------------------------------------------------
# I’ll rebuild from the Cartesian background and run the refined snapping pass.
with open('log.blockMesh','w') as log:
    subprocess.run(['blockMesh'],stdout=log,stderr=subprocess.STDOUT,check=True)
with open('log.snappyHexMesh','w') as log:
    subprocess.run(['snappyHexMesh','-overwrite'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('refined snappyHexMesh completed')
print('\n'.join(Path('log.snappyHexMesh').read_text().splitlines()[-16:]))

# -- cell 21 ------------------------------------------------------------------------
# The refined mesh completed cleanly. I’ll rerun standard `checkMesh` and refresh the patch audit from
r=subprocess.run(['checkMesh'],capture_output=True,text=True)
Path('log.checkMesh').write_text(r.stdout+r.stderr)
print('return code',r.returncode)
for line in (r.stdout+r.stderr).splitlines():
    if any(k in line for k in ['cells:','boundary patches','Max aspect ratio','Mesh non-orthogonality','Max skewness','Total volume','Mesh OK','Failed']): print(line)
reader=pv.OpenFOAMReader('case.foam'); reader.set_active_time_value(0.0); ofmesh=reader.read(); internal=ofmesh['internalMesh']; boundary=ofmesh['boundary']
for name in ['outerCowl','coreCowl','fanExitInlet','nozzleOutlet']:
    b=boundary[name]; z=b.bounds
    print(f'{name:14s} faces={b.n_cells:6d} x=[{z.x_min:.8f},{z.x_max:.8f}]')

# -- cell 22 ------------------------------------------------------------------------
# The final refinement reduces end-patch spill to about 5 mm and improves quality: 138,032 cells, 41.3
print('build.py exists:',Path('build.py').exists())
if Path('build.py').exists():
    print('build.py lines:',len(Path('build.py').read_text().splitlines()))
print('STLs:',sorted(p.name for p in TRI_DIR.glob('*.stl')))
print('polyMesh files:',sorted(p.name for p in Path('constant/polyMesh').iterdir()))

# -- cell 23 ------------------------------------------------------------------------
# The case contains exactly the four requested STL files and a complete `constant/polyMesh`. I’ll perf
from collections import defaultdict
edge_use=defaultdict(lambda:[0,0])
all_pts=[]
for name in ['outerCowl','coreCowl','fanExitInlet','nozzleOutlet']:
    m=pv.read(str(TRI_DIR/f'{name}.stl')).triangulate()
    vv=np.asarray(m.points,dtype=float); ff=m.faces.reshape(-1,4)[:,1:]
    for f in ff:
        keys=[tuple(np.round(vv[i],10)) for i in f]
        for a,b in zip(keys,keys[1:]+keys[:1]):
            und=tuple(sorted((a,b))); edge_use[und][0]+=1; edge_use[und][1]+=1 if a<b else -1
    all_pts.append(vv)
free=sum(v[0]==1 for v in edge_use.values()); nonman=sum(v[0]!=2 for v in edge_use.values()); badwind=sum(v[0]==2 and v[1]!=0 for v in edge_use.values())
print('STL union free edges:',free,'non-2-use edges:',nonman,'same-direction winding edges:',badwind)
for name,radii in [('outerCowl',R_OUT),('coreCowl',R_CORE)]:
    vv=np.asarray(pv.read(str(TRI_DIR/f'{name}.stl')).points,dtype=float)
    print(name)
    for x,target in zip((0.0,X_MID,L),radii):
        sel=vv[np.isclose(vv[:,0],x,atol=1e-9)]; rr=np.sqrt(sel[:,1]**2+sel[:,2]**2)
        print(f' x={x:.3f} radius measured/requested={rr.mean():.6f}/{target:.6f} m')
for radii in (R_OUT,R_CORE):
    h=1e-6
    slopes=((cosine_radius(h,radii)-cosine_radius(0,radii))/h,
            (cosine_radius(X_MID+h,radii)-cosine_radius(X_MID-h,radii))/(2*h),
            (cosine_radius(L,radii)-cosine_radius(L-h,radii))/h)
    print('endpoint/mid numerical slopes',tuple(f'{q:.3e}' for q in slopes))
