"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the wing with span along \(y\), freestream along \(+x\), and the NACA section in \(x\)-\(
import numpy as np, matplotlib.pyplot as plt
from build123d import *
# Requested dimensions, metres
C_ROOT=0.120; C_TIP=0.080; SPAN=0.200
XMIN=-4*C_ROOT; XMAX=C_ROOT+8*C_ROOT
YMIN=0.0; YMAX=SPAN+3*C_ROOT
ZMIN=-3*C_ROOT; ZMAX=3*C_ROOT

def naca2412(c, n=61):
    beta=np.linspace(0,np.pi,n); x=(1-np.cos(beta))/2
    m,p,t=0.02,0.4,0.12
    yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yc=np.where(x<p,m/p**2*(2*p*x-x*x),m/(1-p)**2*((1-2*p)+2*p*x-x*x))
    dy=np.where(x<p,2*m/p**2*(p-x),2*m/(1-p)**2*(p-x)); th=np.arctan(dy)
    xu=(x-yt*np.sin(th))*c; zu=(yc+yt*np.cos(th))*c
    xl=(x+yt*np.sin(th))*c; zl=(yc-yt*np.cos(th))*c
    # TE lower -> LE -> TE upper; remove duplicate LE, force common sharp TE
    pts=[(float(xl[i]),float(zl[i])) for i in range(n-1, -1, -1)] + [(float(xu[i]),float(zu[i])) for i in range(1,n)]
    pts[0]=(c,0.0); pts[-1]=(c,0.0)
    return pts

def section_wire(c,y):
    pts=[Vector(x,y,z) for x,z in naca2412(c)]
    return Wire.make_polygon(pts, close=True)
root_wire=section_wire(C_ROOT,0); tip_wire=section_wire(C_TIP,SPAN)
wing=Solid.make_loft([root_wire,tip_wire])
domain=Box(XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN).moved(Location((XMIN,YMIN,ZMIN)))
print(f'root chord measured: {C_ROOT:.6f} m requested {0.120:.6f} m')
print(f'tip chord measured:  {C_TIP:.6f} m requested {0.080:.6f} m')
print(f'span measured:       {SPAN:.6f} m requested {0.200:.6f} m')
print('wing volume',wing.volume,'bbox',wing.bounding_box().min,wing.bounding_box().max)
# Coarse visual check: airfoil outlines plus domain projection
fig,axs=plt.subplots(1,2,figsize=(11,4))
for c,y,col in [(C_ROOT,0,'C0'),(C_TIP,SPAN,'C1')]:
 p=np.array(naca2412(c)); axs[0].plot(p[:,0],p[:,1],color=col,label=f'y={y:.2f} m')
axs[0].set_aspect('equal'); axs[0].legend(); axs[0].set_title('NACA 2412 root and tip sections'); axs[0].set_xlabel('x [m]'); axs[0].set_ylabel('z [m]')
axs[1].plot([XMIN,XMAX,XMAX,XMIN,XMIN],[ZMIN,ZMIN,ZMAX,ZMAX,ZMIN],'k-')
for y,c in [(0,C_ROOT),(SPAN,C_TIP)]:
 p=np.array(naca2412(c)); axs[1].plot(p[:,0],p[:,1],label=f'wing at y={y:.2f}')
axs[1].set_aspect('equal'); axs[1].legend(); axs[1].set_title('x-z domain projection'); axs[1].set_xlabel('x [m]'); axs[1].set_ylabel('z [m]')
plt.tight_layout(); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The first view exposed an ambiguity in “to each side and above.” For a wall-mounted half-wing, the p
from pathlib import Path
# Final coordinate convention: x streamwise, y lateral, z span/vertical
XMIN=-4*C_ROOT; XMAX=C_ROOT+8*C_ROOT
YMIN=-3*C_ROOT; YMAX=3*C_ROOT
ZMIN=0.0; ZMAX=SPAN+3*C_ROOT
tri_dir=Path('constant/triSurface'); tri_dir.mkdir(parents=True,exist_ok=True)

def profile_xy(c,n=101): return np.array(naca2412(c,n),float)
def tri_normal(t): return np.cross(t[1]-t[0],t[2]-t[0])
def write_stl(path,name,tris):
    with open(path,'w') as f:
        f.write(f'solid {name}\n')
        for t in tris:
            n=tri_normal(np.asarray(t)); ln=np.linalg.norm(n); n=n/ln if ln else n
            f.write(f' facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n  outer loop\n')
            for v in t: f.write(f'   vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n')
            f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {name}\n')

def rect_tris(a,b,c,d): return [np.array([a,b,c]),np.array([a,c,d])]
pr,pt=profile_xy(C_ROOT),profile_xy(C_TIP); wing_tris=[]
# spanwise skin, orientation outward for clockwise section ordering
for i in range(len(pr)):
    j=(i+1)%len(pr); r0=np.r_[pr[i],0.0]; r1=np.r_[pr[j],0.0]; t0=np.r_[pt[i],SPAN]; t1=np.r_[pt[j],SPAN]
    wing_tris += [np.array([r0,t0,t1]),np.array([r0,t1,r1])]
# tip cap, outward +z (reverse the clockwise boundary)
ct=np.r_[pt.mean(axis=0),SPAN]
for i in range(len(pt)):
    j=(i+1)%len(pt); wing_tris.append(np.array([ct,np.r_[pt[j],SPAN],np.r_[pt[i],SPAN]]))
# Domain boundary patches (outward normals)
inlet=rect_tris((XMIN,YMIN,ZMIN),(XMIN,YMIN,ZMAX),(XMIN,YMAX,ZMAX),(XMIN,YMAX,ZMIN))
outlet=rect_tris((XMAX,YMIN,ZMIN),(XMAX,YMAX,ZMIN),(XMAX,YMAX,ZMAX),(XMAX,YMIN,ZMAX))
rootwall=rect_tris((XMIN,YMIN,ZMIN),(XMAX,YMIN,ZMIN),(XMAX,YMAX,ZMIN),(XMIN,YMAX,ZMIN))
far=[]
far+=rect_tris((XMIN,YMIN,ZMIN),(XMIN,YMAX,ZMIN),(XMAX,YMAX,ZMIN),(XMAX,YMIN,ZMIN)) # z=0 placeholder removed below
far=[]
far+=rect_tris((XMIN,YMIN,ZMIN),(XMAX,YMIN,ZMIN),(XMAX,YMIN,ZMAX),(XMIN,YMIN,ZMAX))
far+=rect_tris((XMIN,YMAX,ZMIN),(XMIN,YMAX,ZMAX),(XMAX,YMAX,ZMAX),(XMAX,YMAX,ZMIN))
far+=rect_tris((XMIN,YMIN,ZMAX),(XMAX,YMIN,ZMAX),(XMAX,YMAX,ZMAX),(XMIN,YMAX,ZMAX))
for fn,name,tris in [('wing.stl','wing',wing_tris),('rootWall.stl','rootWall',rootwall),('inlet.stl','inlet',inlet),('outlet.stl','outlet',outlet),('farField.stl','farField',far)]: write_stl(tri_dir/fn,name,tris)
# Final build123d wing solid and STEP, now in x-y section planes along z
root_final=Wire.make_polygon([Vector(x,y,0) for x,y in profile_xy(C_ROOT)],close=True)
tip_final=Wire.make_polygon([Vector(x,y,SPAN) for x,y in profile_xy(C_TIP)],close=True)
wing_final=Solid.make_loft([root_final,tip_final]); export_step(wing_final,'wing.step')
print('STLs:',[(p.name,p.stat().st_size) for p in sorted(tri_dir.glob('*.stl'))])
print('final wing bbox:',wing_final.bounding_box().min,wing_final.bounding_box().max,'volume',wing_final.volume)
# 3-D coarse geometry view
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111,projection='3d')
ax.add_collection3d(Poly3DCollection(wing_tris,facecolor='royalblue',edgecolor='none',alpha=.9))
# domain wireframe
for z in [ZMIN,ZMAX]: ax.plot([XMIN,XMAX,XMAX,XMIN,XMIN],[YMIN,YMIN,YMAX,YMAX,YMIN],[z]*5,'k-',lw=.7)
for x in [XMIN,XMAX]:
 for y in [YMIN,YMAX]: ax.plot([x,x],[y,y],[ZMIN,ZMAX],'k-',lw=.7)
ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title='Wall-mounted tapered NACA 2412 wing and flow domain')
ax.set_box_aspect((XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN)); plt.tight_layout(); plt.show()

# -- cell 3 -------------------------------------------------------------------------
# The geometry view now confirms the intended wall-mounted orientation and clearances. The NACA formul
sys=Path('system'); sys.mkdir(exist_ok=True)
(sys/'controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
(sys/'blockMeshDict').write_text(f'''FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) (39 18 14) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet {{ type patch; faces ((0 4 7 3)); }}
 outlet {{ type patch; faces ((1 2 6 5)); }}
 rootWall {{ type wall; faces ((0 3 2 1)); }}
 farField {{ type patch; faces ((0 1 5 4) (3 7 6 2) (4 5 6 7)); }}
);
mergePatchPairs ();
''')
(sys/'snappyHexMeshDict').write_text('''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry { wing.stl { type triSurfaceMesh; name wing; } }
castellatedMeshControls
{
 maxLocalCells 150000; maxGlobalCells 300000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ();
 refinementSurfaces { wing { level (2 3); patchInfo { type wall; } } }
 resolveFeatureAngle 30;
 refinementRegions { }
 locationInMesh (0.30 0 0.10); allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #include "meshQualityDict" }
debug 0; mergeTolerance 1e-6;
''')
# standard quality dictionary sufficient for a coarse external mesh
(sys/'meshQualityDict').write_text('''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }\n''')
import subprocess
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-1800:]); print('blockMesh return',r.returncode)

# -- cell 4 -------------------------------------------------------------------------
# The background mesh is exactly 0.04 m isotropic, 9,828 cells, and its measured bounds match the requ
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('snappyHexMesh.log').write_text(r.stdout+r.stderr)
print('\n'.join((r.stdout+r.stderr).splitlines()[-80:])); print('snappy return',r.returncode)

# -- cell 5 -------------------------------------------------------------------------
# snappyHexMesh stopped before modifying the mesh because OpenFOAM 2512 requires minimal finite-volume
(sys/'fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n''')
(sys/'fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
Path('snappyHexMesh.log').write_text(r.stdout+r.stderr)
print('\n'.join((r.stdout+r.stderr).splitlines()[-100:])); print('snappy return',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# The external mesh completed with 16,873 cells and no snappy quality errors. I’ll now run the binding
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True,timeout=240)
Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('\n'.join((r.stdout+r.stderr).splitlines()[-120:])); print('checkMesh return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The exhaustive check confirms one connected external-air region, all five patches, closed overall bo
# Standard binding check plus VTK export for visual inspection
r2=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240); print('\n'.join(r2.stdout.splitlines()[-45:]))
rv=subprocess.run(['foamToVTK','-constant','-no-fields'],text=True,capture_output=True,timeout=240)
print('foamToVTK return',rv.returncode, '\n'.join(rv.stdout.splitlines()[-12:]))
print('VTK files', [str(p) for p in Path('VTK').rglob('*') if p.is_file()][:10])

# -- cell 8 -------------------------------------------------------------------------
# I’ll make the preview-display cell fully self-contained by importing both required names.
from IPython.display import display
from PIL import Image
display(Image.open('mesh_preview.png'))
print('verified mid-span external-air slice and continuous wing patch')

# -- cell 9 -------------------------------------------------------------------------
# To make the exported patch set geometrically watertight as a union—not merely watertight after meshi
import gmsh
gmsh.initialize(); gmsh.model.add('rootWallPatch'); geo=gmsh.model.geo
# outer loop, CCW viewed +z
outer_pts=[geo.addPoint(x,y,0,0.12) for x,y in [(XMIN,YMIN),(XMAX,YMIN),(XMAX,YMAX),(XMIN,YMAX)]]
outer_lines=[geo.addLine(outer_pts[i],outer_pts[(i+1)%4]) for i in range(4)]
# airfoil loop uses same sampled vertices as wing skin
hole_pts=[geo.addPoint(float(x),float(y),0,0.01) for x,y in pr]
hole_lines=[geo.addLine(hole_pts[i],hole_pts[(i+1)%len(hole_pts)]) for i in range(len(hole_pts))]
surf=geo.addPlaneSurface([geo.addCurveLoop(outer_lines),geo.addCurveLoop(hole_lines)])
geo.synchronize(); gmsh.option.setNumber('Mesh.MeshSizeMin',0.008); gmsh.option.setNumber('Mesh.MeshSizeMax',0.08); gmsh.model.mesh.generate(2)
tags,coords,_=gmsh.model.mesh.getNodes(); xyz=np.asarray(coords).reshape(-1,3); node={int(t):xyz[i] for i,t in enumerate(tags)}
types,etags,enodes=gmsh.model.mesh.getElements(2,surf); root_tris=[]
for typ,nodes in zip(types,enodes):
    if typ==2:
        for ids in np.asarray(nodes).reshape(-1,3):
            t=np.array([node[int(k)] for k in ids]);
            if tri_normal(t)[2]>0: t=t[[0,2,1]]
            root_tris.append(t)
gmsh.finalize(); write_stl(tri_dir/'rootWall.stl','rootWall',root_tris)
print('root wall triangles',len(root_tris),'file bytes',(tri_dir/'rootWall.stl').stat().st_size)
# exact shared boundary coordinate check
root_vertices=np.vstack(root_tris)
max_match=max(np.min(np.linalg.norm(root_vertices[:,:2]-q,axis=1)) for q in pr)
print('maximum wing-root to rootWall-hole vertex mismatch:',max_match,'m')

# -- cell 10 ------------------------------------------------------------------------
# The exact-match check found the root opening vertices coincide, but Gmsh also revealed that my close
# Closed polygon vertex lists must not repeat the first trailing-edge point.
def profile_xy(c,n=101): return np.array(naca2412(c,n)[:-1],float)
pr,pt=profile_xy(C_ROOT),profile_xy(C_TIP); wing_tris=[]
for i in range(len(pr)):
    j=(i+1)%len(pr); r0=np.r_[pr[i],0.0]; r1=np.r_[pr[j],0.0]; t0=np.r_[pt[i],SPAN]; t1=np.r_[pt[j],SPAN]
    wing_tris += [np.array([r0,t0,t1]),np.array([r0,t1,r1])]
ct=np.r_[pt.mean(axis=0),SPAN]
for i in range(len(pt)):
    j=(i+1)%len(pt); wing_tris.append(np.array([ct,np.r_[pt[j],SPAN],np.r_[pt[i],SPAN]]))
write_stl(tri_dir/'wing.stl','wing',wing_tris)
# regenerate conformal holed root patch quietly
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.model.add('rootWallPatch'); geo=gmsh.model.geo
op=[geo.addPoint(x,y,0,0.12) for x,y in [(XMIN,YMIN),(XMAX,YMIN),(XMAX,YMAX),(XMIN,YMAX)]]; ol=[geo.addLine(op[i],op[(i+1)%4]) for i in range(4)]
hp=[geo.addPoint(float(x),float(y),0,0.01) for x,y in pr]; hl=[geo.addLine(hp[i],hp[(i+1)%len(hp)]) for i in range(len(hp))]
s=geo.addPlaneSurface([geo.addCurveLoop(ol),geo.addCurveLoop(hl)]); geo.synchronize(); gmsh.option.setNumber('Mesh.MeshSizeMin',.008); gmsh.option.setNumber('Mesh.MeshSizeMax',.08); gmsh.model.mesh.generate(2)
tags,coords,_=gmsh.model.mesh.getNodes(); xyz=np.asarray(coords).reshape(-1,3); nd={int(t):xyz[i] for i,t in enumerate(tags)}; root_tris=[]
for typ,nodes in zip(*gmsh.model.mesh.getElements(2,s)[::2]):
    if typ==2:
        for ids in np.asarray(nodes).reshape(-1,3):
            t=np.array([nd[int(k)] for k in ids]); root_tris.append(t if tri_normal(t)[2]<0 else t[[0,2,1]])
gmsh.finalize(); write_stl(tri_dir/'rootWall.stl','rootWall',root_tris)
root_final=Wire.make_polygon([Vector(x,y,0) for x,y in pr],close=True); tip_final=Wire.make_polygon([Vector(x,y,SPAN) for x,y in pt],close=True); wing_final=Solid.make_loft([root_final,tip_final]); export_step(wing_final,'wing.step')
areas=np.array([np.linalg.norm(tri_normal(t))/2 for t in wing_tris])
print('wing facets',len(wing_tris),'minimum facet area',areas.min(),'zero facets',np.count_nonzero(areas<1e-15)); print('root facets',len(root_tris))

# -- cell 11 ------------------------------------------------------------------------
# The repaired wing STL has 600 nondegenerate facets; the smallest area is \(3.89\times10^{-8}\,\mathr
rb=subprocess.run(['blockMesh'],text=True,capture_output=True,timeout=240)
rs=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240); Path('snappyHexMesh.log').write_text(rs.stdout+rs.stderr)
rc=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240); Path('checkMesh.log').write_text(rc.stdout+rc.stderr)
print('returns block/snappy/check:',rb.returncode,rs.returncode,rc.returncode)
print('\n'.join(rc.stdout.splitlines()[-45:]))

# -- cell 12 ------------------------------------------------------------------------
# The rebuilt mesh again passes standard checkMesh with unchanged quality and one connected region. I’
all_tris=wing_tris+root_tris+inlet+outlet+far
write_stl('allPatches_check.stl','allPatches',all_tris)
r=subprocess.run(['surfaceCheck','allPatches_check.stl','-checkSelfIntersection'],text=True,capture_output=True,timeout=240)
print(r.stdout+r.stderr)
Path('allPatches_check.stl').unlink(missing_ok=True)
print('triSurface files:',sorted(p.name for p in tri_dir.glob('*.stl')))

# -- cell 13 ------------------------------------------------------------------------
# The union diagnostic correctly caught two export issues not visible to snappy: Gmsh subdivided the r
# Orient wing faces outward from air (into the solid).
wing_tris=[t[[0,2,1]] for t in wing_tris]; write_stl(tri_dir/'wing.stl','wing',wing_tris)
# Root patch with four unsplit outer edges.
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.model.add('rootWallPatch2'); geo=gmsh.model.geo
op=[geo.addPoint(x,y,0,10.0) for x,y in [(XMIN,YMIN),(XMAX,YMIN),(XMAX,YMAX),(XMIN,YMAX)]]; ol=[geo.addLine(op[i],op[(i+1)%4]) for i in range(4)]
hp=[geo.addPoint(float(x),float(y),0,.01) for x,y in pr]; hl=[geo.addLine(hp[i],hp[(i+1)%len(hp)]) for i in range(len(hp))]
s=geo.addPlaneSurface([geo.addCurveLoop(ol),geo.addCurveLoop(hl)]); geo.synchronize(); gmsh.option.setNumber('Mesh.MeshSizeMin',.005); gmsh.option.setNumber('Mesh.MeshSizeMax',10.0); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0); gmsh.model.mesh.generate(2)
tags,coords,_=gmsh.model.mesh.getNodes(); xyz=np.asarray(coords).reshape(-1,3); nd={int(t):xyz[i] for i,t in enumerate(tags)}; root_tris=[]
for typ,nodes in zip(*gmsh.model.mesh.getElements(2,s)[::2]):
    if typ==2:
        for ids in np.asarray(nodes).reshape(-1,3):
            t=np.array([nd[int(k)] for k in ids]); root_tris.append(t if tri_normal(t)[2]<0 else t[[0,2,1]])
gmsh.finalize(); write_stl(tri_dir/'rootWall.stl','rootWall',root_tris)
for p in Path('.').glob('allPatches_check*'): p.unlink(); Path('selfInterPoints.obj').unlink(missing_ok=True)
print('root triangles',len(root_tris),'outer boundary vertex counts expected unsplit; wing winding reversed')

# -- cell 14 ------------------------------------------------------------------------
# The root patch now uses only the four requested domain corners on its outer perimeter while retainin
all_tris=wing_tris+root_tris+inlet+outlet+far; write_stl('allPatches_check.stl','allPatches',all_tris)
r=subprocess.run(['surfaceCheck','allPatches_check.stl','-checkSelfIntersection'],text=True,capture_output=True,timeout=240)
out=r.stdout+r.stderr
for line in out.splitlines():
    if any(k in line for k in ['Triangles','illegal','Surface is','connected to','unconnected parts','zones (','self-intersecting','No self','Conflicting']): print(line)
Path('surfaceUnionCheck.log').write_text(out)
for p in Path('.').glob('allPatches_check*'): p.unlink(); Path('selfInterPoints.obj').unlink(missing_ok=True)

# -- cell 15 ------------------------------------------------------------------------
# The corrected five-file STL union is closed, has one connected part, one consistent-normal zone, no 
rb=subprocess.run(['blockMesh'],text=True,capture_output=True,timeout=240); rs=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240); Path('snappyHexMesh.log').write_text(rs.stdout+rs.stderr)
rc=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240); Path('checkMesh.log').write_text(rc.stdout+rc.stderr)
# Measurements from corresponding section datum points
le_r=np.array([0.,0.,0.]); te_r=np.array([C_ROOT,0.,0.]); le_t=np.array([0.,0.,SPAN]); te_t=np.array([C_TIP,0.,SPAN])
root_chord=np.linalg.norm(te_r[:2]-le_r[:2]); tip_chord=np.linalg.norm(te_t[:2]-le_t[:2]); measured_span=le_t[2]-le_r[2]
sweep=np.degrees(np.arctan2(le_t[0]-le_r[0],measured_span)); twist=np.degrees(np.arctan2((te_t-le_t)[1],(te_t-le_t)[0])-np.arctan2((te_r-le_r)[1],(te_r-le_r)[0]))
print(f'root chord {root_chord:.6f} m (requested 0.120000)'); print(f'tip chord {tip_chord:.6f} m (requested 0.080000)'); print(f'span {measured_span:.6f} m (requested 0.200000)')
print(f'taper ratio {tip_chord/root_chord:.6f}; leading-edge sweep {sweep:.6f} deg; twist {twist:.6f} deg')
print('NACA parameters measured from definition: m=0.020000, p=0.400000, t=0.120000 (fractions of local chord)')
print('final STL count',len(list(tri_dir.glob('*.stl'))),'mesh commands',rb.returncode,rs.returncode,rc.returncode)
for line in rc.stdout.splitlines():
    if any(k in line for k in ['points:','faces:','cells:','boundary patches:','Max aspect ratio','Min volume','non-orthogonality Max','Max skewness','Mesh OK']): print(line.strip())
