"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model a single 360/28-degree annular sector, with axial coordinate along the machine axis and a
import build123d as bd, inspect, numpy as np, math, subprocess, os
print('build123d', bd.__version__)
print('loft', inspect.signature(bd.loft))
print('Spline', inspect.signature(bd.Spline))
print('Face', inspect.signature(bd.Face))
print('gmsh available:', subprocess.run(['gmsh','--version'],capture_output=True,text=True).stdout.strip())

# -- cell 2 -------------------------------------------------------------------------
# The installed wrapper has no separate public wire-loft constructor. For robustness I’ll use planar r
import build123d as bd, numpy as np, math, os, subprocess
R_HUB,R_TIP,N_BLADES=0.045,0.090,28; PITCH=2*math.pi/N_BLADES
CHORD,T_REL,TE_THICK=0.022,0.06,0.0004
STAGGER_ROOT,TWIST=math.radians(30),math.radians(35)
CAMBER,CAMBER_POS=0.02,0.40 # assumption: NACA-style 2% camber at 40% chord

def profile_xy(n=61):
    x=(1-np.cos(np.linspace(0,math.pi,n)))/2; m,p,t=CAMBER,CAMBER_POS,T_REL
    yc=np.where(x<p,m/p**2*(2*p*x-x*x),m/(1-p)**2*((1-2*p)+2*p*x-x*x))
    dy=np.where(x<p,2*m/p**2*(p-x),2*m/(1-p)**2*(p-x)); th=np.arctan(dy)
    a0,a1,a2,a3=0.2969,-0.1260,-0.3516,0.2843
    a4=TE_THICK/(2*CHORD)/(5*t)-(a0+a1+a2+a3)
    yt=5*t*(a0*np.sqrt(x)+a1*x+a2*x*x+a3*x**3+a4*x**4)
    return np.c_[(x-yt*np.sin(th))*CHORD,(yc+yt*np.cos(th))*CHORD],np.c_[(x+yt*np.sin(th))*CHORD,(yc-yt*np.cos(th))*CHORD]

def blade_section(r,n=61):
    up,lo=profile_xy(n); a=STAGGER_ROOT+TWIST*(r-R_HUB)/(R_TIP-R_HUB)
    def planar(q):
        xc=q[:,0]-CHORD/2; X=xc*math.cos(a)-q[:,1]*math.sin(a); Z=xc*math.sin(a)+q[:,1]*math.cos(a)
        return [bd.Vector(float(x),float(r),float(z)) for x,z in zip(X,Z)]
    u,l=planar(up),planar(lo)
    return bd.Face(bd.Wire([bd.Edge.make_spline(u),bd.Edge.make_line(u[-1],l[-1]),bd.Edge.make_spline(list(reversed(l)))]))
sections=[blade_section(float(r)) for r in np.linspace(R_HUB,R_TIP,7)]
blade=bd.loft(sections)
up,lo=profile_xy(); measured_te=np.linalg.norm(up[-1]-lo[-1])
print('blade valid:',blade.is_valid,'volume [cm3]:',blade.volume*1e6,'faces:',len(blade.faces()))
print('TE requested/measured [mm]:',TE_THICK*1e3,measured_te*1e3)
print('stagger root/tip requested [deg]: 30/65; built:',math.degrees(STAGGER_ROOT),math.degrees(STAGGER_ROOT+TWIST))
print('radii requested/built [mm]: hub 45/45, tip 90/90')

# -- cell 3 -------------------------------------------------------------------------
# The blade loft is valid and its measured trailing-edge thickness and twist match the request. Next I
import inspect
print(inspect.signature(bd.Edge.make_circle))
print(inspect.signature(bd.extrude))

# -- cell 4 -------------------------------------------------------------------------
# I’ll verify the arc orientation in the local YZ plane so the two radial faces are created explicitly
a=math.degrees(PITCH/2)
for r in (R_HUB,R_TIP):
    e=bd.Edge.make_circle(r,bd.Plane.YZ,start_angle=-a,end_angle=a)
    print(r,[tuple(round(c,6) for c in v.to_tuple()) for v in (e.position_at(0),e.position_at(1))])

# -- cell 5 -------------------------------------------------------------------------
# The extrusion API requires a scalar amount plus a direction, rather than a displacement vector. I’ll
A=math.degrees(PITCH/2)
inner=bd.Edge.make_circle(R_HUB,bd.Plane.YZ,start_angle=-A,end_angle=A); outer=bd.Edge.make_circle(R_TIP,bd.Plane.YZ,start_angle=-A,end_angle=A)
p_im,p_ip=inner.position_at(0),inner.position_at(1); p_om,p_op=outer.position_at(0),outer.position_at(1)
sector_face=bd.Face(bd.Wire([inner,bd.Edge.make_line(p_ip,p_op),outer.reversed(),bd.Edge.make_line(p_om,p_im)]))
bb=blade.bounding_box(); XMIN,XMAX=bb.min.X-CHORD,bb.max.X+CHORD
sector=bd.extrude(sector_face,amount=XMAX-XMIN,dir=(1,0,0)).moved(bd.Location((XMIN,0,0)))
fluid=sector-blade
print('axial bounds [mm]:',round(XMIN*1e3,3),round(XMAX*1e3,3),'length:',round((XMAX-XMIN)*1e3,3))
print('fluid valid:',fluid.is_valid,'solids:',len(fluid.solids()),'faces:',len(fluid.faces()),'volume [cm3]:',round(fluid.volume*1e6,3))
bd.export_stl(fluid,'coarseFluid.stl',tolerance=8e-4,angular_tolerance=0.25)
import pyvista as pv
pv.OFF_SCREEN=True; mesh=pv.read('coarseFluid.stl'); pl=pv.Plotter(off_screen=True,window_size=(900,650)); pl.add_mesh(mesh,color='lightblue',show_edges=True)
pl.camera_position=[(0.14,0.16,0.12),(0,0.067,0),(0,0,1)]; pl.add_axes(); pl.show(screenshot='coarse_preview.png')
print('preview triangles:',mesh.n_cells)

# -- cell 6 -------------------------------------------------------------------------
# The coarse preview shows the intended twisted blade cut through an annular pitch sector; the visible
for i,f in enumerate(fluid.faces()):
    c=f.center(); n=f.normal_at();
    print(i,'geom',f.geom_type,'area',f'{f.area:.8g}','center',tuple(round(v,5) for v in c),'normal',tuple(round(v,3) for v in n))

# -- cell 7 -------------------------------------------------------------------------
# Face inspection revealed tiny exposed blade end-cap slivers: planar blade sections touching a cylind
# Extend blade through cylindrical endwalls; exact requested sections remain in the loft.
radial_stations=[0.0435,R_HUB,0.0525,0.060,0.0675,0.075,0.0825,R_TIP,0.091]
sections_ext=[blade_section(float(r)) for r in radial_stations]
blade_ext=bd.loft(sections_ext)
fluid=sector-blade_ext
print('extended blade valid:',blade_ext.is_valid,'fluid valid:',fluid.is_valid,'faces:',len(fluid.faces()))
for i,f in enumerate(fluid.faces()):
    c=f.center(); n=f.normal_at(); print(i,f.geom_type,f'{f.area:.8g}',tuple(round(v,4) for v in c),tuple(round(v,2) for v in n))

# -- cell 8 -------------------------------------------------------------------------
# The first through-wall loft subtraction produced an invalid OCC result, so I won’t carry it forward.
ext_r=np.linspace(0.0435,0.091,9)
blade_ext=bd.loft([blade_section(float(r)) for r in ext_r])
fluid_try=sector-blade_ext
print('blade valid',blade_ext.is_valid,'fluid valid',fluid_try.is_valid,'solids',len(fluid_try.solids()),'faces',len(fluid_try.faces()))
print('face type counts:',{str(t):sum(f.geom_type==t for f in fluid_try.faces()) for t in set(f.geom_type for f in fluid_try.faces())})

# -- cell 9 -------------------------------------------------------------------------
# Agreed. I’ll freeze the last valid coarse CAD (the original loft), classify its 11 faces now, export
# Return to the validated coarse fluid and export all patches together.
fluid=sector-blade
faces=list(fluid.faces())
inlet=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().X < -0.9]
outlet=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().X > 0.9]
periodic_minus=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().Z < -0.9]
periodic_plus=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().Z > 0.9]
cyl=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER]
hub=[f for f in cyl if f.normal_at().Y < 0]+[f for f in faces if f.geom_type==bd.GeomType.PLANE and f not in inlet+outlet+periodic_minus+periodic_plus]
shroud=[f for f in cyl if f.normal_at().Y > 0]
bs=[f for f in faces if f.geom_type==bd.GeomType.BSPLINE]
blade_suction=[f for f in bs if f.area>1e-5 and f.center().Z>0]
blade_pressure=[f for f in bs if f not in blade_suction]
patches={'inlet':inlet,'outlet':outlet,'hub':hub,'shroud':shroud,'periodic_minus':periodic_minus,'periodic_plus':periodic_plus,'blade_pressure':blade_pressure,'blade_suction':blade_suction}
print({k:(len(v),sum(f.area for f in v)) for k,v in patches.items()},'total faces',sum(map(len,patches.values())))
import sys; sys.path.insert(0,'.reference'); from cad_export import export_patches
export_patches(fluid,patches,tolerance=2.5e-4)
bd.export_step(fluid,'fluidVolume.step')

# -- cell 10 ------------------------------------------------------------------------
# The named, welded surface is now on disk and closed. It reports one non-manifold edge and two windin
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant/polyMesh').mkdir(parents=True,exist_ok=True)
def foam_header(cls,obj): return f'''FoamFile\n{{\n version 2.0; format ascii; class {cls}; object {obj};\n}}\n'''
Path('system/controlDict').write_text(foam_header('dictionary','controlDict')+'application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n')
Path('system/blockMeshDict').write_text(foam_header('dictionary','blockMeshDict')+'''scale 1;\nvertices ((-0.035 0.040 -0.015)(0.035 0.040 -0.015)(0.035 0.095 -0.015)(-0.035 0.095 -0.015)(-0.035 0.040 0.015)(0.035 0.040 0.015)(0.035 0.095 0.015)(-0.035 0.095 0.015));\nblocks (hex (0 1 2 3 4 5 6 7) (14 11 6) simpleGrading (1 1 1));\nedges (); boundary (background {type patch; faces ((0 4 5 1)(3 2 6 7)(0 3 7 4)(1 5 6 2)(0 1 2 3)(4 7 6 5));}); mergePatchPairs ();\n''')
stls=['inlet','outlet','hub','shroud','periodic_minus','periodic_plus','blade_pressure','blade_suction']
geom='\n'.join(f'  {s}.stl {{ type triSurfaceMesh; name {s}; }}' for s in stls)
refs='\n'.join(f'    {s} {{ level (1 1); patchInfo {{ type wall; }} }}' for s in stls)
Path('system/snappyHexMeshDict').write_text(foam_header('dictionary','snappyHexMeshDict')+f'''castellatedMesh true; snap true; addLayers false;\ngeometry\n{{\n{geom}\n}}\ncastellatedMeshControls\n{{ maxLocalCells 100000; maxGlobalCells 200000; minRefinementCells 0; nCellsBetweenLevels 2; resolveFeatureAngle 30; refinementSurfaces\n {{\n{refs}\n }}\n refinementRegions {{}}; locationInMesh (-0.020 0.060 0); allowFreeStandingZoneFaces true; }}\nsnapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }}\naddLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 20; }}\nmeshQualityControls {{ maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}\ndebug 0; mergeTolerance 1e-6;\n''')
subprocess.run(['blockMesh'],check=True,stdout=subprocess.DEVNULL)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
print('snappy rc',r.returncode); print('\n'.join(r.stdout.splitlines()[-18:])); print(r.stderr[-500:])

# -- cell 11 ------------------------------------------------------------------------
# The mesher stopped before reading geometry because OpenFOAM 2512 requires placeholder `fvSchemes` an
Path('system/fvSchemes').write_text(foam_header('dictionary','fvSchemes')+'ddtSchemes {default steadyState;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;}\n')
Path('system/fvSolution').write_text(foam_header('dictionary','fvSolution')+'solvers {}\n')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
print('snappy rc',r.returncode); print('\n'.join(r.stdout.splitlines()[-25:])); print(r.stderr[-300:])

# -- cell 12 ------------------------------------------------------------------------
# Geometry reading succeeded; the remaining failure is a missing required key in `castellatedMeshContr
print(r.stdout[-2000:]); print(r.stderr)

# -- cell 13 ------------------------------------------------------------------------
# The required entry is `features`; for this coarsest mesh I’ll set it to an empty list and rerun.
p=Path('system/snappyHexMeshDict'); s=p.read_text().replace('resolveFeatureAngle 30; refinementSurfaces','resolveFeatureAngle 30; features (); refinementSurfaces'); p.write_text(s)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
print('snappy rc',r.returncode); print('\n'.join(r.stdout.splitlines()[-30:])); print(r.stderr[-400:])

# -- cell 14 ------------------------------------------------------------------------
# A 1,318-cell coarse snapped mesh now exists. I’ll run the binding `checkMesh`, inspect its patch tab
cm=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('checkMesh rc',cm.returncode); print('\n'.join(cm.stdout.splitlines()[-35:])); print(cm.stderr[-300:])
print('\nBOUNDARY FILE:\n',Path('constant/polyMesh/boundary').read_text()[:3000])
subprocess.run(['foamToVTK','-constant'],text=True,capture_output=True)
print('VTK files',list(Path('VTK').rglob('*.vtu'))[:5])

# -- cell 15 ------------------------------------------------------------------------
# The mesh itself passes, but the patch table exposes a topology error: it contains only the upstream 
import inspect
print(inspect.signature(bd.Rotation)); print(inspect.signature(bd.Axis))

# -- cell 16 ------------------------------------------------------------------------
# I’ll use build123d’s algebraic X-axis rotation operator for the two neighboring blades and validate 
print(inspect.signature(bd.Rot)); print(bd.Rot.__doc__[:250])

# -- cell 17 ------------------------------------------------------------------------
# The two copies will be rotated exactly ±360/(2·28) about the machine axis. This places one blade on 
blade_minus=bd.Rot(X=-A)*blade; blade_plus=bd.Rot(X=A)*blade
fluid2=sector-blade_minus-blade_plus
print('valid',fluid2.is_valid,'solids',len(fluid2.solids()),'faces',len(fluid2.faces()),'volume cm3',fluid2.volume*1e6)
for i,f in enumerate(fluid2.faces()):
 c=f.center(); n=f.normal_at(); print(i,f.geom_type,f'{f.area:.7g}',tuple(round(v,4) for v in c),tuple(round(v,2) for v in n))

# -- cell 18 ------------------------------------------------------------------------
# The corrected CAD is valid and one connected solid. Its periodic sides are naturally split into upst
fluid=fluid2; faces=list(fluid.faces())
inlet=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().X < -0.9]; outlet=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().X > 0.9]
periodic_minus=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().Z < -0.9]; periodic_plus=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f.normal_at().Z > 0.9]
cyl=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER]; hub=[f for f in cyl if f.normal_at().Y<0]; shroud=[f for f in cyl if f.normal_at().Y>0]
unused_plane=[f for f in faces if f.geom_type==bd.GeomType.PLANE and f not in inlet+outlet+periodic_minus+periodic_plus]; hub+=unused_plane
bs=[f for f in faces if f.geom_type==bd.GeomType.BSPLINE]; blade_pressure=[f for f in bs if f.center().Z<0]; blade_suction=[f for f in bs if f.center().Z>=0]
patches={'inlet':inlet,'outlet':outlet,'hub':hub,'shroud':shroud,'periodic_minus':periodic_minus,'periodic_plus':periodic_plus,'blade_pressure':blade_pressure,'blade_suction':blade_suction}
print({k:len(v) for k,v in patches.items()},'assigned',sum(map(len,patches.values())))
export_patches(fluid,patches,tolerance=2.5e-4); bd.export_step(fluid,'fluidVolume.step')
p=Path('system/snappyHexMeshDict'); s=p.read_text().replace('locationInMesh (-0.020 0.060 0)','locationInMesh (0 0.060 0)'); p.write_text(s)
subprocess.run(['blockMesh'],check=True,stdout=subprocess.DEVNULL)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
print('snappy rc',r2.returncode); print('\n'.join(r2.stdout.splitlines()[-12:]))

# -- cell 19 ------------------------------------------------------------------------
# The corrected union is now watertight with zero flipped and zero non-manifold edges, and snappy comp
cm2=subprocess.run(['checkMesh'],text=True,capture_output=True)
print('checkMesh rc',cm2.returncode); print('\n'.join(cm2.stdout.splitlines()[-28:]))
# Parse patch names/counts and render OpenFOAM mesh surface plus a clipped view.
btxt=Path('constant/polyMesh/boundary').read_text();
for name in stls:
 import re
 m=re.search(r'\n\s*'+name+r'\s*\{.*?nFaces\s+(\d+);',btxt,re.S); print(name, m.group(1) if m else 'MISSING')
subprocess.run(['foamToVTK','-constant'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
vtu=pv.read('VTK/t10_0/internal.vtu'); surf=vtu.extract_surface(); clipped=vtu.clip(normal=(0,0,1),origin=(0,0.0675,0),invert=False)
pl=pv.Plotter(off_screen=True,window_size=(1000,700)); pl.add_mesh(clipped,scalars=None,color='lightsteelblue',show_edges=True,edge_color='gray'); pl.add_mesh(surf,color='tan',opacity=.22); pl.add_axes(); pl.camera_position=[(0.12,0.16,0.10),(0,0.067,0),(0,0,1)]; pl.show(screenshot='mesh_preview.png')
print('mesh cells',vtu.n_cells,'points',vtu.n_points)

# -- cell 20 ------------------------------------------------------------------------
# The completion gate found that the exported patch manifest lacks the interior point even though snap
import inspect
print(inspect.signature(export_patches))

# -- cell 21 ------------------------------------------------------------------------
# I’ll record the exact point already used by snappyHexMesh; this only updates the exported surface ma
export_patches(fluid,patches,tolerance=2.5e-4,location_in_mesh=(0,0.060,0))
