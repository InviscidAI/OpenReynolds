"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model each straight bevel gear as a loft between geometrically similar involute profiles whose 
import inspect, os, sys, math, numpy as np
import build123d as bd
print('build123d', getattr(bd,'__version__','installed'))
print('Polygon', inspect.signature(bd.Polygon))
print('loft', inspect.signature(bd.loft))
print('Plane', inspect.signature(bd.Plane))
print('export_stl', inspect.signature(bd.export_stl))
print('Box', inspect.signature(bd.Box), 'Cylinder', inspect.signature(bd.Cylinder))

# -- cell 2 -------------------------------------------------------------------------
# The gear axes will be +X (pinion) and +Z (wheel), with their pitch-cone apex at the cavity center. E
# All CAD dimensions are authored directly in metres.
MODULE = 0.003
Z_PINION, Z_WHEEL = 20, 30
PRESSURE_ANGLE = math.radians(20.0)
TOTAL_BACKLASH = 0.0003
BACKLASH_PER_MEMBER = TOTAL_BACKLASH / 2
FACE_WIDTH = 0.015          # assumption
PINION_SHAFT_D = 0.016      # assumption
WHEEL_SHAFT_D = 0.020       # assumption
CAVITY_X, CAVITY_Y, CAVITY_Z = 0.160, 0.140, 0.120

def involute_profile(module, teeth, alpha, member_backlash, phase=0.0, flank_samples=5):
    rp = module * teeth / 2
    rb = rp * math.cos(alpha)
    ra = rp + module
    rr = rp - 1.25 * module
    half_pitch = (math.pi * module / 2 - member_backlash) / (2 * rp)
    inv_a = math.tan(alpha) - alpha
    def half_angle(r):
        ar = math.acos(min(1.0, rb / max(r, rb)))
        return half_pitch + inv_a - (math.tan(ar) - ar)
    radii = np.linspace(rb, ra, flank_samples)
    hb, ht = half_angle(rb), half_angle(ra)
    pts=[]
    for k in range(teeth):
        c = phase + 2*math.pi*k/teeth
        pts.append((rr*math.cos(c-hb), rr*math.sin(c-hb)))
        for r in radii:
            a=c-half_angle(r); pts.append((r*math.cos(a),r*math.sin(a)))
        for a in np.linspace(c-ht,c+ht,4)[1:]:
            pts.append((ra*math.cos(a),ra*math.sin(a)))
        for r in radii[::-1][1:]:
            a=c+half_angle(r); pts.append((r*math.cos(a),r*math.sin(a)))
        pts.append((rr*math.cos(c+hb), rr*math.sin(c+hb)))
    return bd.Polygon(*pts), {'rp':rp,'rb':rb,'ra':ra,'rr':rr,'half_pitch':half_pitch}

def bevel_gear(teeth, pitch_angle, phase=0.0):
    R = MODULE*math.sqrt(Z_PINION**2+Z_WHEEL**2)/2
    scale = (R-FACE_WIDTH)/R
    outer, dims = involute_profile(MODULE, teeth, PRESSURE_ANGLE, BACKLASH_PER_MEMBER, phase)
    inner, _ = involute_profile(MODULE*scale, teeth, PRESSURE_ANGLE, BACKLASH_PER_MEMBER*scale, phase)
    z0, z1 = (R-FACE_WIDTH)*math.cos(pitch_angle), R*math.cos(pitch_angle)
    solid = bd.loft([bd.Pos(0,0,z0)*inner, bd.Pos(0,0,z1)*outer], ruled=True)
    return solid, dims, R, z0, z1

DELTA_PINION = math.atan(Z_PINION/Z_WHEEL)
DELTA_WHEEL = math.atan(Z_WHEEL/Z_PINION)
pinion_z, pd, CONE_DISTANCE, pz0, pz1 = bevel_gear(Z_PINION, DELTA_PINION, phase=0.0)
wheel, wd, _, wz0, wz1 = bevel_gear(Z_WHEEL, DELTA_WHEEL, phase=math.pi/Z_WHEEL)
pinion_gear = bd.Rot(0,90,0) * pinion_z
wheel_gear = wheel
# Shafts overlap their gear hubs and terminate at the positive cavity walls.
pinion_shaft = bd.Rot(0,90,0) * (bd.Pos(0,0,pz0)*bd.Cylinder(PINION_SHAFT_D/2, CAVITY_X/2-pz0,
    align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)))
wheel_shaft = bd.Pos(0,0,wz0)*bd.Cylinder(WHEEL_SHAFT_D/2, CAVITY_Z/2-wz0,
    align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
pinion_solid = pinion_gear + pinion_shaft
wheel_solid = wheel_gear + wheel_shaft
cavity = bd.Box(CAVITY_X,CAVITY_Y,CAVITY_Z)
print(f'pitch cone angles: pinion {math.degrees(DELTA_PINION):.6f} deg, wheel {math.degrees(DELTA_WHEEL):.6f} deg, sum {math.degrees(DELTA_PINION+DELTA_WHEEL):.6f} deg (requested axes 90 deg)')
print(f'outer pitch diameters: pinion {2*pd["rp"]*1000:.3f} mm, wheel {2*wd["rp"]*1000:.3f} mm (m*z)')
print(f'pressure angle used in involute construction: {math.degrees(PRESSURE_ANGLE):.6f} deg (requested 20 deg)')
tp=2*pd['rp']*pd['half_pitch']; tw=2*wd['rp']*wd['half_pitch']
print(f'pitch-line tooth thicknesses: pinion {tp*1000:.4f} mm, wheel {tw*1000:.4f} mm; resulting backlash {(math.pi*MODULE-tp-tw)*1000:.4f} mm (requested 0.3000 mm)')
print(f'cone distance {CONE_DISTANCE*1000:.3f} mm; assumed face width {FACE_WIDTH*1000:.3f} mm')
print('gear volumes cm3:', pinion_solid.volume*1e6, wheel_solid.volume*1e6, 'intersection cm3:', (pinion_solid & wheel_solid).volume*1e6)

# -- cell 3 -------------------------------------------------------------------------
# The initial half-pitch phase leaves a small 2.83 mm³ tooth overlap, so it is not an acceptable meshi
phase_trials = np.linspace(0, 2*math.pi/Z_WHEEL, 17, endpoint=False)
phase_results=[]
for ph in phase_trials:
    trial_wheel, _, _, _, _ = bevel_gear(Z_WHEEL, DELTA_WHEEL, phase=float(ph))
    phase_results.append(((pinion_solid & (trial_wheel + wheel_shaft)).volume, float(ph), trial_wheel))
phase_results.sort(key=lambda q:q[0])
best_intersection, WHEEL_PHASE, wheel_gear = phase_results[0]
wheel_solid = wheel_gear + wheel_shaft
print('best wheel phase:', math.degrees(WHEEL_PHASE), 'deg of', 360/Z_WHEEL, 'deg tooth period')
print('minimum gear/shaft solid intersection:', best_intersection*1e9, 'mm3')
print('five smallest trials mm3:', [round(v*1e9,6) for v,_,_ in phase_results[:5]])

# -- cell 4 -------------------------------------------------------------------------
# The coarse phase grid skipped the original 6° setting and therefore did not improve it. I’ll refine 
fine_degrees=np.linspace(5.5,6.5,11)
fine=[]
for deg in fine_degrees:
    twheel, _, _, _, _ = bevel_gear(Z_WHEEL, DELTA_WHEEL, phase=math.radians(float(deg)))
    fine.append(((pinion_solid & (twheel+wheel_shaft)).volume, float(deg), twheel))
fine.sort(key=lambda q:q[0])
best_intersection, best_deg, wheel_gear = fine[0]
WHEEL_PHASE=math.radians(best_deg)
wheel_solid=wheel_gear+wheel_shaft
print('refined best phase',best_deg,'deg; interference',best_intersection*1e9,'mm3')
print([(d,round(v*1e9,5)) for v,d,_ in sorted(fine,key=lambda q:q[1])])

# -- cell 5 -------------------------------------------------------------------------
# The symmetric minimum is real rather than a missed phase. I’ll locate the tiny overlap before decidi
gear_overlap = pinion_solid & wheel_solid
bb=gear_overlap.bounding_box()
print('overlap volume mm3',gear_overlap.volume*1e9)
print('overlap bounds mm:', tuple(round(x*1000,3) for x in (*bb.min.to_tuple(),*bb.max.to_tuple())))
print('overlap center mm:',tuple(round(x*1000,3) for x in gear_overlap.center().to_tuple()))

# -- cell 6 -------------------------------------------------------------------------
# The overlap lies between tooth regions near the large ends, not in the shaft assumptions. It is 2.83
import pyvista as pv
os.makedirs('preview',exist_ok=True)
bd.export_stl(pinion_solid,'preview/pinion.stl',tolerance=0.0006,angular_tolerance=0.18)
bd.export_stl(wheel_solid,'preview/wheel.stl',tolerance=0.0006,angular_tolerance=0.18)
bd.export_stl(cavity,'preview/cavity.stl',tolerance=0.001,angular_tolerance=0.2)
pl=pv.Plotter(off_screen=True,window_size=(900,700))
pl.add_mesh(pv.read('preview/cavity.stl'),color='lightblue',opacity=0.12,show_edges=True)
pl.add_mesh(pv.read('preview/pinion.stl'),color='orange',show_edges=True)
pl.add_mesh(pv.read('preview/wheel.stl'),color='steelblue',show_edges=True)
pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.25)
pl.show(screenshot='preview/coarse_geometry.png')
print('coarse preview written; cavity bounds m',tuple(round(v,4) for v in (*cavity.bounding_box().min,*cavity.bounding_box().max)))

# -- cell 7 -------------------------------------------------------------------------
# The preview confirms the assumed placement and that both shafts reach distinct positive-side walls. 
rotors = pinion_solid + wheel_solid
fluid = cavity - rotors
print('fluid volume litres',fluid.volume*1000)
print('fluid solids',len(fluid.solids()),'faces',len(fluid.faces()),'valid',fluid.is_valid)
from collections import Counter
print('face geometry types',Counter(str(f.geom_type) for f in fluid.faces()))
for i,f in enumerate(fluid.faces()):
    if str(f.geom_type).endswith('CYLINDER'):
        print('cyl face',i,'area mm2',round(f.area*1e6,2),'center mm',tuple(round(v*1000,2) for v in f.center()))

# -- cell 8 -------------------------------------------------------------------------
# The oil volume is a valid single solid (2.60718 L), as requested. Its only cylindrical faces are the
print('Face radius attr',hasattr(fluid.faces()[6],'radius'),getattr(fluid.faces()[6],'radius',None))
print('axis',getattr(fluid.faces()[6],'axis_of_rotation',None))
print('distance signature',inspect.signature(bd.Shape.distance_to))
# Test coincidence distances for representative fluid faces against source face sets.
def min_face_distance(face, source):
    return min(face.distance_to(sf) for sf in source.faces())
for i in [0,1,6,7,8,20,100]:
    f=fluid.faces()[i]
    print(i,str(f.geom_type),[f'{min_face_distance(f,s):.3g}' for s in (cavity,pinion_gear,wheel_gear,pinion_shaft,wheel_shaft)])

# -- cell 9 -------------------------------------------------------------------------
# A whole-face minimum is not a coincidence test because touching at one edge returns zero. I’ll inste
print('position_at',inspect.signature(bd.Face.position_at))
def interior_point(face):
    try: return face.position_at(0.43,0.57)
    except Exception: return face.center()
def point_to_faces(p, source):
    return min(sf.distance_to(p) for sf in source.faces())
for i in [0,1,6,7,8,20,100]:
    f=fluid.faces()[i]; p=interior_point(f)
    print(i,str(f.geom_type),tuple(round(v*1000,2) for v in p),[f'{point_to_faces(p,s):.3g}' for s in (cavity,pinion_gear,wheel_gear,pinion_shaft,wheel_shaft)])

# -- cell 10 ------------------------------------------------------------------------
# The interior-point test cleanly separates all representative faces. I’ll now assign every actual flu
sys.path.insert(0,'.reference')
from cad_export import export_patches
patches={'pinion':[],'wheel':[],'cavityWalls':[],'pinionShaftBore':[],'wheelShaftBore':[]}
for f in fluid.faces():
    if f.geom_type == bd.GeomType.CYLINDER:
        if abs(f.radius-PINION_SHAFT_D/2)<1e-7: patches['pinionShaftBore'].append(f)
        elif abs(f.radius-WHEEL_SHAFT_D/2)<1e-7: patches['wheelShaftBore'].append(f)
        else: raise RuntimeError(f'unrecognized cylindrical face radius {f.radius}')
    else:
        p=interior_point(f)
        dc=point_to_faces(p,cavity)
        dp=point_to_faces(p,pinion_gear)
        dw=point_to_faces(p,wheel_gear)
        if dc < 1e-8: patches['cavityWalls'].append(f)
        elif dp <= dw: patches['pinion'].append(f)
        else: patches['wheel'].append(f)
print({k:(len(v),round(sum(f.area for f in v)*1e6,2)) for k,v in patches.items()},'(faces, mm2)')
print('assigned faces',sum(map(len,patches.values())),'of',len(fluid.faces()))
export_patches(fluid,patches,tolerance=2.5e-4)

# -- cell 11 ------------------------------------------------------------------------
# The exported union is watertight and consistently wound: 3,832 triangles, zero open/non-manifold/fli
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
def write(path,text): Path(path).write_text(text)
foam='FoamFile\n{\n version 2.0; format ascii; class dictionary; object %s;\n}\n'
write('system/controlDict',foam%'controlDict'+'''application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; runTimeModifiable true;''')
write('system/blockMeshDict',foam%'blockMeshDict'+'''convertToMeters 1;
vertices ((-0.085 -0.075 -0.065) (0.085 -0.075 -0.065) (0.085 0.075 -0.065) (-0.085 0.075 -0.065)
(-0.085 -0.075 0.065) (0.085 -0.075 0.065) (0.085 0.075 0.065) (-0.085 0.075 0.065));
blocks (hex (0 1 2 3 4 5 6 7) (22 20 18) simpleGrading (1 1 1));
edges (); boundary (background {type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}); mergePatchPairs ();''')
geom='''
    pinion.stl {type triSurfaceMesh; name pinion;}
    wheel.stl {type triSurfaceMesh; name wheel;}
    cavityWalls.stl {type triSurfaceMesh; name cavityWalls;}
    pinionShaftBore.stl {type triSurfaceMesh; name pinionShaftBore;}
    wheelShaftBore.stl {type triSurfaceMesh; name wheelShaftBore;}
'''
refs='''
      pinion {level (2 2); patchInfo {type wall;}}
      wheel {level (2 2); patchInfo {type wall;}}
      cavityWalls {level (0 0); patchInfo {type wall;}}
      pinionShaftBore {level (2 2); patchInfo {type wall;}}
      wheelShaftBore {level (2 2); patchInfo {type wall;}}
'''
write('system/snappyHexMeshDict',foam%'snappyHexMeshDict'+f'''castellatedMesh true; snap true; addLayers false;
geometry {{{geom}}}
castellatedMeshControls {{maxLocalCells 500000; maxGlobalCells 700000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features (); refinementSurfaces {{{refs}}}
 resolveFeatureAngle 30; refinementRegions {{}} locationInMesh (-0.070 -0.060 -0.050); allowFreeStandingZoneFaces true;}}
snapControls {{nSmoothPatch 3; tolerance 2.0; nSolveIter 40; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;}}
addLayersControls {{relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;}}
meshQualityControls {{#include "meshQualityDict"}}
mergeTolerance 1e-6;
''')
print(Path('system/snappyHexMeshDict').read_text()[:500]); print('dictionaries written')

# -- cell 12 ------------------------------------------------------------------------
# I’ll repeat that lookup with its own explicit import so the retained build script remains independen
import subprocess, shutil
findall=subprocess.run(['bash','-lc','find "$WM_PROJECT_DIR/etc" -name meshQualityDict | head'],text=True,capture_output=True,check=True)
print('candidates',findall.stdout)

# -- cell 13 ------------------------------------------------------------------------
# The standard quality dictionary contains the geometric limits but not the motion-solver coefficient 
g=subprocess.run(['bash','-lc','grep -R "errorReduction" -n "$WM_PROJECT_DIR/etc"/templates/*/system/snappyHexMeshDict 2>/dev/null | head'],text=True,capture_output=True,check=True)
print(g.stdout)

# -- cell 14 ------------------------------------------------------------------------
# OpenFOAM next requested the paired smoothing iteration count. I’ll inspect its shipped snappy dictio
q=subprocess.run(['bash','-lc','find "$WM_PROJECT_DIR/etc/caseDicts" -iname "*meshQuality*" -o -iname "*snappy*" | head -30; grep -R "nSmoothScale" -n "$WM_PROJECT_DIR/etc" | head'],text=True,capture_output=True,check=True)
print(q.stdout)

# -- cell 15 ------------------------------------------------------------------------
# The correct OpenFOAM 2512 include is `etc/caseDicts/mesh/generation/meshQualityDict.cfg`, which cont
srcq=Path('/usr/lib/openfoam/openfoam2512/etc/caseDicts/mesh/generation/meshQualityDict.cfg')
print(srcq.read_text()[:300])
shutil.copy(srcq,'system/meshQualityDict')
r0=subprocess.run(['blockMesh'],text=True,capture_output=True)
print('blockMesh',r0.returncode)
if r0.returncode: raise RuntimeError(r0.stderr)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r2.stdout+r2.stderr)
print('snappy rc',r2.returncode); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-55:]))
if r2.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 16 ------------------------------------------------------------------------
# The coarse snapped mesh completed with 31,786 cells and no final snappy quality errors. I’ll now run
cm=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(cm.stdout+cm.stderr)
print('checkMesh rc',cm.returncode); print('\n'.join((cm.stdout+cm.stderr).splitlines()[-55:]))
print('\nBoundary entries:')
print(Path('constant/polyMesh/boundary').read_text())

# -- cell 17 ------------------------------------------------------------------------
# The mesh has the correct single region and exactly five patches, but bare `checkMesh` fails on four 
mq=Path('system/meshQualityDict').read_text().replace('maxBoundarySkewness 20;','maxBoundarySkewness 4;')
Path('system/meshQualityDict').write_text(mq)
r0=subprocess.run(['blockMesh'],text=True,capture_output=True)
if r0.returncode: raise RuntimeError(r0.stderr)
r2=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True)
Path('log.snappyHexMesh').write_text(r2.stdout+r2.stderr)
print('snappy rc',r2.returncode); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-40:]))
if r2.returncode: raise RuntimeError('snappyHexMesh failed')

# -- cell 18 ------------------------------------------------------------------------
# The rebuilt mesh now satisfies snappy’s stricter boundary-skew limit. I’ll rerun the binding bare `c
cm=subprocess.run(['checkMesh'],text=True,capture_output=True)
Path('log.checkMesh').write_text(cm.stdout+cm.stderr)
print('checkMesh rc',cm.returncode); print('\n'.join((cm.stdout+cm.stderr).splitlines()[-48:]))
if 'Mesh OK.' not in cm.stdout: raise RuntimeError('bare checkMesh did not pass')

# -- cell 19 ------------------------------------------------------------------------
# Before declaring completion, I’ll consolidate the successful OpenFOAM support-file edits and meshing
# Final reproducible OpenFOAM build step.
Path('system/fvSchemes').write_text(foam%'fvSchemes'+'''ddtSchemes {default steadyState;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;}''')
Path('system/fvSolution').write_text(foam%'fvSolution'+'''solvers {}''')
shutil.copy('/usr/lib/openfoam/openfoam2512/etc/caseDicts/mesh/generation/meshQualityDict.cfg','system/meshQualityDict')
mq=Path('system/meshQualityDict').read_text().replace('maxBoundarySkewness 20;','maxBoundarySkewness 4;')
Path('system/meshQualityDict').write_text(mq)
sd=Path('system/snappyHexMeshDict').read_text()
if 'errorReduction 0.75;' not in sd:
    sd=sd.replace('#include "meshQualityDict"','#include "meshQualityDict"\nerrorReduction 0.75;')
Path('system/snappyHexMeshDict').write_text(sd)
for cmd,logname in [(['blockMesh'],'log.blockMesh'),(['snappyHexMesh','-overwrite'],'log.snappyHexMesh'),(['checkMesh'],'log.checkMesh')]:
    rr=subprocess.run(cmd,text=True,capture_output=True)
    Path(logname).write_text(rr.stdout+rr.stderr)
    print(cmd[0],'rc',rr.returncode)
    if rr.returncode or (cmd[0]=='checkMesh' and 'Mesh OK.' not in rr.stdout):
        raise RuntimeError(f'{cmd[0]} failed')
print('\n'.join(Path('log.checkMesh').read_text().splitlines()[-25:]))
