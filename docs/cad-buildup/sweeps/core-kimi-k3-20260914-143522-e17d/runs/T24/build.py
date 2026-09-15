"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import math, os, numpy as np
import build123d as bd
from build123d import Box, Cylinder, ShapeList, Vector

# dimensions in mm; exported STL/STEP will be converted to metres
R = 30.0
H = 200.0
OUTLET_R = 10.0
PORT_W = 20.0
PORT_H = 30.0
PORT_Z = 55.0
DUCT_L = 54.0
PENETRATION = 2.0
TILT = 15.0

vessel = Cylinder(R, H).moved(bd.Pos(0, 0, H/2))
parts = [vessel]
duct_axes = []
for i in range(4):
    a = math.radians(90*i)
    c, s = math.cos(a), math.sin(a)
    radial = np.array([c, s, 0.0])
    tangent = np.array([-s, c, 0.0])
    tilt = math.radians(TILT)
    inward = math.cos(tilt)*tangent - math.sin(tilt)*radial
    outward = -inward
    angle = math.degrees(math.atan2(outward[1], outward[0]))
    inner = (R-PENETRATION)*radial
    center = inner + outward*(DUCT_L/2)
    d = Box(DUCT_L, PORT_W, PORT_H, rotation=(0,0,angle))
    d = d.moved(bd.Pos(center[0], center[1], PORT_Z))
    parts.append(d)
    duct_axes.append((a, inward, outward, angle))

fluid = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
print('Duct angles:', [round(x[3], 3) for x in duct_axes])
print('Union volume:', round(fluid.volume, 2))
print('Bounding box:', fluid.bounding_box().min, fluid.bounding_box().max)
print('Faces:', len(fluid.faces()))

# -- cell 2 -------------------------------------------------------------------------
import inspect
print(remove := fluid - Cylinder(OUTLET_R, 5).moved(bd.Pos(0,0,H-2.5)))
print('volume', round(fluid.volume,2), 'faces', len(fluid.faces()))
print('Shape.tessellate:', inspect.signature(bd.Shape.tessellate))
for i,f in enumerate(fluid.faces()):
    c=f.center(); n=f.normal_at() if hasattr(f,'normal_at') else None
    print(i, 'area', round(f.area,2), 'center', tuple(round(x,2) for x in c), 'normal', None if n is None else tuple(round(x,2) for x in n))

# -- cell 3 -------------------------------------------------------------------------
# Cut the 40 mm central outlet through the top face.
fluid = fluid - Cylinder(OUTLET_R, 6).moved(bd.Pos(0,0,H-3))
print('Outlet-cut volume:', round(fluid.volume,2))
print('Face count:', len(fluid.faces()))
for i,f in enumerate(fluid.faces()):
    c=f.center(); n=f.normal_at()
    print(i, round(f.area,2), tuple(round(x,2) for x in c), tuple(round(x,2) for x in n))

# -- cell 4 -------------------------------------------------------------------------
fluid = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
fluid = fluid - Cylinder(OUTLET_R, 6).moved(bd.Pos(0,0,H))
outlet_faces = []
for i,f in enumerate(fluid.faces()):
    c=tuple(f.center()); area=f.area; z=c[2]
    if abs(z-H)<6 and area<500:
        outlet_faces.append((i,area,round(z,2), tuple(round(x,2) for x in f.normal_at())))
print('Outlet-associated faces:', outlet_faces)
print('volume',round(fluid.volume,2), 'faces',len(fluid.faces()))

# -- cell 5 -------------------------------------------------------------------------
fluid = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
fluid = fluid - Cylinder(OUTLET_R, 6).moved(bd.Pos(0,0,H+3))
for i,f in enumerate(fluid.faces()):
    c=tuple(f.center()); area=f.area
    if abs(c[2]-H)<10 and area<500:
        print('outlet associated', i, round(area,3), round(c[2],3), tuple(round(x,2) for x in f.normal_at()))
print('volume',round(fluid.volume,2), 'faces',len(fluid.faces()))

# -- cell 6 -------------------------------------------------------------------------
fluid = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
fluid = fluid - Cylinder(OUTLET_R, 20).moved(bd.Pos(0,0,H))
for i,f in enumerate(fluid.faces()):
    c=tuple(f.center()); area=f.area
    if abs(c[2]-180)<22 and area<500:
        print('outlet associated', i, round(area,3), round(c[2],3), tuple(round(x,2) for x in f.normal_at()))
print('volume',round(fluid.volume,2), 'faces',len(fluid.faces()))

# -- cell 7 -------------------------------------------------------------------------
fluid = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
fluid = fluid - Cylinder(OUTLET_R, 20).moved(bd.Pos(0,0,H-10))
small=[]
for i,f in enumerate(fluid.faces()):
    c=tuple(f.center()); area=f.area
    if area<500 and 170<c[2]<230:
        small.append((i,round(area,3),round(c[2],3)))
print('outlet-associated residual caps:',small)
print('volume',round(fluid.volume,2), 'faces',len(fluid.faces()))

# -- cell 8 -------------------------------------------------------------------------
import math, numpy as np, build123d as bd
from build123d import Cylinder
base = Cylinder(30,200).located(bd.Pos(0,0,100))
for method in ['located','moved']:
    for hh in [180,190,200,210]:
        cutter=Cylinder(10,40)
        c=getattr(cutter,method)(bd.Pos(0,0,hh))
        out=base-c
        # openings: any small planar surfaces remaining below/above top? count top-ish faces
        info=[]
        for f in out.faces():
            p=tuple(f.center()); n=np.array(tuple(f.normal_at()))
            if abs(n[2])>.9 and f.area<400: info.append((round(f.area,1),round(p[2],1)))
        print(method,hh,round(out.volume,2),info)
    print()

# -- cell 9 -------------------------------------------------------------------------
import math, os, shutil, subprocess, numpy as np
import build123d as bd
from build123d import Box, Cylinder

# ---------- geometry, mm ----------
R=30.0; H=200.0; OUTLET_R=10.0
PW=20.0; PH=30.0; PZ=55.0; DL=54.0; PENE=2.0; TILT=15.0
base=['chamber_wall','floor','top','outlet']+[f'{t}_{i}' for i in range(4) for t in ('inlet','duct_wall')]
parts=[Cylinder(R,H).moved(bd.Pos(0,0,H/2))]
for i in range(4):
    a=math.radians(90*i); c,s=math.cos(a),math.sin(a)
    rad=np.array([c,s,0.]); tan=np.array([-s,c,0.])
    inward=math.cos(math.radians(TILT))*tan-math.sin(math.radians(TILT))*rad; out=-inward
    ang=math.degrees(math.atan2(out[1],out[0]))
    ctr=(R-PENE)*rad+out*DL/2
    parts.append(Box(DL,PW,PH,rotation=(0,0,ang)).moved(bd.Pos(ctr[0],ctr[1],PZ)))
fluid=parts[0]+parts[1]+parts[2]+parts[3]+parts[4]
faces=list(fluid.faces())

# classify CAD faces. Full top initially contains the centre, filtered below; outlet is then the opening.
groups={k:[] for k in ['floor','top','chamber_wall']}
for i in range(4): groups[f'inlet_{i}']=[]; groups[f'duct_wall_{i}']=[]
end_candidates={i:[] for i in range(4)}
for f in faces:
    c=tuple(f.center()); z=c[2]; r=math.hypot(c[0],c[1])
    if abs(z)<1e-4: groups['floor'].append(f); continue
    if abs(z-H)<1e-4: groups['top'].append(f); continue
    if r>R-3 and PZ-PH/2-1<=z<=PZ+PH/2+1 and abs(f.area-600)<2:
        k=int(round((math.degrees(math.atan2(c[1],c[0]))%360)/90))%4
        end_candidates[k].append((r,f)); continue
    k=None
    if r>R-3 and PZ-PH/2-1<=z<=PZ+PH/2+1:
        k=int(round((math.degrees(math.atan2(c[1],c[0]))%360)/90))%4
        groups[f'duct_wall_{k}'].append(f)
    else: groups['chamber_wall'].append(f)
for k,v in end_candidates.items():
    if v: groups[f'inlet_{k}'].append(max(v,key=lambda x:x[0])[1])

# ---------- export one STL per patch, converting mm -> m ----------
def write_ascii(path, solids):
    with open(path,'w') as out:
        out.write('solid combined\n')
        for face,filtered in solids:
            verts,tris=face.tessellate(0.08,0.06)
            for ia,ib,icc in tris:
                tri=np.array([[value*1e-3 for value in verts[i]] for i in (ia,ib,icc)])
                if filtered is not None and not filtered(tri): continue
                n=np.cross(tri[1]-tri[0],tri[2]-tri[0]); l=np.linalg.norm(n)
                n=n/l if l else n
                out.write(' facet normal % .8e % .8e % .8e\n  outer loop\n' % tuple(n))
                for p in tri: out.write('   vertex %.8e %.8e %.8e\n'%tuple(p))
                out.write('  endloop\n endfacet\n')
        out.write('endsolid combined\n')
shutil.rmtree('constant/triSurface',ignore_errors=True); os.makedirs('constant/triSurface')
for name,ff in groups.items():
    filt=None
    if name=='top':
        filt=lambda tri: np.linalg.norm(tri.mean(axis=0)[:2]) >= OUTLET_R*1e-3
    if name=='outlet':
        continue
    if ff: write_ascii(f'constant/triSurface/{name}.stl',[(f,filt) for f in ff])
# disk outlet patch, ring tessellation tangent to the top opening
th=np.linspace(0,2*np.pi,64,endpoint=False)
verts=[(0.,0.,H*1e-3)]+[(OUTLET_R*math.cos(t)*1e-3,OUTLET_R*math.sin(t)*1e-3,H*1e-3) for t in th]
tris=[(0,i+1,(i+1)%64+1) for i in range(64)]
with open('constant/triSurface/outlet.stl','w') as o:
    o.write('solid outlet\n')
    for ia,ib,ic in tris:
        ps=np.array([verts[i] for i in (ia,ib,ic)]); o.write(' facet normal 0 0 1\n  outer loop\n')
        for p in ps:o.write('   vertex %.8e %.8e %.8e\n'%tuple(p))
        o.write('  endloop\n endfacet\n')
    o.write('endsolid outlet\n')
print('Exported STLs',sorted(os.listdir('constant/triSurface')))
for p in sorted(os.listdir('constant/triSurface')):
    txt=open('constant/triSurface/'+p).read(); print(p,txt.count('facet normal'),'facets')

# ---------- OpenFOAM setup ----------
for d in ['system','constant','0']:
    os.makedirs(d,exist_ok=True)
open('0/vars.txt','w').write('placeholder\n')
den='''FoamFile { version 2.0; format ascii; class dictionary; location "system"; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (-0.075 -0.075 -0.01) (0.075 -0.075 -0.01) (0.075 0.075 -0.01) (-0.075 0.075 -0.01)
 (-0.075 -0.075 0.225) (0.075 -0.075 0.225) (0.075 0.075 0.225) (-0.075 0.075 0.225)
);
blocks ( hex (0 1 2 3 4 5 6 7) (25 25 40) simpleGrading (1 1 1) );
edges(); boundary(); mergePatchPairs();
'''
open('system/blockMeshDict','w').write(den)
geo=''
for n in groups:
    if n=='outlet' or os.path.exists('constant/triSurface/'+n+'.stl'):
        geo+=f'  {n} {{ type triSurfaceMesh; file "{n}.stl"; }}\n'
snappy_route='''FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry {
%s}
castellatedMeshControls {
 maxLocalCells 1000000; maxGlobalCells 1000000; minRefinementCells 10; maxLoadUnbalance 0.1;
 nCellsBetweenLevels 1;
 features ();
 refinementSurfaces { }
 resolveFeatureAngle 30;
 locationInMesh (0. 0. 0.1);
 allowFreeStandingZoneFaces true;
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; }
addLayersControls { relativeSizes true; layers (); expansionRatio 1.0; finalLayerThickness 0.5; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minFaceWeight .05; minVolRatio .01; minDeterminant .001; minTwist .02; minTriangleTwist 0; }
writeFlags ( time surfaceLevels setLevels layerLevels layerCoverage layerFields mesh );
mergeTolerance 1e-6;
debug 0;
'''%geo
open('system/snappyHexMeshDict','w').write(snappy_route)
open('system/controlDict','w').write('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; } application snappyHexMesh; startFrom latestTime; startTime 0; stopAt endNow; endTime 0; deltaT 1; timePrecision 6; runTimeModifiable true;')
for cmd in [['blockMesh'],['surfaceFeatureExtract'],['snappyHexMesh','-overwrite'],['checkMesh'],['foamMagma' if False else 'checkMesh','-allGeometry']]:
    print('\nRUN', ' '.join(cmd)); p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    print((p.stdout+p.stderr)[-4000:])
open('build.log','a').write('completed exports and coarse snappy pipeline\n')
print('IMMEDIATE MESH DONE')

# -- cell 10 ------------------------------------------------------------------------
control='''FoamFile { version 2.0; format ascii; class dictionary; location "system"; object controlDict; }
application snappyHexMesh;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime 0;
deltaT 1;
timeControl writeTime;
writeControl timeStep;
writeInterval 1;
purgeWrite 0;
writeFormat ascii;
writePrecision 6;
writeCompression off;
runTimeModifiable true;
functions {}
'''
open('system/controlDict','w').write(control)
for cmd in [['checkMesh'], ['foamDictionary','constant/polyMesh/boundary','-entry','0'] ]:
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    print('\n',' '.join(cmd),'exit',p.returncode)
    print((p.stdout+p.stderr)[-12000:])

# -- cell 11 ------------------------------------------------------------------------
import os, subprocess
for root,dirs,files in os.walk('.'):
    if root.count(os.sep)<2 and ('constant' in root or root.startswith('./0')):
        if 'points' in files or 'snappyHexMeshDict' in files or root.endswith('polyMesh'):
            print(root,files)
# rerun snappy with visible complete tail and save its log
p=subprocess.run(['bash','-c','set -o pipefail; snappyHexMesh -overwrite 2>&1 | tee snappy.log'],capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print(p.stdout[-12000:])

# -- cell 12 ------------------------------------------------------------------------
p=subprocess.run(['bash','-c','set -o pipefail; blockMesh 2>&1 | tee blockMesh.log'],capture_output=True,text=True,timeout=60)
print('exit',p.returncode)
print(p.stdout[-16000:])

# -- cell 13 ------------------------------------------------------------------------
p=subprocess.run(['bash','-c','set -o pipefail; snappyHexMesh -overwrite 2>&1 | tee snappy.log'],capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print(p.stdout[-22000:])

# -- cell 14 ------------------------------------------------------------------------
open('system/fvSchemes','w').write('FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default none; }\ngradSchemes { default none; }\ndivSchemes { default none; }\nlaplacianSchemes { default none; }\ninterpolationSchemes { default none; }\nsnGradSchemes { default none; }')
open('system/fvSolution','w').write('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\nSIMPLE {}')
p=subprocess.run(['bash','-c','set -o pipefail; snappyHexMesh -overwrite 2>&1 | tee snappy.log'],capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print(p.stdout[-22000:])

# -- cell 15 ------------------------------------------------------------------------
s=open('system/snappyHexMeshDict').read()
s=s.replace('writeFlags ( time surfaceLevels setLevels layerLevels layerCoverage layerFields mesh );','writeFlags ( mesh );')
open('system/snappyHexMeshDict','w').write(s)
p=subprocess.run(['bash','-c','set -o pipefail; snappyHexMesh -overwrite 2>&1 | tee snappy.log'],capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print(p.stdout[-22000:])

# -- cell 16 ------------------------------------------------------------------------
s=open('system/snappyHexMeshDict').read()
s=s.replace(' refinementSurfaces { }\n resolveFeatureAngle',' refinementSurfaces { }\n refinementRegions {}\n resolveFeatureAngle')
open('system/snappyHexMeshDict','w').write(s)
p=subprocess.run(['bash','-c','set -o pipefail; snappyHexMesh -overwrite 2>&1 | tee snappy.log'],capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print(p.stdout[-22000:])

# -- cell 17 ------------------------------------------------------------------------
s=open('system/snappyHexMeshDict').read()
s=s.replace('minVol 1e-13;','minVol 1e-13;\n    minArea 1e-13;')
open('system/snappyHexMeshDict','w').write(s)
p=subprocess.run(['bash','-c','set -o pipefail; snappyHexMesh -overwrite 2>&1 | tee snappy.log'],capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print(p.stdout[-26000:])
