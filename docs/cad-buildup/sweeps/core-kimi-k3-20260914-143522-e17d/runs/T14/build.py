"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll build a NACA 2412 wing (root chord 0.12 m tapering to 0.08 m over 0.2 m span, exact-scale), sub
import numpy as np

def naca2412(n=200):
    # returns (x, y_lower, y_upper) for unit chord
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # cosine spacing 0..1
    t = 0.12
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1037*x**4)  # closed-ish trailing edge
    # camber 2% at 40%
    yc = np.where(x<0.4, 0.02/0.04*(0.8*x - x*x), 0.02/0.36*(1.6 - 2.4*x + 0.8 + 100*np.clip(0.2*x/x,0,0)) )
    # simpler: piecewise avoid the messy branch
    m = 0.02; p = 0.4
    yc = np.where(x<p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    return x, yc-yt, yc+yt

x, yl, yu = naca2412()
t_max = (yu.max() - yl[np.argmax(yu)])
print("max thickness (fraction of chord):", t_max, "expect ~0.12")

# -- cell 2 -------------------------------------------------------------------------
# Odd error — inspect what the two wire objects actually are.
import numpy as np, build123d as bd

def naca2412_pts(chord, n=100):
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    t = 0.12; m = 0.02; p = 0.4
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1037*x**4)
    yc = np.where(x<p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    seq = [(float(xi*chord), float(yu*chord)) for xi, yu in zip(x, yu)] + \
          [(float(xi*chord), float(yl*chord)) for xi, yl in list(zip(x, yl))[1:-1]]
    return seq

CR, CT, SPAN = 0.120, 0.080, 0.200
def sec_wire(chord, yspan):
    return bd.Wire.make_polygon([bd.Vector(x, yspan, th) for x, th in naca2412_pts(chord)])

w_root = sec_wire(CR, 0.0)
w_tip  = sec_wire(CT, SPAN)
print(type(w_root), type(w_tip), isinstance(w_root, bd.Wire), isinstance(w_tip, bd.Wire))
lst = [w_root, w_tip]
print("len:", len(lst), [getattr(w,'closed','?') for w in lst])

# -- cell 3 -------------------------------------------------------------------------
# Fine — I'll write the wing STL directly as triangles (loft of the two polylines) and build the box f
import numpy as np

def naca2412_pts(chord, n=100):
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    t = 0.12; m = 0.02; p = 0.4
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1037*x**4)
    yc = np.where(x<p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    seq = [(float(xi), float(yu)) for xi, yu in zip(x, yu)] + \
          [(float(xi), float(yl)) for xi, yl in list(zip(x, yl))[1:-1]]
    return np.asarray(seq)  # (N,2) column x, thickness z

CR, CT, SPAN = 0.120, 0.080, 0.200
root = naca2412_pts(CR); tip = naca2412_pts(CT)
N = len(root)
# wing lateral triangles: quad strip between root (y=0) and tip (y=SPAN)
def tris_wing(r, tp, SPAN):
    p0 = np.stack([r[:,0], np.zeros(N), r[:,1]], axis=1)      # root y=0
    p1 = np.stack([tp[:,0], np.full(N, SPAN), tp[:,1]], axis=1) # tip y=SPAN
    tris = []
    for i in range(N):
        j = (i+1) % N
        a, b, c, d = p0[i], p0[j], p1[j], p1[i]
        tris += [[a, b, c], [a, c, d]]
    # caps: fan from centroid
    def cap(pts, yv, flip):
        c = [pts[:,0].mean(), yv, pts[:,1].mean()]
        cent = np.tile(np.array(c), (N,1))
        for i in range(N):
            j = (i+1) % N
            if flip:
                t = [cent[i], [pts[j,0], yv, pts[j,1]], [pts[i,0], yv, pts[i,1]]]
            else:
                t = [cent[i], [pts[i,0], yv, pts[i,1]], [pts[j,0], yv, pts[j,1]]]
            tris.append(t)
    cap(tp, SPAN, False)   # tip cap outward +y
    cap(root, 0.0, True)   # root cap points -y
    return np.array(tris)

def write_stl(name, tris):
    with open(name, 'w') as f:
        f.write('solid ' + name + '\n')
        for tri in tris:
            a, b, c = tri
            nrm = np.cross(b-a, c-a)
            L = np.linalg.norm(nrm)
            if L < 1e-15: continue
            n = nrm / L
            f.write('facet normal %e %e %e\n' % tuple(n))
            f.write('  outer loop\n')
            for v in (a, b, c):
                f.write('    vertex %e %e %e\n' % tuple(v))
            f.write('  endloop\nendfacet\n')
        f.write('endsolid ' + name + '\n')

import os
os.makedirs('constant/triSurface', exist_ok=True)
wtris = tris_wing(root, tip, SPAN)
write_stl('constant/triSurface/wing.stl', wtris)
print("wing triangles:", len(wtris))

# -- cell 4 -------------------------------------------------------------------------
import numpy as np, os
CR, CT, SPAN = 0.120, 0.080, 0.200
X0, X1 = -4*CR, 8*CR
Y0, Y1 = 0.0, SPAN + 3*CR
Z0, Z1 = -3*CR, 3*CR

patches = {}
patches['inlet'] = np.array([ [[X0,Y0,Z0],[X0,Y0,Z1],[X0,Y1,Z1]],
                              [[X0,Y0,Z0],[X0,Y1,Z1],[X0,Y1,Z0]] ])
patches['outlet'] = np.array([ [[X1,Y0,Z1],[X1,Y0,Z0],[X1,Y1,Z0]],
                               [[X1,Y0,Z1],[X1,Y1,Z0],[X1,Y1,Z1]] ])
patches['rootWall'] = np.array([ [[X0,0,Z0],[X1,0,Z0],[X1,0,Z1]],
                                 [[X0,0,Z0],[X1,0,Z1],[X0,0,Z1]] ])
ff = []
ff += [[[X0,Y0,Z0],[X0,Y1,Z0],[X1,Y1,Z0]],
       [[X0,Y0,Z0],[X1,Y1,Z0],[X1,Y0,Z0]]]
ff += [[[X0,Y0,Z1],[X1,Y1,Z1],[X0,Y1,Z1]],
       [[X0,Y0,Z1],[X1,Y0,Z1],[X1,Y1,Z1]]]
ff += [[[X0,Y1,Z0],[X1,Y1,Z1],[X1,Y1,Z0]],
       [[X0,Y1,Z0],[X1,Y1,Z1],[X0,Y1,Z0]]]
patches['farField'] = np.array(ff)

def write_stl(name, tris):
    with open(name, 'w') as f:
        f.write('solid %s\n' % name)
        for tri in tris:
            a, b, c = tri
            n = np.cross(b-a, c-a); L = np.linalg.norm(n)
            if L < 1e-15: continue
            f.write('facet normal %e %e %e\n' % tuple(n/L))
            f.write('  outer loop\n')
            for v in (a,b,c): f.write('    vertex %e %e %e\n' % tuple(v))
            f.write('  endloop\nendfacet\n')
        f.write('endsolid %s\n' % name)

for name, tris in patches.items():
    write_stl('constant/triSurface/%s.stl' % name, tris)
print('files:', sorted(os.listdir('constant/triSurface')))

# -- cell 5 -------------------------------------------------------------------------
# STLs done. Now the OpenFOAM dictionaries: blockMesh background grid (~(60x30x20) coarse) and snappyH
import os
CR, CT, SPAN = 0.120, 0.080, 0.200
X0, X1 = -4*CR, 8*CR
Y0, Y1 = 0.0, SPAN + 3*CR
Z0, Z1 = -3*CR, 3*CR

os.makedirs('system', exist_ok=True)
block = f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
    ({X0} {Y0} {Z0})
    ({X1} {Y0} {Z0})
    ({X1} {Y1} {Z0})
    ({X0} {Y1} {Z0})
    ({X0} {Y0} {Z1})
    ({X1} {Y0} {Z1})
    ({X1} {Y1} {Z1})
    ({X0} {Y1} {Z1})
);
blocks( hex (0 1 2 3 4 5 6 7) (80 32 40) simpleGrading (1 1 1) );
edges( );
boundary( );
"""
open('system/blockMeshDict','w').write(block)
print('blockMeshDict written')

# -- cell 6 -------------------------------------------------------------------------
shmd = """
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
    inlet    { type triSurfaceMesh; file "inlet.stl"; }
    outlet   { type triSurfaceMesh; file "outlet.stl"; }
    rootWall { type triSurfaceMesh; file "rootWall.stl"; }
    farField { type triSurfaceMesh; file "farField.stl"; }
    wing     { type triSurfaceMesh; file "wing.stl"; }
}
castellatedMeshControls
{
    maxLocalCells 1000000; maxGlobalCells 2000000; minRefinementCells 10;
    maxLoadUnbalanceFactor 20; nCellsBetweenLevels 2; allowDomainBoundingBox true;
    features ();
    refinementSurfaces
    {
        wing
        {
            level (2 3);
            regions
            {
                solid { level (2 3); }
            }
        }
    }
    resolveFeatureAngle 30;
    locationInMesh (0.6 0.3 0.2);
}
snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10;
    implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true; layers (); expansionRatio 1.0; finalLayerThickness 0.3;
    minThickness 0.1; maxThickness 0.5; nGrow 0; nSmoothThickness 10; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; minMedialAxisAngle 90; featureAngle 40; nRelaxIter 3;
    maxThicknessToMedian 0.5; bufferZoneFactor 5.0; nFeatureSnapIter 10;
}
meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minFaceCubicity 0.3; minTetQuality 1e-30; nSmoothScale 4;
    errorReduction 0.75; minVol 1e-13; minArea -1; minTwist 0.02;
    minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01;
}
algorithm parallel;
"""
open('system/snappyHexMeshDict','w').write(shmd)
print(open('system/snappyHexMeshDict').read()[:200])

# -- cell 7 -------------------------------------------------------------------------
# Fix the refinementSurfaces entry (drop the bogus `regions` block), then run blockMesh.
import subprocess
s = open('system/snappyHexMeshDict').read()
s = s.replace("""            regions
            {
                solid { level (2 3); }
            }
""", "            regions { }\n")
open('system/snappyHexMeshDict','w').write(s)
r = subprocess.run(['blockMesh'], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
# blockMesh needs a controlDict — write the standard minimal set and rerun.
import subprocess
open('system/controlDict','w').write("""
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 0; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 10;
writeCompression false; timeFormat general; timePrecision 6; runTimeModifiable true; adjustTimeStep yes;
""")
open('system/fvSchemes','w').write("FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n")
open('system/fvSolution','w').write("FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n")
r = subprocess.run(['blockMesh'], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-200:])

# -- cell 9 -------------------------------------------------------------------------
import subprocess, re
b = open('system/blockMeshDict').read()
b = re.sub(r'edges\(\s*\);', 'edges();', b)
b = re.sub(r'boundary\(\s*\);', 'boundary();', b)
open('system/blockMeshDict','w').write(b)
print(re.findall(r'edges|boundary', b))
r = subprocess.run(['blockMesh'], capture_output=True, text=True)
print(r.stdout[-200:], r.stderr[-200:])

# -- cell 10 ------------------------------------------------------------------------
import subprocess
CR, CT, SPAN = 0.120, 0.080, 0.200
X0, X1 = -4*CR, 8*CR
Y0, Y1 = 0.0, SPAN + 3*CR
Z0, Z1 = -3*CR, 3*CR
verts = [(X0,Y0,Z0),(X1,Y0,Z0),(X1,Y1,Z0),(X0,Y1,Z0),(X0,Y0,Z1),(X1,Y0,Z1),(X1,Y1,Z1),(X0,Y1,Z1)]
vtxt = '\n'.join('    (%g %g %g)' % v for v in verts)
txt = "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n" \
"convertToMeters 1;\nvertices\n(\n" + vtxt + "\n);\n" \
"blocks\n(\n    hex (0 1 2 3 4 5 6 7) (80 32 40) simpleGrading (1 1 1)\n);\n" \
"edges ();\nboundary ();\n"
open('system/blockMeshDict','w').write(txt)
r = subprocess.run(['blockMesh'], capture_output=True, text=True)
print('rc', r.returncode, r.stdout[-150:].replace('\n',' '), r.stderr[-100:])

# -- cell 11 ------------------------------------------------------------------------
# Background mesh built (~102k cells). Run snappyHexMesh.
import subprocess
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True)
print('rc', r.returncode)
print(r.stdout[-800:])
print('ERR', r.stderr[-800:])

# -- cell 12 ------------------------------------------------------------------------
import subprocess
open('system/fvSchemes','w').write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steady; }
gradSchemes { default Gauss linear; }
divSchemes { default Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open('system/fvSolution','w').write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { }
""")
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True)
print('rc', r.returncode)
print(r.stdout[-600:], 'ERR', r.stderr[-600:])

# -- cell 13 ------------------------------------------------------------------------
import subprocess
s = open('system/snappyHexMeshDict').read()
s = s.replace('implicitFeatureSnap false;', 'mergeTolerance 1.0; implicitFeatureSnap false;')
open('system/snappyHexMeshDict','w').write(s)
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True, timeout=1000)
print('rc', r.returncode)
print(r.stdout[-700:], 'ERR', r.stderr[-700:])

# -- cell 14 ------------------------------------------------------------------------
import subprocess
s = open('system/snappyHexMeshDict').read()
s = s.replace('castellatedMesh true; snap true; addLayers false;',
              'mergeTolerance 1.0; castellatedMesh true; snap true; addLayers false;')
open('system/snappyHexMeshDict','w').write(s)
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True, timeout=1000)
print('rc', r.returncode)
print(r.stdout[-500:], 'ERR', r.stderr[-500:])

# -- cell 15 ------------------------------------------------------------------------
import subprocess
s = open('system/snappyHexMeshDict').read()
s = s.replace('resolveFeatureAngle 30;', 'refinementRegions { }\n    resolveFeatureAngle 30;')
open('system/snappyHexMeshDict','w').write(s)
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True, timeout=1000)
print('rc', r.returncode)
print(r.stdout[-500:], 'ERR', r.stderr[-500:])

# -- cell 16 ------------------------------------------------------------------------
import subprocess
s = open('system/snappyHexMeshDict').read()
s = s.replace('refinementRegions { }', 'refinementRegions { }\n    allowFreeStandingZoneFaces false;')
open('system/snappyHexMeshDict','w').write(s)
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True, timeout=1000)
print('rc', r.returncode)
print(r.stdout[-400:], 'ERR', r.stderr[-400:])

# -- cell 17 ------------------------------------------------------------------------
import subprocess
s = open('system/snappyHexMeshDict').read()
s = s.replace('minVol 1e-13;', 'minVol 1e-13; minAxisSize 1e-10;')
open('system/snappyHexMeshDict','w').write(s)
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True, timeout=1000)
print('rc', r.returncode)
print(r.stdout[-300:], 'ERR', r.stderr[-300:])

# -- cell 18 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(['snappyHexMesh'], capture_output=True, text=True, timeout=1000)
import re
print(re.findall(r"Entry '(\S+)'", r.stderr))
