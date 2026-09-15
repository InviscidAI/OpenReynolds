"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by defining the geometry parameters and the NACA 2412 profile, then build the wing solid 
# Key decisions (assumptions I'll record):
# - **Axes**: x downstream, y spanwise, z up. Root at y=0 (the wall plane), leading edges of both sect
# - **Domain**: root chord c=0.12 m → x ∈ [−0.48, 1.08] (4c upstream of LE, 8c downstream of root TE),
# - **Meshing**: export one STL per patch of the air volume (box minus wing), then `blockMesh` backgro
# - NACA 2412 with sharp (zero-thickness) trailing edge via the modified −0.1036 coefficient.
import numpy as np

# ---- Parameters (metres) ----
C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200
DOM = dict(xmin=-4*C_ROOT, xmax=C_ROOT+8*C_ROOT,   # 4c upstream of LE, 8c downstream of root TE
           ymin=0.0,      ymax=3*C_ROOT,            # wall at root plane, 3c beyond tip
           zmin=-3*C_ROOT, zmax=3*C_ROOT)
print("domain:", DOM)

# ---- NACA 2412, sharp TE (coefficient -0.1036) ----
def naca2412(chord, n=121):
    m, p, t = 0.02, 0.4, 0.12
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2          # cosine spacing 0..1
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc  = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x),     2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    # order: TE -> upper -> LE -> lower -> TE
    xs = np.concatenate([xu[::-1], xl[1:]]) * chord
    zs = np.concatenate([yu[::-1], yl[1:]]) * chord
    return xs, zs

xs_r, zs_r = naca2412(C_ROOT)
xs_t, zs_t = naca2412(C_TIP)
print("root max thickness (m):", xs_r.size and (zs_r.max()-zs_r.min()), " expect ~", 0.12*C_ROOT)
print("TE gap root:", abs(zs_r[0]-zs_r[-1]), " tip:", abs(zs_t[0]-zs_t[-1]))
print("root LE/TE x:", xs_r.min(), xs_r.max())

# -- cell 2 -------------------------------------------------------------------------
# The profile checks out (12% thickness, sharp TE). Now I'll build the wing as a ruled loft between th
import build123d as bd
print([n for n in dir(bd) if 'loft' in n.lower()])
help(bd.loft)

# -- cell 3 -------------------------------------------------------------------------
def section_face(xs, zs, y):
    pts = [bd.Vector(float(x), float(y), float(z)) for x, z in zip(xs, zs)]
    wire = bd.Wire(bd.Polyline(*pts, close=True))
    return bd.Face(wire)

root_face = section_face(xs_r, zs_r, 0.0)
tip_face  = section_face(xs_t, zs_t, SPAN)
wing = bd.loft([root_face, tip_face], ruled=True)
print("wing volume (m^3):", wing.volume)
print("wing bbox:", wing.bounding_box().min, wing.bounding_box().max)

# measure taper: chord at root and tip from bbox of section wires
print("root chord:", xs_r.max()-xs_r.min(), "tip chord:", xs_t.max()-xs_t.min())

# -- cell 4 -------------------------------------------------------------------------
# Wing solid looks right (chords 120/80 mm, span 200 mm). Now the air volume: box minus wing, with fac
box = bd.Box(DOM['xmax']-DOM['xmin'], DOM['ymax']-DOM['ymin'], DOM['zmax']-DOM['zmin'])
box = bd.Pos((DOM['xmax']+DOM['xmin'])/2, (DOM['ymax']+DOM['ymin'])/2, (DOM['zmax']+DOM['zmin'])/2) * box
air = box - wing
print("air volume:", air.volume, " expected:", box.volume - wing.volume)
faces = air.faces()
print("n faces:", len(faces))
for i, f in enumerate(faces):
    c = f.center()
    print(i, "area=%.5f" % f.area, "center=(%.3f,%.3f,%.3f)" % (c.X, c.Y, c.Z),
          "normal=", f.normal_at())

# -- cell 5 -------------------------------------------------------------------------
import os
os.makedirs("constant/triSurface", exist_ok=True)
os.makedirs("system", exist_ok=True)

EPS = 1e-6
def face_key(f):
    c = f.center(); n = f.normal_at()
    if f.geom_type == bd.GeomType.PLANE:
        if abs(n.X + 1) < 1e-4 and abs(c.X - DOM['xmin']) < EPS: return "inlet"
        if abs(n.X - 1) < 1e-4 and abs(c.X - DOM['xmax']) < EPS: return "outlet"
        if abs(n.Y + 1) < 1e-4 and abs(c.Y - DOM['ymin']) < EPS: return "wall"
        if abs(n.Y - 1) < 1e-4 and abs(c.Y - DOM['ymax']) < EPS: return "farfield"
        if abs(n.Z - 1) < 1e-4 and abs(c.Z - DOM['zmax']) < EPS: return "farfield"
        if abs(n.Z + 1) < 1e-4 and abs(c.Z - DOM['zmin']) < EPS: return "farfield"
    return "wing"

groups = {}
for f in air.faces():
    groups.setdefault(face_key(f), []).append(f)
for k, v in groups.items():
    print(k, len(v), "faces, area=%.5f" % sum(f.area for f in v))

for name, fs in groups.items():
    bd.export_stl(bd.Compound(fs), f"constant/triSurface/{name}.stl",
                  tolerance=2e-4, angular_tolerance=0.15)
print(sorted(os.listdir("constant/triSurface")))

# -- cell 6 -------------------------------------------------------------------------
# All 5 patches exported, areas sane (wing wetted ≈ 0.0414 m² vs planform 0.020 m²). Now verify the ST
import struct
def read_stl_verts(path):
    with open(path, 'rb') as fh:
        fh.read(80)
        (n,) = struct.unpack('<I', fh.read(4))
        data = np.frombuffer(fh.read(), dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(9,)),('a','<u2')]))
    return data['v'].reshape(-1,3)

from collections import Counter
edge_count = Counter()
tot_tris = 0
for name in groups:
    V = read_stl_verts(f"constant/triSurface/{name}.stl")
    tot_tris += len(V)//3
    Vk = np.round(V/1e-7).astype(np.int64)          # quantise to 0.1 µm
    keys = [tuple(k) for k in Vk]
    for i in range(0, len(keys), 3):
        for a, b in ((i,i+1),(i+1,i+2),(i+2,i)):
            edge_count[tuple(sorted((keys[a], keys[b])))] += 1
n1 = sum(1 for v in edge_count.values() if v == 1)
n2 = sum(1 for v in edge_count.values() if v == 2)
nother = {v: c for v, c in Counter(edge_count.values()).items() if v not in (1,2)}
print("triangles:", tot_tris, "edges used once (free):", n1, " twice:", n2, " other:", nother)

# -- cell 7 -------------------------------------------------------------------------
import pyvista as pv
pv.set_jupyter_backend('static')
pl = pv.Plotter(off_screen=True, window_size=(1100, 700))
colors = dict(wing='tomato', wall='tan', inlet='lightblue', outlet='lightgreen', farfield='white')
for name, col in colors.items():
    m = pv.read(f"constant/triSurface/{name}.stl")
    pl.add_mesh(m, color=col, opacity=1.0 if name in ('wing','wall') else 0.25,
                show_edges=(name=='wing'), label=name)
pl.add_legend()
pl.view_vector((-1.4, -1.2, 0.9))
pl.add_axes()
pl.show()

# -- cell 8 -------------------------------------------------------------------------
pl2 = pv.Plotter(off_screen=True, window_size=(1100, 700))
pl2.add_mesh(pv.read("constant/triSurface/wing.stl"), color='tomato', show_edges=True)
pl2.add_mesh(pv.read("constant/triSurface/wall.stl"), color='tan', opacity=0.4)
pl2.camera_position = [(0.35, -0.25, 0.25), (0.05, 0.1, 0.0), (0, 0, 1)]
pl2.add_axes()
pl2.show()

# numeric check of taper/span/thickness from the STL itself
W = read_stl_verts("constant/triSurface/wing.stl")
print("wing bbox x:[%.4f, %.4f] y:[%.4f, %.4f] z:[%.4f, %.4f]" %
      (W[:,0].min(), W[:,0].max(), W[:,1].min(), W[:,1].max(), W[:,2].min(), W[:,2].max()))
tip = W[np.isclose(W[:,1], SPAN, atol=1e-6)]
root = W[np.isclose(W[:,1], 0.0, atol=1e-6)]
print("root chord from STL: %.5f  tip chord: %.5f  span: %.4f" %
      (root[:,0].max()-root[:,0].min(), tip[:,0].max()-tip[:,0].min(), W[:,1].max()))

# -- cell 9 -------------------------------------------------------------------------
nx, ny, nz = 78, 18, 36   # ~20 mm base cells over 1.56 x 0.36 x 0.72 m
block_mesh = f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
scale 1;

vertices
(
    ({DOM['xmin']} {DOM['ymin']} {DOM['zmin']})
    ({DOM['xmax']} {DOM['ymin']} {DOM['zmin']})
    ({DOM['xmax']} {DOM['ymax']} {DOM['zmin']})
    ({DOM['xmin']} {DOM['ymax']} {DOM['zmin']})
    ({DOM['xmin']} {DOM['ymin']} {DOM['zmax']})
    ({DOM['xmax']} {DOM['ymin']} {DOM['zmax']})
    ({DOM['xmax']} {DOM['ymax']} {DOM['zmax']})
    ({DOM['xmin']} {DOM['ymax']} {DOM['zmax']})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet   {{ type patch; faces ( (0 4 7 3) ); }}
    outlet  {{ type patch; faces ( (1 2 6 5) ); }}
    wall    {{ type wall;  faces ( (0 1 5 4) ); }}
    farfield
    {{
        type patch;
        faces
        (
            (3 7 6 2)
            (4 5 6 7)
            (0 3 2 1)
        );
    }}
);
"""
open("system/blockMeshDict","w").write(block_mesh)
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
# Just needs a controlDict. Adding one plus minimal fvSchemes/fvSolution for checkMesh.
open("system/controlDict","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     snappyHexMesh;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")
open("system/fvSchemes","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open("system/fvSolution","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { }
relaxationFactors { }
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-300:])

# -- cell 11 ------------------------------------------------------------------------
open("system/surfaceFeatureExtractDict","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
wing.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj            yes;
}
""")
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.stdout[-400:], r.stderr[-300:])

snappy = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    wing     { type triSurfaceMesh; file "wing.stl"; }
    wall     { type triSurfaceMesh; file "wall.stl"; }
    inlet    { type triSurfaceMesh; file "inlet.stl"; }
    outlet   { type triSurfaceMesh; file "outlet.stl"; }
    farfield { type triSurfaceMesh; file "farfield.stl"; }
}

castellatedMeshControls
{
    maxLocalCells      1000000;
    maxGlobalCells     3000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    locationInMesh (0.5 0.3 0.2);

    features
    (
        { file "wing.eMesh"; level 2; }
    );

    refinementSurfaces
    {
        wing     { level (2 2); }
        wall     { level (0 0); }
        inlet    { level (0 0); }
        outlet   { level (0 0); }
        farfield { level (0 0); }
    }

    refinementRegions { }
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      100;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap  false;
    explicitFeatureSnap  true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers { }
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
    maxNonOrtho        65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave         80;
    minVol             1e-13;
    minTetQuality      1e-15;
    minArea            -1;
    minTwist           0.02;
    minDeterminant     0.001;
    minFaceWeight      0.05;
    minVolRatio        0.01;
    minTriangleTwist   -1;
    nSmoothScale       4;
    errorReduction     0.75;
}

mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
print("dict written")

# -- cell 12 ------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.stdout[-2500:])
print("STDERR:", r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
# snappyHexMesh completed cleanly (~60k cells). Running checkMesh and looking at the patches.
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2000:])
print("STDERR:", r.stderr[-300:])

# -- cell 14 ------------------------------------------------------------------------
r = subprocess.run(["foamToVTK", "-latestTime"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-200:])
import glob
print(glob.glob("VTK/**/*.vtu", recursive=True)[:3], glob.glob("VTK/**/*.vtp", recursive=True)[:8])

# -- cell 15 ------------------------------------------------------------------------
internal = pv.read("VTK/t14_0/internal.vtu")
wingm = pv.read("VTK/t14_0/boundary/wing.vtp")

# slice at midspan y=0.1
sl = internal.slice(normal=(0,1,0), origin=(0,0.1,0))
pl3 = pv.Plotter(off_screen=True, window_size=(1300, 650))
pl3.add_mesh(sl, show_edges=True, color='lightsteelblue')
pl3.add_mesh(wingm, color='tomato')
pl3.camera_position = [(0.15, 0.1, 0.9), (0.15, 0.1, 0.0), (0, 1, 0)]
pl3.show()

# zoom at wing, slice at y=0.05 (root half)
sl2 = internal.slice(normal=(0,1,0), origin=(0,0.05,0))
pl4 = pv.Plotter(off_screen=True, window_size=(1000, 700))
pl4.add_mesh(sl2, show_edges=True, color='lightsteelblue')
pl4.add_mesh(wingm, color='tomato')
pl4.camera_position = [(0.06, 0.05, 0.35), (0.06, 0.05, 0.0), (0, 1, 0)]
pl4.show()

# -- cell 16 ------------------------------------------------------------------------
# The slice doesn't appear — let me check whether the slice actually contains cells.
print(internal.n_cells, internal.celltypes if hasattr(internal,'celltypes') else '')
print("slice cells:", sl.n_cells, "bounds:", sl.bounds)
cts = internal.celltypes
import collections
print(collections.Counter(cts.tolist() if hasattr(cts,'tolist') else cts))

# -- cell 17 ------------------------------------------------------------------------
pl5 = pv.Plotter(off_screen=True, window_size=(1300, 620))
pl5.add_mesh(sl, show_edges=True, color='lightsteelblue')
pl5.add_mesh(wingm, color='tomato')
pl5.camera_position = [(0.3, 1.1, 0.0), (0.3, 0.1, 0.0), (0, 0, 1)]
pl5.show()

sl2 = internal.slice(normal=(0,1,0), origin=(0,0.05,0))
pl6 = pv.Plotter(off_screen=True, window_size=(900, 650))
pl6.add_mesh(sl2, show_edges=True, color='lightsteelblue')
pl6.add_mesh(wingm, color='tomato')
pl6.camera_position = [(0.06, 0.6, 0.0), (0.06, 0.05, 0.0), (0, 0, 1)]
pl6.show()

# -- cell 18 ------------------------------------------------------------------------
for xq in (0.006, 0.03, 0.06, 0.108, 0.117):
    band = W[(abs(W[:,0]-xq) < 0.002) & (W[:,1] < 0.05)]
    if len(band):
        print("x=%.3f  z range [%.5f, %.5f]  thickness=%.5f" %
              (xq, band[:,2].min(), band[:,2].max(), band[:,2].max()-band[:,2].min()))
    else:
        print("x=%.3f  no points" % xq)

# -- cell 19 ------------------------------------------------------------------------
snappy = open("system/snappyHexMeshDict").read()
snappy = snappy.replace(
    '    farfield { type triSurfaceMesh; file "farfield.stl"; }\n}',
    '''    farfield { type triSurfaceMesh; file "farfield.stl"; }
    wingZone { type searchableBox; min (-0.06 -0.01 -0.08); max (0.25 0.24 0.08); }
    wakeZone { type searchableBox; min (0.0 0.0 -0.07); max (0.9 0.22 0.07); }
}''')
snappy = snappy.replace('{ file "wing.eMesh"; level 2; }', '{ file "wing.eMesh"; level 3; }')
snappy = snappy.replace('wing     { level (2 2); }', 'wing     { level (3 3); }')
snappy = snappy.replace('    refinementRegions { }',
    '''    refinementRegions
    {
        wingZone { mode inside; levels ((1e15 2)); }
        wakeZone { mode inside; levels ((1e15 2)); }
    }''')
open("system/snappyHexMeshDict","w").write(snappy)

r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r2.stdout[-1200:])
print("STDERR:", r2.stderr[-300:])

# -- cell 20 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
i = out.find("Checking patch topology")
print(out[i:i+1200])
j = out.find("Checking geometry")
print(out[j:j+900])

# -- cell 21 ------------------------------------------------------------------------
# checkMesh passes: 341k cells, all five patches present and correctly typed. Final visual check of th
subprocess.run(["foamToVTK", "-latestTime"], capture_output=True, text=True)
internal = pv.read("VTK/t14_0/internal.vtu")
wingm = pv.read("VTK/t14_0/boundary/wing.vtp")
sl = internal.slice(normal=(0,1,0), origin=(0,0.05,0))
pl7 = pv.Plotter(off_screen=True, window_size=(1100, 650))
pl7.add_mesh(sl, show_edges=True, color='lightsteelblue')
pl7.add_mesh(wingm, color='tomato')
pl7.camera_position = [(0.08, 0.7, 0.0), (0.08, 0.05, 0.0), (0, 0, 1)]
pl7.show()
print("cells:", internal.n_cells)

# -- cell 22 ------------------------------------------------------------------------
plt_t = pv.Plotter(off_screen=True, window_size=(900, 300))
plt_t.add_mesh(pv.Sphere(radius=0.05, center=(1.0, 0.05, 0)), color='red')    # downstream (+x)
plt_t.add_mesh(pv.Sphere(radius=0.05, center=(-0.4, 0.05, 0)), color='lime')  # upstream (-x)
plt_t.camera_position = [(0.3, 0.7, 0.0), (0.3, 0.05, 0.0), (0, 0, 1)]
plt_t.show()

# -- cell 23 ------------------------------------------------------------------------
centers = internal.cell_centers().points
vols = internal.compute_cell_sizes()["Volume"]
fine = centers[vols < 3e-7]          # level-2 (5 mm) and finer cells
print("fine cell count:", len(fine))
print("fine cells x extent: [%.3f, %.3f]" % (fine[:,0].min(), fine[:,0].max()))
print("fine cells y extent: [%.3f, %.3f]" % (fine[:,1].min(), fine[:,1].max()))
print("fine cells z extent: [%.3f, %.3f]" % (fine[:,2].min(), fine[:,2].max()))
# coarse/fine boundary distances from wing (wing spans x 0..0.12)
print("=> fine region reaches %.2f m upstream of LE and %.2f m downstream of TE"
      % (0.0 - fine[:,0].min(), fine[:,0].max() - 0.12))
