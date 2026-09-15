"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd
import numpy as np, os

# Dimensions in METRES (OpenFOAM reads raw numbers as metres).
Z = [0.0, 0.45, 0.90]                    # stations: fan exit, mid, nozzle
R_OUT = [0.450, 0.430, 0.410]            # outer cowl inner surface
R_IN  = [0.280, 0.310, 0.250]            # core cowl outer surface

os.makedirs("constant/triSurface", exist_ok=True)

# Generator splines in the r-z plane: (x=r, y=0, z=axial), axis = z
outer_edge = bd.Spline([(r, 0, z) for z, r in zip(Z, R_OUT)])
inner_edge = bd.Spline([(r, 0, z) for z, r in zip(Z, R_IN)])

axis = bd.Axis((0, 0, 0), (0, 0, 1))

outer_cowl = bd.revolve(outer_edge, axis)   # surface of revolution, outer cowl
core_cowl  = bd.revolve(inner_edge, axis)   # surface of revolution, core cowl

ring_in  = bd.Plane(origin=(0, 0, 0.0), z_dir=(0, 0, 1)) * (bd.Circle(R_OUT[0]) - bd.Circle(R_IN[0]))
ring_out = bd.Plane(origin=(0, 0, 0.9), z_dir=(0, 0, 1)) * (bd.Circle(R_OUT[2]) - bd.Circle(R_IN[2]))

# Verify generator curves pass through the requested stations (sample the splines)
def radii_at(edge, zvals):
    pts = np.array([tuple(edge @ t) for t in np.linspace(0, 1, 4001)])
    return [np.hypot(*pts[np.abs(pts[:, 2] - z).argmin()][:2]) for z in zvals]

print("outer radii:", np.round(radii_at(outer_edge, Z), 5), "target:", R_OUT)
print("core  radii:", np.round(radii_at(inner_edge, Z), 5), "target:", R_IN)
print("valid:", outer_cowl.is_valid, core_cowl.is_valid, ring_in.is_valid, ring_out.is_valid)
print("areas m^2: outer %.4f core %.4f rin %.5f rout %.5f" % (
    outer_cowl.area, core_cowl.area, ring_in.area, ring_out.area))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd
import numpy as np
print(type(outer_cowl), type(core_cowl))
print("outer bbox:", outer_cowl.bounding_box().min, outer_cowl.bounding_box().max)
# children?
if hasattr(outer_cowl, "faces"):
    print("n faces:", len(outer_cowl.faces()), [f.area for f in outer_cowl.faces()])

# -- cell 3 -------------------------------------------------------------------------
# `bd.revolve` returned an empty Part — let me check its signature to see what it actually wants.
import build123d as bd, inspect
print(inspect.signature(bd.revolve))
print(bd.revolve.__doc__)

# -- cell 4 -------------------------------------------------------------------------
# Solid revolves fine (0.2764 m³). Minor fix — `Vector.X` not `.x` — and cleaning up the inner spline 
import build123d as bd
import numpy as np, os

# Dimensions in METRES (OpenFOAM reads raw numbers as metres).
Z = [0.0, 0.45, 0.90]                    # stations: fan exit, mid, nozzle
R_OUT = [0.450, 0.430, 0.410]            # outer cowl inner surface
R_IN  = [0.280, 0.310, 0.250]            # core cowl outer surface

os.makedirs("constant/triSurface", exist_ok=True)

# Closed generator profile in the r-z plane (x=r, y=0, z=axial), traversed in order
outer_edge = bd.Spline([(r, 0, z) for z, r in zip(Z, R_OUT)])
inner_edge = bd.Spline([(r, 0, z) for z, r in reversed(list(zip(Z, R_IN)))])
nozzle_line  = bd.Line((R_OUT[2], 0, Z[2]), (R_IN[2], 0, Z[2]))
fanexit_line = bd.Line((R_IN[0], 0, Z[0]), (R_OUT[0], 0, Z[0]))

profile_wire = bd.Wire([outer_edge, nozzle_line, inner_edge, fanexit_line])
profile_face = bd.Face(profile_wire)
axis = bd.Axis((0, 0, 0), (0, 0, 1))
fluid = bd.revolve(profile_face, axis)

r_cg = profile_face.center().X
print("profile closed:", profile_wire.is_closed, "| fluid valid:", fluid.is_valid)
print("volume: %.5f m^3 | Pappus: %.5f m^3 (A=%.5f, r_cg=%.4f)" %
      (fluid.volume, 2*np.pi*r_cg*profile_face.area, profile_face.area, r_cg))

faces = fluid.faces()
print("faces:", len(faces))
for f in faces:
    c = f.center()
    print("  planar=%s center=(%.4f, %.4f, %.4f) r=%.4f z=%.4f area=%.4f" % (
        f.geom_type == "PLANE", c.X, c.Y, c.Z, np.hypot(c.X, c.Y), c.Z, f.area))

# -- cell 5 -------------------------------------------------------------------------
import build123d as bd
import numpy as np

# Classify the four faces of the revolved annulus; axisymmetric => unambiguous
patches = {}
for f in fluid.faces():
    c = f.center()
    r, z = np.hypot(c.X, c.Y), c.Z
    if abs(z - 0.0) < 1e-9 and r < 1e-9:
        patches["fanExit"] = f        # annular ring at the fan exit plane
    elif abs(z - 0.9) < 1e-9 and r < 1e-9:
        patches["nozzleOutlet"] = f   # annular ring at the nozzle plane
    elif r > 0.37:
        patches["outerCowl"] = f      # surface of revolution at r~0.43 mid
    else:
        patches["coreCowl"] = f       # surface of revolution at r~0.31 mid
print("patches:", {k: round(v.area, 5) for k, v in patches.items()})

# Export one STL per patch (units already metres; 0.05 mm linear tolerance)
for name, f in patches.items():
    bd.export_stl(f, f"constant/triSurface/{name}.stl", tolerance=5e-5, angular_tolerance=0.05)
print("exported:", sorted(os.listdir("constant/triSurface")))

# Verify the EXPORTED STL geometry: radii at the three stations, from the STL points
import pyvista as pv
for name, targets in [("outerCowl", R_OUT), ("coreCowl", R_IN)]:
    m = pv.read(f"constant/triSurface/{name}.stl")
    p = m.points
    rad = [np.hypot(*(p[np.abs(p[:, 2] - z) < 0.003][:, :2].T)).max() for z in Z]
    print(name, "radii @stations:", np.round(rad, 4), "target:", targets)
for name, z, ro, ri in [("fanExit", 0.0, 0.450, 0.280), ("nozzleOutlet", 0.9, 0.410, 0.250)]:
    m = pv.read(f"constant/triSurface/{name}.stl")
    r = np.hypot(*(m.points[:, :2].T))
    print(name, "z-range %.4f..%.4f  r: %.4f..%.4f target %.3f..%.3f" %
          (m.points[:, 2].min(), m.points[:, 2].max(), r.min(), r.max(), ri, ro))

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv
pv.set_jupyter_backend("static")
pl = pv.Plotter(off_screen=True, window_size=(1000, 700))
colors = {"outerCowl": "tan", "coreCowl": "steelblue", "fanExit": "lime", "nozzleOutlet": "tomato"}
for name, c in colors.items():
    pl.add_mesh(pv.read(f"constant/triSurface/{name}.stl"), color=c, opacity=0.85, show_edges=False)
pl.add_axes()
pl.view_vector((1.4, -2.2, 0.7))
pl.show()

# -- cell 7 -------------------------------------------------------------------------
import subprocess, os
os.makedirs("system", exist_ok=True)

def w(path, text):
    with open(path, "w") as f: f.write(text)

w("system/controlDict", """
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh;
startFrom latestTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0;
writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
""")

# Background box fully enclosing the duct (duct: r<=0.45, z in [0, 0.9])
w("system/blockMeshDict", """
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
    (-0.47 -0.47 -0.03)
    ( 0.47 -0.47 -0.03)
    ( 0.47  0.47 -0.03)
    (-0.47  0.47 -0.03)
    (-0.47 -0.47  0.93)
    ( 0.47 -0.47  0.93)
    ( 0.47  0.47  0.93)
    (-0.47  0.47  0.93)
);
blocks ( hex (0 1 2 3 4 5 6 7) (48 48 49) simpleGrading (1 1 1) );
edges ();
boundary ( allOutside { type patch; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0) (3 7 6 2) (0 3 2 1) (4 5 6 7)); } );
mergePatchPairs ();
""")

w("system/surfaceFeatureExtractDict", """
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
outerCowl.stl    { extractionMethod extractFromSurface; includedAngle 150; subsetFeatures { nonManifoldEdges no; openEdges yes; } writeObj yes; }
coreCowl.stl     { extractionMethod extractFromSurface; includedAngle 150; subsetFeatures { nonManifoldEdges no; openEdges yes; } writeObj yes; }
fanExit.stl      { extractionMethod extractFromSurface; includedAngle 150; subsetFeatures { nonManifoldEdges no; openEdges yes; } writeObj yes; }
nozzleOutlet.stl { extractionMethod extractFromSurface; includedAngle 150; subsetFeatures { nonManifoldEdges no; openEdges yes; } writeObj yes; }
""")

w("system/snappyHexMeshDict", """
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;

geometry
{
    outerCowl    { type triSurfaceMesh; file "outerCowl.stl"; }
    coreCowl     { type triSurfaceMesh; file "coreCowl.stl"; }
    fanExit      { type triSurfaceMesh; file "fanExit.stl"; }
    nozzleOutlet { type triSurfaceMesh; file "nozzleOutlet.stl"; }
}

castellatedMeshControls
{
    maxLocalCells 1000000; maxGlobalCells 3000000; minRefinementCells 10;
    maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
    features
    (
        { file "outerCowl.eMesh";    level 1; }
        { file "coreCowl.eMesh";     level 1; }
        { file "fanExit.eMesh";      level 1; }
        { file "nozzleOutlet.eMesh"; level 1; }
    );
    refinementSurfaces
    {
        outerCowl    { level (2 2); }
        coreCowl     { level (2 2); }
        fanExit      { level (2 2); }
        nozzleOutlet { level (2 2); }
    }
    resolveFeatureAngle 30;
    refinementRegions { }
    locationInMesh (0.36 0.0 0.45);   // inside the annulus at mid-duct (0.31 < r < 0.43)
    allowFreeStandingZoneFaces false;
}

snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers { }
    expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15;
    minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}
""")

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc:", r.returncode)
print(r.stdout[-600:])
r2 = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print("sfe rc:", r2.returncode, r2.stdout[-200:], r2.stderr[-300:])

# -- cell 8 -------------------------------------------------------------------------
# Background mesh and feature edges are ready. Running `snappyHexMesh` (serial, coarse).
import subprocess
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-3000:])
print("STDERR:", r.stderr[-1000:])

# -- cell 9 -------------------------------------------------------------------------
# snappy needs `fvSchemes`/`fvSolution` for its snap-phase smoothing. Adding minimal ones and re-runni
import subprocess

with open("system/fvSchemes", "w") as f:
    f.write("""
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }
""")

with open("system/fvSolution", "w") as f:
    f.write("""
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { }
""")

r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-2500:])
print("STDERR:", r.stderr[-800:])

# -- cell 10 ------------------------------------------------------------------------
# Missing `mergeTolerance` — adding it to the dict and re-running.
import subprocess, re

with open("system/snappyHexMeshDict") as f:
    txt = f.read()
if "mergeTolerance" not in txt:
    txt = txt.replace("castellatedMesh true; snap true; addLayers false;",
                      "castellatedMesh true; snap true; addLayers false;\nmergeTolerance 1e-06;")
with open("system/snappyHexMeshDict", "w") as f:
    f.write(txt)

r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-2500:])
print("STDERR:", r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-3500:])
print("STDERR:", r.stderr[-500:])

# -- cell 12 ------------------------------------------------------------------------
import subprocess, glob
import pyvista as pv
r = subprocess.run(["foamToVTK", "-noZero", "-constant"], capture_output=True, text=True)
print("rc:", r.returncode, r.stdout[-200:], r.stderr[-200:])

vtm = sorted(glob.glob("VTK/*.vtm"))
print(vtm)
ds = pv.read(vtm[-1])
print(ds)
# slice through the axis (y=0 plane)
sl = ds.slice(normal=(0, 1, 0), origin=(0, 0, 0.45))
pl = pv.Plotter(off_screen=True, window_size=(900, 700))
pl.add_mesh(sl, show_edges=True, scalars=None, color="lightgrey")
pl.add_text("slice y=0 through annulus", font_size=10)
pl.view_vector((0, -1, 0))
pl.camera.up = (0, 0, 1)
pl.show()

# -- cell 13 ------------------------------------------------------------------------
import struct, os
import numpy as np
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_Orientation, TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

# Mesh the whole solid ONCE -> shared edges get identical vertex positions on both faces
BRepMesh_IncrementalMesh(fluid.wrapped, 5e-5, False, 0.05, True)

def face_triangles(bd_face):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(bd_face.wrapped, loc)
    trsf = loc.Transformation()
    n = tri.NbNodes()
    pts = np.empty((n, 3))
    for i in range(1, n + 1):
        p = tri.Node(i).Transformed(trsf)
        pts[i - 1] = (p.X(), p.Y(), p.Z())
    m = tri.NbTriangles()
    tris = np.empty((m, 3), dtype=np.int64)
    rev = bd_face.wrapped.Orientation() == TopAbs_REVERSED
    for i in range(1, m + 1):
        a, b, c = tri.Triangle(i).Get()
        tris[i - 1] = (a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1)
    return pts, tris

def write_stl(path, pts, tris):
    v1 = pts[tris[:, 0]]; v2 = pts[tris[:, 1]]; v3 = pts[tris[:, 2]]
    nrm = np.cross(v2 - v1, v3 - v1)
    ln = np.linalg.norm(nrm, axis=1); ln[ln == 0] = 1.0
    nrm /= ln[:, None]
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack("<12fH", *nrm[i], *v1[i], *v2[i], *v3[i], 0))

for name, f in patches.items():
    pts, tris = face_triangles(f)
    write_stl(f"constant/triSurface/{name}.stl", pts, tris)
    print(name, "tris:", len(tris), "nodes:", len(pts))

# -- cell 14 ------------------------------------------------------------------------
import numpy as np
import pyvista as pv
from collections import Counter

allpts, alltris, off = [], [], 0
for name in ["outerCowl", "coreCowl", "fanExit", "nozzleOutlet"]:
    m = pv.read(f"constant/triSurface/{name}.stl")
    f = m.faces.reshape(-1, 4)[:, 1:]
    allpts.append(m.points); alltris.append(f + off); off += len(m.points)
P = np.vstack(allpts); T = np.vstack(alltris)

# weld by rounding (checker tolerance ~1e-7-ish; use 1e-6 m = 1 micron)
key = np.round(P / 1e-6).astype(np.int64)
_, inv = np.unique(key, axis=0, return_inverse=True)
Tw = inv[T]
edges = np.vstack([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]])
e_sorted = np.sort(edges, axis=1)
cnt = Counter(map(tuple, e_sorted))
free = [e for e, c in cnt.items() if c == 1]
# winding: edges appearing twice with same direction = normals inconsistency
dir_cnt = Counter(map(tuple, edges))
same_dir = sum(1 for (a, b), c in dir_cnt.items() if c == 2)
print("free edges after weld:", len(free))
print("same-direction doubled edges (winding):", same_dir)

# -- cell 15 ------------------------------------------------------------------------
# Where are the free edges? Map welded indices back to coordinates
uw = np.unique(key, axis=0)          # welded grid points (order matches inv? no - get via unique)
uid, inv_full = np.unique(key, axis=0, return_inverse=True)
# representative coordinates of each welded point: mean of members
rep = np.zeros((len(uid), 3))
np.add.at(rep, inv_full, P)
cntp = np.bincount(inv_full)[:, None]
rep /= cntp

fe = np.array(free)
fr = np.hypot(rep[fe[:, 0]], rep[fe[:, 1]]) if False else None
# radius and z of free-edge endpoints
r0 = np.hypot(rep[fe[:, 0], 0], rep[fe[:, 0], 1]); z0 = rep[fe[:, 0], 2]
r1 = np.hypot(rep[fe[:, 1], 0], rep[fe[:, 1], 1]); z1 = rep[fe[:, 1], 2]
import numpy as np
print("free-edge endpoint r range: %.4f..%.4f" % (min(r0.min(), r1.min()), max(r0.max(), r1.max())))
print("free-edge endpoint z range: %.5f..%.5f" % (min(z0.min(), z1.min()), max(z0.max(), z1.max())))
# histogram of z
zs = np.concatenate([z0, z1])
for zc, lab in [(0.0, "z=0 (fan exit)"), (0.9, "z=0.9 (nozzle)")]:
    print(lab, "count:", int((np.abs(zs - zc) < 0.01).sum()))
print("mid-duct count:", int(((np.abs(zs) > 0.01) & (np.abs(zs - 0.9) > 0.01)).sum()))

# -- cell 16 ------------------------------------------------------------------------
# All 1,549 free edges sit on the nozzle plane — the fan-exit rim welded fine. Let me measure the actu
import numpy as np, pyvista as pv

def bnd_pts(name, z, rad_lo, rad_hi):
    m = pv.read(f"constant/triSurface/{name}.stl")
    p = m.points
    sel = p[(np.abs(p[:, 2] - z) < 1e-4)]
    r = np.hypot(sel[:, 0], sel[:, 1])
    return sel[(r > rad_lo) & (r < rad_hi)]

for rad, lab in [(0.41, "outer rim"), (0.25, "inner rim")]:
    ring = bnd_pts("nozzleOutlet", 0.9, rad - 0.005, rad + 0.005)
    cowl = bnd_pts("outerCowl" if rad > 0.3 else "coreCowl", 0.9, rad - 0.005, rad + 0.005)
    print(lab, ": ring nodes", len(ring), " cowl nodes", len(cowl))
    # nearest neighbour distance ring->cowl
    from scipy.spatial import cKDTree
    d = cKDTree(cowl).query(ring)[0]
    print("  nearest-dist ring->cowl: max %.3e mean %.3e" % (d.max(), d.mean()))
