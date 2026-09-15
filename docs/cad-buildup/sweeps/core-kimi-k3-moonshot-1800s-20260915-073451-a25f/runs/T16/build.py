"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# This version also lacks `Face.make_from_wires`. Checking the actual Face constructor:
import build123d as bd, inspect
print(inspect.signature(bd.Face.__init__) if hasattr(bd.Face,'__init__') else '')
print([m for m in dir(bd.Face) if 'make' in m or 'from' in m])
print(inspect.signature(bd.revolve))
print(inspect.signature(bd.Wire.__init__))

# -- cell 2 -------------------------------------------------------------------------
# `bd.Face(wire)` is the constructor here. Re-emitting the full build cell with that fix:
import numpy as np
import build123d as bd

# --- parameters (metres; request gave mm, converted once here) ---
L = 0.900                      # duct length, axis = z, z=0 fan exit (inlet), z=L nozzle (outlet)
Z_STATIONS = [0.0, L/2, L]
R_OUT = [0.450, 0.430, 0.410]  # outer cowl inner-surface radii
R_CORE = [0.280, 0.310, 0.250] # core cowl outer-surface radii

# smooth blends: quadratic through the three stations (outer set is collinear -> linear taper)
co_out  = np.polyfit(Z_STATIONS, R_OUT, 2)
co_core = np.polyfit(Z_STATIONS, R_CORE, 2)
r_out  = lambda z: float(np.polyval(co_out, z))
r_core = lambda z: float(np.polyval(co_core, z))
print("outer poly:", co_out, "\ncore poly:", co_core)

def revolved_solid(rfun):
    """Solid of revolution about z: 0 <= r <= rfun(z), 0 <= z <= L."""
    zs = np.linspace(0, L, 25)
    prof = [bd.Vector(rfun(z), 0, z) for z in zs]
    edges = [
        bd.Spline(prof),                                    # profile curve
        bd.Line(prof[-1], bd.Vector(0, 0, L)),              # top: profile to axis
        bd.Line(bd.Vector(0, 0, L), bd.Vector(0, 0, 0)),    # down the axis
        bd.Line(bd.Vector(0, 0, 0), prof[0]),               # axis back out to profile start
    ]
    wire = bd.Wire(edges)
    face = bd.Face(wire)
    return bd.revolve(face, axis=bd.Axis.Z, revolution_arc=360)

outer_solid = revolved_solid(r_out)
core_solid  = revolved_solid(r_core)
fluid = outer_solid - core_solid

# analytic volume of the annulus: pi * integral (rout^2 - rcore^2) dz
zs = np.linspace(0, L, 2001)
vol_analytic = np.pi * np.trapezoid([r_out(z)**2 - r_core(z)**2 for z in zs], zs)
print("fluid volume: %.6f m^3  (analytic %.6f, diff %.2e)" % (fluid.volume, vol_analytic, fluid.volume - vol_analytic))
print("n faces:", len(fluid.faces()))
for i, f in enumerate(fluid.faces()):
    c = f.center()
    print(i, f.geom_type, "area %.4f" % f.area, "center (r=%.4f, z=%.4f)" % (np.hypot(c.X, c.Y), c.Z))

# -- cell 3 -------------------------------------------------------------------------
import os, numpy as np
os.makedirs("constant/triSurface", exist_ok=True)

# --- identify the four faces by what they were built from, then tag them ---
patches = {}          # name -> Face
for f in fluid.faces():
    c = f.center()
    r_c, z_c = np.hypot(c.X, c.Y), c.Z
    if f.geom_type == bd.GeomType.PLANE:
        patches["inlet" if abs(z_c - 0.0) < 1e-9 else "outlet"] = f
    elif abs(r_c - r_out(z_c)) < abs(r_c - r_core(z_c)):
        patches["outer_cowl"] = f
    else:
        patches["core_cowl"] = f
print({k: round(v.area, 4) for k, v in patches.items()})

# --- measure the surfaces: tessellate each revolved face, compare r(z) to the requested profile ---
for name, rfun in [("outer_cowl", r_out), ("core_cowl", r_core)]:
    pts, _ = patches[name].tessellate(1e-4)   # tolerance in metres
    pts = np.array([[p.X, p.Y, p.Z] for p in pts])
    rad = np.hypot(pts[:, 0], pts[:, 1])
    expected = np.array([rfun(z) for z in pts[:, 2]])
    err = np.abs(rad - expected)
    print("%s: %d pts, max |r_meas - r_profile| = %.2e m; stations r(z=0,.45,.9) = %.4f %.4f %.4f"
          % (name, len(pts), err.max(), rfun(0), rfun(L/2), rfun(L)))
# check monotonic annular gap stays positive
gap = [r_out(z) - r_core(z) for z in np.linspace(0, L, 91)]
print("annular gap: min %.4f m at z=%.3f" % (min(gap), np.linspace(0, L, 91)[np.argmin(gap)]))

# --- export one STL per patch ---
for name, f in patches.items():
    bd.export_stl(f, f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.15)
print(sorted(os.listdir("constant/triSurface")))

# -- cell 4 -------------------------------------------------------------------------
import pyvista as pv
pv.OFF_SCREEN = True
pl = pv.Plotter()
colors = {"outer_cowl": "lightblue", "core_cowl": "tan", "inlet": "lime", "outlet": "tomato"}
for name, col in colors.items():
    m = pv.read(f"constant/triSurface/{name}.stl")
    pl.add_mesh(m, color=col, opacity=0.6 if "cowl" in name else 1.0, show_edges=False)
pl.add_axes()
pl.camera_position = [(2.2, 0.4, 0.45), (0, 0, 0.45), (0, 0, 1)]
pl.show(screenshot="geometry_overview.png")
pl.close()
import matplotlib.pyplot as plt
img = plt.imread("geometry_overview.png")
plt.figure(figsize=(10, 8)); plt.imshow(img); plt.axis("off"); plt.title("green=inlet, red=outlet, blue=outer cowl, tan=core cowl"); plt.show()

# -- cell 5 -------------------------------------------------------------------------
import os
os.makedirs("system", exist_ok=True)

open("system/controlDict","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     snappyHexMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")

# background box: 1.0 x 1.0 x 1.0 m, 20^3 cells -> 50 mm base cells
open("system/blockMeshDict","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale 1;
vertices
(
    (-0.5 -0.5 -0.05)
    ( 0.5 -0.5 -0.05)
    ( 0.5  0.5 -0.05)
    (-0.5  0.5 -0.05)
    (-0.5 -0.5  0.95)
    ( 0.5 -0.5  0.95)
    ( 0.5  0.5  0.95)
    (-0.5  0.5  0.95)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 20) simpleGrading (1 1 1)
);
boundary
(
    background
    {
        type patch;
        faces
        (
            (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)
        );
    }
);
""")

open("system/snappyHexMeshDict","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    outer_cowl  { type triSurfaceMesh; file "outer_cowl.stl"; }
    core_cowl   { type triSurfaceMesh; file "core_cowl.stl"; }
    inlet       { type triSurfaceMesh; file "inlet.stl"; }
    outlet      { type triSurfaceMesh; file "outlet.stl"; }
}

castellatedMeshControls
{
    maxLocalCells       1000000;
    maxGlobalCells      3000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;
    features
    (
        { file "outer_cowl.eMesh"; level 2; }
        { file "core_cowl.eMesh";  level 2; }
        { file "inlet.eMesh";      level 2; }
        { file "outlet.eMesh";     level 2; }
    );
    refinementSurfaces
    {
        outer_cowl  { level (2 2); }
        core_cowl   { level (2 2); }
        inlet       { level (2 2); }
        outlet      { level (2 2); }
    }
    resolveFeatureAngle 30;
    refinementRegions   {}
    locationInMesh      (0.36 0.0 0.45);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      100;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers {}
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
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}

mergeTolerance 1e-6;
""")

open("system/surfaceFeaturesDict","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}
surfaces ("outer_cowl.stl" "core_cowl.stl" "inlet.stl" "outlet.stl");
includedAngle 150;
writeObj yes;
""")
print("dicts written:", sorted(os.listdir("system")))

# -- cell 6 -------------------------------------------------------------------------
# `surfaceFeatureExtract` failed — checking what it couldn't read:
import subprocess
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.stdout[-1500:]); print("STDERR:", r.stderr[-1500:])

# -- cell 7 -------------------------------------------------------------------------
import shutil, subprocess
shutil.copy("system/surfaceFeaturesDict", "system/surfaceFeatureExtractDict")
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print("rc=", r.returncode)
print("\n".join((r.stdout + r.stderr).splitlines()[-12:]))

# -- cell 8 -------------------------------------------------------------------------
import subprocess
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappyHexMesh","w"), stderr=subprocess.STDOUT)
rc = p.wait()
print("snappy rc=", rc)
print("\n".join(open("log.snappyHexMesh").read().splitlines()[-25:]))

# -- cell 9 -------------------------------------------------------------------------
# snappyHexMesh in this version wants `fvSchemes`/`fvSolution` present even though we never solve. Wri
import subprocess
open("system/fvSchemes","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes      {}
gradSchemes     {}
divSchemes      {}
laplacianSchemes {}
interpolationSchemes {}
snGradSchemes   {}
""")
open("system/fvSolution","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
""")
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappyHexMesh","w"), stderr=subprocess.STDOUT)
rc = p.wait()
print("snappy rc=", rc)
print("\n".join(open("log.snappyHexMesh").read().splitlines()[-20:]))

# -- cell 10 ------------------------------------------------------------------------
# The eMesh files didn't land where expected — checking what surfaceFeatureExtract actually wrote:
import subprocess
print(subprocess.run(["find","constant","-name","*eMesh*","-o","-name","*.obj"],capture_output=True,text=True).stdout)
print(subprocess.run(["ls","-R","constant"],capture_output=True,text=True).stdout)

# -- cell 11 ------------------------------------------------------------------------
# It wrote nothing at all — running it again with full output to see what it parsed:
import subprocess
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.stdout)

# -- cell 12 ------------------------------------------------------------------------
import subprocess, os
print(subprocess.run(["bash","-c",'find $FOAM_TUTORIALS -name "surfaceFeatureExtractDict" | head -5'],capture_output=True,text=True).stdout)
print(subprocess.run(["bash","-c",'find $FOAM_TUTORIALS -name "surfaceFeatureExtractDict" | head -1 | xargs cat'],capture_output=True,text=True).stdout)

# -- cell 13 ------------------------------------------------------------------------
import subprocess
surfs = ["outer_cowl.stl", "core_cowl.stl", "inlet.stl", "outlet.stl"]
body = "".join(f"""{s}
{{
    extractionMethod    extractFromSurface;
    writeObj            yes;
    extractFromSurfaceCoeffs
    {{
        includedAngle   150;
    }}
}}

""" for s in surfs)
open("system/surfaceFeatureExtractDict","w").write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}
""" + body)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print("rc=", r.returncode)
print("\n".join(r.stdout.splitlines()[-10:]))
print(subprocess.run(["ls","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 14 ------------------------------------------------------------------------
# eMeshes extracted. Running snappyHexMesh:
import subprocess
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappyHexMesh","w"), stderr=subprocess.STDOUT)
rc = p.wait()
print("snappy rc=", rc)
lines = open("log.snappyHexMesh").read().splitlines()
import re
for l in lines:
    if re.search(r"cells:|Layer mesh|Finished|patch.*faces|Final", l):
        print(l)
print("\n".join(lines[-12:]))

# -- cell 15 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print("rc=", r.returncode)
# print mesh stats + patch table + verdict lines
import re
keep = []
capture = False
for l in out.splitlines():
    if l.startswith("Mesh stats") or l.startswith("Checking patch") or re.match(r"\s+(cells|faces|points|cells per cell|boundary faces)", l):
        capture = True
    if "Mesh stats" in l or "Overall" in l or "boundary" in l.lower() and "faces" in l.lower():
        pass
print("\n".join([l for l in out.splitlines() if re.search(r"(points:|faces:|cells:|Mesh stats|OK\.|FAILED|patch\s|^\s+\w+\s+\d+\s+\d+$|Max aspect|non-orthogonality|skewness)", l)]))
print("---- boundary file ----")
print(open("constant/polyMesh/boundary").read())

# -- cell 16 ------------------------------------------------------------------------
import subprocess
print(subprocess.run(["find","VTK","-type","f"],capture_output=True,text=True).stdout[:800])

# -- cell 17 ------------------------------------------------------------------------
import pyvista as pv, matplotlib.pyplot as plt
pv.OFF_SCREEN = True
grid = pv.read("VTK/t16_0/internal.vtu")
sl = grid.slice(normal=(0,1,0), origin=(0,0,0))
pl = pv.Plotter()
pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
pl.view_vector((0,-1,0), viewup=(0,0,1))
pl.show(screenshot="mesh_slice.png")
pl.close()

# 3D view of boundary patches from the mesh itself
pl2 = pv.Plotter()
for name, col in [("outer_cowl","lightblue"),("core_cowl","tan"),("inlet","lime"),("outlet","tomato")]:
    pl2.add_mesh(pv.read(f"VTK/t16_0/boundary/{name}.vtp"), color=col, show_edges=False,
                 opacity=0.55 if "cowl" in name else 1.0)
pl2.camera_position = [(2.0, 0.5, 0.45), (0, 0, 0.45), (0, 0, 1)]
pl2.show(screenshot="mesh_patches.png")
pl2.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(plt.imread("mesh_slice.png")); ax[0].axis("off"); ax[0].set_title("mesh slice at y=0")
ax[1].imshow(plt.imread("mesh_patches.png")); ax[1].axis("off"); ax[1].set_title("mesh boundary patches")
plt.show()

# -- cell 18 ------------------------------------------------------------------------
import subprocess, re
d = open("system/snappyHexMeshDict").read()
d = d.replace("level 2;", "level 3;").replace("level (2 2);", "level (3 3);")
d = d.replace("maxGlobalCells      3000000;", "maxGlobalCells      6000000;")
open("system/snappyHexMeshDict","w").write(d)
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappyHexMesh","w"), stderr=subprocess.STDOUT)
rc = p.wait()
print("snappy rc=", rc)
lines = open("log.snappyHexMesh").read().splitlines()
for l in lines:
    if re.search(r"Refined mesh|Snapped mesh|Finished meshing|Layer mesh", l):
        print(l)

# -- cell 19 ------------------------------------------------------------------------
print("\n".join(open("log.snappyHexMesh").read().splitlines()[-15:]))

# -- cell 20 ------------------------------------------------------------------------
import re
lines = open("log.snappyHexMesh").read().splitlines()
print("lines so far:", len(lines))
print("\n".join(lines[-8:]))

# -- cell 21 ------------------------------------------------------------------------
import subprocess, re
r = subprocess.run(["blockMesh"], capture_output=True, text=True); assert r.returncode == 0
p = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=open("log.snappyHexMesh","w"), stderr=subprocess.STDOUT)
rc = p.wait()
print("snappy rc=", rc)
for l in open("log.snappyHexMesh").read().splitlines():
    if re.search(r"Refined mesh|Snapped mesh|Finished meshing|FATAL", l):
        print(l)

# -- cell 22 ------------------------------------------------------------------------
import subprocess, re
print("log tail:")
print("\n".join(open("log.snappyHexMesh").read().splitlines()[-3:]))
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("checkMesh rc=", r.returncode)
for l in r.stdout.splitlines():
    if re.search(r"(cells:|Max aspect|Total volume|non-orthogonality Max|Max skewness|Mesh OK|FAILED)", l):
        print(l)

# -- cell 23 ------------------------------------------------------------------------
import subprocess, re, glob, shutil, os
# confirm final patch table
b = open("constant/polyMesh/boundary").read()
print(re.findall(r"\n    (\w+)\n    \{\n        type\s+(\w+);\n        inGroups.*\n        nFaces\s+(\d+);", b))

# refresh VTK for a look at the refined mesh
shutil.rmtree("VTK", ignore_errors=True)
r = subprocess.run(["foamToVTK","-latestTime"], capture_output=True, text=True); print("foamToVTK rc=", r.returncode)
import pyvista as pv, matplotlib.pyplot as plt
pv.OFF_SCREEN = True
grid = pv.read("VTK/t16_0/internal.vtu")
sl = grid.slice(normal=(0,1,0), origin=(0,0,0))
pl = pv.Plotter(); pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
pl.view_vector((0,-1,0), viewup=(0,0,1)); pl.show(screenshot="mesh_slice_final.png"); pl.close()
plt.figure(figsize=(7,8)); plt.imshow(plt.imread("mesh_slice_final.png")); plt.axis("off")
plt.title("refined mesh slice at y=0"); plt.show()

# -- cell 24 ------------------------------------------------------------------------
import numpy as np, build123d as bd, inspect
print(inspect.signature(bd.Shape.tessellate))
# force one consistent BRepMesh over the whole solid, then per-face triangulations share seam nodes
_ = fluid.tessellate(1e-4, angular_tolerance=0.15)
for name, f in patches.items():
    bd.export_stl(f, f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.15)

# weld-check the union in python: count edges used exactly once
import pyvista as pv
all_pts, all_tris, off = [], [], 0
for name in ["outer_cowl","core_cowl","inlet","outlet"]:
    m = pv.read(f"constant/triSurface/{name}.stl")
    all_pts.append(m.points)
    all_tris.append(m.faces.reshape(-1,4)[:,1:] + off)
    off += m.n_points
pts = np.vstack(all_pts); tris = np.vstack(all_tris)
key = np.round(pts / 1e-7).astype(np.int64)
uniq, inv = np.unique(key, axis=0, return_inverse=True)
edges = {}
for t in inv[tris]:
    for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
        e = (min(a,b), max(a,b))
        edges[e] = edges.get(e,0)+1
free = [e for e,c in edges.items() if c==1]
print("welded vertices: %d -> %d unique" % (len(pts), len(uniq)))
print("free edges in welded union:", len(free))
