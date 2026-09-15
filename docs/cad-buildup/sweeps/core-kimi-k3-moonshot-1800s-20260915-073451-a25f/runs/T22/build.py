"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first build cell errored on `geom_type()`, so nothing from it is in the replay script. I'll re-e
import os, math, collections
import build123d as bd

# --- parameters (metres; request gives mm) ---
R_OUT  = 0.160          # disc outside radius (320 mm dia)
R_IN   = 0.095          # disc inside radius (190 mm dia)
GAP    = 0.012          # gap between friction faces
PLATE_T= 0.010          # friction face thickness (context only, not part of fluid)
N_VANE = 36
VANE_W = 0.008          # vane tangential width

# Fluid annulus: ID 190 -> OD 320, height GAP, centred on z=0
annulus = bd.Cylinder(radius=R_OUT, height=GAP) - bd.Cylinder(radius=R_IN, height=GAP)

# One vane: box spanning slightly beyond ID/OD for a clean boolean, 8 mm wide, full gap height
vane_proto = bd.Pos(R_IN - 0.001 + (R_OUT - R_IN + 0.002)/2, 0, 0) * bd.Box(R_OUT - R_IN + 0.002, VANE_W, GAP)
vanes = bd.Compound([bd.Rot(0, 0, i*360.0/N_VANE) * vane_proto for i in range(N_VANE)])

fluid = annulus - vanes
print("fluid volume (m^3):", fluid.volume, " expected-ish:", math.pi*(R_OUT**2-R_IN**2)*GAP - N_VANE*(R_OUT-R_IN)*VANE_W*GAP)
print("n faces:", len(fluid.faces()))

# census faces by type / radius / height / normal
census = collections.Counter()
detail = collections.defaultdict(float)
for f in fluid.faces():
    g = str(f.geom_type)
    c = f.center()
    r = math.hypot(c.X, c.Y)
    n = f.normal_at()
    key = (g.split('.')[-1], round(r,3), round(c.Z,4), (round(n.X,2), round(n.Y,2), round(n.Z,2)))
    census[key] += 1
    detail[key] += f.area
for k, v in sorted(census.items()):
    print("%3d faces of" % v, k, " total area %.6f" % detail[k])

# -- cell 2 -------------------------------------------------------------------------
import inspect
print(inspect.signature(bd.export_stl))

os.makedirs("constant/triSurface", exist_ok=True)

patches = {"inlet": [], "outlet": [], "friction_face_upper": [], "friction_face_lower": [], "vanes": []}
for f in fluid.faces():
    c = f.center(); r = math.hypot(c.X, c.Y); n = f.normal_at()
    g = str(f.geom_type).split('.')[-1]
    if g == "CYLINDER":
        patches["inlet" if r < 0.13 else "outlet"].append(f)
    elif abs(n.Z) > 0.9:
        patches["friction_face_upper" if n.Z > 0 else "friction_face_lower"].append(f)
    else:
        patches["vanes"].append(f)

for name, faces in patches.items():
    comp = bd.Compound(faces)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=2e-5, angular_tolerance=0.1)
    print(name, len(faces), "faces, area %.6f" % sum(f.area for f in faces), "->", os.path.getsize(f"constant/triSurface/{name}.stl"), "bytes")

# sanity totals
tot = sum(sum(f.area for f in v) for v in patches.values())
print("total boundary area %.6f, fluid face area %.6f" % (tot, sum(f.area for f in fluid.faces())))

# -- cell 3 -------------------------------------------------------------------------
# Areas balance exactly against the fluid boundary. Now a look at the assembled patches:
import pyvista as pv
pv.set_jupyter_backend("static")
colors = {"inlet":"red","outlet":"blue","friction_face_upper":"tan","friction_face_lower":"brown","vanes":"grey"}
pl = pv.Plotter(shape=(1,2), window_size=(1400,650))
pl.subplot(0,0)
for name, col in colors.items():
    m = pv.read(f"constant/triSurface/{name}.stl")
    pl.add_mesh(m, color=col, label=f"{name} ({m.n_cells})")
pl.add_legend(); pl.add_text("assembled patches"); pl.view_isometric()
pl.subplot(0,1)
for name in ["inlet","vanes"]:
    pl.add_mesh(pv.read(f"constant/triSurface/{name}.stl"), color=colors[name], label=name)
pl.add_legend(); pl.add_text("inlet + vanes only"); pl.view_isometric()
pl.show()

# -- cell 4 -------------------------------------------------------------------------
# The assembly looks right: red inlet ring, blue outlet ring, tan/brown friction faces, grey vanes sta
os.makedirs("system", exist_ok=True)

# background box: encloses STL (r 0.095..0.160, z +/-0.006) with margin
block = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale 1;
vertices
(
    (-0.166 -0.166 -0.0072)
    ( 0.166 -0.166 -0.0072)
    ( 0.166  0.166 -0.0072)
    (-0.166  0.166 -0.0072)
    (-0.166 -0.166  0.0072)
    ( 0.166 -0.166  0.0072)
    ( 0.166  0.166  0.0072)
    (-0.166  0.166  0.0072)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (80 80 4) simpleGrading (1 1 1)
);
boundary
(
    outerBox
    {
        type patch;
        faces
        (
            (0 3 2 1) (4 5 6 7) (0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7)
        );
    }
);
"""
open("system/blockMeshDict","w").write(block)

import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-500:], r.stderr[-300:])

# -- cell 5 -------------------------------------------------------------------------
# Missing `controlDict`. Adding it plus `fvSchemes`/`fvSolution` stubs, then re-running blockMesh:
control = """FoamFile
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
"""
open("system/controlDict","w").write(control)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-400:])
print(r.stderr[-200:])

# -- cell 6 -------------------------------------------------------------------------
# Background mesh is up (25,600 cells, ~4 mm xy / 3.6 mm z). Now the snappyHexMesh dictionaries — one 
shm = """FoamFile
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
    inlet               { type triSurfaceMesh; file "inlet.stl"; }
    outlet              { type triSurfaceMesh; file "outlet.stl"; }
    friction_face_upper { type triSurfaceMesh; file "friction_face_upper.stl"; }
    friction_face_lower { type triSurfaceMesh; file "friction_face_lower.stl"; }
    vanes               { type triSurfaceMesh; file "vanes.stl"; }
}

castellatedMeshControls
{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;
    features            ();
    refinementSurfaces
    {
        inlet               { level (2 2); }
        outlet              { level (2 2); }
        friction_face_upper { level (2 2); }
        friction_face_lower { level (2 2); }
        vanes               { level (2 2); }
    }
    resolveFeatureAngle 30;
    refinementRegions   ();
    locationInMesh      (0.12701 0.01112 0.0);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers ();
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
"""
open("system/snappyHexMeshDict","w").write(shm)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print("sfe:", r.returncode, r.stdout[-200:], r.stderr[-200:])

# -- cell 7 -------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.returncode)
print(r.stdout[-1500:])
print("STDERR:", r.stderr[-300:])

# -- cell 8 -------------------------------------------------------------------------
# snappy wants `fvSchemes` present. Adding both fv stubs and re-running:
fvs = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes  { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes  { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
fvsol = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
"""
open("system/fvSchemes","w").write(fvs)
open("system/fvSolution","w").write(fvsol)
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.returncode)
print(r.stdout[-1200:])
print("STDERR:", r.stderr[-200:])

# -- cell 9 -------------------------------------------------------------------------
print(r.stderr[-1500:])

# -- cell 10 ------------------------------------------------------------------------
# `refinementRegions` must be a dictionary in this version. Fixing and re-running:
shm = open("system/snappyHexMeshDict").read().replace("refinementRegions   ();", "refinementRegions   {}")
open("system/snappyHexMeshDict","w").write(shm)
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.returncode)
print(r.stdout[-1500:])
print("STDERR:", r.stderr[-200:])

# -- cell 11 ------------------------------------------------------------------------
# Snappy finished cleanly: 8,330 cells, zero quality violations. Running checkMesh and looking at the 
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-2200:])

# -- cell 12 ------------------------------------------------------------------------
pts = []
for i in range(N_VANE):
    a = math.radians((i + 0.5) * 360.0 / N_VANE)
    pts.append("(%.6f %.6f 0.0)" % (0.1275*math.cos(a), 0.1275*math.sin(a)))
loc_block = "locationsInMesh (\n    " + "\n    ".join(pts) + "\n);"
shm = open("system/snappyHexMeshDict").read().replace(
    "locationInMesh      (0.12701 0.01112 0.0);", loc_block)
open("system/snappyHexMeshDict","w").write(shm)
print(loc_block[:300], "...")
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.returncode)
tail = r.stdout
import re
m = re.findall(r"Snapped mesh : cells:(\d+)", tail)
print("snapped cells:", m)
print(tail[-400:])

# -- cell 13 ------------------------------------------------------------------------
print("STDERR:", r.stderr[-800:])
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh:", r.returncode)
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print("snappy:", r.returncode)
print(r.stdout[-600:])
print("STDERR:", r.stderr[-300:])

# -- cell 14 ------------------------------------------------------------------------
# `locationsInMesh` in v2512 wants `(point zoneName)` tuples. Reformatting with zone name `none`:
pts = []
for i in range(N_VANE):
    a = math.radians((i + 0.5) * 360.0 / N_VANE)
    pts.append("((%.6f %.6f 0.0) none)" % (0.1275*math.cos(a), 0.1275*math.sin(a)))
loc_block = "locationsInMesh\n(\n    " + "\n    ".join(pts) + "\n);"
shm = open("system/snappyHexMeshDict").read()
import re as _re
shm = _re.sub(r"locationsInMesh \(\n(?:.*\n)*?\);", loc_block, shm, count=1)
open("system/snappyHexMeshDict","w").write(shm)
subprocess.run(["grep","-n","-A3","locationsInMesh","system/snappyHexMeshDict"])
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print("snappy:", r.returncode)
print(r.stdout[-400:])
print("STDERR:", r.stderr[-200:])

# -- cell 15 ------------------------------------------------------------------------
print(r.stderr[-1200:])
i = r.stdout.find("Refinement phase")
print(r.stdout[i:i+1500])

# -- cell 16 ------------------------------------------------------------------------
# The full 36-passage mesh outran the 240 s window and was killed. Relaunching snappy in the backgroun
subprocess.run(["blockMesh"], capture_output=True)  # ensure clean background mesh
log = open("snappy.log","w")
proc = subprocess.Popen(["snappyHexMesh", "-overwrite"], stdout=log, stderr=subprocess.STDOUT)
print("pid", proc.pid)

# -- cell 17 ------------------------------------------------------------------------
import time
time.sleep(120)
print("running:", proc.poll() is None)
print(open("snappy.log").read()[-800:])

# -- cell 18 ------------------------------------------------------------------------
# Still in the snap phase. Polling again:
time.sleep(150)
print("running:", proc.poll() is None)
print(open("snappy.log").read()[-600:])

# -- cell 19 ------------------------------------------------------------------------
time.sleep(120)
print("running:", proc.poll() is None)
print(open("snappy.log").read()[-500:])

# -- cell 20 ------------------------------------------------------------------------
import re
log_txt = open("snappy.log").read()
m = re.findall(r"(?:Snapped|Morph) mesh : cells:(\d+)", log_txt)
print("cells:", m)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-2000:])

# -- cell 21 ------------------------------------------------------------------------
r = subprocess.run(["foamToVTK"], capture_output=True, text=True)
print(r.returncode, r.stdout[-200:])
import glob
print(glob.glob("VTK/**/*.vtk", recursive=True)[:5])

# -- cell 22 ------------------------------------------------------------------------
import glob
vtks = glob.glob("VTK/t22_0/**/*.vt*", recursive=True)
print(vtks)
mesh_vtk = [f for f in vtks if f.endswith(".vtk") or "internal" in f]
m = pv.read([f for f in vtks if not f.endswith(".vtp")][0])
clip = m.clip(normal="z", origin=(0,0,0.003))
pl = pv.Plotter(shape=(1,2), window_size=(1500,650))
pl.subplot(0,0)
pl.add_mesh(clip, show_edges=False, color="lightblue")
for name, col in [("vanes","grey"),("inlet","red"),("outlet","blue")]:
    pl.add_mesh(pv.read(f"VTK/t22_0/boundary/{name}.vtp"), color=col)
pl.add_text("clipped mesh + walls"); pl.view_isometric()
pl.subplot(0,1)
slice2 = m.slice(normal="z", origin=(0,0,0))
pl.add_mesh(slice2, show_edges=True, color="lightblue")
pl.add_text("mid-gap slice"); pl.view_xy()
pl.show()

# -- cell 23 ------------------------------------------------------------------------
sfe = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}
surfaces
(
    "inlet.stl"
    "outlet.stl"
    "friction_face_upper.stl"
    "friction_face_lower.stl"
    "vanes.stl"
);
includedAngle   150;
subsetFeatures
{
    nonManifoldEdges    no;
    openEdges           yes;
}
writeObj        no;
"""
open("system/surfaceFeatureExtractDict","w").write(sfe)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-200:])
print(glob.glob("constant/extendedFeatureEdgeMesh/*"))

# -- cell 24 ------------------------------------------------------------------------
print(r.stdout)
print(glob.glob("constant/**/*.eMesh", recursive=True), glob.glob("**/*.eMesh", recursive=True))

# -- cell 25 ------------------------------------------------------------------------
r2 = subprocess.run(["surfaceFeatureExtract", "-help"], capture_output=True, text=True)
print(r2.stdout[-1500:])
