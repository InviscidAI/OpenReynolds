"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The `Compound.make_compound` API doesn't exist in this build123d version; I'll just subtract the van
import math
import build123d as bd

R_OUT = 0.160          # 320 mm OD
R_IN  = 0.095          # 190 mm ID
GAP   = 0.012          # 12 mm between friction faces
N_VANES = 36
VANE_W = 0.008         # 8 mm vane width

# annular gap solid, base at z=0
gap_solid = bd.Pos(0, 0, GAP/2) * bd.Cylinder(radius=R_OUT, height=GAP)
gap_solid = gap_solid - (bd.Pos(0, 0, GAP/2) * bd.Cylinder(radius=R_IN, height=GAP + 0.002))

# one radial vane along +X, spanning inner to outer radius with margin, full gap height
vane_len = (R_OUT - R_IN) + 0.010
vane = bd.Pos((R_IN + R_OUT)/2, 0, GAP/2) * bd.Box(vane_len, VANE_W, GAP + 0.002)
fluid = gap_solid
for i in range(N_VANES):
    fluid = fluid - (bd.Rot(0, 0, i * 360.0/N_VANES) * vane)

vol_expected = math.pi*(R_OUT**2 - R_IN**2)*GAP - N_VANES*(R_OUT - R_IN)*VANE_W*GAP
print(f"fluid volume: built {fluid.volume:.6e} m^3, expected {vol_expected:.6e} m^3")
print(f"solids: {len(fluid.solids())}, faces: {len(fluid.faces())}")
print(f"bbox min: {fluid.bounding_box().min}, max: {fluid.bounding_box().max}")

# -- cell 2 -------------------------------------------------------------------------
# `Vector` uses uppercase `.X/.Y/.Z` in this build123d. Fixing once more.
def face_kind(f):
    c = f.center()
    n = f.normal_at()
    if f.geom_type == bd.GeomType.PLANE:
        if abs(n.Z) > 0.999:
            return "lower" if c.Z < GAP/2 else "upper"
        return "vane"
    r = math.hypot(c.X, c.Y)
    return "inlet" if r < (R_IN + R_OUT)/2 else "outlet"

groups = {"lower": [], "upper": [], "vane": [], "inlet": [], "outlet": []}
for f in fluid.faces():
    groups[face_kind(f)].append(f)

for k, v in groups.items():
    area = sum(f.area for f in v)
    print(f"{k:6s}: {len(v):3d} faces, area {area:.6e} m^2")

print(f"expected inlet  = {2*math.pi*R_IN*GAP - N_VANES*VANE_W*GAP:.6e} m^2")
print(f"expected outlet = {2*math.pi*R_OUT*GAP - N_VANES*VANE_W*GAP:.6e} m^2")
print(f"expected friction face = {math.pi*(R_OUT**2-R_IN**2) - N_VANES*(R_OUT-R_IN)*VANE_W:.6e} m^2")
print(f"expected vane sides = {2*N_VANES*(R_OUT-R_IN)*GAP:.6e} m^2")

# -- cell 3 -------------------------------------------------------------------------
# All 216 faces classified: 36+36 friction, 72 vane, 36 inlet, 36 outlet, areas matching hand calculat
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

patch_names = {
    "lower":  "frictionFaceLower",
    "upper":  "frictionFaceUpper",
    "vane":   "vanes",
    "inlet":  "inlet",
    "outlet": "outlet",
}
for key, name in patch_names.items():
    comp = bd.Compound(children=groups[key])
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-5, angular_tolerance=0.1)
    print(name, os.path.getsize(f"constant/triSurface/{name}.stl"), "bytes")

# render the fluid solid
import pyvista as pv
pv.OFF_SCREEN = True
verts, tris = fluid.tessellate(tolerance=1e-4)
import numpy as np
import pyvista as pv
mesh = pv.PolyData(np.array([(v.X, v.Y, v.Z) for v in verts]),
                   np.hstack([[3]+list(t) for t in tris]).astype(np.int64))
pl = pv.Plotter(off_screen=True)
pl.add_mesh(mesh, color="lightsteelblue", show_edges=False)
pl.camera_position = 'iso'
pl.screenshot("fluid_view.png")
print("saved fluid_view.png")

# -- cell 4 -------------------------------------------------------------------------
# Understood — I'll display figures as last expressions or import `display` explicitly from here on. T
# Now the mesh plan: `blockMesh` background box slightly larger than the disc, then `snappyHexMesh` ag
import os
os.makedirs("system", exist_ok=True)

blockMeshDict = r"""/*--------------------------------*- C++ -*----------------------------------*\
  Vented brake disc air passages - background mesh (metres)
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1;

vertices
(
    (-0.170 -0.170 -0.005)
    ( 0.170 -0.170 -0.005)
    ( 0.170  0.170 -0.005)
    (-0.170  0.170 -0.005)
    (-0.170 -0.170  0.017)
    ( 0.170 -0.170  0.017)
    ( 0.170  0.170  0.017)
    (-0.170  0.170  0.017)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (85 85 6) simpleGrading (1 1 1)
);

boundary
(
    outerBox
    {
        type patch;
        faces
        (
            (0 1 5 4)
            (3 2 6 7)
            (0 3 7 4)
            (1 2 6 5)
            (0 1 2 3)
            (4 5 6 7)
        );
    }
);
"""
open("system/blockMeshDict", "w").write(blockMeshDict)

# locationInMesh: r=0.1275 at 5 deg (mid-passage between vane 0 and vane 1), z mid-gap
lx = 0.1275*math.cos(math.radians(5)); ly = 0.1275*math.sin(math.radians(5))
print("locationInMesh:", lx, ly, 0.006)

snappy = f"""/*--------------------------------*- C++ -*----------------------------------*\
  snappyHexMeshDict - vented brake disc air passages
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    frictionFaceLower {{ type triSurfaceMesh; file "frictionFaceLower.stl"; }}
    frictionFaceUpper {{ type triSurfaceMesh; file "frictionFaceUpper.stl"; }}
    vanes             {{ type triSurfaceMesh; file "vanes.stl"; }}
    inlet             {{ type triSurfaceMesh; file "inlet.stl"; }}
    outlet            {{ type triSurfaceMesh; file "outlet.stl"; }}
}}

castellatedMeshControls
{{
    maxLocalCells       10000000;
    maxGlobalCells      20000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;

    features ();

    refinementSurfaces
    {{
        frictionFaceLower {{ level (2 2); patchInfo {{ type wall; }} }}
        frictionFaceUpper {{ level (2 2); patchInfo {{ type wall; }} }}
        vanes             {{ level (2 2); patchInfo {{ type wall; }} }}
        inlet             {{ level (2 2); patchInfo {{ type patch; }} }}
        outlet            {{ level (2 2); patchInfo {{ type patch; }} }}
    }}

    resolveFeatureAngle 30;

    refinementRegions {{}}

    locationInMesh ({lx:.8f} {ly:.8f} 0.006);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      30;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
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
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict", "w").write(snappy)

controlDict = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
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
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
open("system/controlDict", "w").write(controlDict)
print("dictionaries written")

# -- cell 5 -------------------------------------------------------------------------
# The controlDict banner is malformed (unterminated comment would comment out the whole file). Rewriti
controlDict = r"""/*--------------------------------*- C++ -*----------------------------------*\
| snappyHexMesh case                                                          |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
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
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
open("system/controlDict", "w").write(controlDict)
print(open("system/controlDict").read()[:200])

# -- cell 6 -------------------------------------------------------------------------
# Now run `blockMesh` and then `snappyHexMesh` (coarse level 2 refinement).
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:])
print("RC:", r.returncode, r.stderr[-500:])

# -- cell 7 -------------------------------------------------------------------------
# Background mesh: 43,350 cells at 4 mm. Running `snappyHexMesh`.
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.stdout[-2500:])
print("RC:", r.returncode, r.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
# snappyHexMesh wants `fvSchemes`/`fvSolution` present. Writing minimal ones and re-running.
fvSchemes = r"""/*--------------------------------*- C++ -*----------------------------------*\
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""
fvSolution = r"""/*--------------------------------*- C++ -*----------------------------------*\
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
solvers {}
relaxationFactors {}
"""
open("system/fvSchemes", "w").write(fvSchemes)
open("system/fvSolution", "w").write(fvSolution)
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r.stdout[-2000:])
print("RC:", r.returncode, r.stderr[-800:])

# -- cell 9 -------------------------------------------------------------------------
# snappy finished cleanly (8,946 cells, no quality errors). Running `checkMesh` and inspecting the pat
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout)
print("RC:", r.returncode, r.stderr[-300:])

# -- cell 10 ------------------------------------------------------------------------
pts = []
for i in range(N_VANES):
    a = math.radians(5 + i*10)
    pts.append(f"({0.1275*math.cos(a):.8f} {0.1275*math.sin(a):.8f} 0.006)")
loc_block = "\n        ".join(pts)

snappy_txt = open("system/snappyHexMeshDict").read()
snappy_txt = snappy_txt.replace(
    "locationInMesh (0.12701482 0.01111236 0.006);",
    "locationsInMesh\n    (\n        " + loc_block + "\n    );")
assert "locationsInMesh" in snappy_txt
open("system/snappyHexMeshDict", "w").write(snappy_txt)

r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh RC:", r1.returncode)
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
tail = r2.stdout[-1200:]
print(tail)
print("snappy RC:", r2.returncode, r2.stderr[-400:])

# -- cell 11 ------------------------------------------------------------------------
# The 2512 `locationsInMesh` syntax is more structured than a bare point list. Checking the installed 
r = subprocess.run(["grep", "-rn", "locationsInMesh", "/usr/lib/openfoam/openfoam2512/applications/utilities/mesh/generation/snappyHexMesh/"],
                   capture_output=True, text=True)
print(r.stdout[:3000], r.stderr[:500])
r2 = subprocess.run(["bash", "-c", "find / -name 'snappyHexMeshDict' -path '*tutorials*' 2>/dev/null | head -5"],
                    capture_output=True, text=True)
print(r2.stdout)

# -- cell 12 ------------------------------------------------------------------------
r = subprocess.run(["bash", "-c", "grep -rl 'locationsInMesh' /usr/lib/openfoam/openfoam2512/tutorials/ | head -3"],
                   capture_output=True, text=True)
print(r.stdout)
if r.stdout.strip():
    f = r.stdout.split()[0]
    r2 = subprocess.run(["bash", "-c", f"grep -A 12 'locationsInMesh' {f}"], capture_output=True, text=True)
    print(r2.stdout)

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["bash", "-c", "grep -B2 -A 15 'locationsInMesh' /usr/lib/openfoam/openfoam2512/tutorials/mesh/snappyHexMesh/addLayersToFaceZone/system/snappyHexMeshDict"],
                   capture_output=True, text=True)
print(r.stdout)

# -- cell 14 ------------------------------------------------------------------------
r = subprocess.run(["bash", "-c", "grep -B3 -A 20 'locationsInMesh' /usr/lib/openfoam/openfoam2512/tutorials/mesh/snappyHexMesh/block_with_curvature/system/snappyHexMeshDict"],
                   capture_output=True, text=True)
print(r.stdout[:2500])

# -- cell 15 ------------------------------------------------------------------------
# The docs say each entry pairs a point with a cellZone name, with `none` for unzoned. Checking the so
r = subprocess.run(["bash", "-c", "grep -rn 'locationsInMesh' /usr/lib/openfoam/openfoam2512/src/mesh/snappyHexMesh/ 2>/dev/null | head; grep -rn 'locationsInMesh' /usr/lib/openfoam/openfoam2512/src/ -l 2>/dev/null | head"],
                   capture_output=True, text=True)
print(r.stdout[:2000])

# -- cell 16 ------------------------------------------------------------------------
r = subprocess.run(["bash", "-c", "grep -rn -B3 -A25 '\"locationsInMesh\"' /usr/lib/openfoam/openfoam2512/src/mesh/snappyHexMesh/snappyHexMeshDriver/snappyRefineDriver.C | head -80"],
                   capture_output=True, text=True)
print(r.stdout[:3000])

# -- cell 17 ------------------------------------------------------------------------
r = subprocess.run(["bash", "-c", "grep -rn -B5 -A30 'locationsInMesh' /usr/lib/openfoam/openfoam2512/src/mesh/snappyHexMesh/snappyHexMeshDriver/refinementParameters/refinementParameters.C 2>/dev/null | head -100; find /usr/lib/openfoam/openfoam2512/src -name 'refinementParameters*' 2>/dev/null"],
                   capture_output=True, text=True)
print(r.stdout[:4000])

# -- cell 18 ------------------------------------------------------------------------
# Confirmed: `locationsInMesh` takes `((x y z) zoneName)` tuples; `none` keeps the region without crea
pts = []
for i in range(N_VANES):
    a = math.radians(5 + i*10)
    pts.append(f"(({0.1275*math.cos(a):.8f} {0.1275*math.sin(a):.8f} 0.006) none)")
loc_block = "\n        ".join(pts)

snappy_txt = open("system/snappyHexMeshDict").read()
import re
snappy_txt = re.sub(r"locationsInMesh\s*\(.*?\);",
                    "locationsInMesh\n    (\n        " + loc_block + "\n    );",
                    snappy_txt, flags=re.S)
open("system/snappyHexMeshDict", "w").write(snappy_txt)

r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh RC:", r1.returncode)
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print(r2.stdout[-900:])
print("snappy RC:", r2.returncode, r2.stderr[-300:])

# -- cell 19 ------------------------------------------------------------------------
# All 36 passages kept now: 323k cells, zero quality errors. Verifying with `checkMesh` and rendering 
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
# print the interesting parts
for marker in ["Mesh stats", "Number of regions", "Checking patch topology", "Overall domain bounding box",
               "Boundary openness", "Max aspect ratio", "Total volume", "Mesh non-orthogonality", "Max skewness", "Mesh OK", "Mesh not OK", "Failed"]:
    for line in out.splitlines():
        if marker in line:
            print(line)
print("---patches---")
import re
m = re.search(r"Checking patch topology.*?\n(.*?)\n\n", out, re.S)
print(m.group(1) if m else "not found")
print("RC:", r.returncode)

# -- cell 20 ------------------------------------------------------------------------
r = subprocess.run(["foamToVTK", "-noFaceZones", "-noPointZones"], capture_output=True, text=True, timeout=200)
print("RC:", r.returncode, r.stdout[-400:], r.stderr[-200:])
import glob
print(glob.glob("VTK/**/*.vtu", recursive=True)[:5], glob.glob("VTK/*.vtm"))

# -- cell 21 ------------------------------------------------------------------------
r = subprocess.run(["foamToVTK"], capture_output=True, text=True, timeout=200)
print("RC:", r.returncode, r.stdout[-300:], r.stderr[-200:])
import glob
print(glob.glob("VTK/**/*.vtu", recursive=True)[:8])

# -- cell 22 ------------------------------------------------------------------------
import pyvista as pv
import numpy as np
grid = pv.read("VTK/t22_0/internal.vtu")
print(grid.n_cells, "cells")

# horizontal slice at mid-gap
sl = grid.slice(normal=(0,0,1), origin=(0,0,0.006))
pl = pv.Plotter(off_screen=True)
pl.add_mesh(sl, show_edges=False, color="tomato")
# overlay patch boundaries
for name, col in [("inlet","blue"), ("outlet","green")]:
    p = pv.read(f"VTK/t22_0/boundary/{name}.vtp")
    pl.add_mesh(p, color=col, opacity=0.6)
pl.view_xy()
pl.screenshot("mesh_slice.png")

# isometric view of the mesh surface (cut away top half with a clip)
clipped = grid.clip(normal=(0,0,-1), origin=(0,0,0.008))
pl2 = pv.Plotter(off_screen=True)
pl2.add_mesh(clipped, show_edges=False, color="lightsteelblue")
pl2.camera_position = 'iso'
pl2.screenshot("mesh_iso.png")
print("done")
