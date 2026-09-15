"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, build123d as bd
shape = bd.import_step("/home/qiuzi/.openreynolds-buildup/work/T6-20260914-165436-821f/geometry/customer_part.step")
v = shape.volume
box = 200*100*100
bore = np.pi*20**2*200
print("volume:", v, "box-bore:", box-bore, "through-hole" if abs(v-(box-bore)) < box else "blind")

# -- cell 2 -------------------------------------------------------------------------
# Solid = box minus through bore; the fluid region is the Ø40 × 200 mm bore. Verify the STEP unit decl
import subprocess, re
s = subprocess.run(["grep","-i","-m4","si_unit\|conversion\|length_measure\|millimetre\|metre",
                    "/home/qiuzi/.openreynolds-buildup/work/T6-20260914-165436-821f/geometry/customer_part.step"],
                   capture_output=True, text=True).stdout
print(s[:2000])

# -- cell 3 -------------------------------------------------------------------------
import subprocess
s = subprocess.run(["grep","si_unit","/home/qiuzi/.openreynolds-buildup/work/T6-20260914-165436-821f/geometry/customer_part.step"],
                   capture_output=True, text=True).stdout
print(s)

# -- cell 4 -------------------------------------------------------------------------
import subprocess, re
raw = open("/home/qiuzi/.openreynolds-buildup/work/T6-20260914-165436-821f/geometry/customer_part.step").read()
hits = re.findall(r"SI_UNIT\(.*?LENGTH[^;]*?;", raw, re.I|re.S)
print(hits[:5])

# -- cell 5 -------------------------------------------------------------------------
import build123d as bd, numpy as np
R_MM, L_MM = 20.0, 200.0
cyl = bd.Pos(0,50,50)*bd.Rotation(0,90,0)*bd.Cylinder(R_MM, L_MM)  # axis x, base at x=0?
print("bbox:", cyl.bounding_box())
faces = np.array([tuple(f.center()) for f in cyl.faces()])
print(np.round(faces,1))

# -- cell 6 -------------------------------------------------------------------------
import build123d as bd, numpy as np
R_MM, L_MM = 20.0, 200.0
cyl = bd.Pos(100,50,50)*bd.Rotation(0,90,0)*bd.Cylinder(R_MM, L_MM)
groups = {"inlet":[], "outlet":[], "wall":[]}
for f in cyl.faces() if False else list(cyl.faces()):
    cx = f.center().to_tuple()[0]
    if abs(cx) < 1e-6: groups["inlet"].append(f)
    elif abs(cx-L_MM) < 1e-6: groups["outlet"].append(f)
    else: groups["wall"].append(f)
for name, faces in groups.items():
    scaled = bd.scale(bd.Compound(children=faces), 0.001)
    bd.export_stl(scaled, f"{name}.stl", ascii_format=True)
    print(name, len(faces), "exported")

# -- cell 7 -------------------------------------------------------------------------
# STLs exported. Now OpenFOAM files: blockMesh (background box), surfaceFeatureExtract, snappyHexMesh.
import pathlib
# background box
bm = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
scale 1;
vertices
(
    (-0.02 0.03 0.03)
    ( 0.22 0.03 0.03)
    ( 0.22 0.07 0.03)
    (-0.02 0.07 0.03)
    (-0.02 0.03 0.07)
    ( 0.22 0.03 0.07)
    ( 0.22 0.07 0.07)
    (-0.02 0.07 0.07)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (48 8 8) simpleGrading (1 1 1)
);
boundary
(
    bgwalls
    {
        type wall;
        faces ( (0 4 5 1) (2 6 7 3) (0 1 5 4) (3 2 6 7) (0 3 7 4) (1 2 6 5) );
    }
);
"""
pathlib.Path("system/blockMeshDict").parent.mkdir(parents=True, exist_ok=True)
pathlib.Path("system/blockMeshDict").write_text(bm)
print("blockMeshDict written")

# -- cell 8 -------------------------------------------------------------------------
import pathlib
shmd = """FoamFile
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
    inlet.stl { type triSurface; }
    outlet.stl { type triSurface; }
    wall.stl { type triSurface; }
}
castellatedMeshControls
{
    locationInMesh (0.10 0.05 0.05);
    allowFreeStandingZoneFaces false;
    nCellsBetweenLevels 1;
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    resolveFeatureAngle 30;
    features ();
    features
    (
        { file "inlet.eMesh"; level 0; }
        { file "outlet.eMesh"; level 0; }
        { file "wall.eMesh"; level 1; }
    );
    refinementSurfaces
    {
        inlet.stl { level (0 0); }
        outlet.stl { level (0 0); }
        wall.stl { level (1 1); }
    }
    refinementRegions {}
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes false; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 5; nSurfaceLayers 1; }
meshQualityControls
{
    includeBbox 1;
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minFlatness 0.5;
    minVol 1e-13;
    minTetQuality -1;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1E-06;
"""
pathlib.Path("system/snappyHexMeshDict").write_text(shmd)
print("snappyHexMeshDict written")

# -- cell 9 -------------------------------------------------------------------------
import pathlib
fvs = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default none; }
gradSchemes { default none; }
divSchemes { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default none; }
snGradSchemes { default none; }
wallDist { method meshWave; }
"""
fvsol = """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
"""
pathlib.Path("system/fvSchemes").write_text(fvs)
pathlib.Path("system/fvSolution").write_text(fvsol)
cd = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application internal;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
"""
pathlib.Path("system/controlDict").write_text(cd)
import subprocess
for cmd in (["blockMesh"], ["surfaceFeatureExtract"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(cmd[0], "rc", r.returncode); print(r.stdout[-1500:])

# -- cell 10 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stderr[-800:]); print(r.stdout[-800:])

# -- cell 11 ------------------------------------------------------------------------
# Boundary face duplicated; fixing face list properly.
import pathlib, subprocess
p = pathlib.Path("system/blockMeshDict")
t = p.read_text().replace("(0 4 5 1) (2 6 7 3) (0 1 5 4) (3 2 6 7) (0 1 5 4) (1 2 6 5)",
  "(0 3 2 1) (4 5 6 7) (0 1 5 4) (3 2 6 7) (0 4 7 3) (1 2 6 5)")
p.write_text(t)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:], r.stderr[-300:])

# -- cell 12 ------------------------------------------------------------------------
print(open("system/blockMeshDict").read())

# -- cell 13 ------------------------------------------------------------------------
import pathlib, subprocess
p = pathlib.Path("system/blockMeshDict")
t = p.read_text()
t = t.replace("faces ( (0 4 5 1) (2 6 7 3) (0 1 5 4) (3 2 6 7) (0 3 7 4) (1 2 6 5) );",
              "faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (3 2 6 7) (0 4 7 3) (1 2 6 5) );")
p.write_text(t)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stderr[-300:])

# -- cell 14 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-400:])

# -- cell 15 ------------------------------------------------------------------------
import pathlib
d = """FoamFile { version 2.0; format ascii; object surfaceFeatureExtractDict; }
inlet.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
outlet.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
wall.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
"""
pathlib.Path("system/surfaceFeatureExtractDict").write_text(d)
import subprocess
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode); print(r.stderr[-300:]); print(r.stdout[-2500:])

# -- cell 16 ------------------------------------------------------------------------
import pathlib
d = pathlib.Path("system/surfaceFeatureExtractDict")
d.write_text(d.read_text().replace("version 2.0; format ascii; object surfaceFeatureExtractDict;",
  "version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict;"))
import subprocess
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1200:]); print(r.stderr[-200:])

# -- cell 17 ------------------------------------------------------------------------
import shutil, os, subprocess, pathlib
p = pathlib.Path("constant/triSurface"); p.mkdir(parents=True, exist_ok=True)
for f in ["inlet.stl","outlet.stl","wall.stl"]:
    shutil.move(f, p/f)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:], r.stderr[-200:])

# -- cell 18 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(rc := r.returncode)
print(r.stdout[-1500:]); print(r.stderr[-200:])

# -- cell 19 ------------------------------------------------------------------------
import pathlib
p = pathlib.Path("system/snappyHexMeshDict")
d = p.read_text()
d = d.replace("""inlet.stl { type triSurface; }
    outlet.stl { type triSurface; }
    wall.stl { type triSurface; }""",
"""inlet { type triSurface; file "inlet.stl"; }
    outlet { type triSurface; file "outlet.stl"; }
    wall { type triSurface; file "wall.stl"; }""")
d = d.replace("""{ file "inlet.eMesh"; level 0; }
        { file "outlet.eMesh"; level 0; }
        { file "wall.eMesh"; level 1; }""",
"""{ file "inlet.eMesh"; level 1; }
        { file "outlet.eMesh"; level 1; }
        { file "wall.eMesh"; level 1; }""")
d = d.replace("""inlet.stl { level (0 0); }
        outlet.stl { level (0 0); }
        wall.stl { level (1 1); }""",
"""inlet { level (1 1); }
        outlet { level (1 1); }
        wall { level (1 1); }""")
p.write_text(d)
import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1200:]); print(r.stderr[-200:])

# -- cell 20 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
i = r.stdout.find("Error")
print(r.stdout[max(0,len(r.stdout)-4000)-0:])
print("ERR:", r.stderr[:1500])
