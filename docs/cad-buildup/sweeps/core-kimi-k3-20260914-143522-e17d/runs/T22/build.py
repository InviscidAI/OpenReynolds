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
import numpy as np, math

OD, ID, GAP, NVANE, VWID = 320, 190, 12, 36, 8
R_o, R_i, gap, W = OD/2000, ID/2000, GAP/1000, VWID/1000

annulus = bd.Cylinder(R_o, gap) - bd.Cylinder(R_i, gap)
one_vane = bd.Pos(0, (R_i+R_o)/2, 0) * bd.Box(W, R_o-R_i, gap)
vanes = bd.Compound([bd.Rot(0,0,a) * one_vane for a in np.linspace(0,360,NVANE,endpoint=False)])
fluid = annulus - vanes
print("volume m^3:", fluid.volume)

groups = {"inlet":[], "outlet":[], "friction":[], "vanes":[]}
print("geom_type available:", [c for c in fluid.faces()[:2]] )
samples=[]
for f in fluid.faces():
    n = f.normal_at()
    gt = f.geom_type
    samples.append((str(n), gt))
    if abs(n.Z) > 0.9:
        groups["friction"].append(f)
    else:
        r = math.hypot(f.center().X, f.center().Y)
        cyl = gt != bd.GeomType.PLANE
        if cyl and abs(r - R_i) < 2e-4: groups["inlet"].append(f)
        elif cyl and abs(r - R_o) < 2e-4: groups["outlet"].append(f)
        elif not cyl: groups["vanes"].append(f)

for k,v in groups.items():
    print(k, len(v), sum(f.area for f in v))

# -- cell 2 -------------------------------------------------------------------------
# Splitting friction into top/bottom (two inner face surfaces), then export 5 STLs to `constant/triSur
import build123d as bd, math, os
from pathlib import Path
tri = Path("constant/triSurface"); tri.mkdir(parents=True, exist_ok=True)

f_top = [f for f in groups["friction"] if f.center().Z > 0]
f_bot = [f for f in groups["friction"] if f.center().Z < 0]
print(len(f_top), len(f_bot))

patches = {
    "friction_top": bd.Compound(children=f_top),
    "friction_bottom": bd.Compound(children=f_bot),
    "inlet": bd.Compound(children=groups["inlet"]),
    "outlet": bd.Compound(children=groups["outlet"]),
    "vanes": bd.Compound(children=groups["vanes"]),
}
for name, comp in patches.items():
    bd.export_stl(comp, str(tri / (name + ".stl")))
print(sorted(os.listdir(tri)))

# -- cell 3 -------------------------------------------------------------------------
import pyvista as pv
palette = ["red","blue","green","orange","purple"]
pl = pv.Plotter(off_screen=True)
for i,(name) in enumerate(patches):
    m = pv.read(f"constant/triSurface/{name}.stl")
    pl.add_mesh(m, color=palette[i%len(palette)], label=name)
pl.view_xy(); pl.show(screenshot="view.png")
print("ok")

# -- cell 4 -------------------------------------------------------------------------
# Formatting braces tripped it — rewriting cleanly with concatenation:
import math, os
from pathlib import Path
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

x0,x1,y0,y1,z0,z1 = -0.175,0.175,-0.175,0.175,-0.02,0.02
header = "FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object %s;\n}\n"
bm_body = f"""
convertToMeters 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) (58 58 4) simpleGrading (1 1 1) );
boundary ( dummy {{ type patch; faces (((0 1 2 3))) ;}} );
"""
Path("system/blockMeshDict").write_text(header%"blockMeshDict" + bm_body)

ang=math.radians(5); rm=0.1275
lx,ly,lz = rm*math.sin(ang), rm*math.cos(ang), 0.0

def region(name, ptype):
    return f'    {name} {{ level (4 4); patchInfo {{ type {ptype}; }} }}\n'
geom = "".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}\n' for n in patches)
regions = "".join(region(n, "wall") for n in ["vanes","friction_top","friction_bottom"]) + \
          "".join(region(n, "patch") for n in ["inlet","outlet"])

snappy = header%"snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom} }}
castellatedMeshControls
{{
 maxLocalCells 100000; maxGlobalCells 2000000; minRefinementCells 10;
 maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ();
 refinementSurfaces
 {{
{regions} }}
 resolveFeatureAngle 30;
 refinementRegions {{ }}
 locationInMesh ({lx} {ly} {lz});
 allowFreeStandingZoneFaces true;
}}
snapControls
{{
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10;
 implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}}
meshQualityControls
{{
 maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minFlatness 0.5; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02;
 minDeterminant 0.001; checkFaceSimpleEdges true; minLengthScale 0.0001;
 minVolRatio 0.01; maxFaceWeightRatio 0.2; maxVolWeightRatio 0.2;
 nSmoothScale 4; errorReduction 0.75;
}}
"""
Path("system/snappyHexMeshDict").write_text(snappy)
print("dicts written")

# -- cell 5 -------------------------------------------------------------------------
# Running blockMesh then snappyHexMesh:
import subprocess
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc", r1.returncode, r1.stdout[-300:], r1.stderr[-300:])
if r1.returncode==0:
    r2 = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
    print("snappy rc", r2.returncode, r2.stdout[-1200:], r2.stderr[-1200:])

# -- cell 6 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[:1500])
print("---")
head = open("system/blockMeshDict").read()[:500]
print(head)

# -- cell 7 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:])

# -- cell 8 -------------------------------------------------------------------------
import subprocess, os
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("rc", r.returncode)
print(r.stdout[-800:])
print("stderr", r.stderr[-800:])
print(os.listdir("."))

# -- cell 9 -------------------------------------------------------------------------
from pathlib import Path
header = "FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object %s;\n}\n"
cd = header%"controlDict" + """
application foo;
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
Path("system/controlDict").write_text(cd)

import subprocess, os
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc", r.returncode, r.stdout[-200:], r.stderr[-200:])

# -- cell 10 ------------------------------------------------------------------------
from pathlib import Path
header = "FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object %s;\n}\n"
x0,x1,y0,y1,z0,z1 = -0.175,0.175,-0.175,0.175,-0.02,0.02
bm_body = f"""
convertToMeters 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) (58 58 4) simpleGrading (1 1 1) );
"""
Path("system/blockMeshDict").write_text(header%"blockMeshDict" + bm_body)

import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc", r.returncode, r.stdout[-300:], r.stderr[-300:])

# -- cell 11 ------------------------------------------------------------------------
# Background mesh OK (13k cells). Running snappyHexMesh:
import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print("rc", r.returncode)
print(r.stdout[-2000:])
print("ERR", r.stderr[-500:])

# -- cell 12 ------------------------------------------------------------------------
from pathlib import Path
header = "FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object %s;\n}\n"
Path("system/fvSchemes").write_text(header%"fvSchemes" + "ddtSchemes {}\ndivSchemes {}\ngradSchemes {}\nlaplacianSchemes {}\ninterpolationSchemes {}\nsnGradSchemes {}\n")
Path("system/fvSolution").write_text(header%"fvSolution" + "solvers {}\n")

import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print("rc", r.returncode, "out", r.stdout[-500:], "err", r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
from pathlib import Path
s = Path("system/snappyHexMeshDict").read_text()
s = s.replace("snap true; addLayers false;",
              "snap true; addLayers false;\naddLayersControls { relativeSizes true; layers { } expansionRatio 1.0; finalLayerThickness 0.5; minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; nOptionalLayerIterations 1; }")
Path("system/snappyHexMeshDict").write_text(s)

import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print("rc", r.returncode, "err", r.stderr[-300:])
print(r.stdout[-800:])

# -- cell 14 ------------------------------------------------------------------------
from pathlib import Path
s = Path("system/snappyHexMeshDict").read_text()
for i,l in enumerate(s.splitlines()):
    if 8 <= i <= 25: print(i+1, l)
r2 = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(r2.stderr[:1200])

# -- cell 15 ------------------------------------------------------------------------
from pathlib import Path
s = Path("system/snappyHexMeshDict").read_text()
s = s.replace("implicitFeatureSnap false;", "explicitFeatureSnap initialTempted TRUE;\n implicitFeatureSnap false; mergeTolerance 1e-06;")
Path("system/snappyHexMeshDict").write_text(s)

import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print("rc", r.returncode, "err", r.stderr[-400:])
print(r.stdout[-400:])

# -- cell 16 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(r.stderr[:600])

# -- cell 17 ------------------------------------------------------------------------
from pathlib import Path
s = Path("system/snappyHexMeshDict").read_text()
for i,l in enumerate(s.splitlines()):
    if "snapControls" in l or 30<=i<=40: print(i+1,l)

# -- cell 18 ------------------------------------------------------------------------
# Rewriting the snappy dict cleanly with mergeTolerance in both places and removing the bogus line:
import math
from pathlib import Path
header = "FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object %s;\n}\n"
ang=math.radians(5); rm=0.1275
lx,ly,lz = rm*math.sin(ang), rm*math.cos(ang), 0.0
patches = ["friction_top","friction_bottom","inlet","outlet","vanes"]
geom = "".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}\n' for n in patches)
def region(name, ptype):
    return f'    {name} {{ level (4 4); patchInfo {{ type {ptype}; }} }}\n'
regions = "".join(region(n, "wall") for n in ["vanes","friction_top","friction_bottom"]) + \
          "".join(region(n, "patch") for n in ["inlet","outlet"])

snappy = header%"snappyHexMeshDict" + f"""
mergeTolerance 1e-06;
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom} }}
castellatedMeshControls
{{
 maxLocalCells 100000; maxGlobalCells 2000000; minRefinementCells 10;
 maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ();
 refinementSurfaces
 {{
{regions} }}
 resolveFeatureAngle 30;
 refinementRegions {{ }}
 locationInMesh ({lx} {ly} {lz});
 allowFreeStandingZoneFaces true;
}}
snapControls
{{
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10;
 implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}}
addLayersControls
{{
 relativeSizes true; layers {{ }} expansionRatio 1.0; finalLayerThickness 0.5;
 minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 5;
 nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
 maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
 nBufferCellsNoExtrude 0; nLayerIter 50; nOptionalLayerIterations 1;
}}
meshQualityControls
{{
 maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minFlatness 0.5; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02;
 minDeterminant 0.001; checkFaceSimpleEdges true; minLengthScale 0.0001;
 minVolRatio 0.01; maxFaceWeightRatio 0.2; maxVolWeightRatio 0.2;
 nSmoothScale 4; errorReduction 0.75;
}}
"""
Path("system/snappyHexMeshDict").write_text(snappy)

import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print("rc", r.returncode, "err", r.stderr[-300:])
print(r.stdout[-400:])

# -- cell 19 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(((r.stderr).splitlines())[:4])

# -- cell 20 ------------------------------------------------------------------------
from pathlib import Path
s = Path("system/snappyHexMeshDict").read_text()
s = s.replace("maxFaceWeightRatio 0.2; maxVolWeightRatio 0.2;", "minFaceWeight 0.2; minVolRatio 0.01;").replace("minVolRatio 0.01;\n\n minVolRatio 0.01;", "minVolRatio 0.01;")
# actually old dict had minVolRatio separately; remove dup
s = s.replace("minVolRatio 0.01;", "")
s = s.replace("minFaceWeight 0.2;", "minVolRatio 0.01; minFaceWeight 0.2;")
Path("system/snappyHexMeshDict").write_text(s)

import subprocess
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print("rc", r.returncode, "stderr:", r.stderr[:400])
print(r.stdout[-600:])
