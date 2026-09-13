"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Re-sending as a tool call.
import os, subprocess
from build123d import *

# --- parameters (metres) ---
LX, LY, LZ = 0.160, 0.100, 0.040      # internal box
TW = 0.003                             # wall thickness
SD, SH = 0.010, 0.025                  # standoff dia, height
HD, HDP = 0.004, 0.012                 # blind hole dia, depth (from standoff top)
INSET = 0.015                          # standoff centre inset from each corner
VW, VH, VZ = 0.030, 0.020, 0.010       # vent width(y), height(z), bottom above floor

air = Pos(LX/2, LY/2, LZ/2) * Box(LX, LY, LZ)
duct = Pos(-TW/2, LY/2, VZ + VH/2) * Box(TW, VW, VH)
fluid = air + duct

for cx in (INSET, LX - INSET):
    for cy in (INSET, LY - INSET):
        post = Pos(cx, cy, SH/2) * Cylinder(SD/2, SH)
        hole = Pos(cx, cy, SH - HDP/2) * Cylinder(HD/2, HDP)
        fluid = fluid - (post - hole)

print(fluid.volume, len(fluid.faces()), fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Volume checks out (6.4e-4 + duct 1.8e-6 − 4 posts 7.25e-6 = 6.3455e-4). Now classify faces by exact 
from collections import defaultdict
import numpy as np
groups = defaultdict(list)
for f in fluid.faces():
    gt = f.geom_type
    c = f.center()
    if gt == GeomType.CYLINDER:
        r = f.radius
        groups["standoffs" if abs(r-SD/2) < 1e-6 else "holes"].append(f)
    else:
        n = f.normal_at(c)
        if abs(n.Z) > 0.9:
            z = c.Z
            if abs(z) < 1e-9: groups["floor"].append(f)
            elif abs(z-LZ) < 1e-9: groups["top"].append(f)
            elif abs(z-SH) < 1e-9: groups["standoffs"].append(f)   # standoff top annulus
            elif abs(z-(SH-HDP)) < 1e-9: groups["holes"].append(f) # blind hole bottom
            else: groups["walls"].append(f)
        elif abs(n.X) > 0.9 and abs(c.X + TW) < 1e-9:
            groups["vent"].append(f)
        else:
            groups["walls"].append(f)
for k,v in groups.items():
    print(k, len(v), round(sum(f.area for f in v),8))

# -- cell 3 -------------------------------------------------------------------------
# All six areas match analytic values. Export one STL per patch with a fine tessellation tolerance (th
os.makedirs("constant/triSurface", exist_ok=True)
for name, fs in groups.items():
    shp = Compound(children=[Face(f.wrapped) for f in fs]) if len(fs) > 1 else fs[0]
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.15, ascii_format=True)
    # rename solid to patch name so snappy gets sensible region names
    p = f"constant/triSurface/{name}.stl"
    txt = open(p).read().replace("solid ", f"solid {name} ", 1)
    open(p,"w").write(txt)
print(subprocess.run("ls -la constant/triSurface", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run("head -2 constant/triSurface/holes.stl", shell=True, capture_output=True, text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# Look at it before meshing.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(1000,700))
cols = dict(walls="lightgray", floor="tan", top="skyblue", vent="red", standoffs="green", holes="magenta")
for k,c in cols.items():
    pl.add_mesh(pv.read(f"constant/triSurface/{k}.stl"), color=c, opacity=0.45 if k in ("walls","top") else 1.0, show_edges=(k=="holes"))
pl.camera_position=[(0.3,-0.25,0.2),(0.08,0.05,0.02),(0,0,1)]
pl.screenshot("view.png"); print("ok")

# -- cell 5 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 6 -------------------------------------------------------------------------
# Geometry looks right. Now the case: blockMesh background aligned to the box planes (5 mm base cells)
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
def w(p,s): open(p,"w").write(s)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
w("system/controlDict", HDR%"controlDict" + "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
w("system/fvSchemes", HDR%"fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
w("system/fvSolution", HDR%"fvSolution" + "solvers{} \n")
X0,X1,Y0,Y1,Z0,Z1 = -0.005,0.165,-0.005,0.105,-0.005,0.045
nx,ny,nz = 34,22,10
w("system/blockMeshDict", HDR%"blockMeshDict" + f"""
scale 1;
vertices
(
 ({X0} {Y0} {Z0}) ({X1} {Y0} {Z0}) ({X1} {Y1} {Z0}) ({X0} {Y1} {Z0})
 ({X0} {Y0} {Z1}) ({X1} {Y0} {Z1}) ({X1} {Y1} {Z1}) ({X0} {Y1} {Z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""")
print(subprocess.run("blockMesh > log.blockMesh 2>&1; tail -4 log.blockMesh", shell=True, capture_output=True, text=True).stdout)

# -- cell 7 -------------------------------------------------------------------------
names = ["walls","floor","top","vent","standoffs","holes"]
w("system/surfaceFeatureExtractDict", HDR%"surfaceFeatureExtractDict" +
  "".join(f"{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}\n" for n in names))
r=subprocess.run("surfaceFeatureExtract > log.sfe 2>&1; ls constant/triSurface/*.eMesh", shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
LEV = {"walls":(1,1),"floor":(1,1),"top":(1,1),"vent":(1,1),"standoffs":(2,2),"holes":(3,3)}
geom = "".join(f"{n}.stl {{ type triSurfaceMesh; name {n}; }}\n" for n in names)
refs = "".join(f"{n} {{ level ({LEV[n][0]} {LEV[n][1]}); patchInfo {{ type wall; }} }}\n" for n in names)
feats = "".join(f'{{ file "{n}.eMesh"; level {max(LEV[n])}; }}\n' for n in names)
w("system/snappyHexMeshDict", HDR%"snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry {{ {geom} }}
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ( {feats} );
    refinementSurfaces {{ {refs} }}
    refinementRegions {{}}
    locationInMesh (0.0813 0.0517 0.0351);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 15;
    implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers{{}} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
r=subprocess.run("snappyHexMesh -overwrite > log.snappy 2>&1; tail -25 log.snappy", shell=True, capture_output=True, text=True)
print(r.stdout)

# -- cell 9 -------------------------------------------------------------------------
# Meshed. Check patch areas against the analytic surface areas and run checkMesh.
r=subprocess.run("checkMesh -allTopology > log.checkMesh 2>&1; grep -E 'patch|Mesh OK|\\*\\*\\*|cells:|Overall' -i log.checkMesh | head -40", shell=True, capture_output=True, text=True)
print(r.stdout)
print(open("constant/polyMesh/boundary").read()[-900:])

# -- cell 10 ------------------------------------------------------------------------
# Mesh OK. Now the final pass: `top`/`vent` should be `patch` type, not wall, and the Ø4 blind holes d
LEV = {"walls":(1,1),"floor":(1,1),"top":(1,1),"vent":(2,2),"standoffs":(2,3),"holes":(4,4)}
PTYPE = {"top":"patch","vent":"patch"}
refs = "".join(f"{n} {{ level ({LEV[n][0]} {LEV[n][1]}); patchInfo {{ type {PTYPE.get(n,'wall')}; }} }}\n" for n in names)
feats = "".join(f'{{ file "{n}.eMesh"; level {max(LEV[n])}; }}\n' for n in names)
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"features \( .*? \);", f"features ( {feats} );", d, flags=re.S)
d = re.sub(r"refinementSurfaces \{\{?.*?\}\}?\n\s*refinementRegions", f"refinementSurfaces {{ {refs} }}\n    refinementRegions", d, flags=re.S)
w("system/snappyHexMeshDict", d)
r=subprocess.run("snappyHexMesh -overwrite > log.snappy 2>&1; tail -6 log.snappy; checkMesh > log.checkMesh 2>&1; grep -E 'cells:|Mesh OK|\\*\\*\\*' log.checkMesh", shell=True, capture_output=True, text=True)
print(r.stdout)

# -- cell 11 ------------------------------------------------------------------------
print(subprocess.run("grep -n -A3 'FOAM FATAL' log.snappy | head -20", shell=True, capture_output=True, text=True).stdout)
print(open("system/snappyHexMeshDict").read()[:1400])

# -- cell 12 ------------------------------------------------------------------------
# The dict is fine — snappy was re-run on the already-snapped mesh. Rerun blockMesh first.
r=subprocess.run("blockMesh > log.blockMesh 2>&1 && snappyHexMesh -overwrite > log.snappy 2>&1; tail -8 log.snappy; checkMesh -allTopology > log.checkMesh 2>&1; grep -E 'cells:|Mesh OK|\\*\\*\\*|<<' log.checkMesh", shell=True, capture_output=True, text=True)
print(r.stdout)

# -- cell 13 ------------------------------------------------------------------------
# 56.9k cells, Mesh OK. Now verify the meshed patch areas against the analytic values (this is the rea
open("t21.foam","w").write("")
rd = pv.OpenFOAMReader("t21.foam"); rd.enable_all_patch_arrays(); m = rd.read()
bd = m["boundary"]
analytic = dict(walls=0.0205, floor=0.01568584, top=0.016, vent=0.0006, standoffs=0.00340549, holes=0.00065345)
for k in bd.keys():
    a = bd[k].compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{k:10s} mesh={a:.6f}  analytic={analytic[k]:.6f}  err={100*(a-analytic[k])/analytic[k]:+.2f}%")
print("internal volume:", m["internalMesh"].compute_cell_sizes(length=False,area=False)["Volume"].sum(), "vs CAD", fluid.volume)

# -- cell 14 ------------------------------------------------------------------------
# All patches within 0.2% of analytic. Final dimensional check from the meshed surfaces themselves, pl
hb = bd["holes"].bounds; sb = bd["standoffs"].bounds
print("holes bbox z: %.4f..%.4f (expect 0.0130..0.0250, depth %.4f vs 0.012)"%(hb[4],hb[5],hb[5]-hb[4]))
print("standoffs bbox z: %.4f..%.4f (expect 0..0.025)"%(sb[4],sb[5]))
pts = bd["standoffs"].points
import numpy as np
for cx in (0.015,0.145):
    for cy in (0.015,0.085):
        sel = pts[(abs(pts[:,0]-cx)<0.006)&(abs(pts[:,1]-cy)<0.006)]
        rr = np.hypot(sel[:,0]-cx, sel[:,1]-cy)
        print(f"post ({cx},{cy}): n={len(sel)} max radius={rr.max():.5f} (expect 0.0050)")
sl = m["internalMesh"].slice(normal="y", origin=(0.08,0.015,0.02))
pl = pv.Plotter(off_screen=True, window_size=(1100,450)); pl.add_mesh(sl, show_edges=True, color="w")
pl.view_xz(); pl.camera.zoom(1.4); pl.screenshot("slice.png"); display(Image("slice.png"))
