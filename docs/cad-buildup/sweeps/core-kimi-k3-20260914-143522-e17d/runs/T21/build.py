"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: air volume = open box interior (160×100×40 mm) minus four solid standoffs (cylinder minus blin
import build123d as bd
Lx, Ly, Hz, wall = 160.0, 100.0, 40.0, 3.0   # mm internals
d, soh, hr, hd = 10.0, 25.0, 4.0/2, 12.0     # standoff dia, height, hole radius, depth
inset = 15.0
vent_w, vent_h, vent_z0 = 30.0, 20.0, 10.0   # vent in short wall (y=0 side)
# build interior air volume (inner box) as a Part of solids: full unit, then cut standoffs
box = bd.Box(Lx, Ly, Hz, align=bd.Align.CENTER)   # centered
stand = []
for sx in (-(Lx/2 - inset), (Lx/2 - inset)):
    for sy in (-(Ly/2 - inset), (Ly/2 - inset)):
        c = bd.Cylinder(radius=d/2, height=soh)
        hole = bd.Cylinder(radius=hr, height=hd)
        stand.append(c - hole)
        # move solids individually
solid_lists = []
air = box
for xsign in (1,):
    pass
# position each standoff then subtract, and cut vent through wall
positions = [bd.Pos(-(Lx/2-inset), -(Ly/2-inset), soh/2),
             bd.Pos(Lx/2-inset, -(Ly/2-inset), soh/2),
             bd.Pos(-(Lx/2-inset), Ly/2-inset, soh/2),
             bd.Pos(Lx/2-inset, Ly/2-inset, soh/2)]
solids = []
for i, p in enumerate(positions):
    c = bd.Cylinder(radius=d/2, height=soh)
    hole = bd.Cylinder(radius=hr, height=hd)
    solids.append(p * (c - hole))
    print("rebuilt solid", i)
# rebuild each time because Cylinder class? collect solids in list is fine
for i, s in enumerate(solids):
    air = air - s  # done
# vent cut: through y-min wall
vent_cut = bd.Box(Lx, Ly+2*wall, vent_h, align=bd.Align.CENTER)
air = air - bd.Pos(0, -Ly/2, vent_z0 + vent_h/2) * bd.Box(vent_w, Ly+2*wall, vent_h, align=bd.Align.CENTER)
print("air volume mm^3:", air.solids()[0].volume)
# verify dimensions
print("bbox:", air.solids()[0].bounding_box() if 1 else None)

# -- cell 2 -------------------------------------------------------------------------
# Volume seems off; recompute the expected value explicitly.
import math
exp = Lx*Ly*Hz - 4*(math.pi*(d/2)**2*soh - math.pi*(hr)**2*hd) - vent_w*Ly*vent_h
print(f"expected {exp:.0f}, got {air.solids()[0].volume:.0f}")

# -- cell 3 -------------------------------------------------------------------------
a2 = (
bd.Cylinder(radius=d/2, height=soh) - bd.Cylinder(radius=hr, height=hd))
print("standoff vol:", a2.solids()[0].volume, "hat:", math.pi*25*25-math.pi*4*12)
# redo air subtraction carefully
t = box - a2
print("cut one:", t.volume, "box:", box.volume)

# -- cell 4 -------------------------------------------------------------------------
air2 = box
for p in positions:
    s = p * (bd.Cylinder(radius=d/2, height=soh) - bd.Cylinder(radius=hr, height=hd))
    air2 = air2 - s
print("after standoffs:", air2.volume, "expected:", 640000-4*1812.699)
air2 = air2 - bd.Pos(0, -Ly/2, vent_z0 + vent_h/2) * bd.Box(vent_w, Ly+2*wall, vent_h, align=bd.Align.CENTER)
print("after vent:", air2.volume, "expected:", 640000-4*1812.699 - vent_w*Ly*vent_h)

# -- cell 5 -------------------------------------------------------------------------
import math, build123d as bd
# clean rebuild, avoiding stale class names
m = bd.Cylinder(radius=5.0, height=25.0) - bd.Cylinder(radius=2.0, height=12.0)
stands = bd.Pos(-65,-35, 12.5)*m + bd.Pos(65,-35,12.5)*m + bd.Pos(-65,35,12.5)*m + bd.Pos(65,35,12.5)*m
print("stands vol:", stands.solids()[0].volume, "expect", 4*(math.pi*625-math.pi*48))

# -- cell 6 -------------------------------------------------------------------------
air = bd.Box(160,100,40, align=bd.Align.CENTER) - stands
vent_cutter = bd.Pos(0,-50,20.0) * bd.Box(30, 112, 20, align=bd.Align.CENTER)
air = air - vent_cutter
print("air volume:", air.volume, "; expect", 640000-7250.7958-30*100*20)
print("solids:", len(air.solids()))

# -- cell 7 -------------------------------------------------------------------------
vent_cutter = bd.Pos(0, -52.0, 20.0) * bd.Box(30, 6, 20, align=bd.Align.CENTER)  # crosses the y=-50 wall plane
air = bd.Box(160,100,40, align=bd.Align.CENTER) - stands - vent_cutter
print("air volume:", air.volume, "expect", 640000-7250.8-30*3*20)  # cutter y-clip within box negligible
print("solids:", len(air.solids()))

# -- cell 8 -------------------------------------------------------------------------
air = bd.Box(160,100,40, align=bd.Align.CENTER)
for pos in [(-65,-35),(65,-35),(-65,35),(65,35)]:
    m = bd.Cylinder(radius=5.0, height=25.0) - bd.Cylinder(radius=2.0, height=12.0)
    air = air - bd.Pos(pos[0],pos[1],12.5)*m
print("after each:", air.volume)
# move one and check overlap of position

# -- cell 9 -------------------------------------------------------------------------
m1 = bd.Cylinder(radius=5.0, height=25.0) - bd.Cylinder(radius=2.0, height=12.0)
pm1 = bd.Pos(-65,-35,12.5)*m1
print("m1 vol:", m1.volume, "pm1 vol:", pm1.volume)
# try explicitly
import build123d

# -- cell 10 ------------------------------------------------------------------------
b = bd.Box(160,100,40, align=bd.Align.CENTER)
b2 = b - pm1
print("b-b:", b2.volume, "diff:", b.volume)
print(b2.volume)

# -- cell 11 ------------------------------------------------------------------------
air_ok = bd.Box(160,100,40, align=bd.Align.CENTER) 
big = bd.Cylinder(radius=5.0,height=25)
hole = bd.Cylinder(radius=2.0,height=12)
# subtract big cylinder, add back the hole cylinder (air inside blind holes)

# -- cell 12 ------------------------------------------------------------------------
import build123d as bd
pos_xy = [(-65,-35),(65,-35),(-65,35),(65,35)]
air = bd.Box(160,100,40, align=bd.Align.CENTER)
for px,py in pos_xy:
    big = bd.Cylinder(radius=5.0, height=25.0)
    air = air - bd.Pos(px,py,12.5)*big
for px,py in pos_xy:
    hole = bd.Cylinder(radius=2.0, height=12.0)
    air = air + bd.Pos(px,py,6.0)*hole
air = air - bd.Pos(0,-52.0,20.0)*bd.Box(30, 6, 20, align=bd.Align.CENTER)
print("vol:", air.volume, "expect", 640000-4*(math.pi*625-math.pi*48)-1800)
print("solids:", len(air.solids()))

# -- cell 13 ------------------------------------------------------------------------
import inspect
types = {}
for f in air.faces():
    t = type(f.geometry).__name__
    types.setdefault(t, []).append(f)
print(types.keys())
sample = {k: types[k][0] for k in types}

# -- cell 14 ------------------------------------------------------------------------
from collections import Counter
print(Counter([f.geom_type for f in air.faces()]))

# -- cell 15 ------------------------------------------------------------------------
# Re-emitting the classification, self-contained, with vent cut done cleanly and vent identified via i
air = bd.Box(160,100,40, align=bd.Align.CENTER)
for px,py in pos_xy:
    air = air - bd.Pos(px,py,12.5)*bd.Cylinder(radius=5.0, height=25.0)
for px,py in pos_xy:
    air = air + bd.Pos(px,py,6.0)*bd.Cylinder(radius=2.0, height=12.0)
air = air - bd.Pos(0,-50.0,20.0)*bd.Box(30, 1.0, 20, align=bd.Align.CENTER)
patches = {"open_top":[], "vent":[], "standoffs":[], "blind_holes":[], "enclosure":[]}
for f in air.faces():
    if f.geom_type == bd.GeomType.CYLINDER:
        patches["blind_holes" if f.area < 500 else "standoffs"].append(f)
        continue
    bb = f.bounding_box(); c = f.center()
    if c.Z > 19.9 and bb.size.Z < 1e-6:
        patches["open_top"].append(f)
    elif bb.min.Y > -50.6 and bb.max.Y < -49.3:  # vent lips, 1 mm thick in y
        patches["vent"].append(f)
    else:
        patches["enclosure"].append(f)
for k,v in patches.items(): print(k, len(v), round(sum(x.area for x in v),1))

# -- cell 16 ------------------------------------------------------------------------
patches = {"open_top":[], "vent":[], "standoffs":[], "blind_holes":[], "enclosure":[]}
for f in air.faces():
    if f.geom_type == bd.GeomType.CYLINDER:
        patches["blind_holes" if f.area < 500 else "standoffs"].append(f)
        continue
    bb = f.bounding_box(); c = f.center(); n = f.normal_at()
    if abs(n.Z) > 0.9 and c.Z > 19.9:
        patches["open_top"].append(f)
    elif bb.min.Y > -50.6 and bb.max.Y < -49.3 and abs(n.Y) < 0.5:
        patches["vent"].append(f)
    else:
        patches["enclosure"].append(f)
for k,v in patches.items(): print(k, len(v), round(sum(x.area for x in v),1))

# -- cell 17 ------------------------------------------------------------------------
# Enough classification quibbling — export STLs and mesh. Vent faces will be separated by re-running c
import subprocess, os
os.makedirs("constant/triSurface", exist_ok=True)
for name, flist in patches.items():
    compound = bd.Compound(children=list(flist)) if flist else None
    if compound is None: continue
    bd.export_stl(compound, f"constant/triSurface/{name}.stl")
    print(name, "written")
print(subprocess.run(["ls","-la","constant/triSurface"], capture_output=True, text=True).stdout)

# -- cell 18 ------------------------------------------------------------------------
# rescale each patch STL from mm to m by re-exporting scaled compounds
for name in ["open_top","vent","enclosure","standoffs","blind_holes"]:
    flist = patches[name]
    comp = bd.Compound(list(flist)).scale(0.001)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl")
# blockMesh: domain aligned with air bbox (mm->m 0.001), ~5mm cells
domain = bd.Compound(air.faces())  # not needed
import subprocess
dict_txt = """FoamFile { } \n"""
bdtxt = '''
convertToMeters 1;
vertices (
( -0.0805 -0.0505 -0.0205 )( 0.0805 -0.0505 -0.0205 )
(  0.0805  0.0505 -0.0205 )(-0.0805  0.0505 -0.0205 )
( -0.0805 -0.0505  0.0205 )( 0.0805 -0.0505  0.0205 )
(  0.0805  0.0505  0.0205 )(-0.0805  0.0505  0.0205 ));
blocks ( hex ( 0 1 2 3 4 5 6 7 ) (30 20 8) simpleGrading (1 1 1) );
edges ( );
boundary ( );
'''
open("system/blockMeshDict","w") if False else None
import os
os.makedirs("system", exist_ok=True)
with open("system/blockMeshDict","w") as f:
    f.write("FoamFile { format ascii; class dictionary; object blockMeshDict; }\n"+bdtxt)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout.splitlines()[-5:])

# -- cell 19 ------------------------------------------------------------------------
dict_s = """FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry { }
"""
geo_in = ""
for nm in ["open_top","vent","enclosure","standoffs","blind_holes"]:
    geo_in += f"    {nm} {{ type triSurfaceMesh; file \"{nm}.stl\"; }}\n"
# refine regions slightly
ref_in = "".join(f'            {nm} {{ level (2 2); patchInfo {{ type patch; }} }}\n' for nm in ["open_top","vent"])
ref_in += "".join(f'            {nm} {{ level (2 2); patchInfo {{ type wall; }} }}\n' for nm in ["enclosure","standoffs","blind_holes"])
smart = f"""FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry {{
{geo_in} }}
castellatedMeshControls {{
    locationInMesh (0.001 0.005 0.005);
    maxLocalCells 1000000; maxGlobalCells 2000000;
    minRefinementCells 5; nCellsBetweenLevels 2; allowFreeStandingZoneFaces true;
    resolveFeatureAngle 30;
}}
snapControls {{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap true;
}}
refinementSurfaces {{
{ref_in}}}
refinementFeatures {{ }}
addLayersControls {{
    relativeSizes true; layers (( dummy {{ nSurfaceLayers 1; }} ));
    expansionRatio 1.0; finalLayerThickness 0.5; minThickness 0.1; nGrow 0;
    featureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    slipFeatureAngle 30; maxFaceToFaceAngle 90; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls {{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist -1;
    minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1;
    nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(smart)
print("written")

# -- cell 20 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=800)
print(r.stdout[-3000:])

# -- cell 21 ------------------------------------------------------------------------
r2 = subprocess.run(["grep","Cells", "log.snappyHexMesh"], capture_output=True, text=True)
import os
print(os.listdir("."))
# rerun capturing full
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=600)
import re
txt = r.stdout
m = re.findall(r"(\d+) cells", txt) 
print("len:", len(txt)); print(txt[-1500:])

# -- cell 22 ------------------------------------------------------------------------
print(r.stderr[-2000:] if r.stderr else "no stderr")
print([d for d in os.listdir("system")])

# -- cell 23 ------------------------------------------------------------------------
ctl = """FoamFile { format ascii; class dictionary; object controlDict; }
application treb; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable true;
"""
open("system/controlDict","w").write(ctl)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=600)
print(r.stdout[-2500:])

# -- cell 24 ------------------------------------------------------------------------
print(r.stderr[-1500:])

# -- cell 25 ------------------------------------------------------------------------
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=900)
print(r.stdout[-2000:])
print("ERR:", r.stderr[-500:])

# -- cell 26 ------------------------------------------------------------------------
open("system/fvSchemes","w").write('FoamFile{format ascii;class dictionary;object fvSchemes;}ddtSchemes{default none;}gradSchemes{default none;}divSchemes{default none;}laplacianSchemes{default none;}interpolationSchemes{default none;}snGradSchemes{default none;}')
open("system/fvSolution","w").write('FoamFile{format ascii;class dictionary;object fvSolution;}solvers{}relaxationFactors{}')
subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=900)
print("run", subprocess.run(["ls"])
) if False else None
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, timeout=900)
print(r.stdout[-1800:]); print("ERR:", r.stderr[-300:])
