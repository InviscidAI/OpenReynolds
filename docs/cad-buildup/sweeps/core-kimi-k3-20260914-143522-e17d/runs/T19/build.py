"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The boolean on coincident surfaces is producing garbage. I'll rebuild as an extruded 2D profile inst
import build123d as bd, numpy as np, os
p="constant/triSurface"; os.makedirs(p, exist_ok=True)
BORE_R, SHAFT_R, L = 20.05, 20.0, 60.0
KW_W, KW_D = 12.0, 5.0
S=0.001
outer = bd.make_face(bd.Circle(BORE_R))
inner = bd.make_face(bd.Circle(SHAFT_R))
# keyway slot: width 12 (x +/-6), from r = shaft_r-5 to bore_r
slot = bd.Plane.XY * bd.Rectangle(KW_W, BORE_R -(SHAFT_R-KW_D))
slot = bd.Pos(0, (BOSS:=0), 0) * slot
ys, ye = SHAFT_R-KW_D, BORE_R
slot = bd.Rectangle(KW_W, ye-ys, align=(bd.Align.CENTER, bd.Align.MIN, bd.Align.CENTER))
slot = bd.Pos(0, ys, 0) * slot
profile = (outer - inner) + slot
fluid = bd.extrude(profile, dir=(0,0,1), amount=L)
print("vol mm3:", round(fluid.volume,3), "faces:", len(fluid.faces()))
for f in fluid.faces():
    c = tuple(f.center())
    print("area %.2f r=%.2f z=%.2f"%(f.area,(c[0]**2+c[1]**2)**.5,c[2]))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd
groups = {"bore":[], "shaft":[], "kw_floor":[], "kw_flanks":[], "inlet_HP":[], "outlet_LP":[]}
for f in fluid.faces():
    c = tuple(f.center()); r = (c[0]**2+c[1]**2)**.5; a=f.area
    if c[2] < 1:        groups["outlet_LP"].append(f)
    elif c[2] > 59:     groups["inlet_HP"].append(f)
    elif abs(r-15.0) < 0.2 and a > 300: groups["kw_floor"].append(f)
    elif abs(r-20.05) < 0.2 and a > 5000: groups["bore"].append(f)
    elif abs(r-20.0) < 0.2 and a > 5000: groups["shaft"].append(f)
    elif a < 310: groups["kw_flanks"].append(f)
    else: print("UNCLASSIFIED", a, r, c[2])
for k,v in groups.items():
    print(k, len(v), "area %.2f mm2"%sum(f.area for f in v))
    if v:
        bd.export_stl(bd.scale(bd.Compound(children=v), S), f"constant/triSurface/{k}.stl")

# -- cell 3 -------------------------------------------------------------------------
import build123d as bd
groups = {"bore":[], "shaft":[], "kw_floor":[], "kw_flanks":[], "inlet_HP":[], "outlet_LP":[]}
for f in fluid.faces():
    c = tuple(f.center()); r = (c[0]**2+c[1]**2)**.5; a=f.area
    if c[2] < 1:   groups["outlet_LP"].append(f)
    elif c[2] > 59: groups["inlet_HP"].append(f)
    elif 20.02 < r < 20.09: groups["bore"].append(f)
    elif 19.95 < r <= 20.02: groups["shaft"].append(f)
    elif abs(r-15.0) < 0.2: groups["kw_floor"].append(f)
    elif a < 310: groups["kw_flanks"].append(f)
    else: print("UNCLASSIFIED", a, r, c[2])
for k,v in groups.items():
    print(k, len(v), "area %.2f mm2"%sum(f.area for f in v))
    if v:
        bd.export_stl(bd.scale(bd.Compound(children=v), S), f"constant/triSurface/{k}.stl")
import subprocess
print(subprocess.run(["ls","-la","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# Visual check good — keyway slot visible with red floor and orange flanks. Now setting up blockMesh +
import os, subprocess
os.makedirs("system", exist_ok=True)
bmin,bmax = (-0.021,), (0.021,); 
z0,z1 = -0.001, 0.061
xs=ys=105; zs=155  # ~0.2mm cells
bm = f"""FoamFile {{ version 2.0; class dictionary; }}
vertices (({bmin[0]} {bmin[0]} {z0})({bmax[0]} {bmin[0]} {z0})({bmax[0]} {bmax[0]} {z0})({bmin[0]} {bmax[0]} {z0})
          ({bmin[0]} {bmin[0]} {z1})({bmax[0]} {bmin[0]} {z1})({bmax[0]} {bmax[0]} {z1})({bmin[0]} {bmax[0]} {z1}));
blocks (hex (0 1 2 3 4 5 6 7) ({xs} {ys} {zs}) simpleGrading (1 1 1));
edges (); boundary ();
"""
open("system/blockMeshDict","w").write(bm)

shm = """FoamFile { version 2.0; class dictionary; }
castellatedMesh true; snap true; addLayers false;
geometry {
 bore { type triSurfaceMesh; file "bore.stl"; }
 shaft { type triSurfaceMesh; file "shaft.stl"; }
 kw_floor { type triSurfaceMesh; file "kw_floor.stl"; }
 kw_flanks { type triSurfaceMesh; file "kw_flanks.stl"; }
 inlet_HP { type triSurfaceMesh; file "inlet_HP.stl"; }
 outlet_LP { type triSurfaceMesh; file "outlet_LP.stl"; }
}
castellatedMeshControls {
 maxLocalCells 1000000; maxGlobalCells 4000000; minRefinementCells 10;
 nCellsBetweenLevels 2; resolveFeatureAngle 30;
 features ( { file "bore.eMesh"; level 2; } { file "shaft.eMesh"; level 2; }
            { file "kw_floor.eMesh"; level 2; } { file "kw_flanks.eMesh"; level 2; } );
 refinementSurfaces (
   bore { level (2 2); }  shaft { level (2 2); }
   kw_floor { level (2 2); } kw_flanks { level (2 2); }
   inlet_HP { level (2 2); } outlet_HP_replace { level (2 2); }
   outlet_LP { level (2 2); }
 );
 locationsInMesh ((0 0.02025 0.03));
 meshQuality { }
}
"""
open("system/snappyHexMeshDict","w").write(shm)
cd = """FoamFile { version 2.0; class dictionary; }
application snappyHexMesh; """
open("system/controlDict","w").write(cd)
print("wrote dicts")

# -- cell 5 -------------------------------------------------------------------------
# The snappyHexMeshDict I wrote has a typo and sloppy structure — writing it properly now, then runnin
import subprocess
shm = """FoamFile { version 2.0; format ascii; class dictionary; }
castellatedMesh true;
snap true;
addLayers false;
geometry {
  bore       { type triSurfaceMesh; file "bore.stl"; }
  shaft      { type triSurfaceMesh; file "shaft.stl"; }
  kw_floor   { type triSurfaceMesh; file "kw_floor.stl"; }
  kw_flanks  { type triSurfaceMesh; file "kw_flanks.stl"; }
  inlet_HP   { type triSurfaceMesh; file "inlet_HP.stl"; }
  outlet_LP  { type triSurfaceMesh; file "outlet_LP.stl"; }
}
castellatedMeshControls {
  maxLocalCells 500000; maxGlobalCells 2000000;
  minRefinementCells 10; nCellsBetweenLevels 2; resolveFeatureAngle 30;
  features ();
  refinementSurfaces (
    bore       { level (3 3); }
    shaft      { level (3 3); }
    kw_floor   { level (3 3); }
    kw_flanks  { level (3 3); }
    inlet_HP   { level (2 2); }
    outlet_LP  { level (2 2); }
  );
  refinementRegions ();
  locationInMesh (0 0.02025 0.03);
  allowFreeStandingZoneFaces false;
}
snapControls {
  nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers (); expansionRatio 1.0; finalLayerThickness 0.1; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; mergePatchFacesAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls {
  maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minFlatness 0.05; minArea -1; minTwist 0.05;
  nSmoothScale 4; errorEstimationScale 2;
}
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(shm)
for tool,args in [("blockMesh",[]),("snappyHexMesh",["-overwrite"])]:
    r = subprocess.run([tool]+args, capture_output=True, text=True)
    print(tool, r.returncode, (r.stdout+r.stderr)[-1500:])

# -- cell 6 -------------------------------------------------------------------------
import subprocess
hdr = "FoamFile {\n  version 2.0;\n  format ascii;\n  class dictionary;\n}\n"
for path,body in [("system/blockMeshDict", None), ("system/controlDict", None)]:
    bdy = open(path).read()
    # rewrite header portion up to vertices/application keyword
    kw = "vertices" if "block" in path else "application"
    rest = bdy[bdy.index(kw):]
    open(path,"w").write(hdr+rest)
open("system/controlDict","w").write(hdr+"application snappyHexMesh;\n")
for tool,args in [("blockMesh",[]),("snappyHexMesh",["-overwrite"])]:
    r = subprocess.run([tool]+args, capture_output=True, text=True)
    print(tool, r.returncode, (r.stdout+r.stderr)[-800:])

# -- cell 7 -------------------------------------------------------------------------
import subprocess
def rehdr(path, obj):
    b = open(path).read()
    kw = [k for k in ("vertices","application","castellatedMesh") if k in b][0]
    hdr = f"FoamFile {{\n version 2.0;\n format ascii;\n class dictionary;\n object {obj};\n}}\n"
    open(path,"w").write(hdr+b[b.index(kw):])
rehdr("system/blockMeshDict","blockMeshDict")
rehdr("system/controlDict","controlDict")
rehdr("system/snappyHexMeshDict","snappyHexMeshDict")
for tool,args in [("blockMesh",[]),("snappyHexMesh",["-overwrite"])]:
    r = subprocess.run([tool]+args, capture_output=True, text=True)
    print(tool, r.returncode, (r.stdout+r.stderr)[-600:])

# -- cell 8 -------------------------------------------------------------------------
import subprocess
open("system/controlDict","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0;
writeFormat ascii; writePrecision 6; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh", r.returncode, r.stderr[-400:])
r2 = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("snappy", r2.returncode, (r2.stdout+r2.stderr)[-400:])

# -- cell 9 -------------------------------------------------------------------------
import subprocess
h="""FoamFile { version 2.0; format ascii; class dictionary; object FNAME; }\n"""
open("system/fvSchemes","w").write(h.replace("FNAME","fvSchemes")+"d2dt2Schemes { default Euler; }\n")
open("system/fvSolution","w").write(h.replace("FNAME","fvSolution")+"\n")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("snappy", r.returncode, (r.stdout+r.stderr)[-500:])

# -- cell 10 ------------------------------------------------------------------------
import subprocess
open("system/fvSchemes","w").write("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes      { default steady; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("snappy", r.returncode, (r.stdout+r.stderr)[-700:])

# -- cell 11 ------------------------------------------------------------------------
import subprocess
s=open("system/snappyHexMeshDict").read().replace("refinementSurfaces (","refinementSurfaces {").replace(");","}")
s=s.replace("refinementRegions (","refinementRegions ( 0 )")  # avoid accidental replace of following parens... simpler:
s=s.replace("refinementRegions ( 0 )","refinementRegions ();")
open("system/snappyHexMeshDict","w").write(s)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print("snappy", r.returncode, (r.stdout+r.stderr)[-600:])

# -- cell 12 ------------------------------------------------------------------------
shm = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap true;
addLayers false;
geometry {
 bore { type triSurfaceMesh; file "bore.stl"; }
 shaft { type triSurfaceMesh; file "shaft.stl"; }
 kw_floor { type triSurfaceMesh; file "kw_floor.stl"; }
 kw_flanks { type triSurfaceMesh; file "kw_flanks.stl"; }
 inlet_HP { type triSurfaceMesh; file "inlet_HP.stl"; }
 outlet_LP { type triSurfaceMesh; file "outlet_LP.stl"; }
}
castellatedMeshControls {
 maxLocalCells 500000; maxGlobalCells 2000000; minRefinementCells 10;
 nCellsBetweenLevels 2; resolveFeatureAngle 30;
 features ();
 refinementSurfaces {
  bore { level (3 3); }
  shaft { level (3 3); }
  kw_floor { level (3 3); }
  kw_flanks { level (3 3); }
  inlet_HP { level (2 2); }
  outlet_LP { level (2 2); }
 }
 refinementRegions ();
 locationInMesh (0 0.02025 0.03);
 allowFreeStandingZoneFaces false;
}
snapControls {
 nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers (); expansionRatio 1.0; finalLayerThickness 0.1; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; mergePatchFacesAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls {
 maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minFlatness 0.05; minArea -1; minTwist 0.05;
 nSmoothScale 4; errorEstimationScale 2;
}
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(shm)
import subprocess
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print("snappy",r.returncode,(r.stdout+r.stderr)[-500:])

# -- cell 13 ------------------------------------------------------------------------
s=open("system/snappyHexMeshDict").read().replace("refinementRegions ();","refinementRegions { }")
open("system/snappyHexMeshDict","w").write(s)
import subprocess
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print("snappy",r.returncode,(r.stdout+r.stderr)[-400:])

# -- cell 14 ------------------------------------------------------------------------
s=open("system/snappyHexMeshDict").read()
import re
s=re.sub(r"meshQualityControls \{[^}]*\}","meshQualityControls { }",s,flags=re.S)
open("system/snappyHexMeshDict","w").write(s)
import subprocess
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, (r.stdout+r.stderr)[-400:])
