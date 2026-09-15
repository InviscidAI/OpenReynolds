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
import numpy as np

# Dimensions in mm (converted to m only at STL export -> actually keep geometry in mm, scale STL on export)
PL_L, PL_W, PL_H = 300., 120., 80.   # plenum internal size (y, x, z)
R_THROTTLE = 35.                       # 70 dia throttle inlet
R_RUN = 19.                            # 38 dia runner
R_BEND = 60.                           # elbow centreline radius
PORT_X = 90.                           # port face distance from plenum wall
GROUND_VERT = 60.                      # vertical drop after bend (assumption)
INLET_LEN = 40.                        # throttle pipe length (assumption)

ys = [-112.5, -37.5, 37.5, 112.5]      # runner stations (assumption: even spacing)

def runner(side, y):
    # centreline: horizontal out of wall (+x or -x), 30mm straight, R60 90-deg arc, then vertical down
    s = side  # +1 right, -1 left
    ls = bd.Line(bd.Vector(0, 0, 0), bd.Vector(30*s, 0, 0))
    arc = bd.RadiusArc(bd.Vector(30*s, 0, 0), bd.Vector(0, 0, -R_BEND), radius=..., )  # placeholder
    return ls

print("placeholder, testing API")
import inspect
print(inspect.signature(bd.Line))
print(inspect.signature(bd.RadiusArc))

# -- cell 2 -------------------------------------------------------------------------
def runner_wire():
    # centreline in x-z plane, y=0, for +x side
    p1 = bd.Line((0,0,0), (30,0,0))
    a1 = bd.RadiusArc((30,0,0), (90,0,-60), radius=R_BEND)
    v1 = bd.Line((90,0,-60), (90,0,-60-GROUND_VERT))
    w = bd.Wire([p1, a1, v1])
    return w

def runner_solid(side, y):
    # face normal must be along wire start tangent
    w = runner_wire()
    profile = bd.Pos(0,0,0) * bd.Circle(R_RUN)  # circle plane normal +X? check
    return bd.sweep(profile, path=w)

# test: circle normal is +Z normally; need normal along +X
circ = bd.Circle(R_RUN)
print(circ.faces()[0].normal_at().to_tuple())
# rotate into proper plane
def sweep_tube(side, y):
    w = bd.Pos(side, 0 if side<0 else 0) * bd.Wire([])  # placeholder
    return None

# -- cell 3 -------------------------------------------------------------------------
def make_tube(side):
    w = runner_wire()
    prof = bd.Pos(0,0,0) * bd.Rot(0,90/side if side<0 else 90,0) * bd.Circle(R_RUN)
    tube = bd.sweep(prof, path=w)
    return tube

t = make_tube(1)
print("volume mm3:", t.volume)
bb = t.bounding_box()
print("bbox:", bb.min.to_tuple() if hasattr(bb,'min') else bb)

# -- cell 4 -------------------------------------------------------------------------
bb = t.bounding_box()
print(tuple(bb.min), tuple(bb.max))
print("n faces", len(t.faces()))
for f in t.faces():
    print(f.geom_type, f.area)

# -- cell 5 -------------------------------------------------------------------------
plenum = bd.Pos(0, 0, 0) * bd.Box(PL_W, PL_L, PL_H)
inlet_pipe = bd.Pos(0, PL_L/2 + INLET_LEN/2, 0) * bd.Rot(90,0,0) * bd.Cylinder(R_THROTTLE, INLET_LEN)

solids = [plenum, inlet_pipe]
for side in (1, -1):
    for y0 in ys:
        tb = make_tube(side)
        if side == -1:
            tb = bd.Rot(0,0,180) * tb
        tb = tb.moved(bd.Location((side*PL_W/2, y0 if side==1 else -y0 if False else y0, PL_H/2)))
        solids.append(tb)

air = solids[0]
for s in solids[1:]:
    air = air + s
print("volume:", air.volume, " faces:", len(air.faces()))
import pyvista
v,f = air.tessellate(1.0)
v = np.array([tuple(p) for p in v],float)
fidx = np.asarray(f)
flat = np.concatenate([np.full((len(fidx),1),3), fidx], axis=1)
mesh = pyvista.PolyData(v, flat)
pl = pyvista.Plotter(off_screen=True)
pl.add_mesh(mesh)
pl.view_isometric()
pl.screenshot("manifold.png"); pl.close()
from IPython.display import Image
Image("manifold.png")

# -- cell 6 -------------------------------------------------------------------------
# Now emit the consolidated accepted cell (definitions + build + classify + export) so the script is c
import build123d as bd, numpy as np, math, os, glob, struct
PL_L, PL_W, PL_H = 300., 120., 80.
R_THROTTLE = 35.; R_RUN = 19.; R_BEND = 60.; PORT_X = 90.; GROUND_VERT = 60.; INLET_LEN = 40.
ys = [-112.5, -37.5, 37.5, 112.5]
TOL = 0.5
def runner_wire():
    w = bd.Wire([bd.Line((0,0,0),(30,0,0)),
                 bd.RadiusArc((30,0,0),(90,0,-60),radius=R_BEND),
                 bd.Line((90,0,-60),(90,0,-60-GROUND_VERT))])
    return w
def make_tube():
    w = runner_wire()
    prof = bd.Pos(0,0,0)*bd.Rot(0,90,0)*bd.Circle(R_RUN)
    return bd.sweep(prof, path=w)
plenum = bd.Pos(0,0,0)*bd.Box(PL_W, PL_L, PL_H)
inlet_pipe = bd.BoundedBox if False else bd.Pos(0, PL_L/2+INLET_LEN/2, 0)*bd.Rot(90,0,0)*bd.Cylinder(R_THROTTLE, INLET_LEN)
solids = [plenum, inlet_pipe]
for side in (1,-1):
    for y0 in ys:
        tb = make_tube()
        if side == -1: tb = bd.Rot(0,0,180)*tb
        tb = tb.moved(bd.Location((side*PL_W/2, y0, PL_H/2)))
        solids.append(tb)
air = solids[0]
for s in solids[1:]: air = air + s

def assign(f):
    gt, area, c = f.geom_type, f.area, tuple(f.center())
    if gt==bd.GeomType.PLANE and abs(area-math.pi*R_THROTTLE**2)<500: return "inlet"
    if gt==bd.GeomType.CYLINDER and f.radius and abs(f.radius-R_THROTTLE)<TOL: return "inletWall"
    if gt==bd.GeomType.PLANE and abs(area-math.pi*R_RUN**2)<150:
        side = 'R' if c[0]>0 else 'L'
        yi = ys.index(min(ys, key=lambda y: abs(y-c[1])))
        return f"port_{side}{yi+1}"
    return "plenumWalls" if gt==bd.GeomType.PLANE else "runnerWalls"
patches = {}
for f in air.faces(): patches.setdefault(assign(f),[]).append(f)
assert sum(len(v) for v in patches.values())==len(air.faces())
os.makedirs("constant/triSurface", exist_ok=True)
for old in glob.glob("constant/triSurface/*.stl"): os.remove(old)
for name,fl in patches.items():
    bd.export_stl(bd.Compound(children=list(fl)), f"constant/triSurface/{name}.stl", tolerance=0.05)
def scale_stl(path, fc=0.001):
    d = open(path,'rb').read(); n = struct.unpack_from('<I',d,80)[0]; b = bytearray(d)
    for i in range(n):
        base = 84+i*50; vals = struct.unpack_from('<12fH',d,base)
        struct.pack_into('<12fH', b, base, *[v*fc for v in vals[:12]], vals[12])
    open(path,'wb').write(b)
for p in glob.glob("constant/triSurface/*.stl"): scale_stl(p)
print("volume mm3:", air.volume, "| faces:", len(air.faces()), "| patches:", {k:len(v) for k,v in patches.items()})

# -- cell 7 -------------------------------------------------------------------------
# STLs done — 12 patches. Now write the stub of the OpenFOAM case: blockMesh background and snappyHexM
import os, json
# bounding box in metres (from STL extents)
xs = [-0.30, 0.30]; ys_ = [-0.35, 0.35]; zs = [-0.25, 0.20]
CELL = 0.015  # 15 mm cells, coarse
nx = int((xs[1]-xs[0])/CELL); ny = int((ys_[1]-ys_[0])/CELL); nz = int((zs[1]-zs[0])/CELL)
print(nx,ny,nz, nx*ny*nz)

blockMeshDict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    object      blockMeshDict;
}}
convertToMeters 1;
vertices
(
    ({xs[0]} {ys_[0]} {zs[0]}) ({xs[1]} {ys_[0]} {zs[0]})
    ({xs[1]} {ys_[1]} {zs[0]}) ({xs[0]} {ys_[1]} {zs[0]})
    ({xs[0]} {ys_[0]} {zs[1]}) ({xs[1]} {ys_[0]} {zs[1]})
    ({xs[1]} {ys_[1]} {zs[1]}) ({xs[0]} {ys_[1]} {zs[1]})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges (); patches (); defaultPatch {{ name walls; type wall; }}
"""
open("system/blockMeshDict","w").write(blockMeshDict) if os.path.isdir("system") else None
os.makedirs("system", exist_ok=True)
open("system/blockMeshDict","w").write(blockMeshDict)
print("written")

# -- cell 8 -------------------------------------------------------------------------
blockMeshDict = f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0; format ascii; object  blockMeshDict; class dictionary;
}}
convertToMeters 1;
vertices
(
    ({xs[0]} {ys_[0]} {zs[0]}) ({xs[1]} {ys_[0]} {zs[0]})
    ({xs[1]} {ys_[1]} {zs[0]}) ({xs[0]} {ys_[1]} {zs[0]})
    ({xs[0]} {ys_[0]} {zs[1]}) ({xs[1]} {ys_[0]} {zs[1]})
    ({xs[1]} {ys_[1]} {zs[1]}) ({xs[0]} {ys_[1]} {zs[1]})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
patches
(
    wall walls ( (0 4 7 3) (2 6 5 1) (1 5 4 0) (3 7 6 2) (0 1 2 3) (4 5 6 7) )
);
"""
open("system/blockMeshDict","w").write(blockMeshDict)

patchNames = ["inlet","inletWall","plenumWalls","runnerWalls"] + [f"port_{s}{i+1}" for s in 'RL' for i in range(4)]
geom = "\n".join(f"        {name}\n        {{\n            type triSurfaceMesh;\n            file \"{name}.stl\";\n            regions {{ surface {name}; }}\n        }}" for name in patchNames)
refRef = "\n".join(f"            {{\n                name {name};\n                level  {{ min 1; max 2; }}\n            }}" for name in patchNames)
patchNames_str = " ".join(patchNames)
shm = f"""FoamFile {{ version 2.0; format ascii; object snappyHexMeshDict; class dictionary; }}
castellated true; snap true; addLayers false;
geometry
{{
{geom}
}};
castellatedMeshControls
{{
    maxLocalCells 6000000; maxGlobalCells 12000000; minRefinementCells 10;
    nCellsBetweenLevels 1; resolveFeatureAngle 30; allowFreeStandingZoneFaces false;
    locationsInMesh ((0 0 0.04)); planarTolerance 0.5; checkGapLevels ();
    features ();
    refinementSurfaces
    {{
        manifold {{ level ((0 1)(2 2)); patchInfo {{ type wall; }} }}
    }}
    refinementRegions {{ }}
}}
snapControls
{{
    nSmoothPatch 3; snapTolerance 4; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true; expansionRatio 1; finalLayerThickness 0.6; layers ();
    minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30;
    nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedial 0.3; minMedialAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50; nRelaxedIter 20;
}}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; features {{ surfaceGlen }}; }}
writeFlags ();
"""
# group geometry into one geometry 'manifold' hmm simpler: each stl separately with named surface
open("system/snappyHexMeshDict","w").write(shm)
os.makedirs("constant", exist_ok=True)
open("system/decomposeParDict","w").write("FoamFile {version 2.0; format ascii; object decomposeParDict;} numberOfSubdomains 1; method scotch;")
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
# Add controlDict and fix snappyHexMeshDict with per-surface refinement entries.
ctrl = """FoamFile {version 2.0; format ascii; object controlDict; class dictionary;}
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0;
writeFormat ascii; writePrecision 6; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""
open("system/controlDict","w").write(ctrl)

geom_entries = "\n".join(f"    {name}\n    {{\n        type triSurfaceMesh;\n        file \"{name}.stl\";\n    }}" for name in patchNames)
refine = "\n".join(f"        {name} {{ level ((0 1) (2 2)); patchInfo {{ type wall; }} }}" for name in patchNames)
# inlet should be patch, ports patch too; walls wall
def pinfo(n):
    return "patch" if n in ("inlet",) or n.startswith("port") else "wall"
refine = "\n".join(f"        {name} {{ level ((0 1) (2 2)); patchInfo {{ type {pinfo(name)}; }} }}" for name in patchNames)
shm = """FoamFile {version 2.0; format ascii; object snappyHexMeshDict; class dictionary;}
castellated true; snap true; addLayers false;
geometry
{
GEOM
};
castellatedMeshControls
{
    maxLocalCells 6000000; maxGlobalCells 12000000; minRefinementCells 10;
    maxLoadUnbalance 0.1; nCellsBetweenLevels 1; resolveFeatureAngle 30;
    allowFreeStandingZoneFaces false;
    locationsInMesh ((0 0 0.04));
    features ();
    refinementSurfaces
    {
REF
    }
    refinementRegions {}
}
snapControls
{
    nSmoothPatch 3; snapTolerance 4; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true; expansionRatio 1.0; finalLayerThickness 0.6; layers ();
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedial 0.3; minMedialAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
writeFlags ();
""".replace("GEOM", geom_entries).replace("REF", refine)
open("system/snappyHexMeshDict","w").write(shm)
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-200:], r.stderr[-300:])

# -- cell 10 ------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1500:]); print(r.stderr[-1000:])

# -- cell 11 ------------------------------------------------------------------------
with open("system/fvSchemes","w") as fh: fh.write("FoamFile {version 2.0; format ascii; object fvSchemes;}\n")
with open("system/fvSolution","w") as fh: fh.write("FoamFile {version 2.0; format ascii; object fvSolution;}\n")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1200:]); print(r.stderr[-800:])

# -- cell 12 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("""FoamFile {version 2.0; format ascii; object fvSchemes; class dictionary;}
ddtSchemes { default none; }
divSchemes { default none; }
gradSchemes { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default none; }
snGradSchemes { default none; }
""")
open("system/fvSolution","w").write("""FoamFile {version 2.0; format ascii; object fvSolution; class dictionary;}
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1500:]); print(r.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.stderr[-1500:])
print(r.stdout[-t] if False else r.stdout[-900:])

# -- cell 14 ------------------------------------------------------------------------
shm_text = open("system/snappyHexMeshDict").read()
shm_text = shm_text.replace("castellated true; snap true; addLayers false;",
                            "castellated true; snap true; addLayers false;\nmergeTolerance 1e-6;")
open("system/snappyHexMeshDict","w").write(shm_text)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stderr[-800:])
print(r.stdout[-1500:])

# -- cell 15 ------------------------------------------------------------------------
shm_text = open("system/snappyHexMeshDict").read()
shm_text = shm_text.replace("level ((0 1)(2 2))", "level { min 1; max 2; }")
open("system/snappyHexMeshDict","w").write(shm_text)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stderr[-800:])
print(r.stdout[-1200:])

# -- cell 16 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["grep","-roh","level .*;","/opt/openfoam" if False else "$WM_PROJECT_DIR/tutorials","-H"], capture_output=True, text=True, env={})
r = subprocess.run(["bash","-c","grep -rPo 'level\s+\(.*?\)\s*;' $WM_PROJECT_DIR/tutorials 2>/dev/null | sort -u | head"], capture_output=True, text=True)
print(r.stdout)
r = subprocess.run(["bash","-c","grep -rP -A2 'refinementSurfaces' $WM_PROJECT_DIR/tutorials | grep 'level' | sort -u | head"], capture_output=True, text=True)
print(r.stdout)
