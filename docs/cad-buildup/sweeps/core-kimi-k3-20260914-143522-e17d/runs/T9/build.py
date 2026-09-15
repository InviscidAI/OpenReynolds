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
outer = bd.Cylinder(radius=0.030, height=0.010)
spinner = (bd.Cylinder(radius=0.022, height=0.008)
           - bd.Cylinder(radius=0.015, height=0.008))
oil = outer - spinner
faces = list(oil.faces())
for i,f in enumerate(faces):
    (loc, nrm) = f.center_location
    c = f.center()
    print(i, f.geom_type, "area=%.6f"%f.area, "nrm=",tuple(round(v,3) for v in (nrm if isinstance(nrm,bd.Vector) else nrm.to_tuple())) , "xyz=",(round(c.X,4),round(c.Y,4),round(c.Z,4)))

# -- cell 2 -------------------------------------------------------------------------
# One correction: the ring is an annulus (30 outer, 23 inner), so the oil's outer wall must be at r=23
ring = bd.Cylinder(radius=0.030, height=0.010) - bd.Cylinder(radius=0.023, height=0.010)
spinner = bd.Cylinder(radius=0.022, height=0.008) - bd.Cylinder(radius=0.015, height=0.008)
bare = bd.Cylinder(radius=0.023, height=0.010)
oil = bare - spinner
for i,f in enumerate(oil.faces()):
    (loc, nrm) = f.center_location
    c = f.center()
    r=(c.X**2+c.Y**2)**0.5
    print(i, f.geom_type, "area=%.6f"%f.area, "r= %.4f"%r, "z= %.4f"%c.Z)

# -- cell 3 -------------------------------------------------------------------------
# Clean classification. Export one STL per patch.
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)
groups = {
  "ring_wetted": [oil.faces()[0]],
  "spinner_wetted": oil.faces()[3:4]+oil.faces()[5:7]+[oil.faces()[4]],
  "endGapTop": [oil.faces()[1]],
  "endGapBottom": [oil.faces()[2]],
}
groups["spinner_wetted"] = [oil.faces()[i] for i in (3,4,5,6)]
import build123d as bd
for name, fl in groups.items():
    comp = bd.Compound(label=name, children=fl)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl")
    print(name, len(fl), "faces")

# -- cell 4 -------------------------------------------------------------------------
cd_d = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application none; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 100000; purgeWrite 0;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""
open("system/controlDict","w").write(cd_d)
subprocess.run(["blockMesh"], check=False)
print("ok")

# -- cell 5 -------------------------------------------------------------------------
snappy = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
    ring_wetted.stl { type triSurfaceMesh; name ring_wetted; }
    spinner_wetted.stl { type triSurfaceMesh; name spinner_wetted; }
    endGapTop.stl { type triSurfaceMesh; name endGapTop; }
    endGapBottom.stl { type triSurfaceMesh; name endGapBottom; }
}
castellatedMeshControls
{
    maxLocalCells 1000000; maxGlobalCells 2000000;
    minRefinementCells 10; maxLoadUncertainty 0.10;
    nCellsBetweenLevels 3;
    explicitFeatureEdgeRefinement false;
    features ();
    refinementSurfaces
    {
        ring_wetted    { level (3 3); patchInfo { type patch; } }
        spinner_wetted { level (3 3); patchInfo { type patch; } }
        endGapTop      { level (3 3); patchInfo { type patch; } }
        endGapBottom   { level (3 3); patchInfo { type patch; } }
    }
    resolveFeatureAngle 30;
    refinementRegions {}
    locationInMesh ( 0.0225 0.0 0.0 );
    allowFreeStandingZoneFaces false;
}
snapControls
{
    nSmoothPatch 3; tolerance 4.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    expansionRatio 1.0; finalLayerThickness 0.5; minThickness 0.2;
    nGrow 0; featureAngle 150; slipFeatureAngle 45;
    nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nMedialAxisIterations 10; nBufferCellsNoExtrude 0;
    nLayerIter 50;
    layers {}
}
meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minFlatness 0.5;
    minVol -1.; minTetQuality 1e-15;
    minArea -1; minTwist 0.05; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist 0.002;
    nSmoothScale 4; errorReduction 0.75;
}
advanced
{
    verbose 0;
}
"""
open("system/snappyHexMeshDict","w").write(snappy)
p = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print(p.stdout[-2000:]); print("rc", p.returncode)

# -- cell 6 -------------------------------------------------------------------------
import re
idx = p.stdout.find("geometry {")
tail = p.stdout if idx<0 else p.stdout
print(tail[-4000:])

# -- cell 7 -------------------------------------------------------------------------
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(p.stdout[-3000:], p.stderr[-1000:], "rc", p.returncode)

# -- cell 8 -------------------------------------------------------------------------
open("system/fvSchemes","w").write("FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }")
open("system/fvSolution","w").write("FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }")
p = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
s = p.stdout
print("rc", p.returncode)
i = s.find("Extract cells")
print(s[-3500:])

# -- cell 9 -------------------------------------------------------------------------
print(p.stderr[-2500:])

# -- cell 10 ------------------------------------------------------------------------
fvs = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {}
interpolationSchemes {} snGradSchemes {}
"""
fso = """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {} PISO {}
"""
open("system/fvSchemes","w").write(fvs)
open("system/fvSolution","w").write(fso)
p = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", p.returncode)
print(p.stdout[-2500:]); print(p.stderr[-1500:])

# -- cell 11 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace("meshQualityControls", "mergeTolerance 1e-6;\nmeshQualityControls")
open("system/snappyHexMeshDict","w").write(s)
p = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", p.returncode)
print(p.stdout[-1800:]); print(p.stderr[-1500:])

# -- cell 12 ------------------------------------------------------------------------
# snappy ended with 1 cell — the cell extraction failed. Checking the full log.
out = p.stdout
import re
for m in re.finditer(r"(Extract cells|cells|fluid|inside|outside)", out):
    pass
i = out.find("Selected")
print([l for l in out.splitlines() if re.search("cel|extract|inside|outside|region|Iterate", l, re.I)][:60])

# -- cell 13 ------------------------------------------------------------------------
i = out.find("After introducing baffles")
print(out[i-2000:i+500] if i>=0 else "")
print("-----")
open("/tmp/shm.log","w").write(out)

# -- cell 14 ------------------------------------------------------------------------
import gmsh
gmsh.initialize()
print("gmsh ready")

# -- cell 15 ------------------------------------------------------------------------
import build123d as bd
bd.export_step(oil, "oil.step")
gmsh.model.add("oil")
shapes = gmsh.model.occ.importShapes("oil.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
for dim,tag in surfs:
    cx,cy,cz = gmsh.model.occ.getCenterOfMass(dim, tag)
    r = (cx*cx+cy*cy)**0.5
    print(tag, "r=%.5f"%r, "z=%.4f"%cz)

# -- cell 16 ------------------------------------------------------------------------
groups_map = {"ring_wetted":[], "spinner_wetted":[], "endGapTop":[], "endGapBottom":[]}
for dim,tag in gmsh.model.getEntities(2):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(dim, tag)
    rmax = max(abs(x0),abs(x1))
    zc = (z0+z1)/2
    if rmax > 1e-4:  # cylinder
        r = round(rmax, 5)
        if abs(r-0.0230) < 1e-4: groups_map["ring_wetted"].append(tag)
        else: groups_map["spinner_wetted"].append(tag)
    else:  # plane
        if abs(zc-0.005) < 1e-4: groups_map["endGapTop"].append(tag)
        elif abs(zc+0.005) < 1e-4: groups_map["endGapBottom"].append(tag)
        else: groups_map["spinner_wetted"].append(tag)
for k,v in groups_map.items():
    print(k, v)
    gmsh.model.addPhysicalGroup(2, v, name=k)
vols = gmsh.model.getEntities(3)
print("volumes", vols)
gmsh.model.addPhysicalGroup(3, [v[-1] for _,v in [(3,t[1]) for t in [vols[0]]]] if False else [vols[0][1]], name="oil")

# -- cell 17 ------------------------------------------------------------------------
for pg in gmsh.model.getPhysicalGroups(2):
    gmsh.model.removePhysicalGroups([pg])
groups_map = {"ring_wetted":[], "spinner_wetted":[], "endGapTop":[], "endGapBottom":[]}
for dim,tag in gmsh.model.getEntities(2):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(dim, tag)
    dz = z1-z0
    r = max(abs(x0),abs(x1))
    zc=(z0+z1)/2
    if dz > 1e-4:  # cylinder
        name = "ring_wetted" if abs(r-0.023)<1e-4 else "spinner_wetted"
    else:
        name = "endGapTop" if zc>0.0045 else ("endGapBottom" if zc<-0.0045 else "spinner_wetted")
    groups_map[name].append(tag)
for k,v in groups_map.items():
    gmsh.model.addPhysicalGroup(2, v, name=k)
    print(k, v)

# -- cell 18 ------------------------------------------------------------------------
# Classified correctly (7 surfaces: 2 cylinders ring, 4 spinner, 1 top, 1 bottom). Now mesh. A coarse 
import math
expected = math.pi*(0.023)**2*0.010 - math.pi*((0.022**2-0.015**2))*0.008
print("expected volume (m^3): %.14f"%expected)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0005)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.0020)
gmsh.model.mesh.generate(3)
gmsh.write("oil.msh")
gmsh.finalize()
print("meshed to oil.msh")
p = subprocess.run(["gmshToFoam", "oil.msh"], capture_output=True, text=True)
print(p.stdout[-800:], p.stderr[-300:], "rc", p.returncode)

# -- cell 19 ------------------------------------------------------------------------
# Mesh converted; two patch names came through as `patch1/patch2` instead of endGap names. Check the b
print(open("constant/polyMesh/boundary").read())

# -- cell 20 ------------------------------------------------------------------------
import subprocess, re
# look at named entities in the msh file
out = subprocess.run(["grep","-n","PhysicalTags|Entities","oil.msh"], capture_output=True, text=True)
# easier: just count occurrences of names
txt = open("oil.msh").read()
for nm in ["ring_wetted","spinner_wetted","endGapTop","endGapBottom","$ElementData"]:
    print(nm, txt.count(nm))

# -- cell 21 ------------------------------------------------------------------------
import numpy as np, re
def read_block(path):
    txt = open(path).read()
    m = re.search(r"\n\s*(\d+)\s*\(", txt)
    start = m.start()
    n = int(m.group(1))
    body = txt[m.end()-1:]
    end = body.index(")\n")
    return n, body[1:end]
n, ptxt = read_block("constant/polyMesh/points")
ptokens = re.findall(r"[-\d.eE+]+", ptxt)
arr = np.array([float(t) for t in ptokens]).reshape(-1,3)
n2, ftxt = read_block("constant/polyMesh/faces")
grp = re.findall(r"\((.*?)\)", ftxt, re.S)
faces = [list(map(int, g.split())) for g in grp]
print("nfaces parsed", len(faces), "total expected", 19167+1013 and None)
