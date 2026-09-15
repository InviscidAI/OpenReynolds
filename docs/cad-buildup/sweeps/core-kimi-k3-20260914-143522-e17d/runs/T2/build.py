"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
D = 0.04
sph = gmsh.model.occ.addSphere(0, 0, 0, D/2)
box = gmsh.model.occ.addBox(-5*D, -5*D, -5*D, 20*D, 10*D, 10*D)
fluid, _ = gmsh.model.occ.cut([(3, box)], [(3, sph)])
gmsh.model.occ.synchronize()
surfs = gmsh.model.getBoundary(fluid)
print("fluid:", fluid)
print("surfaces:", surfs)

# -- cell 2 -------------------------------------------------------------------------
# Six box faces plus the sphere, as expected. Now tag the patches by their centroid positions — inlet 
tags = {"inlet": [], "outlet": [], "farfield": [], "sphere": []}
for dim, s in surfs:
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, s)
    if abs(cx - (-5*D)) < 1e-9:
        tags["inlet"].append(s)
    elif abs(cx - 15*D) < 1e-9:
        tags["outlet"].append(s)
    elif abs(abs(cy) - 5*D) < 1e-2 or abs(abs(cz) - 5*D) < 1e-2:
        tags["farfield"].append(s)
    else:
        tags["sphere"].append(s)
for name, ss in tags.items():
    gmsh.model.addPhysicalGroup(2, ss, name=name)
    print(name, ss)
gmsh.model.addPhysicalGroup(3, [fluid[0][1]], name="internal")
os = __import__("os")
os.makedirs("geo", exist_ok=True)
gmsh.write("geo/fluid.geo_unrolled")
print({k:(gmsh.model.occ.getBoundingBox(*surfs[-1]))[0:3] for k in ["last_surf"]})

# -- cell 3 -------------------------------------------------------------------------
# Patches identified. Now set mesh sizes — fine on the sphere, coarse far field — plus a 4-layer bound
import time
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)  # 0.5 D far field
phs = gmsh.model.getEntitiesForPhysicalName("sphere")
sph_surf = tags["sphere"][0]
# Boundary layer field
bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(bl, "CurvesList", [1,2,3,4])  # placeholder; use EdgeList? try surface-level via GeometryCurves
gmsh.model.mesh.field.setNumber(bl, "Size", 1)
print("fields api:" , [x for x in dir(gmsh.model.mesh.field)][:5])

# -- cell 4 -------------------------------------------------------------------------
curves = gmsh.model.getBoundary([(2, sph_surf)])
edge_ids = [c for d, c in curves if d == 1]
bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(bl, "CurvesList", edge_ids)
gmsh.model.mesh.field.setNumber(bl, "Size", 0.0005)    # first layer 0.5 mm
gmsh.model.mesh.field.setNumber(bl, "Ratio", 1.3)
gmsh.model.mesh.field.setNumber(bl, "Thickness", 0.004)  # ~4 mm total
gmsh.model.mesh.field.setAsBackgroundMesh(bl)
gmsh.model.mesh.generate(3)
gmsh.write("fluid.msh")
print("nodes", len(gmsh.model.mesh.getNodes()[0]),
      "elems", len(gmsh.model.mesh.getElementsByType(4)[-1]) if True else 0)

# -- cell 5 -------------------------------------------------------------------------
# Mesh generated: ~86k volume elements. Now convert to OpenFOAM and run checkMesh.
import subprocess
r = subprocess.run(["gmshToFoam", "fluid.msh"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:])
r2 = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r2.stdout[-2500:], r2.stderr)

# -- cell 6 -------------------------------------------------------------------------
# Need a minimal case skeleton (system/controlDict). Writing that, then rerunning gmshToFoam + checkMe
import os
os.makedirs("system", exist_ok=True)
open("system/controlDict","w").write("""
FoamFile
{
    version 2.0; format ascii; class dictionary; object controlDict;
}
application none;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
runTimeModifiable yes;
""")
r = subprocess.run(["gmshToFoam", "fluid.msh"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr)

# -- cell 7 -------------------------------------------------------------------------
# Four named patches mapped. Checking the mesh (the `defaultFaces` warning needs explanation).
r2 = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r2.stdout[-3500:], r2.stderr)

# -- cell 8 -------------------------------------------------------------------------
open("system/fvSchemes","w").write("""
ddtSchemes { default( steadyState ); }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open("system/fvSolution","w").write("""
solvers { p { solver GAMG; tolerance 1e-6; relTol 0.1; smoother DICGaussSeidel; } }
SIMPLE { residualControl { p 1e-4; } }
relaxationFactors {}
""")
r3 = subprocess.run(["checkMesh"], capture_output=True, text=True)
txt = (r3.stdout[-6000:]+r3.stderr)
idx = txt.find("Create mesh")
print(txt[idx:])

# -- cell 9 -------------------------------------------------------------------------
def foamify(path, obj):
    s = open(path).read()
    header = '''FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "system";
    object %s;
}
// ******************************************************************* //
''' % obj
    open(path, "w").write(header + s)
foamify("system/fvSchemes", "fvSchemes")
foamify("system/fvSolution", "fvSolution")
r4 = subprocess.run(["checkMesh"], capture_output=True, text=True)
txt = r4.stdout + r4.stderr
i = txt.find("Create mesh")
print(txt[i:])

# -- cell 10 ------------------------------------------------------------------------
open("system/fvSchemes","w").write("""FoamFile { version 2.0; format ascii; class dictionary; location "system"; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open("system/fvSolution","w").write("""FoamFile { version 2.0; format ascii; class dictionary; location "system"; object fvSolution; }
solvers { p { solver GAMG; tolerance 1e-6; relTol 0.1; smoother DICGaussSeidel; } }
SIMPLE { residualControl { p 1e-4; } }
""")
r5 = subprocess.run(["checkMesh"], capture_output=True, text=True)
txt = r5.stdout + r5.stderr
i = txt.find("Create mesh")
print(txt[i:])

# -- cell 11 ------------------------------------------------------------------------
# Mesh is OK but the tagging failed: the boundary-layer field left most of the sphere surface faces in
# Test without BL field to isolate the tagging issue
gmsh.model.mesh.field.remove(bl)
gmsh.model.mesh.generate(3)
gmsh.write("fluid.msh")
r = subprocess.run(["gmshToFoam", "fluid.msh"], capture_output=True, text=True)
print(r.stdout[-700:])
r6 = subprocess.run(["checkMesh"], capture_output=True, text=True)
txt = r6.stdout + r6.stderr
i = txt.find("Checking patch topology")
print(txt[i:i+500])

# -- cell 12 ------------------------------------------------------------------------
pg2 = gmsh.model.getPhysicalGroups(2)
pg3 = gmsh.model.getPhysicalGroups(3)
print(pg2, pg3)
for dim, tag in pg2 + pg3:
    name = gmsh.model.getPhysicalName(dim, tag)
    ent = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
    print(name, ent)
    for s in ent:
        ts, es, _ = gmsh.model.mesh.getElements(dim, s)
        print("   entity", s, sum(len(e) for e in es))

# -- cell 13 ------------------------------------------------------------------------
allsurf = gmsh.model.getEntities(2)
print(allsurf)
for d, s in allsurf:
    ts, es, _ = gmsh.model.mesh.getElements(2, s)
    n = sum(len(e) for e in es)
    print("surface", s, "elements", n)
# also check: surface element counts might be misleading; use getBoundary with recursive?

# -- cell 14 ------------------------------------------------------------------------
for d, s in allsurf:
    bb = gmsh.model.occ.getBoundingBox(d, s)
    print("surf", s, bb)

# -- cell 15 ------------------------------------------------------------------------
# Surface 1's bounding box is exactly the sphere (±0.02), so the tagging was right and the element cou
ts, es, ns = gmsh.model.mesh.getElements(2, 1)
print(ts)
for t, e, n in zip(ts, es, ns):
    print("type", t, "nels", len(e))

# -- cell 16 ------------------------------------------------------------------------
vols = gmsh.model.getEntities(3)
print(vols)
for d, v in vols:
    bnd = gmsh.model.getBoundary([(d, v)])
    print("vol", v, "surfaces", [s for dd, s in bnd])
    ts,es,ns = gmsh.model.mesh.getElements(3, v)
    print("  vol elements", sum(len(e) for e in es))

# -- cell 17 ------------------------------------------------------------------------
# The gmsh conversion is producing duplicated boundary faces (an anonymous `".*"` patch), which isn't 
gmsh.model.mesh.clear()
gmsh.finalize()
pyvista_script = '''
import pyvista as pv
sph = pv.Sphere(center=(0,0,0), radius=0.02, theta_resolution=64, phi_resolution=48)
import os
os.makedirs("constant/triSurface", exist_ok=True)
sph.save("constant/triSurface/sphere.stl")
print("written")
'''
r = subprocess.run(["python3", "-c", pyvista_script], capture_output=True, text=True)
print(r.stdout, r.stderr)

# -- cell 18 ------------------------------------------------------------------------
bm = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices
(
    (-0.2 -0.2 -0.2)
    ( 0.6 -0.2 -0.2)
    ( 0.6  0.2 -0.2)
    (-0.2  0.2 -0.2)
    (-0.2 -0.2  0.2)
    ( 0.6 -0.2  0.2)
    ( 0.6  0.2  0.2)
    (-0.2  0.2  0.2)
);
blocks ( hex (0 1 2 3 4 5 6 7) (80 25 25) simpleGrading (1 1 1) );
boundary
(
    inlet   { type patch; faces ( (0 4 7 3) ); }
    farfield
    {
        type patch;
        faces ( (0 1 5 4) (3 2 6 7) (0 3 7 4) (1 2 6 5) );
    }
    outlet  { type patch; faces ( (1 2 6 5) wait }
);
"""
# correct faces: z-:(0 3 7 4)? order around face. Use proper vertices: z- face (0 1 2 3), z+ (4 5 6 7)
bm = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
 vertices
 (
    (-0.2 -0.2 -0.2) //0
    ( 0.6 -0.2 -0.2) //1
    ( 0.6  0.2 -0.2) //2
    (-0.2  0.2 -0.2) //3
    (-0.2 -0.2  0.2) //4
    ( 0.6 -0.2  0.2) //5
    ( 0.6  0.2  0.2) //6
    (-0.2  0.2  0.2) //7
 );
 blocks ( hex (0 1 2 3 4 5 6 7) (80 25 25) simpleGrading (1 1 1) );
 boundary
 (
     inlet   { type patch; faces ( (0 4 7 3) ); }
     outlet  { type patch; faces ( (1 5 6 2) ); }
     farfield
     {
         type patch;
         faces ( (0 1 5 4) (3 2 6 7) (0 3 7 4) (1 5 6 2) );
     }
     sphere { type wall; faces (); }
 );
"""
# outlet duplicated; fix: far = y- (0 1 5 4), y+ (3 2 6 7)? no z faces. sides: z-(0 3 7 4)->y... Let's just compute: y- face=(0 1 5 4); y+= (3 2 6 7); z-= (0 1 2 3); z+=(4 5 6 7)
bm = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices
(
    (-0.2 -0.2 -0.2)
    ( 0.6 -0.2 -0.2)
    ( 0.6  0.2 -0.2)
    (-0.2  0.2 -0.2)
    (-0.2 -0.2  0.2)
    ( 0.6 -0.2  0.2)
    ( 0.6  0.2  0.2)
    (-0.2  0.2  0.2)
);
blocks ( hex (0 1 2 3 4 5 6 7) (80 25 25) simpleGrading (1 1 1) );
boundary
(
    inlet   { type patch; faces ( (0 4 7 3) ); }
    outlet  { type patch; faces ( (1 5 6 2) ); }
    farfield
    {
        type patch;
        faces ( (0 1 5 4) (3 2 6 7) (0 1 2 3) (4 5 6 7) );
    }
);
"""
open("system/blockMeshDict", "w").write(bm)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-400:], r.stderr)

# -- cell 19 ------------------------------------------------------------------------
# Background mesh is good. Now the snappyHexMeshDict with a refinement box around the sphere and 4 pri
shm = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       true;
geometry
{
    sphere.stl { type triSurfaceMesh; name sphere; }
    box { type searchableBox; min (-0.1 -0.1 -0.1); max (0.1 0.1 0.1); }
}
castellatedMeshControls
{
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        sphere { level (2 3);  }
    }
    resolveFeatureAngle 30;
    locationInsideMesh (0.5 0 0);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    layers
    {
        sphere { nSurfaceLayers 4; }
    }
    expansionRatio 1.3;
    finalLayerThickness 0.5;
    minThickness 0.02;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    slipConcaveFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    preventExtrusion false;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
    nAdditionalRatio 1;
}
meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    maxFlatRatio 0.5;
    maxFaceAspectRatio 4;
}
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict", "w").write(shm)
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(r.stdout[-1200:], r.stderr)

# -- cell 20 ------------------------------------------------------------------------
s = open("system/snappyHexMeshDict").read()
s = s.replace("""    resolveFeatureAngle 30;""", """resolveFeatureAngle 30;
    refinementRegions { box { mode inside; levels ((1E15 2)); } }""")
open("system/snappyHexMeshDict","w").write(s)
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr)

# -- cell 21 ------------------------------------------------------------------------
import pyvista as pv
m = pv.read("constant/triSurface/sphere.stl")
m.compute_normals(inplace=True)
pt = m.cell_centers().points[0]
n = m.face_normals[0]
print("dot:", n @ pt)

# -- cell 22 ------------------------------------------------------------------------
# Normals are outward. Checking the full snappy log to see the cell counts before the subsetting step.
import subprocess
r = subprocess.run(["rm","-rf","0"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
txt = r.stdout + r.stderr
lines = [l for l in txt.splitlines() if "cells" in l.lower() or "Faces" in l or "refine" in l.lower() or "Split" in l]
for l in lines: print(l[:140])

# -- cell 23 ------------------------------------------------------------------------
# The location point got dropped from the list during the region split. Giving it list syntax and reru
s = open("system/snappyHexMeshDict").read()
s = s.replace("locationInsideMesh (0.5 0 0);", "locationInsideMesh ((0.5 0 0));")
open("system/snappyHexMeshDict","w").write(s)
subprocess.run(["rm","-rf","0"], capture_output=True)
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
txt = r.stdout + r.stderr
lines = [l for l in txt.splitlines() if "After" in l or "keeping" in l.lower() or "Selected" in l]
print("\n".join(l[:150] for l in lines[-8:]))
print("finish?", "End" in txt)

# -- cell 24 ------------------------------------------------------------------------
r = subprocess.run(["grep","-rn","insidePoint","/usr/lib/openfoam/openfoam2512/bin","2>/devnull"], capture_output=True, text=True)
print(r.stdout[:500], r.stderr)
s = open("system/snappyHexMeshDict").read()
s = s.replace("locationInsideMesh ((0.5 0 0));", "locationInsideMesh ((0.5 0 0));\n")
open("system/snappyHexMeshDict","w").write(s)
subprocess.run(["grep","-n","locationInsideMesh","system/snappyHexMeshDict"])

# -- cell 25 ------------------------------------------------------------------------
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh"], capture_output=True, text=True)
txt = r.stdout + r.stderr
print([l[:150] for l in txt.splitlines() if "keeping" in l.lower()][-3:])
print("success:", "End" in txt)
