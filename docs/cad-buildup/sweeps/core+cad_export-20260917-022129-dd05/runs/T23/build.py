"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water directly in metres as the union of the central 16 mm sphere and three orthogona
import build123d as bd
from build123d import Box, Sphere, Cylinder, Pos, Rot, Align
import matplotlib.pyplot as plt
import pyvista as pv

# All dimensions are metres (request values converted once from mm).
CUBE = 0.040
HOLLOW_R = 0.016
BALL_R = 0.015
HOLE_R = 0.012

cube = Box(CUBE, CUBE, CUBE, align=(Align.CENTER, Align.CENTER, Align.CENTER))
hollow = Sphere(HOLLOW_R)
ball = Sphere(BALL_R)
bore_z = Cylinder(HOLE_R, CUBE, align=(Align.CENTER, Align.CENTER, Align.CENTER))
bore_x = Rot(0, 90, 0) * bore_z
bore_y = Rot(90, 0, 0) * bore_z
wet_cavity = (hollow + bore_x + bore_y + bore_z) & cube
fluid = wet_cavity - ball

bb = fluid.bounding_box()
print(f"Requested cube side: {CUBE:.6f} m; model bounds: {bb.size.X:.6f}, {bb.size.Y:.6f}, {bb.size.Z:.6f} m")
print(f"Requested hollow radius: {HOLLOW_R:.6f} m; ball radius: {BALL_R:.6f} m; radial clearance: {(HOLLOW_R-BALL_R):.6f} m")
print(f"Requested hole radius: {HOLE_R:.6f} m")
print(f"Fluid volume: {fluid.volume:.9e} m^3; solids: {len(fluid.solids())}; boundary faces: {len(fluid.faces())}")

# Coarse visual inspection via a temporary preview tessellation.
bd.export_stl(fluid, "fluid_preview.stl", tolerance=7.5e-4, angular_tolerance=0.25)
mesh_preview = pv.read("fluid_preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
plotter.add_mesh(mesh_preview, color="deepskyblue", opacity=0.72, show_edges=True, edge_color="navy")
plotter.add_axes()
plotter.view_isometric()
plotter.show(screenshot="fluid_preview.png")
print(f"Preview: {mesh_preview.n_cells} triangles; saved fluid_preview.png")

# -- cell 2 -------------------------------------------------------------------------
# The preview shows the intended connected six-port cavity and the enclosed ball boundary. The 21 CAD 
for i, f in enumerate(fluid.faces()):
    c = f.center()
    print(f"{i:02d} type={str(f.geom_type):12s} area={f.area:.9e} center=({c.X:+.6f},{c.Y:+.6f},{c.Z:+.6f})")

# -- cell 3 -------------------------------------------------------------------------
# The analytic types confirm the expected topology and also show why a face-center shortcut is unsafe:
import inspect
print([n for n in dir(fluid.faces()[1]) if any(k in n.lower() for k in ("axis", "normal", "surface", "adaptor"))])
print("normal_at signature:", inspect.signature(fluid.faces()[1].normal_at))
print("axis_of_rotation:", getattr(fluid.faces()[1], "axis_of_rotation", None))

# -- cell 4 -------------------------------------------------------------------------
# The intrinsic cylinder axis is available and already carries the generating half-bore direction. I’l
for i, f in enumerate(fluid.faces()):
    if f.geom_type == bd.GeomType.CYLINDER:
        a = f.axis_of_rotation
        print(f"cyl {i:02d}: axis origin=({a.position.X:+.3f},{a.position.Y:+.3f},{a.position.Z:+.3f}), dir=({a.direction.X:+.0f},{a.direction.Y:+.0f},{a.direction.Z:+.0f})")
    elif f.geom_type == bd.GeomType.PLANE:
        n = f.normal_at()
        c = f.center()
        print(f"plane {i:02d}: normal=({n.X:+.0f},{n.Y:+.0f},{n.Z:+.0f}), center=({c.X:+.3f},{c.Y:+.3f},{c.Z:+.3f})")

# -- cell 5 -------------------------------------------------------------------------
# Each full generating cylinder retains an unsigned axis, but the trimmed face’s area centroid has a c
for i in [0, 20]:
    f = fluid.faces()[i]
    ga = f.geom_adaptor()
    print(i, type(ga), [n for n in dir(ga) if "phere" in n or "adius" in n])
    try:
        sph = ga.Sphere()
        print(" sphere radius", sph.Radius())
    except Exception as e:
        print(" sphere query failed", e)

# -- cell 6 -------------------------------------------------------------------------
# The spherical adaptor reports exactly 16 mm for the hollow wall pieces and 15 mm for the ball, so al
import sys, inspect
sys.path.insert(0, ".reference")
from cad_export import export_patches
print(inspect.signature(export_patches))
print(export_patches.__doc__)

# -- cell 7 -------------------------------------------------------------------------
# I’ll name signed ports `xPlus/xMinus`, `yPlus/yMinus`, and `zPlus/zMinus`. Cylinder-wall signs come 
import numpy as np

def signed_axis_name(v):
    a = np.array([v.X, v.Y, v.Z], dtype=float)
    j = int(np.argmax(np.abs(a)))
    return "xyz"[j] + ("Plus" if a[j] > 0 else "Minus")

sphere_faces = [f for f in fluid.faces() if f.geom_type == bd.GeomType.SPHERE]
ball_faces = [f for f in sphere_faces if abs(f.geom_adaptor().Radius() - BALL_R) < 1e-9]
hollow_faces = [f for f in sphere_faces if abs(f.geom_adaptor().Radius() - HOLLOW_R) < 1e-9]
cyl_faces = [f for f in fluid.faces() if f.geom_type == bd.GeomType.CYLINDER]
plane_faces = [f for f in fluid.faces() if f.geom_type == bd.GeomType.PLANE]

hole_walls = {}
for f in cyl_faces:
    axis = f.axis_of_rotation.direction
    c = f.center()
    av = np.array([axis.X, axis.Y, axis.Z])
    cv = np.array([c.X, c.Y, c.Z])
    signed = av * np.sign(np.dot(av, cv))
    hole_walls[signed_axis_name(bd.Vector(*signed))] = [f]

openings = {signed_axis_name(f.normal_at()): [f] for f in plane_faces}
patches = {"ball": ball_faces, "hollowWall": hollow_faces}
for side in ("xPlus", "xMinus", "yPlus", "yMinus", "zPlus", "zMinus"):
    patches["holeWall_" + side] = hole_walls[side]
    patches["opening_" + side] = openings[side]

print("Patch face counts:", {k: len(v) for k, v in patches.items()})
report = export_patches(fluid, patches, tolerance=2.5e-4, angular_tolerance=0.12)

# -- cell 8 -------------------------------------------------------------------------
# The conformal export is closed (0 open, flipped, or non-manifold edges) and contains all 14 requeste
from pathlib import Path
import json, subprocess, math
Path("system").mkdir(exist_ok=True)
Path("constant").mkdir(exist_ok=True)

control = '''FoamFile { format ascii; class dictionary; object controlDict; version 2.0; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
'''
block = '''FoamFile { format ascii; class dictionary; object blockMeshDict; version 2.0; }
convertToMeters 1;
vertices
(
 (-0.0225 -0.0225 -0.0225) (0.0225 -0.0225 -0.0225)
 (0.0225 0.0225 -0.0225) (-0.0225 0.0225 -0.0225)
 (-0.0225 -0.0225 0.0225) (0.0225 -0.0225 0.0225)
 (0.0225 0.0225 0.0225) (-0.0225 0.0225 0.0225)
);
blocks (hex (0 1 2 3 4 5 6 7) (18 18 18) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); });
mergePatchPairs ();
'''
manifest = json.loads(Path("constant/triSurface/patches.json").read_text())
stl_names = [p.stem for p in sorted(Path("constant/triSurface").glob("*.stl"))]
geom = "\n".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in stl_names)
refs = "\n".join(f'        {n} {{ level (3 3); patchInfo {{ type wall; }} }}' for n in stl_names)
q = 0.0155/math.sqrt(3)
snappy = f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; version 2.0; }}
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 500000; maxGlobalCells 1000000; minRefinementCells 0;
    nCellsBetweenLevels 2; resolveFeatureAngle 30;
    refinementSurfaces
    {{
{refs}
    }}
    refinementRegions {{}}
    locationInMesh ({q:.9f} {q:.9f} {q:.9f});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls
{{
    #include "meshQualityDict"
}}
mergeTolerance 1e-6;
'''
Path("system/controlDict").write_text(control)
Path("system/blockMeshDict").write_text(block)
Path("system/snappyHexMeshDict").write_text(snappy)
res = subprocess.run(["blockMesh"], text=True, capture_output=True)
print(res.stdout[-1800:])
print("blockMesh return code", res.returncode, "; locationInMesh", (q,q,q), "m")

# -- cell 9 -------------------------------------------------------------------------
# The background mesh is correct at 45 mm extent with 5,832 cells and 2.5 mm spacing. I’ll now run `sn
import subprocess, time
start=time.time()
res_snappy = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(res_snappy.stdout + "\nSTDERR\n" + res_snappy.stderr)
print(res_snappy.stdout[-5000:])
print("snappy return code", res_snappy.returncode, "elapsed", round(time.time()-start,1), "s")

# -- cell 10 ------------------------------------------------------------------------
# `snappyHexMesh` stopped during dictionary loading before any meshing stage, so this is a setup error
print(Path("snappyHexMesh.log").read_text()[-3500:])

# -- cell 11 ------------------------------------------------------------------------
# The failure is only because this otherwise empty case lacks the standard `fvSchemes` file that the s
fv_schemes='''FoamFile { format ascii; class dictionary; object fvSchemes; version 2.0; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
'''
fv_solution='''FoamFile { format ascii; class dictionary; object fvSolution; version 2.0; }
solvers
{
    cellDisplacement { solver GAMG; tolerance 1e-7; relTol 0; smoother GaussSeidel; cacheAgglomeration true; agglomerator faceAreaPair; nCellsInCoarsestLevel 10; mergeLevels 1; }
}
'''
quality='''FoamFile { format ascii; class dictionary; object meshQualityDict; version 2.0; }
maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1;
minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
relaxed { maxNonOrtho 75; }
'''
Path("system/fvSchemes").write_text(fv_schemes)
Path("system/fvSolution").write_text(fv_solution)
Path("system/meshQualityDict").write_text(quality)
start=time.time()
res_snappy = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(res_snappy.stdout + "\nSTDERR\n" + res_snappy.stderr)
print((res_snappy.stdout + res_snappy.stderr)[-5000:])
print("snappy return code", res_snappy.returncode, "elapsed", round(time.time()-start,1), "s")

# -- cell 12 ------------------------------------------------------------------------
# The geometry loaded successfully; OpenFOAM 2512 additionally requires an explicit (possibly empty) `
p=Path("system/snappyHexMeshDict")
txt=p.read_text().replace("    refinementSurfaces\n", "    features ();\n    refinementSurfaces\n")
p.write_text(txt)
start=time.time()
res_snappy = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(res_snappy.stdout + "\nSTDERR\n" + res_snappy.stderr)
print((res_snappy.stdout + res_snappy.stderr)[-5500:])
print("snappy return code", res_snappy.returncode, "elapsed", round(time.time()-start,1), "s")

# -- cell 13 ------------------------------------------------------------------------
# Snapping completed with 266,332 cells and no mesher quality errors. I’ll now run the binding bare `c
res_check = subprocess.run(["checkMesh"], text=True, capture_output=True)
Path("checkMesh.log").write_text(res_check.stdout + "\n" + res_check.stderr)
print((res_check.stdout + res_check.stderr)[-6000:])
print("checkMesh return code", res_check.returncode)
print("\nBoundary patch entries:")
print(Path("constant/polyMesh/boundary").read_text()[:5000])

# -- cell 14 ------------------------------------------------------------------------
# The binding check passes: one connected region, 266,332 cells, 14 non-empty named patches, maximum n
res_vtk=subprocess.run(["foamToVTK", "-constant", "-no-fields"], text=True, capture_output=True)
print("foamToVTK return", res_vtk.returncode, (res_vtk.stdout+res_vtk.stderr)[-1200:])
vtus=list(Path("VTK").rglob("*.vtu"))
print("VTU files:", [str(x) for x in vtus[:8]])
internal=[x for x in vtus if "internal" in x.name.lower()]
vmesh=pv.read(str(internal[0] if internal else vtus[0]))
# Remove the positive octant to expose the ball and shell while retaining three ports.
cut=vmesh.clip(normal=(1,0,0), origin=(0,0,0), invert=True)
cut=cut.clip(normal=(0,1,0), origin=(0,0,0), invert=True)
plotter=pv.Plotter(off_screen=True, window_size=(1000,760))
plotter.add_mesh(cut, color="lightskyblue", show_edges=True, edge_color="midnightblue", line_width=0.25)
plotter.add_axes()
plotter.view_isometric()
plotter.show(screenshot="mesh_cutaway.png")
print("Rendered mesh_cutaway.png; displayed cells", cut.n_cells)
