"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid region as a rectangular box minus a 40 mm sphere centered at the origin. The st
import build123d as bd
import pyvista as pv
from pathlib import Path

D = 0.040
R = D / 2
XMIN, XMAX = -5*D, 15*D
YMIN, YMAX = -5*D, 5*D
ZMIN, ZMAX = -5*D, 5*D

outer_box = bd.Pos(XMIN, YMIN, ZMIN) * bd.Box(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
sphere_solid = bd.Sphere(R)
fluid = outer_box - sphere_solid

print(f"Sphere diameter requested = {D:.6f} m; CAD measured = {2*sphere_solid.bounding_box().size.X:.6f} m")
print(f"Domain x requested/measured = [{XMIN:.3f}, {XMAX:.3f}] m / [{fluid.bounding_box().min.X:.3f}, {fluid.bounding_box().max.X:.3f}] m")
print(f"Domain y,z requested = [{YMIN:.3f}, {YMAX:.3f}] m")
print(f"CAD fluid volume = {fluid.volume:.9f} m^3; expected = {(XMAX-XMIN)*(YMAX-YMIN)*(ZMAX-ZMIN)-4/3*3.141592653589793*R**3:.9f} m^3")

bd.export_stl(sphere_solid, "sphere_preview.stl", tolerance=2.5e-4, angular_tolerance=0.15)
mesh = pv.read("sphere_preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
plotter.add_mesh(mesh, color="lightgray", smooth_shading=True)
plotter.add_mesh(pv.Box(bounds=(XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX)).extract_all_edges(), color="navy", line_width=2)
plotter.add_axes(); plotter.view_isometric(); plotter.camera.zoom(1.15)
plotter.show(screenshot="geometry_preview.png")
print("Wrote geometry_preview.png")

# -- cell 2 -------------------------------------------------------------------------
# The first inspection exposed two measurement/construction mistakes rather than a valid geometry: I d
import inspect
print("Box", inspect.signature(bd.Box))
print("Sphere", inspect.signature(bd.Sphere))
print("Pos", inspect.signature(bd.Pos))
print("outer bbox", outer_box.bounding_box())
print("sphere bbox", sphere_solid.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
# `Box` is centered by default, so the correct box translation is its center at \(x=0.20\) m, not its 
BOX_CENTER_X = (XMIN + XMAX) / 2
outer_box = bd.Pos(BOX_CENTER_X, 0, 0) * bd.Box(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
sphere_solid = bd.Sphere(R)
fluid = outer_box - sphere_solid

bb_s = sphere_solid.bounding_box(); bb_f = fluid.bounding_box()
measured_D = bb_s.size.X
expected_volume = (XMAX-XMIN)*(YMAX-YMIN)*(ZMAX-ZMIN) - 4/3*3.141592653589793*R**3
print(f"Sphere diameter requested/measured: {D:.6f} / {measured_D:.6f} m")
print(f"x extent requested/measured: [{XMIN:.3f},{XMAX:.3f}] / [{bb_f.min.X:.3f},{bb_f.max.X:.3f}] m")
print(f"y extent requested/measured: [{YMIN:.3f},{YMAX:.3f}] / [{bb_f.min.Y:.3f},{bb_f.max.Y:.3f}] m")
print(f"z extent requested/measured: [{ZMIN:.3f},{ZMAX:.3f}] / [{bb_f.min.Z:.3f},{bb_f.max.Z:.3f}] m")
print(f"fluid volume expected/measured: {expected_volume:.9f} / {fluid.volume:.9f} m^3")
print("fluid faces:", len(fluid.faces()), "solids:", len(fluid.solids()))

# -- cell 4 -------------------------------------------------------------------------
# The corrected dimensions and volume now match exactly. I’m replacing the preview with a view of the 
bd.export_stl(sphere_solid, "sphere_preview.stl", tolerance=2.5e-4, angular_tolerance=0.15)
mesh = pv.read("sphere_preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
plotter.add_mesh(mesh, color="silver", smooth_shading=True)
plotter.add_mesh(pv.Box(bounds=(XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX)).extract_all_edges(), color="navy", line_width=2)
plotter.add_point_labels([[XMIN,0,0],[XMAX,0,0]], ["inlet (-5D)","outlet (+15D)"], point_size=5, font_size=14)
plotter.add_axes(); plotter.view_isometric(); plotter.camera.zoom(1.15)
plotter.show(screenshot="geometry_preview.png")
print("Updated geometry_preview.png with corrected extents")

# -- cell 5 -------------------------------------------------------------------------
# Next I’m identifying the seven final fluid faces by their actual surface geometry and outward normal
for i, face in enumerate(fluid.faces()):
    c = face.center()
    try:
        n = face.normal_at()
    except Exception:
        n = None
    print(i, "type=", face.geom_type, "area=", round(face.area,8), "center=", tuple(round(v,5) for v in c), "normal=", None if n is None else tuple(round(v,3) for v in n))

# -- cell 6 -------------------------------------------------------------------------
# The face inventory is exactly as expected: six planes and one spherical cavity, with the sphere area
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches

plane_faces = [f for f in fluid.faces() if f.geom_type == bd.GeomType.PLANE]
sphere_faces = [f for f in fluid.faces() if f.geom_type == bd.GeomType.SPHERE]
inlet_faces = [f for f in plane_faces if f.normal_at().X < -0.9]
outlet_faces = [f for f in plane_faces if f.normal_at().X > 0.9]
farfield_faces = [f for f in plane_faces if abs(f.normal_at().X) < 0.1]
assert (len(inlet_faces), len(outlet_faces), len(farfield_faces), len(sphere_faces)) == (1,1,4,1)
export_patches(fluid, {"inlet": inlet_faces, "outlet": outlet_faces, "farField": farfield_faces, "sphere": sphere_faces}, tolerance=2.5e-5)
print("Named face areas [m^2]:", {"inlet":sum(f.area for f in inlet_faces), "outlet":sum(f.area for f in outlet_faces), "farField":sum(f.area for f in farfield_faces), "sphere":sum(f.area for f in sphere_faces)})

# -- cell 7 -------------------------------------------------------------------------
# `blockMesh` correctly refused to run because an OpenFOAM case also requires `system/controlDict`. I’
import os, subprocess
from pathlib import Path
os.makedirs("system", exist_ok=True)
Path("system/controlDict").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application snappyHexMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
runTimeModifiable true;
''')
Path("system/blockMeshDict").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 1;
vertices
(
    (-0.2 -0.2 -0.2) (0.6 -0.2 -0.2) (0.6 0.2 -0.2) (-0.2 0.2 -0.2)
    (-0.2 -0.2  0.2) (0.6 -0.2  0.2) (0.6 0.2  0.2) (-0.2 0.2  0.2)
);
blocks (hex (0 1 2 3 4 5 6 7) (40 20 20) simpleGrading (1 1 1));
edges ();
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    farField
    {
        type patch;
        faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7));
    }
);
mergePatchPairs ();
''')
result = subprocess.run(["blockMesh"], text=True, capture_output=True)
print(result.stdout[-2200:])
if result.returncode: print(result.stderr)
result.check_returncode()

# -- cell 8 -------------------------------------------------------------------------
# The coarse box mesh is valid at 16,000 cells. For the first fitted mesh I’ll refine the sphere surfa
Path("system/snappyHexMeshDict").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}
castellatedMesh true;
snap true;
addLayers true;
geometry
{
    sphere.stl
    {
        type triSurfaceMesh;
        name sphere;
    }
}
castellatedMeshControls
{
    maxLocalCells 300000;
    maxGlobalCells 500000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        sphere
        {
            level (2 2);
            patchInfo { type wall; }
        }
    }
    resolveFeatureAngle 30;
    refinementRegions
    {
        sphere
        {
            mode distance;
            levels ((0.04 2) (0.08 1));
        }
    }
    locationInMesh (-0.10 0 0);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 5;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    layers
    {
        sphere { nSurfaceLayers 4; }
    }
    expansionRatio 1.2;
    finalLayerThickness 0.25;
    minThickness 0.05;
    nGrow 0;
    featureAngle 120;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}
meshQualityControls
{
    #includeEtc "caseDicts/meshQualityDict"
    relaxed { maxNonOrtho 75; }
}
mergeTolerance 1e-6;
''')
print(Path("system/snappyHexMeshDict").read_text())

# -- cell 9 -------------------------------------------------------------------------
# The mesher reached the snap phase and exposed a version-specific required quality-control entry (`er
Path("system/fvSchemes").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path("system/fvSolution").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
sdict = Path("system/snappyHexMeshDict").read_text()
sdict = sdict.replace('    relaxed { maxNonOrtho 75; }', '    nSmoothScale 4;\n    errorReduction 0.75;\n    relaxed { maxNonOrtho 75; }')
Path("system/snappyHexMeshDict").write_text(sdict)
subprocess.run(["blockMesh"], check=True, stdout=subprocess.DEVNULL)
snappy = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(snappy.stdout + "\n" + snappy.stderr)
print(snappy.stdout[-8000:])
if snappy.returncode: print(snappy.stderr)
snappy.check_returncode()

# -- cell 10 ------------------------------------------------------------------------
# The coarse fitted mesh completed with 26,768 cells. All 200 sphere surface faces received all four r
check = subprocess.run(["checkMesh"], text=True, capture_output=True)
Path("checkMesh.log").write_text(check.stdout + "\n" + check.stderr)
print(check.stdout[-6000:])
if check.returncode: print(check.stderr)
check.check_returncode()

# -- cell 11 ------------------------------------------------------------------------
# `checkMesh` passes: one connected region, all four named patches present, maximum non-orthogonality 
Path("t2.foam").touch()
reader = pv.OpenFOAMReader("t2.foam")
reader.set_active_time_value(reader.time_values[-1])
foam_data = reader.read()
internal = foam_data["internalMesh"]
cut = internal.slice(normal=(0,0,1), origin=(0,0,0))
plotter = pv.Plotter(off_screen=True, window_size=(1100,650))
plotter.add_mesh(cut, show_edges=True, color="white", edge_color="steelblue", line_width=0.6)
plotter.add_axes(); plotter.view_xy(); plotter.camera.tight(padding=0.03)
plotter.show(screenshot="mesh_centerplane.png")
print("Center-plane slice:", cut.n_cells, "cells; wrote mesh_centerplane.png")

# -- cell 12 ------------------------------------------------------------------------
# The coarse center plane confirms the correct external region, smooth local refinement, and a complet
sdict = Path("system/snappyHexMeshDict").read_text()
sdict = sdict.replace('level (2 2);', 'level (3 3);')
sdict = sdict.replace('levels ((0.04 2) (0.08 1));', 'levels ((0.04 3) (0.08 2) (0.12 1));')
sdict = sdict.replace('nSurfaceLayers 4;', 'nSurfaceLayers 5;')
sdict = sdict.replace('finalLayerThickness 0.25;', 'finalLayerThickness 0.20;')
Path("system/snappyHexMeshDict").write_text(sdict)
subprocess.run(["blockMesh"], check=True, stdout=subprocess.DEVNULL)
snappy_final = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(snappy_final.stdout + "\n" + snappy_final.stderr)
print(snappy_final.stdout[-6500:])
if snappy_final.returncode: print(snappy_final.stderr)
snappy_final.check_returncode()

# -- cell 13 ------------------------------------------------------------------------
# The final mesh has 107,872 cells. All 744 sphere faces received all five layers, with 0.241 mm avera
check_final = subprocess.run(["checkMesh"], text=True, capture_output=True)
Path("checkMesh.log").write_text(check_final.stdout + "\n" + check_final.stderr)
print(check_final.stdout[-5200:])
if check_final.returncode: print(check_final.stderr)
check_final.check_returncode()
Path("t2.foam").touch()
reader = pv.OpenFOAMReader("t2.foam")
reader.set_active_time_value(reader.time_values[-1])
foam_data = reader.read()
internal = foam_data["internalMesh"]
print("Final point bounds:", internal.bounds)
print("Available mesh blocks:", list(foam_data.keys()))

# -- cell 14 ------------------------------------------------------------------------
# The final mesh passes `checkMesh` with maximum non-orthogonality 34.26°, maximum skewness 0.560, and
boundary = foam_data["boundary"]
print("Boundary blocks:", list(boundary.keys()))
sphere_patch = boundary["sphere"]
sp = sphere_patch.points
radii = (sp[:,0]**2 + sp[:,1]**2 + sp[:,2]**2)**0.5
print(f"Sphere patch points={sphere_patch.n_points}, faces={sphere_patch.n_cells}")
print(f"Sphere radius min/mean/max = {radii.min():.7f} / {radii.mean():.7f} / {radii.max():.7f} m")
print(f"Fitted diameter from extrema x/y/z = {(sp.max(axis=0)-sp.min(axis=0))}")
cut = internal.slice(normal=(0,0,1), origin=(0,0,0))
plotter = pv.Plotter(off_screen=True, window_size=(1100,650))
plotter.add_mesh(cut, show_edges=True, color="white", edge_color="royalblue", line_width=0.45)
plotter.add_axes(); plotter.view_xy(); plotter.camera.tight(padding=0.03)
plotter.show(screenshot="mesh_centerplane_final.png")
print("Wrote mesh_centerplane_final.png")

# -- cell 15 ------------------------------------------------------------------------
# The mesh artifacts are already present; `build.py` is assembled from accepted cells by the desk fram
for p in ["system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution", "constant/polyMesh/boundary", "constant/triSurface/patches.json", "constant/triSurface/sphere.stl", "checkMesh.log", "snappyHexMesh.log", "mesh_centerplane_final.png"]:
    q = Path(p)
    assert q.exists(), p
    print(f"{p}: {q.stat().st_size} bytes")

# -- cell 16 ------------------------------------------------------------------------
# The completion gate requires the external-region seed point to be recorded in the surface manifest, 
import inspect
print(inspect.signature(export_patches))

# -- cell 17 ------------------------------------------------------------------------
# The exporter supports this directly. I’m regenerating the conformal patch set unchanged, adding only
export_patches(fluid, {"inlet": inlet_faces, "outlet": outlet_faces, "farField": farfield_faces, "sphere": sphere_faces}, tolerance=2.5e-5, location_in_mesh=(-0.10, 0.0, 0.0))
