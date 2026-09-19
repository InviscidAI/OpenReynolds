"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first preview cell had a transcription error before execution, so I’m resending the self-contain
import build123d as bd
import numpy as np
import pyvista as pv
from pathlib import Path

D = 0.040
R = D / 2
UPSTREAM = 5 * D
DOWNSTREAM = 15 * D
SIDE = 5 * D
XMIN, XMAX = -UPSTREAM, DOWNSTREAM
YMIN, YMAX = -SIDE, SIDE
ZMIN, ZMAX = -SIDE, SIDE

sphere_solid = bd.Sphere(R)
domain_solid = bd.Pos(XMIN, YMIN, ZMIN) * bd.Box(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
fluid = domain_solid - sphere_solid

bb = fluid.bounding_box()
expected_volume = (XMAX-XMIN)*(YMAX-YMIN)*(ZMAX-ZMIN) - 4*np.pi*R**3/3
print(f"Sphere diameter measured: {2*R:.6f} m; requested: {D:.6f} m")
print(f"Domain X extents: {bb.min.X:.3f}, {bb.max.X:.3f} m; requested: {XMIN:.3f}, {XMAX:.3f} m")
print(f"Domain Y extents: {bb.min.Y:.3f}, {bb.max.Y:.3f} m; requested: {YMIN:.3f}, {YMAX:.3f} m")
print(f"Domain Z extents: {bb.min.Z:.3f}, {bb.max.Z:.3f} m; requested: {ZMIN:.3f}, {ZMAX:.3f} m")
print(f"Fluid volume: {fluid.volume:.9f} m^3; analytic: {expected_volume:.9f} m^3")

bd.export_stl(fluid, "fluid_preview.stl", tolerance=8e-4, angular_tolerance=0.15)
mesh_preview = pv.read("fluid_preview.stl")
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(mesh_preview, color="lightsteelblue", opacity=0.35, show_edges=True)
pl.add_axes()
pl.camera_position = [(0.65,-0.65,0.45),(0.10,0,0),(0,0,1)]
pl.show(screenshot="coarse_geometry_preview.png")
print(f"Preview surface: {mesh_preview.n_cells} triangles; image coarse_geometry_preview.png")

# -- cell 2 -------------------------------------------------------------------------
# The preview exposed a placement issue: `Box` is center-aligned by default, so translating by the min
DOMAIN_CENTER = ((XMIN+XMAX)/2, (YMIN+YMAX)/2, (ZMIN+ZMAX)/2)
domain_solid = bd.Pos(*DOMAIN_CENTER) * bd.Box(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
fluid = domain_solid - sphere_solid
bb = fluid.bounding_box()
print(f"Corrected domain center: {DOMAIN_CENTER}")
print(f"Measured extents X [{bb.min.X:.3f}, {bb.max.X:.3f}], Y [{bb.min.Y:.3f}, {bb.max.Y:.3f}], Z [{bb.min.Z:.3f}, {bb.max.Z:.3f}] m")
print(f"Fluid volume: {fluid.volume:.9f} m^3; analytic: {expected_volume:.9f} m^3; error {(fluid.volume-expected_volume)/expected_volume:.2e}")
print(f"Boundary face count: {len(fluid.faces())}")
for i, face in enumerate(fluid.faces()):
    c = face.center()
    n = face.normal_at(c)
    print(f"face {i}: area={face.area:.8f}, center=({c.X:.4f},{c.Y:.4f},{c.Z:.4f}), normal=({n.X:.2f},{n.Y:.2f},{n.Z:.2f})")

bd.export_stl(fluid, "fluid_preview.stl", tolerance=5e-4, angular_tolerance=0.12)
mesh_preview = pv.read("fluid_preview.stl")
pl = pv.Plotter(off_screen=True, window_size=(900,650))
pl.add_mesh(mesh_preview, color="lightsteelblue", opacity=0.22, show_edges=False)
pl.add_mesh(pv.Sphere(radius=R, center=(0,0,0), theta_resolution=48, phi_resolution=48), color="crimson")
pl.add_axes()
pl.camera_position = [(0.70,-0.65,0.48),(0.12,0,0),(0,0,1)]
pl.show(screenshot="coarse_geometry_preview_corrected.png")
print(f"Corrected preview: {mesh_preview.n_cells} triangles; image coarse_geometry_preview_corrected.png")

# -- cell 3 -------------------------------------------------------------------------
# The corrected geometry now matches all requested extents and the analytic fluid volume. I’ll assign 
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches

boundary_faces = list(fluid.faces())
inlet = [f for f in boundary_faces if f.geom_type == bd.GeomType.PLANE and f.normal_at(f.center()).X < -0.9]
outlet = [f for f in boundary_faces if f.geom_type == bd.GeomType.PLANE and f.normal_at(f.center()).X > 0.9]
sphere_patch = [f for f in boundary_faces if f.geom_type != bd.GeomType.PLANE]
far_field = [f for f in boundary_faces if f not in inlet + outlet + sphere_patch]
print(f"Patch face counts: inlet={len(inlet)}, outlet={len(outlet)}, far_field={len(far_field)}, sphere={len(sphere_patch)}")
export_patches(fluid, {"inlet": inlet, "outlet": outlet, "far_field": far_field, "sphere": sphere_patch}, tolerance=2.5e-4)

# -- cell 4 -------------------------------------------------------------------------
# For the first mesh I’m choosing a 20 mm isotropic background grid (40×20×20 cells), surface refineme
from pathlib import Path
Path("system").mkdir(exist_ok=True)
Path("constant").mkdir(exist_ok=True)
Path("0").mkdir(exist_ok=True)

block_mesh_dict = r'''FoamFile
{
    format ascii;
    class dictionary;
    object blockMeshDict;
}
convertToMeters 1;
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
blocks
(
    hex (0 1 2 3 4 5 6 7) (40 20 20) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet
    {
        type patch;
        faces ((0 4 7 3));
    }
    outlet
    {
        type patch;
        faces ((1 2 6 5));
    }
    far_field
    {
        type patch;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
mergePatchPairs ();
'''
Path("system/blockMeshDict").write_text(block_mesh_dict)

snappy_dict = r'''FoamFile
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
    maxLocalCells 500000;
    maxGlobalCells 1000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    {
        sphere
        {
            level (3 4);
            patchInfo { type wall; }
        }
    }
    resolveFeatureAngle 30;
    refinementRegions
    {
        sphere
        {
            mode distance;
            levels ((0.040 2));
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
    nRelaxIter 8;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    layers
    {
        sphere { nSurfaceLayers 5; }
    }
    expansionRatio 1.20;
    finalLayerThickness 0.35;
    minThickness 0.08;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 60;
}
meshQualityControls
{
    #include "meshQualityDict"
    relaxed
    {
        maxNonOrtho 75;
    }
}
mergeTolerance 1e-6;
'''
Path("system/snappyHexMeshDict").write_text(snappy_dict)
mesh_quality = r'''maxNonOrtho 65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-16;
minTetQuality -1e30;
minArea -1;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.02;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;
'''
Path("system/meshQualityDict").write_text(mesh_quality)
print("Wrote blockMeshDict: 40 x 20 x 20 = 16000 background cells")
print("Wrote snappyHexMeshDict: sphere levels 3-4 and 5 prism layers")

# -- cell 5 -------------------------------------------------------------------------
# OpenFOAM requires a minimal `controlDict` even for meshing utilities. I’ll add that case metadata, t
control_dict = r'''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application blockMesh;
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
'''
Path("system/controlDict").write_text(control_dict)
print("Wrote minimal system/controlDict for mesh utilities")

# -- cell 6 -------------------------------------------------------------------------
# The diagnostic shows this OpenFOAM build loads `fvSchemes` during `checkMesh`. I’ll add minimal vali
import subprocess
fv_schemes = r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''
fv_solution = r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
'''
Path("system/fvSchemes").write_text(fv_schemes)
Path("system/fvSolution").write_text(fv_solution)
print("Wrote minimal fvSchemes and fvSolution required by checkMesh")

# -- cell 7 -------------------------------------------------------------------------
# I’m checking the background mesh again now that the required dictionaries are present.
cm0 = subprocess.run(["checkMesh"], text=True, capture_output=True)
print(cm0.stdout[-2600:])
if cm0.returncode != 0:
    print(cm0.stderr)
cm0.check_returncode()

# -- cell 8 -------------------------------------------------------------------------
# The background mesh is clean. I’ll now cut and snap it to the sphere and extrude the five requested 
sn = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(sn.stdout + "\n" + sn.stderr)
print(sn.stdout[-6500:])
if sn.returncode != 0:
    print(sn.stderr[-2500:])
sn.check_returncode()

# -- cell 9 -------------------------------------------------------------------------
# Layer generation succeeded on 100% of the 744 sphere faces, producing five layers with a 3.14 mm tot
cm = subprocess.run(["checkMesh"], text=True, capture_output=True)
Path("checkMesh.log").write_text(cm.stdout + "\n" + cm.stderr)
print(cm.stdout[-5200:])
if cm.returncode != 0:
    print(cm.stderr)
cm.check_returncode()
print("\nBoundary file patch names/sizes:")
boundary_text = Path("constant/polyMesh/boundary").read_text()
print(boundary_text[:5000])

# -- cell 10 ------------------------------------------------------------------------
# The final mesh passes `checkMesh` with 32,167 cells, four correctly named patches, maximum non-ortho
Path("t2.foam").touch()
reader = pv.OpenFOAMReader("t2.foam")
reader.set_active_time_value(reader.time_values[-1])
mesh_data = reader.read()
internal = mesh_data["internalMesh"]
center_slice = internal.slice(normal=(0,0,1), origin=(0,0,0))
sphere_surface = mesh_data["boundary"]["sphere"]
print(f"Rendered internal cells: {internal.n_cells}; center slice cells: {center_slice.n_cells}; sphere patch faces: {sphere_surface.n_cells}")
pl = pv.Plotter(off_screen=True, window_size=(1100,700))
pl.add_mesh(center_slice, color="aliceblue", show_edges=True, edge_color="slategray", line_width=0.35)
pl.add_mesh(sphere_surface, color="crimson", show_edges=True, edge_color="darkred")
pl.add_axes()
pl.view_xy()
pl.camera.zoom(1.35)
pl.show(screenshot="final_mesh_center_slice.png")
print("Saved final_mesh_center_slice.png")

# -- cell 11 ------------------------------------------------------------------------
# The full-domain slice confirms the refinement region and unobstructed external domain. I’ll make a c
sphere_pts = np.asarray(sphere_surface.points)
radii = np.linalg.norm(sphere_pts, axis=1)
sp_bounds = sphere_surface.bounds
print(f"Sphere mesh radial range: {radii.min():.8f} to {radii.max():.8f} m; target radius {R:.8f} m")
print(f"Sphere patch extents: X [{sp_bounds[0]:.6f},{sp_bounds[1]:.6f}], Y [{sp_bounds[2]:.6f},{sp_bounds[3]:.6f}], Z [{sp_bounds[4]:.6f},{sp_bounds[5]:.6f}] m")
close_slice = center_slice.clip_box(bounds=(-0.065,0.065,-0.065,0.065,-0.002,0.002), invert=False)
pl2 = pv.Plotter(off_screen=True, window_size=(800,800))
pl2.add_mesh(close_slice, color="aliceblue", show_edges=True, edge_color="black", line_width=0.7)
pl2.add_mesh(sphere_surface, color="crimson", show_edges=True, edge_color="darkred")
pl2.view_xy()
pl2.add_axes()
pl2.show(screenshot="sphere_layers_closeup.png")
print(f"Close-up cells: {close_slice.n_cells}; saved sphere_layers_closeup.png")

# -- cell 12 ------------------------------------------------------------------------
# The close-up confirms the five-layer ring around the sphere and the snapped radius is within 25 µm o
bm_final = subprocess.run(["blockMesh"], text=True, capture_output=True)
print(bm_final.stdout[-1800:])
if bm_final.returncode != 0:
    print(bm_final.stderr)
bm_final.check_returncode()

# -- cell 13 ------------------------------------------------------------------------
# The reproducible background-mesh step is now recorded. I’ll regenerate the snapped and layered final
sn_final = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("snappyHexMesh.log").write_text(sn_final.stdout + "\n" + sn_final.stderr)
print(sn_final.stdout[-3200:])
if sn_final.returncode != 0:
    print(sn_final.stderr[-2000:])
sn_final.check_returncode()

# -- cell 14 ------------------------------------------------------------------------
# The regenerated mesh again has complete five-layer coverage. I’ll run one final binding `checkMesh` 
cm_final = subprocess.run(["checkMesh"], text=True, capture_output=True)
Path("checkMesh.log").write_text(cm_final.stdout + "\n" + cm_final.stderr)
print(cm_final.stdout[-4200:])
if cm_final.returncode != 0:
    print(cm_final.stderr)
cm_final.check_returncode()

# -- cell 15 ------------------------------------------------------------------------
# The completion gate requires the external-fluid seed point to be recorded in the surface manifest as
export_patches(fluid, {"inlet": inlet, "outlet": outlet, "far_field": far_field, "sphere": sphere_patch}, tolerance=2.5e-4, location_in_mesh=(-0.10, 0.0, 0.0))
