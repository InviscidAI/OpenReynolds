"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the air volume as a rectangular box minus a 40 mm sphere centered at the origin. The box 
import build123d as bd
import numpy as np
import matplotlib.pyplot as plt

# All dimensions are in metres.
D = 0.040
R = D / 2
XMIN, XMAX = -5*D, 15*D
YMIN, YMAX = -5*D, 5*D
ZMIN, ZMAX = -5*D, 5*D

domain_box = bd.Pos(XMIN, YMIN, ZMIN) * bd.Box(XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN)
sphere_solid = bd.Sphere(R)
fluid = domain_box - sphere_solid

bb = fluid.bounding_box()
print(f"Sphere diameter: measured {2*R:.6f} m; requested {D:.6f} m")
print(f"Upstream extent: measured {-XMIN/D:.1f} D; requested 5.0 D")
print(f"Downstream extent: measured {XMAX/D:.1f} D; requested 15.0 D")
print(f"Side/vertical extents: measured {YMAX/D:.1f} D; requested 5.0 D")
print(f"Fluid bounds: ({bb.min.X:.3f},{bb.min.Y:.3f},{bb.min.Z:.3f}) to ({bb.max.X:.3f},{bb.max.Y:.3f},{bb.max.Z:.3f}) m")
print(f"Fluid volume: {fluid.volume:.9f} m^3")

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')
# Domain wireframe
for y in (YMIN,YMAX):
    for z in (ZMIN,ZMAX): ax.plot([XMIN,XMAX],[y,y],[z,z],color='steelblue',lw=0.8)
for x in (XMIN,XMAX):
    for z in (ZMIN,ZMAX): ax.plot([x,x],[YMIN,YMAX],[z,z],color='steelblue',lw=0.8)
for x in (XMIN,XMAX):
    for y in (YMIN,YMAX): ax.plot([x,x],[y,y],[ZMIN,ZMAX],color='steelblue',lw=0.8)
u=np.linspace(0,2*np.pi,48); v=np.linspace(0,np.pi,24)
xs=R*np.outer(np.cos(u),np.sin(v)); ys=R*np.outer(np.sin(u),np.sin(v)); zs=R*np.outer(np.ones_like(u),np.cos(v))
ax.plot_surface(xs,ys,zs,color='crimson',alpha=0.9,linewidth=0)
ax.set_box_aspect((XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN)); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]'); ax.set_title('External-flow domain and 40 mm sphere')
plt.tight_layout(); plt.savefig('geometry_preview.png',dpi=160); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The preview showed the intended dimensions, but the CAD bounding-box measurement exposed a placement
# Correct the CAD primitive: place its declared corner at the domain minimum.
domain_box = bd.Pos(XMIN, YMIN, ZMIN) * bd.Box(
    XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN,
    align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN)
)
fluid = domain_box - sphere_solid
bb = fluid.bounding_box()
print(f"Corrected CAD bounds: ({bb.min.X:.3f},{bb.min.Y:.3f},{bb.min.Z:.3f}) to ({bb.max.X:.3f},{bb.max.Y:.3f},{bb.max.Z:.3f}) m")
print(f"Target CAD bounds:    ({XMIN:.3f},{YMIN:.3f},{ZMIN:.3f}) to ({XMAX:.3f},{YMAX:.3f},{ZMAX:.3f}) m")
print(f"Sphere diameter: measured {2*R:.6f} m; requested {D:.6f} m")
print(f"Fluid volume: measured {fluid.volume:.9f} m^3; analytic {(XMAX-XMIN)*(YMAX-YMIN)*(ZMAX-ZMIN)-4*np.pi*R**3/3:.9f} m^3")

# -- cell 3 -------------------------------------------------------------------------
# The corrected CAD now matches the requested extents and analytic volume. Next I’m inspecting the exa
faces = list(fluid.faces())
print(f"Fluid has {len(faces)} boundary faces")
for i, face in enumerate(faces):
    c = face.center()
    try:
        n = face.normal_at()
        ntext = f"({n.X:+.1f},{n.Y:+.1f},{n.Z:+.1f})"
    except Exception:
        ntext = "curved"
    print(f"face {i}: area={face.area:.9f} m^2, center=({c.X:+.4f},{c.Y:+.4f},{c.Z:+.4f}), normal={ntext}")

# -- cell 4 -------------------------------------------------------------------------
# The subtraction produced exactly seven faces: inlet face 0, four far-field faces 1–4, outlet face 5,
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches

inlet_faces = [faces[0]]
farfield_faces = [faces[1], faces[2], faces[3], faces[4]]
outlet_faces = [faces[5]]
sphere_faces = [faces[6]]
export_patches(
    fluid,
    {"inlet": inlet_faces, "outlet": outlet_faces, "farField": farfield_faces, "sphere": sphere_faces},
    tolerance=5.0e-4,
)
print("Named CAD boundary patches exported in metres.")

# -- cell 5 -------------------------------------------------------------------------
# I’ll use a 20 mm base Cartesian spacing (40×20×20 cells) for the coarse mesh, with three local refin
from pathlib import Path
Path("system").mkdir(exist_ok=True)
Path("constant").mkdir(exist_ok=True)

block_dict = r'''FoamFile
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
    farField
    {
        type patch;
        faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7));
    }
);
mergePatchPairs ();
'''
Path("system/blockMeshDict").write_text(block_dict)

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
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    {
        sphere
        {
            level (2 3);
            patchInfo { type wall; }
        }
    }
    resolveFeatureAngle 30;
    refinementRegions {};
    locationInMesh (-0.10 0 0);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
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
    expansionRatio 1.2;
    finalLayerThickness 0.30;
    minThickness 0.05;
    nGrow 0;
    featureAngle 130;
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
    maxNonOrtho 70;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-15;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed { maxNonOrtho 75; }
}
debug 0;
mergeTolerance 1e-6;
'''
Path("system/snappyHexMeshDict").write_text(snappy_dict)
print("Wrote coarse blockMesh and sphere refinement/layer dictionaries.")

# -- cell 6 -------------------------------------------------------------------------
# `blockMesh` did not start because OpenFOAM requires a minimal `system/controlDict` even for meshing 
control_dict = r'''FoamFile
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
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
'''
Path("system/controlDict").write_text(control_dict)
print("Wrote minimal meshing-only system/controlDict.")

# -- cell 7 -------------------------------------------------------------------------
# The background mesh itself succeeded, but that cell was rejected from the reproducible script becaus
import subprocess
bm = subprocess.run(["blockMesh"], text=True, capture_output=True, check=True)
print(bm.stdout[-3000:])

# -- cell 8 -------------------------------------------------------------------------
# `snappyHexMesh` stopped immediately while reading case input, before altering the background mesh. I
print(Path("log.snappyHexMesh").read_text()[-5000:])

# -- cell 9 -------------------------------------------------------------------------
# This OpenFOAM build also requires placeholder `fvSchemes` and `fvSolution` dictionaries when constru
Path("system/fvSchemes").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
''')
Path("system/fvSolution").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
''')
print("Wrote required empty meshing dictionaries fvSchemes and fvSolution.")

# -- cell 10 ------------------------------------------------------------------------
# The utility requires the standard scheme subdictionaries even though it does not solve equations. I’
Path("system/fvSchemes").write_text(r'''FoamFile
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
wallDist { method meshWave; }
''')
Path("system/fvSolution").write_text(r'''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
print("Completed the standard scheme subdictionaries required by snappyHexMesh.")

# -- cell 11 ------------------------------------------------------------------------
# I’m retrying `snappyHexMesh` with the now-complete mesh-initialization dictionaries.
shm = subprocess.run(["snappyHexMesh", "-overwrite"], text=True, capture_output=True)
Path("log.snappyHexMesh").write_text(shm.stdout + "\n" + shm.stderr)
print(shm.stdout[-7000:])
if shm.returncode != 0:
    print(shm.stderr[-2000:])
    raise RuntimeError("snappyHexMesh failed; see log.snappyHexMesh")

# -- cell 12 ------------------------------------------------------------------------
# The coarse final mesh has 26,000 cells. All 744 sphere faces received all five requested layers, wit
vtk_run = subprocess.run(["foamToVTK", "-constant", "-no-fields"], text=True, capture_output=True, check=True)
print(vtk_run.stdout[-1500:])
print("VTK outputs:")
for p in sorted(Path("VTK").rglob("*")):
    if p.is_file(): print(" ", p)

# -- cell 13 ------------------------------------------------------------------------
# The converted mesh contains exactly the four requested boundary patches. I’ll render both the full m
import pyvista as pv
pv.OFF_SCREEN = True
mesh_vtk = pv.read("VTK/t2_0/internal.vtu")
mid = mesh_vtk.slice(normal=(0,0,1), origin=(0,0,0))
sphere_vtk = pv.read("VTK/t2_0/boundary/sphere.vtp")
pl = pv.Plotter(shape=(1,2), off_screen=True, window_size=(1500,650))
pl.subplot(0,0); pl.add_mesh(mid, color="white", show_edges=True, edge_color="gray", line_width=0.4); pl.add_mesh(sphere_vtk, color="crimson"); pl.view_xy(); pl.add_text("Full z=0 mesh cut", font_size=12)
pl.subplot(0,1); pl.add_mesh(mid, color="white", show_edges=True, edge_color="black", line_width=0.7); pl.add_mesh(sphere_vtk, color="crimson"); pl.view_xy(); pl.camera.SetParallelScale(0.045); pl.camera.position=(0,0,1); pl.camera.focal_point=(0,0,0); pl.add_text("Sphere refinement and 5 prism layers", font_size=12)
pl.link_views(); pl.subplot(0,1); pl.camera.SetParallelScale(0.045); pl.screenshot("mesh_preview.png"); pl.show()

# -- cell 14 ------------------------------------------------------------------------
# The full cut confirms the sphere is centered and refinement remains localized, but the linked camera
sb = sphere_vtk.bounds
mesh_diameters = (sb[1]-sb[0], sb[3]-sb[2], sb[5]-sb[4])
sphere_area_mesh = sphere_vtk.area
print(f"Meshed sphere diameters x/y/z: {mesh_diameters[0]:.6f}, {mesh_diameters[1]:.6f}, {mesh_diameters[2]:.6f} m; requested {D:.6f} m")
print(f"Meshed sphere area: {sphere_area_mesh:.9f} m^2; analytic {4*np.pi*R**2:.9f} m^2; error {(sphere_area_mesh/(4*np.pi*R**2)-1)*100:.3f}%")
print(f"Sphere boundary faces: {sphere_vtk.n_cells}")

# -- cell 15 ------------------------------------------------------------------------
# The snapped sphere measures 39.976×39.963×40.000 mm, with a 0.49% tessellated-area deficit—appropria
pl2 = pv.Plotter(off_screen=True, window_size=(850,750))
pl2.add_mesh(mid, color="white", show_edges=True, edge_color="black", line_width=0.8)
pl2.add_mesh(sphere_vtk, color="crimson", opacity=0.45)
pl2.view_xy()
pl2.camera.focal_point = (0,0,0)
pl2.camera.SetParallelScale(0.040)
pl2.add_text("Center-plane close-up: sphere and prism-layer stack", font_size=12)
pl2.screenshot("sphere_layers_closeup.png")
pl2.show()

# -- cell 16 ------------------------------------------------------------------------
# The close-up now shows the concentric prism stack around the sphere and the graded transition to the
cm = subprocess.run(["checkMesh"], text=True, capture_output=True)
Path("log.checkMesh").write_text(cm.stdout + "\n" + cm.stderr)
print(cm.stdout[-6000:])
if cm.returncode != 0:
    print(cm.stderr[-1500:])
    raise RuntimeError("checkMesh failed")
