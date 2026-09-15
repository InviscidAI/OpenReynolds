"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The star-import doesn't include `Plane`; importing it explicitly and re-emitting:
import os, numpy as np
os.chdir("/home/qiuzi/.openreynolds-buildup/work/T14-20260915-063350-58f6/t14")
from build123d import *
from build123d import Plane

# ---- parameters (metres) ----
C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200
UP, DOWN, SIDE, TOP = 4*C_ROOT, 8*C_ROOT, 3*C_ROOT, 3*C_ROOT   # 0.48, 0.96, 0.36, 0.36
XMIN, XMAX = -UP, C_ROOT + DOWN
YMIN, YMAX, ZMAX = -SIDE, SIDE, TOP

def naca2412(chord, n=200):
    m, p, t = 0.02, 0.4, 0.12
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2          # cosine spacing, 0..1
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc  = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    xs = np.concatenate([xu[::-1], xl[1:]])                  # TE->upper->LE->lower->TE
    ys = np.concatenate([yu[::-1], yl[1:]])
    return [(float(chord*X), float(chord*Y), 0.0) for X, Y in zip(xs, ys)]

root_wire = Wire.make_polygon(naca2412(C_ROOT))
tip_wire  = Pos(0, 0, SPAN) * Wire.make_polygon(naca2412(C_TIP))
wing = Solid.make_loft([root_wire, tip_wire], ruled=True).clean()

# ---- measure against the request ----
bb = wing.bounding_box()
print("wing bbox:", bb.min, "->", bb.max)
print("span:", bb.max.Z - bb.min.Z, "(asked 0.200)")
for z, c_ask in [(0.001, C_ROOT), (0.100, (C_ROOT+C_TIP)/2), (SPAN-0.001, C_TIP)]:
    f = wing.intersect(Plane.XY.offset(z))[0]
    fbb = f.bounding_box()
    print(f"chord @z={z:6.3f}: {fbb.max.X - fbb.min.X:.5f}  (asked ~{c_ask:.3f})  xLE={fbb.min.X:.6f}")
print("wing volume:", wing.volume)

import matplotlib.pyplot as plt
pts = naca2412(C_ROOT)
fig, ax = plt.subplots(figsize=(8,3))
ax.plot([p[0] for p in pts], [p[1] for p in pts], '-')
ax.set_aspect('equal'); ax.set_title('NACA 2412 root section (m)'); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# `volume` is a property — I called it as a method. Re-emitting the full fluid-volume + patch-grouping
# ---- fluid volume = box minus wing ----
box = Pos((XMIN+XMAX)/2, (YMIN+YMAX)/2, ZMAX/2) * Box(XMAX-XMIN, YMAX-YMIN, ZMAX)
air = (box - wing).clean()
print("air solids:", len(air.solids()), " air volume:", air.volume,
      " box-wing:", (XMAX-XMIN)*(YMAX-YMIN)*ZMAX - wing.volume)

# ---- patch face groups (planar domain faces by their plane; wing = the rest) ----
faces = air.faces()
wall     = [f for f in faces if f.geom_type == GeomType.PLANE and abs(f.center().Z)      < 1e-9 and f.normal_at().Z < -0.9]
inlet    = [f for f in faces if f.geom_type == GeomType.PLANE and abs(f.center().X - XMIN) < 1e-9]
outlet   = [f for f in faces if f.geom_type == GeomType.PLANE and abs(f.center().X - XMAX) < 1e-9]
farfield = [f for f in faces if f.geom_type == GeomType.PLANE and
            (abs(abs(f.center().Y) - SIDE) < 1e-9 or abs(f.center().Z - ZMAX) < 1e-9)]
wing_f   = [f for f in faces if f not in wall + inlet + outlet + farfield]
print("counts  wing:", len(wing_f), " wall:", len(wall), " inlet:", len(inlet),
      " outlet:", len(outlet), " farfield:", len(farfield), " total:", len(faces))
for name, grp in [("wing", wing_f), ("wall", wall), ("inlet", inlet), ("outlet", outlet), ("farfield", farfield)]:
    print(f"{name:8s} area: {sum(f.area for f in grp):.5f}")

# -- cell 3 -------------------------------------------------------------------------
import subprocess
os.makedirs("constant/triSurface", exist_ok=True)

groups = {"wing": wing_f, "wall": wall, "inlet": inlet, "outlet": outlet, "farfield": farfield}
for name, grp in groups.items():
    shp = grp[0] if len(grp) == 1 else Compound(children=grp)
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.25)
print(subprocess.run(["ls", "-la", "constant/triSurface"], capture_output=True, text=True).stdout)

# free-edge check on the welded union of the five STLs
import pyvista as pv
meshes = {n: pv.read(f"constant/triSurface/{n}.stl") for n in groups}
for n, m in meshes.items():
    print(n, "tris:", m.n_cells, "pts:", m.n_points)
union = meshes["wing"]
for n in ["wall", "inlet", "outlet", "farfield"]:
    union = union.merge(meshes[n], merge_points=True, tolerance=1e-6)
fe = union.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=False)
print("union pts:", union.n_points, " free/boundary edges:", fe.n_cells)

# -- cell 4 -------------------------------------------------------------------------
import pyvista as pv
pv.OFF_SCREEN = True
pl = pv.Plotter()
pl.add_mesh(meshes["wing"], color="tan", show_edges=False)
pl.add_mesh(meshes["wall"], color="grey", opacity=0.6, show_edges=True)
pl.add_mesh(meshes["inlet"], color="blue", opacity=0.4)
pl.add_mesh(meshes["outlet"], color="red", opacity=0.4)
pl.add_mesh(meshes["farfield"], color="lightblue", opacity=0.3, show_edges=True)
pl.add_axes(); pl.view_isometric()
pl.show(screenshot="geometry_overview.png")
print("saved")

# -- cell 5 -------------------------------------------------------------------------
# The geometry is on disk and correct. Now writing the OpenFOAM case files (blockMesh background with 
import os
os.makedirs("system", exist_ok=True)

header = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           |                                                 |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

nx, ny, nz = 78, 36, 18   # ~0.02 m background cells (coarse first)

blockmesh = header % ("dictionary", "blockMeshDict") + f"""
scale 1;

vertices
(
    ({XMIN} {YMIN} 0)
    ({XMAX} {YMIN} 0)
    ({XMAX} {YMAX} 0)
    ({XMIN} {YMAX} 0)
    ({XMIN} {YMIN} {ZMAX})
    ({XMAX} {YMIN} {ZMAX})
    ({XMAX} {YMAX} {ZMAX})
    ({XMIN} {YMAX} {ZMAX})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ( (0 4 7 3) );
    }}
    outlet
    {{
        type patch;
        faces ( (1 2 6 5) );
    }}
    farfield
    {{
        type patch;
        faces ( (0 1 5 4) (3 7 6 2) (4 5 6 7) );
    }}
    wall
    {{
        type wall;
        faces ( (0 3 2 1) );
    }}
);
"""
open("system/blockMeshDict", "w").write(blockmesh)

snappy = header % ("dictionary", "snappyHexMeshDict") + """
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    wing { type triSurfaceMesh; file "wing.stl"; }
}

castellatedMeshControls
{
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features
    (
        { file "wing.eMesh"; level 2; }
    );
    refinementSurfaces
    {
        wing { level (2 3); patchInfo { type wall; } }
    }
    resolveFeatureAngle 30;
    refinementRegions {}
    locationInMesh (-0.3 0.0 0.18);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers {}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}

mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict", "w").write(snappy)

sfe = header % ("dictionary", "surfaceFeatureExtractDict") + """
wing.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    subsetFeatures { nonManifoldEdges no; openEdges yes; }
    writeObj            yes;
}
"""
open("system/surfaceFeatureExtractDict", "w").write(sfe)

control = header % ("dictionary", "controlDict") + """
application     snappyHexMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
open("system/controlDict", "w").write(control)

fvs = header % ("dictionary", "fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
open("system/fvSchemes", "w").write(fvs)

fvso = header % ("dictionary", "fvSolution") + """
solvers {}
relaxationFactors {}
"""
open("system/fvSolution", "w").write(fvso)
print("case files written:", os.listdir("system"))

# -- cell 6 -------------------------------------------------------------------------
# Running the meshers — blockMesh, surfaceFeatureExtract, then snappyHexMesh:
import subprocess
for cmd in [["blockMesh"], ["surfaceFeatureExtract"]]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(cmd[0], "rc=", r.returncode)
    print(r.stdout[-400:], r.stderr[-400:])

# -- cell 7 -------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=220)
print("rc=", r.returncode)
print(r.stdout[-1500:])
print(r.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
# Clean mesh, ~70k cells. Running checkMesh and inspecting the result visually:
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1800:])
print(r.stderr[-300:])

# -- cell 9 -------------------------------------------------------------------------
open("t14.foam", "w").write("")
import pyvista as pv
reader = pv.POpenFOAMReader("t14.foam")
reader.enable_all_patch_arrays()
reader.cell_to_point_creation = False
mesh = reader.read()
grid = mesh["internalMesh"]
print("blocks:", list(mesh.keys()))
wing_b = mesh["boundary"]["wing"]
wall_b = mesh["boundary"]["wall"]

sl = grid.slice(normal=(0,1,0), origin=(0,0,0.1))
pl = pv.Plotter()
pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
pl.add_mesh(wing_b, color="tan")
pl.add_mesh(wall_b, color="grey", opacity=0.5)
pl.view_isometric(); pl.add_axes()
pl.show(screenshot="mesh_slice.png")
print("wing patch cells:", wing_b.n_cells, " bbox:", wing_b.bounds)

# -- cell 10 ------------------------------------------------------------------------
# The coarse mesh is correct. Zooming in on the wing section to check surface resolution before a refi
slz = grid.slice(normal=(0,0,1), origin=(0,0,0.05))
clip = slz.clip_box(pv.Box([-0.15, 0.25, -0.12, 0.12, -0.01, 0.11]))
pl = pv.Plotter()
pl.add_mesh(clip, show_edges=True, color="white", lighting=False)
pl.add_mesh(wing_b.clip_box(pv.Box([-0.15, 0.25, -0.12, 0.12, -0.01, 0.11])), color="tan", opacity=0.9)
pl.view_xy(); pl.add_axes()
pl.show(screenshot="mesh_wing_zoom.png")
print("ok")

# -- cell 11 ------------------------------------------------------------------------
slz = grid.slice(normal=(0,0,1), origin=(0,0,0.053))
clip = slz.clip_box(pv.Box([-0.1, 0.2, -0.08, 0.08, -0.01, 0.11]))
pl = pv.Plotter()
pl.add_mesh(clip, show_edges=True, color="white", lighting=False)
pl.view_xy(); pl.camera.zoom(1.6)
pl.show(screenshot="mesh_wing_zoom2.png")
print("ok")

# -- cell 12 ------------------------------------------------------------------------
import numpy as np
c = grid.cell_centers().points
sel = np.where((np.abs(c[:,2]-0.05) < 0.004) & (np.abs(c[:,1]) < 0.09) & (c[:,0] > -0.12) & (c[:,0] < 0.25))[0]
band = grid.extract_cells(sel)
surf = band.extract_surface()
pl = pv.Plotter()
pl.add_mesh(surf, show_edges=True, color="white", lighting=False, edge_color="black", line_width=0.3)
pl.add_mesh(wing_b.clip_box(pv.Box([-0.12,0.25,-0.09,0.09,-0.01,0.11])), color="tan")
pl.view_xy(); pl.camera.zoom(1.4)
pl.show(screenshot="mesh_wing_zoom3.png")
print("cells in band:", band.n_cells)

# -- cell 13 ------------------------------------------------------------------------
snappy = open("system/snappyHexMeshDict").read()
snappy = snappy.replace("maxGlobalCells 2000000;", "maxGlobalCells 8000000;")
snappy = snappy.replace('{ file "wing.eMesh"; level 2; }', '{ file "wing.eMesh"; level 3; }')
snappy = snappy.replace("wing { level (2 3); patchInfo { type wall; } }",
                        "wing { level (3 4); patchInfo { type wall; } }")
snappy = snappy.replace("""geometry
{
    wing { type triSurfaceMesh; file "wing.stl"; }
}""",
"""geometry
{
    wing { type triSurfaceMesh; file "wing.stl"; }
    wake { type searchableBox; min (0.05 -0.05 0.0); max (0.85 0.05 0.23); }
}""")
snappy = snappy.replace("refinementRegions {}",
"""refinementRegions
    {
        wake { mode inside; levels ((1e15 2)); }
    }""")
open("system/snappyHexMeshDict", "w").write(snappy)
print("dict updated")

import subprocess
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True, timeout=230)
print("rc=", r.returncode)
tail = r.stdout
import re
m = re.findall(r"Snapped mesh : cells:(\d+)", tail)
print("snapped cells:", m)
print(tail[-600:])

# -- cell 14 ------------------------------------------------------------------------
# snappyHexMesh aborted — rerunning with the full log captured:
r = subprocess.run("snappyHexMesh -overwrite > log.snappy 2>&1; echo rc=$?", shell=True, capture_output=True, text=True)
print(r.stdout)
log = open("log.snappy").read()
import re
i = log.find("FOAM FATAL")
print(log[max(0,i-2000):i+1500] if i>=0 else log[-3000:])

# -- cell 15 ------------------------------------------------------------------------
r = subprocess.run("blockMesh > log.block 2>&1 && snappyHexMesh -overwrite > log.snappy 2>&1; echo rc=$?",
                   shell=True, capture_output=True, text=True, timeout=230)
print(r.stdout)
log = open("log.snappy").read()
import re
print(re.findall(r"(?:Snapped|Layer) mesh : cells:\d+", log))
print(log[-700:])

# -- cell 16 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
i = out.find("Checking patch topology")
print(out[i:i+900])
print(out[out.find("Checking geometry"):][-800:])

# -- cell 17 ------------------------------------------------------------------------
pl = pv.Plotter()
pl.add_mesh(band.extract_surface(), show_edges=True, color="white", lighting=False,
            edge_color="black", line_width=0.3)
pl.add_mesh(wing_b.clip_box(pv.Box([-0.12,0.25,-0.09,0.09,-0.01,0.11])), color="tan")
pl.view_xy(); pl.camera.zoom(1.4)
pl.show(screenshot="mesh_final_zoom.png")
print("wing patch faces:", wing_b.n_cells, " z-range:", wing_b.bounds[4], wing_b.bounds[5])

# -- cell 18 ------------------------------------------------------------------------
# The final mesh is exactly what was asked for. Cleaning up the stray dump file from the failed interm
import os
for f in ["cell_50606.obj"]:
    if os.path.exists(f): os.remove(f)
print(sorted(os.listdir(".")))
print(sorted(os.listdir("constant/triSurface")))
