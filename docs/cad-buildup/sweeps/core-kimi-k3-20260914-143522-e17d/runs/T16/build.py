"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Planning first: the geometry is an annular passage between two surfaces of revolution about the engi
import build123d as bd
import numpy as np, os

L = 0.9
outer_pts = [(0.0, 0.450), (0.45, 0.430), (0.9, 0.410)]
inner_pts = [(0.0, 0.280), (0.45, 0.310), (0.9, 0.250)]

def revolved_solid(pts):
    spline = bd.Spline(*[bd.Vector(x, 0, r) for x, r in pts])
    l1 = bd.Line(spline @ 0, bd.Vector(0, 0, 0))
    l2 = bd.Line(spline @ 1, bd.Vector(L, 0, 0))
    face = bd.Face(bd.Wire([spline, l1, l2, bd.Line(bd.Vector(0,0,0), bd.Vector(L,0,0))]).close())
    return bd.revolve(face, axis=bd.Axis.X, revolution_arc=360)

outer_s = revolved_solid(outer_pts)
inner_s = revolved_solid(inner_pts)
air = outer_s - inner_s
print("outer faces:", [(f.geom_type, f.area) for f in outer_s.faces()])
print("air faces:", [(f.geom_type, round(f.area,4)) for f in air.faces()])
air

# -- cell 2 -------------------------------------------------------------------------
# `centroid` isn't a method — use `center()` instead; also fix dir creation ordering.
import shutil
os.makedirs("constant/triSurface", exist_ok=True)
rev = sorted([f for f in air.faces() if f.geom_type == bd.GeomType.REVOLUTION], key=lambda f: f.area)
planes = sorted([f for f in air.faces() if f.geom_type == bd.GeomType.PLANE], key=lambda f: f.center().X)

patches = {
    "coreCowl":  rev[0],
    "outerCowl": rev[1],
    "inlet":     planes[0],
    "outlet":    planes[1],
}
for name, face in patches.items():
    bd.export_stl(face, f"constant/triSurface/{name}.stl")
    print(name, "area", round(face.area,4), "center x", round(face.center().X,3))

# -- cell 3 -------------------------------------------------------------------------
# Binary STL — use pyvista to read it.
import pyvista as pv, numpy as np
for name, xs, expected in [("outerCowl",[0.0,0.45,0.9],(0.450,0.430,0.410)),
                           ("coreCowl",[0.0,0.45,0.9],(0.280,0.310,0.250))]:
    m = pv.read(f"constant/triSurface/{name}.stl")
    pts = np.asarray(m.points)
    r_all = np.hypot(pts[:,1], pts[:,2])
    for x, r_exp in zip(xs, expected):
        sel = np.abs(pts[:,0]-x) < 1e-3
        r = r_all[sel]
        if len(r)==0:
            # no vertex exactly at x (spline interior); use nearest
            sel = np.abs(pts[:,0]-x) < 0.02
            r = r_all[sel]
        print(name, "x=", x, "measured r=", round(float(np.median(r)),6), "(spread %.2e)"%(r.max()-r.min()), "expected", r_exp)
    print()

# -- cell 4 -------------------------------------------------------------------------
# Geometry verified to ~1e-4 tolerance (spline hits all specified radii; 0.310 vs 0.31023 at one point
import subprocess, textwrap, pathlib
os.makedirs("system", exist_ok=True)

pathlib.Path("system/blockMeshDict").write_text(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    (-0.05 -0.50 -0.50)
    ( 0.95 -0.50 -0.50)
    ( 0.95  0.50 -0.50)
    (-0.05  0.50 -0.50)
    (-0.05 -0.50  0.50)
    ( 0.95 -0.50  0.50)
    ( 0.95  0.50  0.50)
    (-0.05  0.50  0.50)
);
blocks ( hex (0 1 2 3 4 5 6 7) (20 20 20) simpleGrading (1 1 1) );
edges (); boundary (); 
""").lstrip())

pathlib.Path("system/snappyHexMeshDict").write_text(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
    outerCowl.stl { type triSurfaceMesh; name outerCowl; appendRegionName false; }
    coreCowl.stl  { type triSurfaceMesh; name coreCowl;  appendRegionName false; }
    inlet.stl     { type triSurfaceMesh; name inlet;     appendRegionName false; }
    outlet.stl    { type triSurfaceMesh; name outlet;    appendRegionName false; }
}
castellatedMeshControls
{
    maxLocalCells 100000; maxGlobalCells 2000000; minRefinementCells 10;
    maxLoadUnbalance 0.10; nCellsBetweenLevels 2; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces
    {
        outerCowl { level (2 2); patchInfo { type wall; } }
        coreCowl  { level (2 2); patchInfo { type wall; } }
        inlet     { level (2 2); }
        outlet    { level (2 2); }
    }
    resolveFeatureAngle 30; refinementRegions (); locationInMesh (0.45 0.35 0.0);
}
snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers (); expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1;
    nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; maxLayerIter 30; }
meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}
mergeTolerance 1e-6; debug 0;
""").lstrip())
print("written")

# -- cell 5 -------------------------------------------------------------------------
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r1.returncode, r1.stdout[-1500:], r1.stderr[-800:])

# -- cell 6 -------------------------------------------------------------------------
# Missing `controlDict` — add a minimal one, rerun blockMesh, then snappyHexMesh.
pathlib.Path("system/controlDict").write_text(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
purgeWrite 0; writeFormat ascii; writePrecision 6; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
""").lstrip())
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc", r1.returncode)
print(r1.stdout[-600:])

# -- cell 7 -------------------------------------------------------------------------
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", r2.returncode)
print(r2.stdout[-2500:])
print(r2.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
pathlib.Path("system/fvSchemes").write_text(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""").lstrip())
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", r2.returncode); print(r2.stdout[-2000:]); print(r2.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
pathlib.Path("system/fvSolution").write_text(textwrap.dedent("""
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {}
""").lstrip())
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", r2.returncode); print(r2.stdout[-2200:]); print(r2.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
# `refinementRegions ();` parsed as a bare entry — needs a `dict` body. Fix both empty-dict entries an
p = pathlib.Path("system/snappyHexMeshDict")
s = p.read_text().replace("refinementRegions ();", "refinementRegions { }").replace("features ();", "features ( ;)")
s = s.replace("features ( ;)", "features ();")  # features () is fine
p.write_text(s)
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc", r2.returncode); print(r2.stdout[-2500:]); print(r2.stderr[-500:])

# -- cell 11 ------------------------------------------------------------------------
# Mesh built: 79,301 cells, all quality checks pass per snappy. Now `checkMesh` and a visual slice.
r3 = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r3.returncode); print(r3.stdout[-2600:])

# -- cell 12 ------------------------------------------------------------------------
r4 = subprocess.run(["foamToVTK", "-latestTime"], capture_output=True, text=True)
print("rc", r4.returncode, r4.stdout[-500:], r4.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
import pyvista as pv
pv.set_plot_theme("document")
m = pv.read("VTK/t16_0/internal.vtu")
sl = m.slice(normal=(0,0,1), origin=(0.45,0,0))
pl = pv.Plotter()
pl.add_mesh(sl, color="lightblue", show_edges=True, edge_color="gray", opacity=0.8)
for name, col in [("outerCowl","tan"),("coreCowl","brown"),("inlet","blue"),("outlet","green")]:
    b = pv.read(f"VTK/t16_0/boundary/{name}.vtp")
    pl.add_mesh(b, color=col, opacity=0.5, label=name)
pl.add_legend()
pl.view_yx(); pl.camera.up = (1,0,0); pl.camera.position = (0.45, 0, 6)
pl.show()

# -- cell 14 ------------------------------------------------------------------------
import numpy as np, pyvista as pv, math
def r_of_x(name):
    pts = np.asarray(pv.read(f"constant/triSurface/{name}.stl").points)
    r = np.hypot(pts[:,1], pts[:,2])
    xs = np.linspace(0,0.9,181)
    rr = [np.median(r[np.abs(pts[:,0]-x)<0.02]) for x in xs]
    return xs, rr
xs, ro = r_of_x("outerCowl"); _, ri = r_of_x("coreCowl")
vol = np.trapezoid(math.pi*(np.array(ro)**2-np.array(ri)**2), xs)
print("analytic volume", round(vol,6), " mesh volume 0.274741")

# -- cell 15 ------------------------------------------------------------------------
import numpy as np, pyvista as pv, math
def r_of_x(name):
    pts = np.asarray(pv.read(f"constant/triSurface/{name}.stl").points)
    r = np.hypot(pts[:,1], pts[:,2])
    edges = np.linspace(0,0.9,31)
    out = [np.median(r[(pts[:,0]>=edges[i]) & (pts[:,0]<edges[i+1])]) for i in range(len(edges)-1)]
    c = (edges[:-1]+edges[1:])/2
    return c, np.array(out)
xs, ro = r_of_x("outerCowl"); _, ri = r_of_x("coreCowl")
vol = np.trapezoid(math.pi*(ro**2-ri**2), xs)
print("analytic volume", round(vol,6), " mesh volume 0.274741  (diff %.2f%%)"%(100*(vol-0.274741)/vol))

# -- cell 16 ------------------------------------------------------------------------
import numpy as np, math
for name in ["outerCowl","coreCowl"]:
    pass
print("nan locations outer:", np.isnan(ro).nonzero()[0])
ok = ~np.isnan(ro) & ~np.isnan(ri)
print("ok count", ok.sum())
vol = np.trapezoid(math.pi*(ro[ok]**2-ri[ok]**2), xs[ok])
print("analytic volume", round(vol,6))

# -- cell 17 ------------------------------------------------------------------------
# 3% gap — refine the quadrature before trusting it.
import numpy as np, pyvista as pv, math
def r_of_x(name, nbins=300):
    pts = np.asarray(pv.read(f"constant/triSurface/{name}.stl").points)
    r = np.hypot(pts[:,1], pts[:,2])
    edges = np.linspace(0,0.9,nbins+1)
    rr = [np.nanmedian(r[(pts[:,0]>=edges[i]) & (pts[:,0]<edges[i+1])]) for i in range(nbins)]
    c=(edges[:-1]+edges[1:])/2
    return c, np.array(rr)
xs, ro = r_of_x("outerCowl"); _, ri = r_of_x("coreCowl")
ok = ~np.isnan(ro) & ~np.isnan(ri)
print("ok", ok.sum())
vol = np.trapezoid(math.pi*(ro[ok]**2-ri[ok]**2), xs[ok])
print("analytic volume", vol)
