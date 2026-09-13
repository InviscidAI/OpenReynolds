"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, numpy as np, subprocess
from build123d import *

os.makedirs("constant/triSurface", exist_ok=True)

def naca2412(n=80):
    m, p, t = 0.02, 0.4, 0.12
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y

X, Y = naca2412()
print(X[:3], Y[:3], X[-3:], Y[-3:], len(X), "thickness span =", (Y.max()-Y.min()))

# -- cell 2 -------------------------------------------------------------------------
C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200
X0_ROOT, X0_TIP = 0.0, 0.25*C_ROOT - 0.25*C_TIP   # quarter-chord line unswept

def section(c, x0, y):
    pts = [(X[i]*c + x0, y, Y[i]*c) for i in range(len(X)-1)]
    return Face(Wire(Polyline(*pts, close=True)))

root_face = section(C_ROOT, X0_ROOT, 0.0)
tip_face  = section(C_TIP,  X0_TIP,  SPAN)
wing = loft([root_face, tip_face], ruled=True)
bb = wing.bounding_box()
print("wing bbox", bb.min, bb.max, "vol", wing.volume)

# -- cell 3 -------------------------------------------------------------------------
print("root section bbox", root_face.bounding_box().min, root_face.bounding_box().max)
print("tip  section bbox", tip_face.bounding_box().min, tip_face.bounding_box().max)
print("root chord", root_face.bounding_box().size.X, "tip chord", tip_face.bounding_box().size.X,
      "span", wing.bounding_box().size.Y)
print("root t/c", root_face.bounding_box().size.Z/C_ROOT, "tip t/c", tip_face.bounding_box().size.Z/C_TIP)
qc_root = X0_ROOT + 0.25*C_ROOT; qc_tip = X0_TIP + 0.25*C_TIP
print("quarter-chord x root/tip", qc_root, qc_tip, "-> sweep deg",
      np.degrees(np.arctan2(qc_tip-qc_root, SPAN)))

# -- cell 4 -------------------------------------------------------------------------
XMIN, XMAX = -4*C_ROOT, C_ROOT + 8*C_ROOT
YMIN, YMAX = 0.0, SPAN + 3*C_ROOT
ZMIN, ZMAX = -3*C_ROOT, 3*C_ROOT
print("domain", (XMIN,XMAX),(YMIN,YMAX),(ZMIN,ZMAX))

def rect(origin, zdir, xdir, w, h):
    pl = Plane(origin=origin, z_dir=zdir, x_dir=xdir)
    return pl * Rectangle(w, h)

cx, cy, cz = (XMIN+XMAX)/2, (YMIN+YMAX)/2, (ZMIN+ZMAX)/2
LX, LY, LZ = XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN

inlet  = rect((XMIN, cy, cz), (-1,0,0), (0,1,0), LY, LZ)
outlet = rect((XMAX, cy, cz), ( 1,0,0), (0,1,0), LY, LZ)
root_plane = rect((cx, YMIN, cz), (0,-1,0), (1,0,0), LX, LZ)
rootWall = root_plane - root_face          # wall plane with the wing root section removed
farfield = Compound(children=[
    rect((cx, YMAX, cz), (0,1,0), (1,0,0), LX, LZ),
    rect((cx, cy, ZMIN), (0,0,-1), (1,0,0), LX, LY),
    rect((cx, cy, ZMAX), (0,0, 1), (1,0,0), LX, LY)])
print("areas", inlet.area, outlet.area, rootWall.area, LX*LZ - root_face.area, farfield.area)

# -- cell 5 -------------------------------------------------------------------------
for name, shp in [("wing", wing), ("inlet", inlet), ("outlet", outlet),
                  ("rootWall", rootWall), ("farfield", farfield)]:
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=2e-5, angular_tolerance=0.1)
    print(name, os.path.getsize(f"constant/triSurface/{name}.stl"))

# -- cell 6 -------------------------------------------------------------------------
os.makedirs("system", exist_ok=True)
BASE = 0.03
nx, ny, nz = [max(1,int(round(L/BASE))) for L in (LX, LY, LZ)]
head = """FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"""
blockdict = head % "blockMeshDict" + f"""
scale 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet    {{ type patch; faces ((0 4 7 3)); }}
 outlet   {{ type patch; faces ((1 2 6 5)); }}
 rootWall {{ type wall; faces ((0 1 5 4)); }}
 farfield {{ type patch; faces ((3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(blockdict)
ctrl = head % "controlDict" + """
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 8;
runTimeModifiable true;
"""
open("system/controlDict","w").write(ctrl)
open("system/fvSchemes","w").write(head % "fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head % "fvSolution" + "solvers{} \n")
print(nx,ny,nz, nx*ny*nz)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-600:])

# -- cell 7 -------------------------------------------------------------------------
open("system/surfaceFeatureExtractDict","w").write(head % "surfaceFeatureExtractDict" + """
wing.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
""")
snappy = head % "snappyHexMeshDict" + """
castellatedMesh true; snap true; addLayers false;
geometry { wing.stl { type triSurfaceMesh; name wing; } }
castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 2;
    maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ( { file "wing.eMesh"; level 4; } );
    refinementSurfaces { wing { level (3 4); patchInfo { type wall; } } }
    refinementRegions {}
    locationInMesh (-0.30 0.30 0.20);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers {} expansionRatio 1.2; finalLayerThickness 0.5;
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:])

# -- cell 8 -------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:])

# -- cell 9 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read()
bnd = m["boundary"]
print(list(bnd.keys()))
p = pv.Plotter(off_screen=True, window_size=(1100,750))
for k, c in [("wing","red"),("rootWall","lightgray")]:
    p.add_mesh(bnd[k], color=c, show_edges=True, line_width=0.4)
p.camera_position = [(0.5,-0.6,0.5),(0.15,0.12,0.0),(0,0,1)]
p.screenshot("view_wing.png")
print("ok")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view_wing.png"))

# -- cell 11 ------------------------------------------------------------------------
p = pv.Plotter(off_screen=True, window_size=(1100,750))
p.add_mesh(bnd["wing"], color="salmon", show_edges=True, line_width=0.3)
p.camera_position = [(0.35,-0.25,0.28),(0.05,0.1,0.0),(0,0,1)]
p.screenshot("view_wing2.png")
display(Image("view_wing2.png"))
w = bnd["wing"].extract_surface()
print("wing patch bbox", np.round(w.bounds,4), "area", w.area, "STL area", wing.area)

# -- cell 12 ------------------------------------------------------------------------
snappy_fine = snappy.replace("level (3 4)", "level (4 5)").replace('"wing.eMesh"; level 4', '"wing.eMesh"; level 5').replace("nCellsBetweenLevels 2", "nCellsBetweenLevels 3")
open("system/snappyHexMeshDict","w").write(snappy_fine)
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-700:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 14 ------------------------------------------------------------------------
print("box - wing volume =", LX*LY*LZ - wing.volume, " mesh total = 0.62882641")
rd2 = pv.OpenFOAMReader("case.foam"); rd2.set_active_time_value(0.0)
m2 = rd2.read(); b2 = m2["boundary"]
p = pv.Plotter(off_screen=True, window_size=(1100,750))
p.add_mesh(b2["wing"], color="salmon", show_edges=True, line_width=0.2)
p.add_mesh(b2["rootWall"].extract_surface(algorithm="dataset_surface"), color="lightblue", opacity=0.4)
p.camera_position = [(0.35,-0.25,0.25),(0.05,0.1,0.0),(0,0,1)]
p.screenshot("view_fine.png"); display(Image("view_fine.png"))
print("cells:", m2["internalMesh"].n_cells)
