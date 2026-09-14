"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, build123d as bd, os
os.makedirs("constant/triSurface", exist_ok=True)

C_ROOT, C_TIP, SPAN = 0.120, 0.080, 0.200
def naca4(m, p, t, n=80):
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
    return np.column_stack([X, Y])

pts = naca4(0.02, 0.4, 0.12)
print(pts.shape, pts[0], pts[-1], "thickness/c:", (pts[:,1].max()-pts[:,1].min()))

# -- cell 2 -------------------------------------------------------------------------
def section(chord, z):
    P = [bd.Vector(float(x)*chord, float(y)*chord, z) for x, y in pts]
    n = len(P)//2
    up = bd.Spline(P[:n+1])          # TE -> LE (upper)
    lo = bd.Spline(P[n:])            # LE -> TE (lower)
    return bd.Wire([up.edge(), lo.edge()])

Z_ROOT_EXT = -0.010   # root section extended below the wall plane so it cuts the floor cleanly
w0 = section(C_ROOT, Z_ROOT_EXT)
w1 = section(C_ROOT, 0.0)
w2 = section(C_TIP,  SPAN)
for w in (w0, w1, w2):
    print(w.is_closed, w.length, w.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
wing = bd.Solid.make_loft([w0, w1, w2], ruled=True)   # ruled -> linear taper between sections
print(type(wing), "volume", wing.volume, "valid", wing.is_valid, wing.bounding_box())
for z in (0.0, 0.1, 0.1999):
    sec = bd.Plane(origin=(0,0,z)).intersect(wing)
    bb = bd.Compound(list(sec)).bounding_box() if not hasattr(sec,'bounding_box') else sec.bounding_box()
    print(f"z={z:.4f} chord={bb.max.X-bb.min.X:.5f} x_LE={bb.min.X:.6f} thick={bb.max.Y-bb.min.Y:.5f} t/c={(bb.max.Y-bb.min.Y)/(bb.max.X-bb.min.X):.4f}")

# -- cell 4 -------------------------------------------------------------------------
XMIN, XMAX = -4*C_ROOT, C_ROOT + 8*C_ROOT        # 4 chords upstream of LE, 8 downstream of TE
YMIN, YMAX = -3*C_ROOT, 3*C_ROOT                 # 3 chords each side
ZMIN, ZMAX = 0.0, SPAN + 3*C_ROOT                # root wall at z=0, 3 chords above the tip
print(XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

def rect_face(origin, normal, u, v, du, dv):
    p = bd.Plane(origin=bd.Vector(origin), x_dir=bd.Vector(u), z_dir=bd.Vector(normal))
    return (p * bd.Rectangle(du, dv)).faces()[0]

cx, cy, cz = (XMIN+XMAX)/2, (YMIN+YMAX)/2, (ZMIN+ZMAX)/2
LX, LY, LZ = XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN
faces = {
 "inlet":    rect_face((XMIN, cy, cz), (1,0,0), (0,1,0), None, LY, LZ),
 "outlet":   rect_face((XMAX, cy, cz), (1,0,0), (0,1,0), None, LY, LZ),
 "rootWall": rect_face((cx, cy, ZMIN), (0,0,1), (1,0,0), None, LX, LY),
 "top":      rect_face((cx, cy, ZMAX), (0,0,1), (1,0,0), None, LX, LY),
 "sideA":    rect_face((cx, YMIN, cz), (0,1,0), (1,0,0), None, LX, LZ),
 "sideB":    rect_face((cx, YMAX, cz), (0,1,0), (1,0,0), None, LX, LZ),
}
for k, f in faces.items():
    print(k, f.area, f.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
from build123d import export_stl
tri = "constant/triSurface/"
export_stl(wing, tri+"wing.stl", tolerance=2e-5, angular_tolerance=0.1)
export_stl(faces["inlet"], tri+"inlet.stl")
export_stl(faces["outlet"], tri+"outlet.stl")
export_stl(faces["rootWall"], tri+"rootWall.stl")
export_stl(bd.Compound([faces["top"], faces["sideA"], faces["sideB"]]), tri+"farField.stl")
import subprocess
print(subprocess.run(["ls","-la",tri],capture_output=True,text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
w = pv.read(tri+"wing.stl"); print("wing tris", w.n_cells, w.bounds)
p = pv.Plotter(off_screen=True, window_size=(900,650))
p.add_mesh(w, color="tan", show_edges=False)
for n,c in [("inlet","blue"),("outlet","red"),("rootWall","grey"),("farField","lightblue")]:
    p.add_mesh(pv.read(tri+n+".stl"), color=c, opacity=0.18)
p.camera_position = "iso"; p.screenshot("view_all.png")

# -- cell 7 -------------------------------------------------------------------------
import matplotlib.pyplot as plt
p = pv.Plotter(off_screen=True, window_size=(900,650))
p.add_mesh(w, color="tan")
for n,c in [("inlet","blue"),("outlet","red"),("rootWall","grey"),("farField","lightblue")]:
    p.add_mesh(pv.read(tri+n+".stl"), color=c, opacity=0.2)
p.set_background("white"); p.camera_position=[(1.2,-1.2,1.0),(0.3,0,0.15),(0,0,1)]
img = p.screenshot("view_all.png", return_img=True)
print(img.shape, (img<250).sum())
plt.figure(figsize=(9,6.5)); plt.imshow(img); plt.axis("off"); plt.show()

# -- cell 8 -------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
HEAD = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
NX, NY, NZ = 39, 18, 14   # ~0.04 m base cells
bm = HEAD % "blockMeshDict" + f"""
scale 1;
vertices
(
 ({XMIN} {YMIN} {ZMIN}) ({XMAX} {YMIN} {ZMIN}) ({XMAX} {YMAX} {ZMIN}) ({XMIN} {YMAX} {ZMIN})
 ({XMIN} {YMIN} {ZMAX}) ({XMAX} {YMIN} {ZMAX}) ({XMAX} {YMAX} {ZMAX}) ({XMIN} {YMAX} {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet    {{ type patch; faces ((0 4 7 3)); }}
 outlet   {{ type patch; faces ((1 2 6 5)); }}
 rootWall {{ type wall;  faces ((0 3 2 1)); }}
 farField {{ type patch; faces ((4 5 6 7) (0 1 5 4) (3 7 6 2)); }}
);
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(bm)
ctrl = HEAD % "controlDict" + "\napplication simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n"
open("system/controlDict","w").write(ctrl)
open("system/fvSchemes","w").write(HEAD % "fvSchemes" + "\nddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(HEAD % "fvSolution" + "\nsolvers{} \n")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
sfe = HEAD % "surfaceFeatureExtractDict" + """
wing.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
"""
open("system/surfaceFeatureExtractDict","w").write(sfe)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:], r.stderr[-300:])

# -- cell 10 ------------------------------------------------------------------------
LEV_MIN, LEV_MAX = 3, 4
snap = HEAD % "snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry
{{
    wing.stl {{ type triSurfaceMesh; name wing; }}
    refBox {{ type searchableBox; min (-0.05 -0.08 -0.02); max (0.25 0.08 0.24); }}
}};
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 2;
    features ( {{ file "wing.eMesh"; level {LEV_MAX}; }} );
    refinementSurfaces {{ wing {{ level ({LEV_MIN} {LEV_MAX}); patchInfo {{ type wall; }} }} }};
    refinementRegions {{ refBox {{ mode inside; levels ((1e15 2)); }} }};
    resolveFeatureAngle 30; locationInMesh (0.5 0.2 0.3); allowFreeStandingZoneFaces true;
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
  implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
   nGrow 0; featureAngle 120; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
   maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #includeEtc "caseDicts/meshQualityDict" }}
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snap)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 11 ------------------------------------------------------------------------
snap = snap.replace('meshQualityControls {{', 'X').replace(
  'meshQualityControls { #includeEtc "caseDicts/meshQualityDict" }',
  'meshQualityControls { #includeEtc "caseDicts/meshQualityDict" \n errorReduction 0.75; nSmoothScale 4; }')
open("system/snappyHexMeshDict","w").write(snap)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-900:], r.stderr[-300:])

# -- cell 12 ------------------------------------------------------------------------
print(r.stdout[-6000:-900])

# -- cell 13 ------------------------------------------------------------------------
import glob
for f in glob.glob("cell_*.obj"): os.remove(f)
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1200:])

# -- cell 14 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 15 ------------------------------------------------------------------------
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
mesh = rd.read(); b = mesh["boundary"]
print(b.keys())
wp = b["wing"]
print("wing patch bounds", wp.bounds, "area", wp.area)
pl = pv.Plotter(off_screen=True, window_size=(1000,500)); pl.set_background("white")
pl.add_mesh(wp, color="tan", show_edges=True, line_width=0.5)
pl.add_mesh(b["rootWall"].clip_box([-0.1,0.3,-0.15,0.15,-1,1], invert=True), color="lightgrey", opacity=0.4, show_edges=True)
pl.camera_position=[(0.45,-0.45,0.45),(0.06,0,0.1),(0,0,1)]
img = pl.screenshot("wing_patch.png", return_img=True)
plt.figure(figsize=(10,5)); plt.imshow(img); plt.axis("off"); plt.show()

# -- cell 16 ------------------------------------------------------------------------
snap_final = snap.replace("wing {{ level", "X").replace(f"level ({LEV_MIN} {LEV_MAX})", "level (4 5)") \
                 .replace(f'file "wing.eMesh"; level {LEV_MAX}', 'file "wing.eMesh"; level 5') \
                 .replace("levels ((1e15 2))", "levels ((1e15 3))")
assert "level (4 5)" in snap_final and "level 5;" in snap_final and "1e15 3" in snap_final
open("system/snappyHexMeshDict","w").write(snap_final)
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-700:])

# -- cell 17 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[-1600:])

# -- cell 18 ------------------------------------------------------------------------
print([l for l in r.stdout.splitlines() if "cells:" in l or "cells " in l][:5])
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
mesh = rd.read(); wp = mesh["boundary"]["wing"]
pts_w = wp.points
for z0 in (0.005, 0.10, 0.195):
    m = np.abs(pts_w[:,2]-z0) < 0.002
    print(f"z={z0}: chord={pts_w[m,0].max()-pts_w[m,0].min():.5f} LE x={pts_w[m,0].min():.5f} thick={pts_w[m,1].max()-pts_w[m,1].min():.5f}")
sl = mesh["internalMesh"].slice(normal="z", origin=(0,0,0.05))
pl = pv.Plotter(off_screen=True, window_size=(900,500)); pl.set_background("white")
pl.add_mesh(sl, color="white", show_edges=True, line_width=0.4)
pl.view_xy(); pl.camera.zoom(1.0); pl.camera.focal_point=(0.06,0,0.05); pl.camera.position=(0.06,0,1.0); pl.camera.parallel_projection=True; pl.camera.parallel_scale=0.12
img = pl.screenshot("slice_z05.png", return_img=True)
plt.figure(figsize=(9,5)); plt.imshow(img); plt.axis("off"); plt.show()

# -- cell 19 ------------------------------------------------------------------------
for f in glob.glob("cell_*.obj"): os.remove(f)
print(sorted(os.listdir(".")), sorted(os.listdir("constant/triSurface")))
