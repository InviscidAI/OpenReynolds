"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os, subprocess, inspect
print(bd.__version__ if hasattr(bd,'__version__') else 'bd')
print(inspect.signature(bd.loft))
print(inspect.signature(bd.revolve))
s = bd.Solid.make_box(1,1,1)
print(type(s.rotate(bd.Axis.X, 30)))
print(os.listdir('.'))

# -- cell 2 -------------------------------------------------------------------------
# The envelope cell errored on `is_valid()` (a property), so it never entered the script. Re-emitting 
import build123d as bd, numpy as np
from build123d import Axis, Plane, Vector

def revolve_profile(pts):
    """pts: list of (x, r) in the XY plane, closed automatically; revolved about the X axis."""
    wire = bd.Polyline([(p[0], p[1], 0) for p in pts] + [(pts[0][0], pts[0][1], 0)])
    return bd.revolve(bd.make_face(wire), Axis.X, 360)

# --- outer flow envelope: nacelle inner wall (bypass duct) then core casing inner wall ---
X_IN, X_BYP_OUT, X_CORE_OUT = 0.0, 2.80, 3.60
outer_pts = [(X_IN,0.0),(X_IN,1.05),(0.35,1.00),(0.90,1.00),(1.60,0.98),(2.20,0.94),
             (X_BYP_OUT,0.90),(X_BYP_OUT,0.45),(3.10,0.43),(X_CORE_OUT,0.35),(X_CORE_OUT,0.0)]
envelope = revolve_profile(outer_pts)
print(envelope.volume, envelope.is_valid, envelope.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
# Now the two internal solid bodies: the centrebody (spinner → compressor/turbine hub → exhaust plug) 
"""Centrebody: pointed spinner at x=0.10, fan hub r=0.35, compressor hub rising,
turbine hub falling, exhaust plug closing to a point at x=3.50."""
center_pts = [(0.10,0.00),(0.20,0.18),(0.45,0.35),(0.90,0.35),(1.30,0.345),(1.80,0.34),
              (2.10,0.335),(2.40,0.32),(2.80,0.30),(3.05,0.28),(3.50,0.00)]
centerbody = revolve_profile(center_pts)

"""Splitter / core cowl: sharp leading edge at (0.90, 0.50); outer surface bounds the
bypass duct, inner surface is the core casing.  Ends at the bypass nozzle exit plane."""
X_SPLIT = 0.90
split_outer = [(X_SPLIT,0.50),(1.10,0.60),(1.60,0.62),(2.20,0.60),(X_BYP_OUT,0.55)]
split_inner = [(X_BYP_OUT,0.45),(2.40,0.45),(2.10,0.42),(1.80,0.42),(1.30,0.46)]
splitter = revolve_profile(split_outer + split_inner)
print(centerbody.volume, centerbody.is_valid, splitter.volume, splitter.is_valid)
print(centerbody.bounding_box(), splitter.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
"""One blade = loft between an elliptical hub section and a twisted tip section,
built with span along +Z, then laid down so span is radial (+Y) and chord axial (+X)."""
def make_blade(x_c, r_hub, r_tip, chord, thick, stag_hub, stag_tip):
    f_h = bd.Rot(0,0,stag_hub) * bd.Pos(0,0,r_hub) * bd.Ellipse(chord/2, thick/2)
    f_t = bd.Rot(0,0,stag_tip) * bd.Pos(0,0,r_tip) * bd.Ellipse(chord*0.7/2, thick*0.6/2)
    blade = bd.loft([f_h.face(), f_t.face()])
    return bd.Pos(x_c,0,0) * bd.Rot(-90,0,0) * blade      # span -> +Y, chord -> +X

def make_row(n, **kw):
    b = make_blade(**kw)
    return [b.rotate(Axis.X, 360.0*i/n) for i in range(n)]

_b = make_blade(0.55, 0.30, 1.02, 0.30, 0.05, 40, 62)
print(_b.volume, _b.is_valid, _b.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
import time
t0=time.time()
fan   = make_row(20, x_c=0.55, r_hub=0.30, r_tip=1.02, chord=0.30, thick=0.05, stag_hub=40, stag_tip=62)
ogv   = make_row(24, x_c=1.35, r_hub=0.55, r_tip=1.00, chord=0.18, thick=0.030, stag_hub=18, stag_tip=10)
comp  = [b for x in (1.10,1.30,1.50,1.70)
         for b in make_row(24, x_c=x, r_hub=0.28, r_tip=0.50, chord=0.09, thick=0.014,
                           stag_hub=35, stag_tip=50)]
turb  = [b for x in (2.30,2.55)
         for b in make_row(20, x_c=x, r_hub=0.26, r_tip=0.50, chord=0.13, thick=0.022,
                           stag_hub=30, stag_tip=45)]
print(len(fan),len(ogv),len(comp),len(turb), round(time.time()-t0,1),"s")

# -- cell 6 -------------------------------------------------------------------------
t0=time.time()
fluid0 = envelope - centerbody - splitter
print("annulus", round(fluid0.volume,4), fluid0.is_valid, round(time.time()-t0,1),"s")

# -- cell 7 -------------------------------------------------------------------------
t0=time.time()
blades = fan + ogv + comp + turb
fluid = fluid0 - blades
print("fluid", round(fluid.volume,4), fluid.is_valid, len(fluid.faces()), round(time.time()-t0,1),"s")

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
bd.export_stl(fluid, "fluid_all.stl", tolerance=2e-3, angular_tolerance=0.3)
m = pv.read("fluid_all.stl")
print(m.n_cells, m.bounds)
p = pv.Plotter(off_screen=True, window_size=(1400,600))
p.add_mesh(m.clip('z', origin=(0,0,0)), color='lightsteelblue', show_edges=False)
p.camera_position = [(1.8, 4.5, 3.5), (1.8, 0, 0), (0,0,1)]
p.show(screenshot="view1.png")

# -- cell 9 -------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sl = m.slice(normal='z', origin=(0,0,0))
fig, ax = plt.subplots(figsize=(13,5))
pts = sl.points; lines = sl.lines.reshape(-1,3)[:,1:]
for a,b in lines:
    ax.plot([pts[a,0],pts[b,0]],[pts[a,1],pts[b,1]],'k-',lw=0.6)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('x [m]'); ax.set_ylabel('r [m]')
ax.set_title('fluid volume, meridional slice z=0')
fig.savefig("slice.png", dpi=110); plt.show()

# -- cell 10 ------------------------------------------------------------------------
import matplotlib.pyplot as plt
sl = m.slice(normal='z', origin=(0,0,0))
fig, ax = plt.subplots(figsize=(13,5))
pts = sl.points; lines = sl.lines.reshape(-1,3)[:,1:]
for a,b in lines:
    ax.plot([pts[a,0],pts[b,0]],[pts[a,1],pts[b,1]],'k-',lw=0.6)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('x [m]'); ax.set_ylabel('r [m]')
ax.set_title('fluid volume, meridional slice z=0')
fig

# -- cell 11 ------------------------------------------------------------------------
print([a for a in dir(fluid) if 'dist' in a.lower()])
import build123d as bd
v = bd.Vertex(0.0,0.0,0.5)
print(centerbody.distance_to(v))

# -- cell 12 ------------------------------------------------------------------------
grp_solids = {"spinnerHub": centerbody, "splitter": splitter,
              "fanBlades": bd.Compound(fan), "ogv": bd.Compound(ogv),
              "compressor": bd.Compound(comp), "turbine": bd.Compound(turb)}
TOL = 1e-6
patch_faces = {k: [] for k in list(grp_solids)+["inlet","bypassOutlet","coreOutlet","nacelle"]}
for f in fluid.faces():
    c = f.center(); v = bd.Vertex(c.X, c.Y, c.Z)
    nx = abs(f.normal_at(c).X) if f.geom_type == bd.GeomType.PLANE else 0.0
    if nx > 0.99:
        if abs(c.X-X_IN) < 1e-6: patch_faces["inlet"].append(f); continue
        if abs(c.X-X_BYP_OUT) < 1e-6: patch_faces["bypassOutlet"].append(f); continue
        if abs(c.X-X_CORE_OUT) < 1e-6: patch_faces["coreOutlet"].append(f); continue
    d = {k: s.distance_to(v) for k, s in grp_solids.items()}
    best = min(d, key=d.get)
    patch_faces[best if d[best] < 1e-4 else "nacelle"].append(f)
print({k: len(v) for k, v in patch_faces.items()})

# -- cell 13 ------------------------------------------------------------------------
import os
os.makedirs("constant/triSurface", exist_ok=True)
for k, fl in patch_faces.items():
    bd.export_stl(bd.Compound(fl), f"constant/triSurface/{k}.stl", tolerance=2e-3, angular_tolerance=0.3)
print(subprocess.run(["ls","-la","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 14 ------------------------------------------------------------------------
import os, textwrap, subprocess
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(HDR%"controlDict"+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime;\n"
 "endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(HDR%"fvSchemes"+"gradSchemes{default Gauss linear;}\n"
 "divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(HDR%"fvSolution"+"solvers{}\n")
NX,NY,NZ = 67,40,40
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+f"""
scale 1;
vertices ((-0.2 -1.2 -1.2)(3.8 -1.2 -1.2)(3.8 1.2 -1.2)(-0.2 1.2 -1.2)
          (-0.2 -1.2 1.2)(3.8 -1.2 1.2)(3.8 1.2 1.2)(-0.2 1.2 1.2));
blocks (hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1));
edges (); boundary (); mergePatchPairs ();
""")
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-400:], r.returncode)

# -- cell 15 ------------------------------------------------------------------------
PATCHES = ["inlet","bypassOutlet","coreOutlet","nacelle","spinnerHub","splitter",
           "fanBlades","ogv","compressor","turbine"]
LEV = {p:(1,1) for p in PATCHES}
for p in ("fanBlades","ogv","compressor","turbine","splitter"): LEV[p]=(2,2)
geom = "\n".join(f'  {p}.stl {{ type triSurfaceMesh; name {p}; }}' for p in PATCHES)
surf = "\n".join(f'    {p} {{ level ({LEV[p][0]} {LEV[p][1]}); patchInfo {{ type '
                 f'{"patch" if p in ("inlet","bypassOutlet","coreOutlet") else "wall"}; }} }}'
                 for p in PATCHES)
open("system/snappyHexMeshDict","w").write(HDR%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls {{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10;
  nCellsBetweenLevels 2; maxLoadUnbalance 0.1; resolveFeatureAngle 45;
  allowFreeStandingZoneFaces true; features ();
  refinementSurfaces {{
{surf}
  }}
  refinementRegions {{}}
  locationInMesh (2.0 0.75 0.0);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
  multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2;
  finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60;
  nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
  nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
  maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02;
  minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1;
  nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
import time; t0=time.time()
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, round(time.time()-t0,1)); print(r.stdout[-1500:])

# -- cell 16 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-2600:])

# -- cell 17 ------------------------------------------------------------------------
r=subprocess.run(["foamToVTK","-constant","-excludePatches","(inlet bypassOutlet coreOutlet)"],capture_output=True,text=True)
print(r.returncode, r.stdout[-300:])
import glob; print(glob.glob("VTK/**/*.vt*", recursive=True)[:20])

# -- cell 18 ------------------------------------------------------------------------
import pyvista as pv
cols = {"nacelle":"silver","spinnerHub":"khaki","splitter":"lightblue",
        "fanBlades":"tomato","ogv":"limegreen","compressor":"orange","turbine":"violet"}
p = pv.Plotter(off_screen=True, window_size=(1500,700))
for k,c in cols.items():
    s = pv.read(f"VTK/t25_0/boundary/{k}.vtp").clip('z', origin=(0,0,0.02))
    p.add_mesh(s, color=c, show_edges=False)
p.camera_position=[(1.6,3.6,3.0),(1.7,0,0),(0,0,1)]
p.show(screenshot="mesh_cut.png")

# -- cell 19 ------------------------------------------------------------------------
import re
d = open("system/snappyHexMeshDict").read()
for p in ("fanBlades","ogv","compressor","turbine"):
    d = re.sub(rf"{p} {{{{? ?level \(2 2\)", f"{p} {{ level (3 3)", d)
    d = d.replace(f"{p} {{ level (2 2);", f"{p} {{ level (3 3);")
d = d.replace("maxInternalSkewness 4;","maxInternalSkewness 3.5;").replace("nSolveIter 30;","nSolveIter 50;")
open("system/snappyHexMeshDict","w").write(d)
print([l.strip() for l in d.splitlines() if "level" in l or "Skewness" in l])

# -- cell 20 ------------------------------------------------------------------------
subprocess.run(["blockMesh"],capture_output=True,text=True)
log = open("log.snappy","w")
proc = subprocess.Popen(["snappyHexMesh","-overwrite"], stdout=log, stderr=subprocess.STDOUT)
import time; time.sleep(180)
print(proc.poll()); print(subprocess.run(["tail","-5","log.snappy"],capture_output=True,text=True).stdout)
