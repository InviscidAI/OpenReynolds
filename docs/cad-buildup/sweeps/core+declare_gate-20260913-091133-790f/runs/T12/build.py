"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll send that as a tool call.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_IN, R_OUT, WRAP = 16.0, 58.0, 30.0      # mm, deg wrap (backward)
T_BLADE = 3.0
R_EXT = 58.6                               # blade overshoot, trimmed at outlet

def camber(r):                             # theta linear in r, backward = negative
    return -np.deg2rad(WRAP)*(r-R_IN)/(R_OUT-R_IN)

r = np.linspace(R_IN, R_EXT, 120)
th = camber(r)
P = np.c_[r*np.cos(th), r*np.sin(th)]
d = np.gradient(P, axis=0); d /= np.linalg.norm(d,axis=1)[:,None]
n = np.c_[-d[:,1], d[:,0]]
A, B = P + 0.5*T_BLADE*n, P - 0.5*T_BLADE*n

print("wrap angle deg:", np.rad2deg(th[0]-camber(R_OUT)))
print("thickness check:", np.linalg.norm(A[60]-B[60]))
fig,ax=plt.subplots(figsize=(5,5))
for k in range(7):
    c,s=np.cos(2*np.pi*k/7),np.sin(2*np.pi*k/7); M=np.array([[c,-s],[s,c]])
    ax.plot(*(A@M.T).T,'b'); ax.plot(*(B@M.T).T,'b')
t=np.linspace(0,2*np.pi,200)
for rr in (12,16,58): ax.plot(rr*np.cos(t),rr*np.sin(t),'k--',lw=.6)
ax.set_aspect('equal'); fig.savefig("profile.png",dpi=90)

# -- cell 2 -------------------------------------------------------------------------
from IPython.display import display
display(fig)

# -- cell 3 -------------------------------------------------------------------------
from build123d import *

H_BLADE, R_HUB, H_HUB, R_EYE = 14.0, 12.0, 18.0, 16.0

pts = [tuple(p) for p in A] + [tuple(p) for p in B[::-1]]
blade_face = make_face(Polyline(*pts, close=True))
blade1 = extrude(blade_face, H_BLADE)                      # z = 0..14
blades = Compound([blade1.rotate(Axis.Z, 360/7*k) for k in range(7)])
print("one blade volume mm^3:", blade1.volume)

disc  = Pos(0,0,H_BLADE/2) * Cylinder(R_OUT, H_BLADE)      # z 0..14, r<=58
hub   = Pos(0,0,H_HUB/2)   * Cylinder(R_HUB, H_HUB)        # z 0..18
eye   = Pos(0,0,(H_HUB+H_BLADE)/2) * Cylinder(R_EYE, H_HUB-H_BLADE)  # z 14..18
fluid = (disc + eye) - hub - blades
print("fluid volume mm^3:", fluid.volume, " solids:", len(fluid.solids()))
print("bbox:", fluid.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
import collections, math
groups = collections.defaultdict(list)
for f in fluid.faces():
    g = f.geom_type; c = f.center()
    if g == GeomType.PLANE:
        n = f.normal_at(c)
        if abs(n.Z) > 0.99:
            z = c.Z
            key = ("backplate" if abs(z)<1e-6 else "inlet" if abs(z-H_HUB)<1e-6
                   else "shroud" if abs(z-H_BLADE)<1e-6 else "?planez%.2f"%z)
        else: key = "blades"
    elif g == GeomType.CYLINDER:
        rad = math.hypot(c.X, c.Y)
        key = ("hub" if abs(rad-R_HUB)<1e-3 else "outlet" if abs(rad-R_OUT)<1e-3
               else "shroud" if abs(rad-R_EYE)<1e-3 else "?cylr%.2f"%rad)
    else: key = "blades"
    groups[key].append(f)
for k,v in sorted(groups.items()): print(k, len(v), round(sum(f.area for f in v),1))

# -- cell 5 -------------------------------------------------------------------------
import os, math
os.makedirs("constant/triSurface", exist_ok=True)
SC = 0.001   # mm -> m, OpenFOAM reads metres
for k, fl in groups.items():
    shp = Compound(fl).scale(SC)
    export_stl(shp, f"constant/triSurface/{k}.stl", tolerance=5e-5, angular_tolerance=0.2)
    print(k, os.path.getsize(f"constant/triSurface/{k}.stl"))
# in-fluid seed point: mid-passage at r=40mm, z=7mm
th_mid = camber(40.0) + math.pi/7
seed = (40*math.cos(th_mid)*SC, 40*math.sin(th_mid)*SC, 7*SC)
print("seed", seed, "inside:", fluid.scale(SC).is_inside(Vector(*seed)))

# -- cell 6 -------------------------------------------------------------------------
import textwrap, pathlib, subprocess
pathlib.Path("system").mkdir(exist_ok=True); pathlib.Path("constant").mkdir(exist_ok=True)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
pathlib.Path("system/controlDict").write_text(HDR%"controlDict"+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;"
 "deltaT 1; writeControl timeStep; writeInterval 1;\n")
pathlib.Path("system/fvSchemes").write_text(HDR%"fvSchemes"+"gradSchemes{}divSchemes{}laplacianSchemes{}ddtSchemes{}\n")
pathlib.Path("system/fvSolution").write_text(HDR%"fvSolution"+"solvers{}\n")
pathlib.Path("system/blockMeshDict").write_text(HDR%"blockMeshDict"+textwrap.dedent("""
scale 1;
vertices ((-0.0601 -0.0601 -0.0021)(0.0601 -0.0601 -0.0021)(0.0601 0.0601 -0.0021)(-0.0601 0.0601 -0.0021)
          (-0.0601 -0.0601 0.0201)(0.0601 -0.0601 0.0201)(0.0601 0.0601 0.0201)(-0.0601 0.0601 0.0201));
blocks (hex (0 1 2 3 4 5 6 7) (61 61 11) simpleGrading (1 1 1));
edges(); boundary(( bg { type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); } )); mergePatchPairs();
"""))
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 7 -------------------------------------------------------------------------
pathlib.Path("system/blockMeshDict").write_text(HDR%"blockMeshDict"+textwrap.dedent("""
scale 1;
vertices ((-0.0601 -0.0601 -0.0021)(0.0601 -0.0601 -0.0021)(0.0601 0.0601 -0.0021)(-0.0601 0.0601 -0.0021)
          (-0.0601 -0.0601 0.0201)(0.0601 -0.0601 0.0201)(0.0601 0.0601 0.0201)(-0.0601 0.0601 0.0201));
blocks (hex (0 1 2 3 4 5 6 7) (61 61 11) simpleGrading (1 1 1));
edges ();
boundary ( bg { type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); } );
mergePatchPairs ();
"""))
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-300:], r.returncode)

# -- cell 8 -------------------------------------------------------------------------
names = ["inlet","outlet","blades","backplate","hub","shroud"]
pathlib.Path("system/surfaceFeatureExtractDict").write_text(HDR%"surfaceFeatureExtractDict"+
  "".join(f'{n}.stl {{ extractionMethod extractFromSurface; includedAngle 150; writeObj no; }}\n' for n in names))
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.returncode, r.stdout[-800:])

# -- cell 9 -------------------------------------------------------------------------
geom = "".join(f'{n}.stl {{ type triSurfaceMesh; name {n}; }}\n' for n in names)
refs = "".join(f'{n} {{ level ({1} {1}); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}\n' for n in names)
feat = "".join(f'{{ file "{n}.eMesh"; level 1; }}\n' for n in names)
snappy = HDR%"snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry {{ {geom} }}
castellatedMeshControls
{{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; nCellsBetweenLevels 2;
  maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ( {feat} );
  refinementSurfaces {{ {refs} }}
  refinementRegions {{}}
  locationInMesh ({seed[0]} {seed[1]} {seed[2]});
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 12; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.4;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
"""
pathlib.Path("system/snappyHexMeshDict").write_text(snappy)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 10 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-2200:])

# -- cell 11 ------------------------------------------------------------------------
import pyvista as pv, numpy as np
pv.start_xvfb() if False else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
rd.disable_all_patch_arrays()
for n in names: rd.enable_patch_array("patch/"+n)
m = rd.read()
cols = dict(inlet="red",outlet="orange",blades="blue",backplate="grey",hub="green",shroud="pink")
p = pv.Plotter(off_screen=True, window_size=(900,700))
for n in names:
    b = m["boundary"][n]
    print(n, "area m^2:", round(float(np.sum(b.compute_cell_sizes(length=False,volume=False)["Area"])),6))
    if n!="shroud": p.add_mesh(b, color=cols[n], show_edges=(n=="blades"), line_width=1)
p.camera_position=[(0.12,-0.12,0.13),(0,0,0.007),(0,0,1)]
p.show()

# -- cell 12 ------------------------------------------------------------------------
bl = m["boundary"]["blades"].points
print("blades z range mm:", bl[:,2].min()*1000, bl[:,2].max()*1000)
rr = np.hypot(bl[:,0],bl[:,1]); print("blade r range mm:", rr.min()*1000, rr.max()*1000)
# thickness at r=40mm, z=7mm: angular spread of the nearest blade
sel = bl[(abs(rr-0.040)<3e-4)&(abs(bl[:,2]-0.007)<1e-3)]
th = np.arctan2(sel[:,1],sel[:,0]); k = np.argmin(abs(th-(camber(40.0)+0*np.pi)))
one = th[abs(th-th[k])<0.2]
print("blade thickness at r=40 mm:", (one.max()-one.min())*40, " camber theta deg:", np.rad2deg(one.mean()),
      " expected:", np.rad2deg(camber(40.0)))
print("mesh z extent mm:", m["boundary"]["backplate"].points[:,2].max()*1000,
      m["boundary"]["shroud"].points[:,2].min()*1000, m["boundary"]["shroud"].points[:,2].max()*1000)

# -- cell 13 ------------------------------------------------------------------------
th_all = np.arctan2(bl[:,1],bl[:,0])
band = (abs(rr-0.040)<5e-4)&(abs(bl[:,2]-0.007)<1.5e-3)
t0 = camber(40.0)                      # blade 0 camber angle at r=40
sel = th_all[band & (abs(th_all-t0)<0.15)]
print("n pts", sel.size, "angular half-widths deg:", np.rad2deg(sel.min()-t0), np.rad2deg(sel.max()-t0))
print("measured blade thickness at r=40mm [mm]:", (sel.max()-sel.min())*40.0, " target 3.0")

# -- cell 14 ------------------------------------------------------------------------
lev = {n: (2,2) if n=="blades" else (1,1) for n in names}
refs = "".join(f'{n} {{ level ({lev[n][0]} {lev[n][1]}); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}\n' for n in names)
feat = "".join(f'{{ file "{n}.eMesh"; level {lev[n][0]}; }}\n' for n in names)
pathlib.Path("system/snappyHexMeshDict").write_text(snappy.replace(
    snappy[snappy.index("  features ("):snappy.index("  refinementRegions")],
    f"  features ( {feat} );\n  refinementSurfaces {{ {refs} }}\n"))
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-350:])

# -- cell 15 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
o=r.stdout
print(o[o.index("Checking geometry"):][:1400])
print([l for l in o.splitlines() if "cells:" in l or "Mesh OK" in l or "***" in l][:12])

# -- cell 16 ------------------------------------------------------------------------
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
rd.disable_all_patch_arrays()
for n in names: rd.enable_patch_array("patch/"+n)
m2 = rd.read()
for n in names:
    b=m2["boundary"][n]; print(n, b.n_cells, round(float(np.sum(b.compute_cell_sizes(length=False,volume=False)["Area"])),6))
p = pv.Plotter(off_screen=True, window_size=(900,700))
for n in names:
    b = m2["boundary"][n].clip(normal="y", origin=(0,0.0,0))
    p.add_mesh(b, color=cols[n], show_edges=True, line_width=0.4)
p.camera_position=[(0.13,-0.15,0.11),(0,0,0.007),(0,0,1)]; p.show()
