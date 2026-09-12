"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, numpy as np
from build123d import *

R_SHAFT, H, R_COL = 1.2, 3.0, 0.1
N_TREAD, TREAD_DEPTH, TREAD_THK, RISE, TURN = 20, 0.25, 0.04, 0.15, 18.0
TREAD_RADIAL = R_SHAFT - R_COL   # 1.1 m exposed

shaft  = Pos(0, 0, H/2) * Cylinder(R_SHAFT, H)
column = Pos(0, 0, H/2) * Cylinder(R_COL, H)

treads = []
for i in range(N_TREAD):
    ztop = RISE * (i + 1)
    plank = Pos(R_SHAFT/2, 0, ztop - TREAD_THK/2) * Box(R_SHAFT, TREAD_DEPTH, TREAD_THK)
    treads.append(Rot(0, 0, TURN * i) * plank)

tread_solid = treads[0]
for t in treads[1:]:
    tread_solid = tread_solid + t

fluid = shaft - column - tread_solid
print("fluid volume m^3:", round(fluid.volume, 5))
print("bbox:", fluid.bounding_box())
print("n faces:", len(fluid.faces()), "n solids:", len(fluid.solids()))

# -- cell 2 -------------------------------------------------------------------------
# Volume 13.2504 m³ matches the analytic estimate (13.478 shaft−column minus ~0.227 of treads). Now cl
from collections import Counter
rows=[]
for f in fluid.faces():
    c=f.center(); n=f.normal_at(c)
    rows.append((f.geom_type, round(f.area,4), round((c.X**2+c.Y**2)**0.5,4), round(c.Z,4)))
print(Counter(r[0] for r in rows))
for r in sorted(rows, key=lambda r:(str(r[0]), -r[1]))[:12]: print(r)
print("total area", round(sum(r[1] for r in rows),4))

# -- cell 3 -------------------------------------------------------------------------
groups={"shaft":[], "column":[], "floor":[], "top":[], "treads":[]}
for f in fluid.faces():
    c=f.center(); r=(c.X**2+c.Y**2)**0.5
    if f.geom_type==GeomType.CYLINDER:
        groups["shaft" if r>0.5 else "column"].append(f)
    elif abs(c.Z)<1e-9 and f.area>1.0:
        groups["floor"].append(f)
    elif abs(c.Z-3.0)<1e-9 and f.area>1.0:
        groups["top"].append(f)
    else:
        groups["treads"].append(f)
for k,v in groups.items(): print(k, len(v), round(sum(f.area for f in v),4))
areas=sorted(f.area for f in groups["treads"])
print("smallest tread-group faces:", [round(a,6) for a in areas[:8]])

# -- cell 4 -------------------------------------------------------------------------
from collections import Counter
print(Counter(round(f.area,5) for f in groups["treads"]))
sm=[f for f in groups["treads"] if f.area<0.002][:3]
for f in sm:
    c=f.center(); print("center",[round(v,4) for v in (c.X,c.Y,c.Z)],"normal",[round(v,3) for v in f.normal_at(c)],"bbox",f.bounding_box())

# -- cell 5 -------------------------------------------------------------------------
# Geometry is clean: each plank is embedded into the column and trimmed by it, so the tread's inner bo
import math
# measure a representative tread (i=5) from the built solid
tf=[f for f in groups["treads"] if abs(f.area-0.28375)<1e-4 and 0.85<f.center().Z<0.95]
bb=[f.bounding_box() for f in tf]
f0=tf[0]; c=f0.center()
rmax=max(math.hypot(v.X,v.Y) for v in f0.vertices()); rmin=min(math.hypot(v.X,v.Y) for v in f0.vertices())
print("tread face z:",[round(f.center().Z,3) for f in tf])
print("radial extent of tread underside: rmin %.4f rmax %.4f -> exposed length %.4f (asked 1.100)"%(rmin,rmax,rmax-R_COL))
# depth = width across
ang=[math.degrees(math.atan2(v.Y,v.X)) for v in f0.vertices()]
print("tread centre-line angle %.2f deg (tread index %d -> %.1f)"%(math.degrees(math.atan2(c.Y,c.X)), 5, 5*TURN))
zs=sorted(set(round(f.center().Z,4) for f in groups["treads"] if abs(f.area-0.28375)<1e-4))
print("consecutive rise:", round(zs[1]-zs[0],4), "thickness:", round(TREAD_THK,3))
print("shaft ID %.3f (asked 2.400), column D %.3f (asked 0.200), height %.3f"%(2*R_SHAFT,2*R_COL,H))

# -- cell 6 -------------------------------------------------------------------------
# Rise was mis-measured (list mixed tops and bottoms). Re-measure rise, depth and turn properly.
import math, numpy as np
tops=sorted(set(round(f.center().Z,4) for f in groups["treads"] if abs(f.area-0.28375)<1e-4))
tops=np.array(tops); tops_only=tops[1::2] if tops[0]<tops[1] else tops
print("all tread plate z:", tops)
print("rise between successive treads:", round(tops[2]-tops[0],4), "(asked 0.150)")
# depth: width of a side-to-side across a tread underside, and turn between tread centre angles
angs=[]
for f in groups["treads"]:
    if abs(f.area-0.28375)<1e-4 and abs(f.normal_at(f.center()).Z)>0.9:
        c=f.center(); angs.append((round(c.Z,3), round(math.degrees(math.atan2(c.Y,c.X))%360,2)))
angs.sort()
print("first 6 (z, centre angle):", angs[:6])
f0=[f for f in groups["treads"] if abs(f.area-0.28375)<1e-4 and abs(f.center().Z-0.86)<1e-3][0]
vs=np.array([[v.X,v.Y] for v in f0.vertices()])
d=max(np.hypot(*(vs[i]-vs[j])) for i in range(len(vs)) for j in range(len(vs)))
# depth = distance between the two parallel side planes
n=np.array([-math.sin(math.radians(90)), math.cos(math.radians(90))])
print("tread depth (spread along its own y):", round(vs@n).max()-round(vs@n).min() if False else round(float((vs@n).max()-(vs@n).min()),4), "(asked 0.250)")

# -- cell 7 -------------------------------------------------------------------------
# All five requested dimensions check out (rise 0.150, turn 18°, depth 0.250, thickness 0.040, exposed
import os
from build123d import Compound, export_stl
os.makedirs("constant/triSurface", exist_ok=True)
stl_paths={}
for name, fs in groups.items():
    comp = Compound(children=fs) if len(fs)>1 else fs[0]
    p = f"constant/triSurface/{name}.stl"
    export_stl(comp, p, tolerance=0.002, angular_tolerance=0.2)
    stl_paths[name]=p
    print(name, len(fs), "faces ->", p, os.path.getsize(p), "bytes")

# -- cell 8 -------------------------------------------------------------------------
# Look at the exported surfaces before meshing.
import pyvista as pv
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(900,800))
cols={"shaft":"lightsteelblue","column":"firebrick","floor":"tan","top":"palegreen","treads":"goldenrod"}
for n,p in stl_paths.items():
    m=pv.read(p)
    pl.add_mesh(m, color=cols[n], opacity=0.35 if n=="shaft" else 1.0, show_edges=(n=="treads"), line_width=0.3)
    print(n, m.n_cells, "tris, bounds", [round(b,3) for b in m.bounds])
pl.camera_position=[(4.5,-4.5,4.0),(0,0,1.5),(0,0,1)]
pl.show_axes(); pl.screenshot("surfaces.png")

# -- cell 9 -------------------------------------------------------------------------
import numpy as np, struct, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
cols={"shaft":"lightsteelblue","column":"firebrick","floor":"tan","top":"palegreen","treads":"goldenrod"}
def read_stl_tris(path):
    with open(path,'rb') as fh:
        head=fh.read(84); n=struct.unpack('<I', head[80:84])[0]
        data=np.frombuffer(fh.read(n*50), dtype=np.uint8).reshape(n,50)
        return data[:,12:48].copy().view('<f4').reshape(n,3,3).astype(float)
fig=plt.figure(figsize=(9,9)); ax=fig.add_subplot(111, projection='3d')
for n,p in stl_paths.items():
    if n=="shaft": continue
    ax.add_collection3d(Poly3DCollection(read_stl_tris(p), facecolor=cols[n], edgecolor='k', linewidths=0.15))
ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_zlim(0,3.0)
ax.set_box_aspect((2.6,2.6,3.0)); ax.view_init(elev=22, azim=35)
plt.tight_layout(); plt.savefig("surfaces.png", dpi=110); plt.show()

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("surfaces.png"))

# -- cell 11 ------------------------------------------------------------------------
# The surfaces are right: 20 planks spiralling a full turn, column, floor, and the top opening with th
import os, textwrap
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
def w(p,s): open(p,"w").write(s.lstrip()+"\n")
HEAD='FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n'
w("system/controlDict", HEAD%"controlDict"+"""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;
runTimeModifiable true;""")
w("system/fvSchemes", HEAD%"fvSchemes"+"""
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}
laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}
snGradSchemes{default corrected;}""")
w("system/fvSolution", HEAD%"fvSolution"+"""
solvers{} relaxationFactors{}""")
BASE=0.1
nx=int(round(2.6/BASE)); nz=int(round(3.2/BASE))
w("system/blockMeshDict", HEAD%"blockMeshDict"+f"""
scale 1;
vertices
(
 (-1.3 -1.3 -0.1) (1.3 -1.3 -0.1) (1.3 1.3 -0.1) (-1.3 1.3 -0.1)
 (-1.3 -1.3 3.1) (1.3 -1.3 3.1) (1.3 1.3 3.1) (-1.3 1.3 3.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3)); }} );
mergePatchPairs ();""")
import subprocess
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-500:], r.stderr[-800:])

# -- cell 12 ------------------------------------------------------------------------
# Extract feature edges, then write the snappy dict (coarse: base 0.1 m, treads refined to level 3 = 1
import subprocess
names=["shaft","column","floor","top","treads"]
sfe=HEAD%"surfaceFeatureExtractDict"+"\n".join(
 f'{n}.stl {{ extractionMethod extractFromSurface; includedAngle 150; writeObj no; }}' for n in names)
w("system/surfaceFeatureExtractDict", sfe)
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1200:], r.stderr[-500:])

# -- cell 13 ------------------------------------------------------------------------
LEV={"shaft":(1,1),"column":(2,2),"floor":(1,1),"top":(0,0),"treads":(3,3)}
geo="\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs="\n".join(f'      {n} {{ level ({LEV[n][0]} {LEV[n][1]}); patchInfo {{ type wall; }} }}' for n in names)
feats="\n".join(f'      {{ file "{n}.eMesh"; level {max(LEV[n])}; }}' for n in names)
snap=HEAD%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geo}
}}
castellatedMeshControls
{{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; nCellsBetweenLevels 2;
  maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features
  (
{feats}
  );
  refinementSurfaces
  {{
{refs}
  }}
  refinementRegions {{}}
  locationInMesh (0.9877 0.1564 1.5);
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
  nGrow 0; featureAngle 130; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50; }}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13;
  minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
writeFlags (); mergeTolerance 1e-6;"""
w("system/snappyHexMeshDict", snap)
print(open("system/snappyHexMeshDict").read()[:900])

# -- cell 14 ------------------------------------------------------------------------
# `top` is an opening, not a wall — set its patchInfo type to `patch`, then run snappy in the backgrou
import re, subprocess, time
s=open("system/snappyHexMeshDict").read().replace(
  "top { level (0 0); patchInfo { type wall; } }", "top { level (0 0); patchInfo { type patch; } }")
open("system/snappyHexMeshDict","w").write(s)
print([l for l in s.splitlines() if l.strip().startswith("top")])
log=open("log.snappy","w"); proc=subprocess.Popen(["snappyHexMesh","-overwrite"],stdout=log,stderr=subprocess.STDOUT)
time.sleep(120); print("running" if proc.poll() is None else proc.poll())
print(subprocess.run(["tail","-25","log.snappy"],capture_output=True,text=True).stdout)

# -- cell 15 ------------------------------------------------------------------------
# Snapped mesh: 354k cells, no errors reported. Run checkMesh and list patches.
import subprocess
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-2600:])

# -- cell 16 ------------------------------------------------------------------------
# Parsing polyMesh by hand is fiddly; convert to VTK and use pyvista purely for numbers and a matplotl
import subprocess, glob
r=subprocess.run(["foamToVTK","-constant","-no-internal"],capture_output=True,text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-300:])
print(sorted(glob.glob("VTK/**/*.vt*", recursive=True))[:20])

# -- cell 17 ------------------------------------------------------------------------
import pyvista as pv, numpy as np
cad_area={k: sum(f.area for f in v) for k,v in groups.items()}
patch={}
for n in names:
    m=pv.read(f"VTK/t26_0/boundary/{n}.vtp"); patch[n]=m
    a=m.compute_cell_sizes(length=False, volume=False)["Area"].sum()
    c=m.cell_centers().points; r=np.hypot(c[:,0],c[:,1])
    print(f"{n:7s} mesh area {a:8.4f}  CAD area {cad_area[n]:8.4f}   r range {r.min():.3f}-{r.max():.3f}  z {c[:,2].min():.3f}-{c[:,2].max():.3f}")

# -- cell 18 ------------------------------------------------------------------------
# Patch areas match CAD to <0.2% and each sits at the right radius/height — naming is verified. Now lo
import subprocess, pyvista as pv, numpy as np
subprocess.run(["foamToVTK","-constant"],capture_output=True,text=True)
vol=pv.read("VTK/t26_0/internal.vtu")
sl=vol.slice(normal=(0,1,0), origin=(0,0,1.5))
print(vol.n_cells, sl.n_cells)
tri=sl.extract_surface().triangulate()
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
pts=tri.points; faces=tri.faces.reshape(-1,4)[:,1:]
polys=[pts[f][:,[0,2]] for f in faces]
fig,ax=plt.subplots(figsize=(9,11))
ax.add_collection(PolyCollection(polys, facecolor="#cfe2f3", edgecolor="k", linewidths=0.15))
ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.1,3.1); ax.set_aspect("equal"); ax.set_title("mesh slice y=0")
fig

# -- cell 19 ------------------------------------------------------------------------
# Clean the VTK dump out of the case before finishing.
import shutil, subprocess
shutil.rmtree("VTK", ignore_errors=True)
print(subprocess.run(["du","-sh","."],capture_output=True,text=True).stdout)
print(subprocess.run(["ls","constant/triSurface"],capture_output=True,text=True).stdout)
