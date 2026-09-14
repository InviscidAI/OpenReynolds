"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, math, os, subprocess
print(bd.__version__ if hasattr(bd,'__version__') else 'ok')
print(os.getcwd(), os.listdir('.'))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd, math
from build123d import Pos, Axis, GeomType

mm = 1e-3
LX, LY, LZ = 160*mm, 100*mm, 40*mm      # internal cavity
TW  = 3*mm                               # wall thickness
SD, SH = 10*mm, 25*mm                    # standoff dia, height
HD, HDEP = 4*mm, 12*mm                   # blind hole dia, depth
INSET = 15*mm                            # standoff axis inset from each corner
VW, VH, VZ = 30*mm, 20*mm, 10*mm         # vent width(y), height(z), bottom above floor

cavity = Pos(LX/2, LY/2, LZ/2) * bd.Box(LX, LY, LZ)
centers = [(x, y) for x in (INSET, LX-INSET) for y in (INSET, LY-INSET)]
posts = [Pos(x, y, SH/2) * bd.Cylinder(SD/2, SH) for x, y in centers]
holes = [Pos(x, y, SH-HDEP/2) * bd.Cylinder(HD/2, HDEP) for x, y in centers]
vent  = Pos(-TW/2, LY/2, VZ+VH/2) * bd.Box(TW, VW, VH)

fluid = cavity
for p in posts: fluid = fluid - p
for h in holes: fluid = fluid + h
fluid = fluid + vent

exp = LX*LY*LZ - 4*math.pi*(SD/2)**2*SH + 4*math.pi*(HD/2)**2*HDEP + TW*VW*VH
print("volume", fluid.volume, "expected", exp)
print("bbox", fluid.bounding_box())
print("nfaces", len(fluid.faces()), "nsolids", len(fluid.solids()))

# -- cell 3 -------------------------------------------------------------------------
# Fluid solid matches the expected volume exactly. Now classify the 27 faces into named patch groups u
TOL=1e-9
faces = fluid.faces()
cyl = faces.filter_by(GeomType.CYLINDER)
post_cyl = [f for f in cyl if abs(f.radius - SD/2) < TOL]
hole_cyl = [f for f in cyl if abs(f.radius - HD/2) < TOL]
pl  = faces.filter_by(GeomType.PLANE)
def at(f, axis, val): return abs(f.center().to_tuple()[axis] - val) < TOL
top   = [f for f in pl if at(f,2,LZ)]
floor = [f for f in pl if at(f,2,0)]
ventf = [f for f in pl if at(f,0,-TW)]
post_top = [f for f in pl if at(f,2,SH)]
hole_bot = [f for f in pl if at(f,2,SH-HDEP)]
named = set(id(f) for f in top+floor+ventf+post_top+hole_bot)
walls = [f for f in pl if id(f) not in named]
groups = {"top":top, "vent":ventf, "floor":floor, "walls":walls,
          "standoffs":post_cyl+post_top, "blindholes":hole_cyl+hole_bot}
for k,v in groups.items(): print(k, len(v), "area=", sum(f.area for f in v))
print("total faces classified", sum(len(v) for v in groups.values()), "of", len(faces))

# -- cell 4 -------------------------------------------------------------------------
# Areas all match hand calculations (top 160×100, vent 30×20, floor minus four footprints, etc.). Now 
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)
for k,v in groups.items():
    shp = bd.Compound(children=[bd.Face(f.wrapped) for f in v])
    bd.export_stl(shp, f"constant/triSurface/{k}.stl", tolerance=5e-5, angular_tolerance=0.2, ascii_format=True)
    # rename solid label to patch name
    p=f"constant/triSurface/{k}.stl"
    s=open(p).read().replace("solid ", f"solid {k} ",1) if False else open(p).read()
    lines=s.splitlines()
    lines[0]=f"solid {k}"; lines[-1]=f"endsolid {k}"
    open(p,"w").write("\n".join(lines)+"\n")
    print(k, os.path.getsize(p), lines[0], sum(1 for l in lines if l.strip().startswith("facet")))

# -- cell 5 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(1000,750))
cols={"top":"lightblue","vent":"red","floor":"grey","walls":"wheat","standoffs":"green","blindholes":"magenta"}
tot=0
for k,c in cols.items():
    m=pv.read(f"constant/triSurface/{k}.stl"); tot+=m.n_cells
    pl.add_mesh(m, color=c, opacity=0.45 if k in("top","walls") else 1.0, show_edges=True, line_width=0.5)
pl.camera_position=[(0.3,-0.25,0.25),(0.08,0.05,0.02),(0,0,1)]
pl.screenshot("view.png"); print("tris",tot)

# -- cell 6 -------------------------------------------------------------------------
import matplotlib.pyplot as plt, matplotlib.image as mpimg
img=mpimg.imread("view.png"); plt.figure(figsize=(9,7)); plt.imshow(img); plt.axis('off'); plt.show()

# -- cell 7 -------------------------------------------------------------------------
# Shape is right. Check the welded union is closed, then write the blockMesh/snappy case.
import pyvista as pv
union=None
for k in groups:
    m=pv.read(f"constant/triSurface/{k}.stl")
    union = m if union is None else union.merge(m)
u=union.clean(tolerance=1e-7)
be=u.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges:", be.n_cells, "tris:", u.n_cells)

# -- cell 8 -------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True)
def head(cls, obj, loc):
    return f"""FoamFile{{version 2.0; format ascii; class {cls}; location "{loc}"; object {obj};}}\n"""
open("system/controlDict","w").write(head("dictionary","controlDict","system")+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes","system")+
 "gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution","system")+"solvers{}\n")

BC = 0.005  # base cell size 5 mm (coarse first)
x0,x1 = -0.005, 0.165
y0,y1 = -0.005, 0.105
z0,z1 = -0.005, 0.045
nx,ny,nz = [int(round((b-a)/BC)) for a,b in ((x0,x1),(y0,y1),(z0,z1))]
bm = head("dictionary","blockMeshDict","system") + f"""
scale 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary ( outer {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
"""
open("system/blockMeshDict","w").write(bm)
print(nx,ny,nz,nx*ny*nz)

# -- cell 9 -------------------------------------------------------------------------
patches = ["top","vent","floor","walls","standoffs","blindholes"]
lev = {"top":(0,0),"vent":(1,1),"floor":(1,1),"walls":(1,1),"standoffs":(2,2),"blindholes":(3,3)}
geom = "\n".join(f'  {p}.stl {{ type triSurfaceMesh; name {p}; }}' for p in patches)
refs = "\n".join(f'    {p} {{ level ({lev[p][0]} {lev[p][1]}); patchInfo {{ type wall; }} }}' for p in patches)
shm = head("dictionary","snappyHexMeshDict","system") + f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2;
  features ();
  refinementSurfaces
  {{
{refs}
  }}
  resolveFeatureAngle 30;
  refinementRegions {{}}
  locationInMesh (0.080 0.050 0.035);
  allowFreeStandingZoneFaces false;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
debug 0;
"""
open("system/snappyHexMeshDict","w").write(shm)
import subprocess
r=subprocess.run(["blockMesh","-case",".","-dict","system/blockMeshDict"],capture_output=True,text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-2500:]); print(r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-3000:])

# -- cell 12 ------------------------------------------------------------------------
# Mesh is OK. Now verify the blind holes are actually resolved: convert and slice through a standoff a
r=subprocess.run(["foamToVTK","-constant","-ascii"],capture_output=True,text=True); print(r.returncode, r.stdout[-200:])
import glob, pyvista as pv
print(glob.glob("VTK/**/*.vt*", recursive=True)[:20])

# -- cell 13 ------------------------------------------------------------------------
cad_area = {k: sum(f.area for f in v) for k,v in groups.items()}
for k in patches:
    m=pv.read(f"VTK/t21_0/boundary/{k}.vtp"); a=m.compute_cell_sizes(length=False,volume=False)["Area"].sum()
    print(f"{k:11s} mesh {a:.6e}  cad {cad_area[k]:.6e}  diff {100*(a-cad_area[k])/cad_area[k]:+.2f}%")
vol=pv.read("VTK/t21_0/internal.vtu").compute_cell_sizes(length=False,area=False)["Volume"].sum()
print("mesh volume", vol, "cad", fluid.volume)

# -- cell 14 ------------------------------------------------------------------------
g=pv.read("VTK/t21_0/internal.vtu")
sl=g.slice(normal=(0,1,0), origin=(0,0.015,0))
p=pv.Plotter(off_screen=True, window_size=(1200,500))
p.add_mesh(sl, color="lightsteelblue", show_edges=True, line_width=0.6)
p.view_xz(); p.camera.zoom(1.4); p.screenshot("slice.png")
img=mpimg.imread("slice.png"); plt.figure(figsize=(13,5.5)); plt.imshow(img); plt.axis('off'); plt.show()

# -- cell 15 ------------------------------------------------------------------------
# Holes are resolved (~6 cells across). Now one refinement step: base cell 2.5 mm, keeping the hole ce
BC = 0.0025
nx,ny,nz = [int(round((b-a)/BC)) for a,b in ((x0,x1),(y0,y1),(z0,z1))]
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict","system") + f"""
scale 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary ( outer {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""")
lev = {"top":(0,0),"vent":(1,1),"floor":(1,1),"walls":(1,1),"standoffs":(1,1),"blindholes":(2,2)}
refs = "\n".join(f'    {p} {{ level ({lev[p][0]} {lev[p][1]}); patchInfo {{ type wall; }} }}' for p in patches)
shm2 = shm.split("refinementSurfaces")[0] + "refinementSurfaces\n  {\n" + refs + "\n  }\n" + \
       shm.split("resolveFeatureAngle",1)[1].join(["  resolveFeatureAngle",""])
open("system/snappyHexMeshDict","w").write(shm2)
print(nx*ny*nz); print(open("system/snappyHexMeshDict").read()[400:1200])

# -- cell 16 ------------------------------------------------------------------------
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.returncode, r.stdout[-150:])
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-900:])

# -- cell 17 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Checking topology"):])

# -- cell 18 ------------------------------------------------------------------------
# Mesh OK at 90k cells. Final visual check: slice through the vent plane (y = 50 mm) and a patch-colou
r=subprocess.run(["foamToVTK","-constant","-ascii","-overwrite"],capture_output=True,text=True)
g=pv.read("VTK/t21_0/internal.vtu")
sl=g.slice(normal=(0,1,0), origin=(0,0.05,0))
p=pv.Plotter(off_screen=True, window_size=(1200,450)); p.add_mesh(sl,color="lightsteelblue",show_edges=True,line_width=0.4)
p.view_xz(); p.camera.zoom(1.5); p.screenshot("slice_vent.png")
p2=pv.Plotter(off_screen=True, window_size=(1000,750))
for k,c in cols.items():
    p2.add_mesh(pv.read(f"VTK/t21_0/boundary/{k}.vtp"), color=c, opacity=0.35 if k in("top","walls") else 1.0)
p2.camera_position=[(0.30,-0.24,0.22),(0.08,0.05,0.02),(0,0,1)]; p2.screenshot("patches.png")
f,ax=plt.subplots(2,1,figsize=(12,11))
for a,fn in zip(ax,["slice_vent.png","patches.png"]): a.imshow(mpimg.imread(fn)); a.axis('off')
plt.show()

# -- cell 19 ------------------------------------------------------------------------
# Measure the requested dimensions directly off the built geometry before closing.
print("standoff cyl radii (mm):", sorted({round(f.radius*1e3,4) for f in post_cyl}), "asked 5.0")
print("standoff top z (mm):", sorted({round(f.center().Z*1e3,4) for f in post_top}), "asked 25")
print("hole radii (mm):", sorted({round(f.radius*1e3,4) for f in hole_cyl}), "asked 2.0")
print("hole bottom z (mm):", sorted({round(f.center().Z*1e3,4) for f in hole_bot}), "-> depth", 25-13, "asked 12")
print("standoff axes (mm):", sorted([(round(x*1e3),round(y*1e3)) for x,y in centers]), "-> 15 mm in from each corner")
bb=ventf[0].bounding_box()
print("vent face: x=%.4f mm, y %.1f..%.1f, z %.1f..%.1f mm -> %0.0f x %0.0f, sill %0.0f above floor"%(
  bb.min.X*1e3, bb.min.Y*1e3, bb.max.Y*1e3, bb.min.Z*1e3, bb.max.Z*1e3,
  (bb.max.Y-bb.min.Y)*1e3,(bb.max.Z-bb.min.Z)*1e3, bb.min.Z*1e3))
tb=top[0].bounding_box(); print("open top: %.0f x %.0f mm at z=%.0f mm"%((tb.max.X-tb.min.X)*1e3,(tb.max.Y-tb.min.Y)*1e3,tb.max.Z*1e3))
