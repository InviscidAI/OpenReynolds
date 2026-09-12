"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os
from build123d import *
p = "/home/qiuzi/.openreynolds-buildup/work/T5-20260912-125139-b2e4/geometry/assembly.step"
print(os.path.getsize(p))
imported = import_step(p)
print(type(imported))
print(imported.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
solids = imported.solids()
print(len(solids))
for i,s in enumerate(solids):
    print(i, round(s.volume,2), s.bounding_box(), s.is_valid)
units = [l for l in open(p).read().split("\n") if "UNIT" in l][:6]
print(units)

# -- cell 3 -------------------------------------------------------------------------
print([l for l in open(p).read().split("\n") if l.startswith("#2299") or l.startswith("#2300") or l.startswith("#2301")])

# -- cell 4 -------------------------------------------------------------------------
t = open(p).read()
i = t.find("#2301=(")
print(t[i:i+300])

# -- cell 5 -------------------------------------------------------------------------
import re
pts = re.findall(r"CARTESIAN_POINT\('',\(([^)]*)\)\)", t)[:8]
print(pts)

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv, numpy as np
pv.start_xvfb() if False else None
pv.set_plot_theme("document")
def mesh_of(shape, tol=0.2):
    v,f = shape.tessellate(tol)
    va = np.array([(p.X,p.Y,p.Z) for p in v])
    fa = np.hstack([[3,*t] for t in f])
    return pv.PolyData(va, fa)
ms = [mesh_of(s) for s in solids]
pl = pv.Plotter(off_screen=True, window_size=(900,700), shape=(1,2))
cols = ["red","green","blue"]
pl.subplot(0,0)
for m,c in zip(ms,cols): pl.add_mesh(m, color=c, opacity=0.5, show_edges=False)
pl.view_isometric()
pl.subplot(0,1)
for m,c in zip(ms,cols): pl.add_mesh(m, color=c, opacity=0.5)
pl.view_xz()
pl.screenshot("as_import.png")
print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("as_import.png"))

# -- cell 8 -------------------------------------------------------------------------
fused = solids[0].fuse(solids[1]).fuse(solids[2]).clean()
print(fused.volume, len(fused.solids()))
fm = mesh_of(fused, 0.2)
pl = pv.Plotter(off_screen=True, window_size=(1000,500), shape=(1,2))
pl.subplot(0,0)
pl.add_mesh(fm.clip(normal='x', origin=(0,0,0)), color="lightblue", show_edges=True)
pl.view_yz(); pl.camera.azimuth=20
pl.subplot(0,1)
pl.add_mesh(fm.clip(normal='y', origin=(0,0,0)), color="wheat", show_edges=True)
pl.view_xz()
pl.screenshot("slice.png"); display(Image("slice.png"))

# -- cell 9 -------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(1000,500), shape=(1,2))
pl.subplot(0,0); pl.add_mesh(fm, color="lightgray", show_edges=True); pl.view_xy()
pl.subplot(0,1); pl.add_mesh(fm, color="lightgray", show_edges=True); pl.view_xy(); pl.camera.roll=0; pl.camera.position=(0,0,-200)
pl.screenshot("tb.png"); display(Image("tb.png"))

# -- cell 10 ------------------------------------------------------------------------
half = Box(200,200,200).moved(Location((0,-100,0)))  # keeps y>? box centered at y=-100 covers y -200..0
cut = fused - half
cm = mesh_of(cut, 0.15)
pl = pv.Plotter(off_screen=True, window_size=(1000,500), shape=(1,2))
pl.subplot(0,0); pl.add_mesh(cm, color="steelblue", show_edges=False); pl.view_xz()
pl.subplot(0,1); pl.add_mesh(cm, color="steelblue"); pl.view_isometric()
pl.screenshot("sec.png"); display(Image("sec.png"))

# -- cell 11 ------------------------------------------------------------------------
half2 = Box(200,200,200).moved(Location((0,100,0)))
cut2 = fused - half2
cm2 = mesh_of(cut2, 0.15)
pl = pv.Plotter(off_screen=True, window_size=(1000,500), shape=(1,2))
pl.subplot(0,0); pl.add_mesh(cm2, color="indianred"); pl.view_xz()
pl.subplot(0,1); pl.add_mesh(cm2, color="indianred"); pl.view_isometric()
pl.screenshot("sec2.png"); display(Image("sec2.png"))

# -- cell 12 ------------------------------------------------------------------------
bb = fused.bounding_box()
blk = Box(bb.size.X+20, bb.size.Y+20, bb.size.Z+20).moved(Location(bb.center()))
air = blk - fused
asol = air.solids()
print(len(asol), [round(s.volume,1) for s in asol], round(blk.volume,1))

# -- cell 13 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(1200,420), shape=(1,3))
for i,s in enumerate(solids):
    c = s - half
    pl.subplot(0,i); pl.add_mesh(mesh_of(c,0.12), color=cols[i]); pl.view_isometric()
    pl.add_text(f"solid{i} V={s.volume:.0f}", font_size=9)
pl.screenshot("each.png"); display(Image("each.png"))

# -- cell 14 ------------------------------------------------------------------------
for z in [1.0,2.5,5,10,15,19,21,25,30]:
    slab = Box(200,200,0.2).moved(Location((0,0,z)))
    inter = fused & slab
    v = inter.volume if inter else 0
    print(z, round(v/0.2,1))

# -- cell 15 ------------------------------------------------------------------------
print(len(imported.children) if hasattr(imported,'children') else None)
from build123d import Shape
print(len(imported.shells()), len(imported.faces()), len(imported.compounds()))
for c in imported:
    print(c, c.label if hasattr(c,'label') else '', len(c.solids()) if hasattr(c,'solids') else '')

# -- cell 16 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(1200,800), shape=(2,3))
views = [("xy",0,0),("xz",0,1),("yz",0,2)]
for k,(v,r,c) in enumerate(views):
    pl.subplot(r,c); pl.add_mesh(fm, color="lightgray", show_edges=True, edge_color="k", line_width=1)
    getattr(pl,"view_"+v)()
for k,(az,el) in enumerate([(180,0),(0,-89),(45,30)]):
    pl.subplot(1,k); pl.add_mesh(fm, color="lightgray", show_edges=True, edge_color="k")
    pl.view_isometric(); pl.camera.azimuth=az; pl.camera.elevation=el
pl.screenshot("views.png"); display(Image("views.png"))

# -- cell 17 ------------------------------------------------------------------------
sec = fused.faces().filter_by(Plane.XY)  # not that
pln = Plane.XY.offset(10)
slab = Box(200,200,0.02).moved(Location((0,0,10)))
inter = (fused & slab)
f = inter.faces().filter_by(Plane.XY)
for ff in f:
    print(round(ff.area,2), len(ff.wires()), ff.center())

# -- cell 18 ------------------------------------------------------------------------
# The assembly is three touching solids, all fully solid — no internal cavity, no openings (confirmed 
part_mm = fused
part = part_mm.scale(0.001)
bb = part.bounding_box()
print("part bbox (m):", bb, "size", bb.size)
rad = []
for f in part.faces():
    try:
        if f.geom_type == GeomType.CYLINDER:
            rad.append(round(f.radius*1000,3))
    except Exception as e:
        pass
import collections
print("cylindrical face radii (mm):", collections.Counter(rad))

# -- cell 19 ------------------------------------------------------------------------
from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCP.TopTools import TopTools_ListOfShape
def rad_of(f):
    try:
        return f.radius if f.geom_type == GeomType.CYLINDER else None
    except Exception:
        return None
small = [f for f in part.faces() if (rad_of(f) or 9) < 0.0008]
lst = TopTools_ListOfShape()
for f in small: lst.Append(f.wrapped)
df = BRepAlgoAPI_Defeaturing(); df.SetShape(part.wrapped); df.AddFacesToRemove(lst)
df.SetRunParallel(True); df.Build()
ok = df.IsDone()
part_df = Solid(df.Shape()) if ok else part
print("defeature done:", ok, "removed", len(small), "fillet faces of r<=0.5mm")
print("volume mm^3 before/after:", round(part.volume*1e9,2), round(part_df.volume*1e9,2))
print("faces before/after:", len(part.faces()), len(part_df.faces()), "valid:", part_df.is_valid)

# -- cell 20 ------------------------------------------------------------------------
print("orig scaled valid:", part.is_valid, "solids", len(part.solids()))
print("defeatured valid:", part_df.is_valid, "solids", len(part_df.solids()))
rem = sorted(round((rad_of(f) or 0)*1000,3) for f in part_df.faces() if (rad_of(f) or 9)<0.0008)
print("remaining small radii:", rem)

# -- cell 21 ------------------------------------------------------------------------
small2 = [f for f in part_df.faces() if (rad_of(f) or 9) < 0.0008]
lst2 = TopTools_ListOfShape()
for f in small2: lst2.Append(f.wrapped)
df2 = BRepAlgoAPI_Defeaturing(); df2.SetShape(part_df.wrapped); df2.AddFacesToRemove(lst2); df2.Build()
cand = Solid(df2.Shape()) if df2.IsDone() else part_df
n_small = len([f for f in cand.faces() if (rad_of(f) or 9) < 0.0008])
part_clean = cand if (df2.IsDone() and abs(cand.volume-part_df.volume) < 1e-9) else part_df
print("pass2 done:", df2.IsDone(), "small left:", n_small, "vol mm^3:", round(part_clean.volume*1e9,2),
      "faces:", len(part_clean.faces()))

# -- cell 22 ------------------------------------------------------------------------
import os, numpy as np
case = "/home/qiuzi/.openreynolds-buildup/work/T5-20260912-125139-b2e4/t5"
for d in ["system","constant/triangulation","constant/geometry"]:
    os.makedirs(os.path.join(case,d), exist_ok=True)
def write_stl(shape, path, name, tol=5e-5, ang=0.3):
    vs, ts = shape.tessellate(tol, ang)
    V = np.array([(p.X,p.Y,p.Z) for p in vs])
    with open(path,"w") as fh:
        fh.write(f"solid {name}\n")
        for a,b,c in ts:
            p0,p1,p2 = V[a],V[b],V[c]
            n = np.cross(p1-p0,p2-p0); L=np.linalg.norm(n); n = n/L if L>0 else n
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for p in (p0,p1,p2): fh.write(f"   vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
    return len(ts)
stl = os.path.join(case,"constant/triangulation/walls.stl")
n = write_stl(part_clean, stl, "walls")
print("triangles", n, os.path.getsize(stl))

# -- cell 23 ------------------------------------------------------------------------
XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX = -0.080,0.160,-0.070,0.070,0.0,0.090
NX,NY,NZ = 60,35,23   # 4 mm background cells
hdr = lambda cls,obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
verts = [(x,y,z) for z in (ZMIN,ZMAX) for y in (YMIN,YMAX) for x in (XMIN,XMAX)]
order = [0,1,3,2,4,5,7,6]
vstr = "\n".join(f"({verts[i][0]} {verts[i][1]} {verts[i][2]})" for i in order)
bm = hdr("dictionary","blockMeshDict") + f"""
scale 1;
vertices ( {vstr} );
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    walls {{ type wall; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2)); }}
);
"""
open(os.path.join(case,"system/blockMeshDict"),"w").write(bm)
open(os.path.join(case,"system/controlDict"),"w").write(hdr("dictionary","controlDict")+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open(os.path.join(case,"system/fvSchemes"),"w").write(hdr("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open(os.path.join(case,"system/fvSolution"),"w").write(hdr("dictionary","fvSolution")+"solvers{} \n")
import subprocess
r = subprocess.run(["blockMesh","-case",case],capture_output=True,text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 24 ------------------------------------------------------------------------
shm = hdr("dictionary","snappyHexMeshDict") + """
castellatedMesh true; snap true; addLayers false;
geometry { walls { type triSurfaceMesh; file "walls.stl"; name walls; } }
castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces { walls { level (2 3); patchInfo { type wall; } } }
    refinementRegions {}
    locationInMesh (-0.060 0.050 0.070);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
    implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers{} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1; nGrow 0;
    featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
mergeTolerance 1e-6;
"""
open(os.path.join(case,"system/snappyHexMeshDict"),"w").write(shm)
import subprocess
p_shm = subprocess.Popen(["snappyHexMesh","-overwrite","-case",case],stdout=open(case+"/log.snappy","w"),stderr=subprocess.STDOUT)
print("started", p_shm.pid)
