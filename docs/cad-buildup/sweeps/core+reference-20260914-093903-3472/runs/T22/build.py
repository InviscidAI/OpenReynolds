"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, math, os
print(bd.__version__ if hasattr(bd,'__version__') else 'ok')
os.makedirs("constant/triSurface", exist_ok=True)

# --- parameters (all from the request, in METRES) ---
R_out   = 0.320/2      # 320 mm OD
R_in    = 0.190/2      # 190 mm ID
GAP     = 0.012        # 12 mm between friction faces
N_VANE  = 36
W_VANE  = 0.008        # 8 mm vane width (tangential)

annulus = bd.extrude(bd.Plane.XY * (bd.Circle(R_out) - bd.Circle(R_in)), GAP)
vane0 = bd.Box(2*R_out, W_VANE, GAP, align=(bd.Align.CENTER,)*3).moved(bd.Pos(0,0,GAP/2))
vanes = [vane0.moved(bd.Rot(0,0,i*360/N_VANE)) for i in range(N_VANE)]
fluid = annulus - vanes
print(fluid.volume, len(fluid.faces()), fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
import collections
def clas(f):
    c = f.center(); r = math.hypot(c.X, c.Y)
    if isinstance(f.geom_type, str): gt = f.geom_type
    else: gt = str(f.geom_type)
    return gt, round(c.Z,4), round(r,4)
cnt = collections.Counter(clas(f) for f in fluid.faces())
for k,v in sorted(cnt.items()): print(k,v)

# -- cell 3 -------------------------------------------------------------------------
fl = fluid.faces()
groups = {
 "friction_bottom": [f for f in fl if str(f.geom_type)=="GeomType.PLANE" and abs(f.center().Z)<1e-9],
 "friction_top":    [f for f in fl if str(f.geom_type)=="GeomType.PLANE" and abs(f.center().Z-GAP)<1e-9],
 "vanes":           [f for f in fl if str(f.geom_type)=="GeomType.PLANE" and abs(f.center().Z-GAP/2)<1e-9],
 "inlet":           [f for f in fl if str(f.geom_type)=="GeomType.CYLINDER" and math.hypot(f.center().X,f.center().Y)<0.12],
 "outlet":          [f for f in fl if str(f.geom_type)=="GeomType.CYLINDER" and math.hypot(f.center().X,f.center().Y)>0.12],
}
for name, faces in groups.items():
    sh = bd.Compound(children=faces)
    bd.export_stl(sh, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.1, ascii_format=True)
    # rename solid to patch name
    p=f"constant/triSurface/{name}.stl"; t=open(p).read().replace("solid ","solid %s "%name,1)
    open(p,"w").write(t)
    print(name, len(faces), sum(f.area for f in faces), os.path.getsize(p))

# -- cell 4 -------------------------------------------------------------------------
import textwrap, subprocess, os
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
head = lambda cls,obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
open("system/controlDict","w").write(head("dictionary","controlDict")+
 "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes")+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution")+"solvers{} \n")
BASE=0.0025
xmin,xmax=-0.1650,0.1650; zmin,zmax=-0.005,0.017
nx=int(round((xmax-xmin)/BASE)); nz=int(round((zmax-zmin)/BASE))
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ( ({xmin} {xmin} {zmin}) ({xmax} {xmin} {zmin}) ({xmax} {xmax} {zmin}) ({xmin} {xmax} {zmin})
           ({xmin} {xmin} {zmax}) ({xmax} {xmin} {zmax}) ({xmax} {xmax} {zmax}) ({xmin} {xmax} {zmax}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1) );
edges (); boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} ); mergePatchPairs ();
""")
print(nx,nz, subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 5 -------------------------------------------------------------------------
names=["friction_bottom","friction_top","vanes","inlet","outlet"]
geom="\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs="\n".join(f'      {n} {{ level (1 1); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in names)
open("system/snappyHexMeshDict","w").write(head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}};
castellatedMeshControls
{{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces
  {{
{refs}
  }}
  refinementRegions {{}};
  locationInMesh (0.12701 0.011113 0.006);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers{{}} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
""")
ang=math.radians(5); print(0.1275*math.cos(ang),0.1275*math.sin(ang))
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.stdout[-1500:])

# -- cell 6 -------------------------------------------------------------------------
pts=[]
for i in range(N_VANE):
    a=math.radians((i+0.5)*360/N_VANE); r=0.1275
    pts.append(f"      ( ({r*math.cos(a):.6f} {r*math.sin(a):.6f} 0.006) fluid )")
locs="locationsInMesh (\n"+"\n".join(pts)+"\n  );\n  singleRegionName true;"
d=open("system/snappyHexMeshDict").read().replace("locationInMesh (0.12701 0.011113 0.006);", locs)
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.stdout[-900:], r.stderr[-500:])

# -- cell 7 -------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2600:])

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pv.OFF_SCREEN=True
os.makedirs("VTK",exist_ok=True)
subprocess.run(["foamToVTK","-latestTime","-no-internal"],capture_output=True,text=True)
import glob
print(glob.glob("VTK/**/*.vt*",recursive=True)[:10])

# -- cell 9 -------------------------------------------------------------------------
cols={"vanes":"orange","inlet":"green","outlet":"red","friction_top":"lightblue","friction_bottom":"grey"}
p=pv.Plotter(off_screen=True,window_size=(900,800))
for n,c in cols.items():
    m=pv.read(f"VTK/t22_0/boundary/{n}.vtp")
    if n=="friction_top": m=m.clip(normal='y',origin=(0,0,0))
    p.add_mesh(m,color=c,show_edges=(n!="friction_top"),line_width=0.3)
p.camera_position=[(0.3,-0.3,0.35),(0,0,0.006),(0,0,1)]
p.screenshot("view.png"); print("ok")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 11 ------------------------------------------------------------------------
import numpy as np
def rad(m): 
    p=m.points; return np.hypot(p[:,0],p[:,1])
for n in ["inlet","outlet"]:
    m=pv.read(f"VTK/t22_0/boundary/{n}.vtp"); r=rad(m); print(n, r.min(), r.max())
# vane width from CAD: the two side faces of the vane at angle 0 are at y=+-0.004
vf=groups["vanes"]; ys=sorted(set(round(f.center().Y,6) for f in vf if abs(f.center().X-0.1275)<0.02))
print("vane side face y offsets near theta=0:", ys, "-> width", ys[-1]-ys[0])
print("passages:",36, "gap z:", 0.012, "cells:",156364)
