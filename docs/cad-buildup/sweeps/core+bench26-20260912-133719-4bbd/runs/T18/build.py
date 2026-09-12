"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess
from build123d import *

R_B, H_B = 0.045, 0.120          # barrel radius, height
R_F, T_F, GAP, N_F = 0.070, 0.003, 0.006, 12   # fin radius, thickness, gap, count
PITCH = T_F + GAP
STACK = N_F*T_F + (N_F-1)*GAP
Z0 = (H_B - STACK)/2             # first fin bottom

barrel = Cylinder(R_B, H_B, align=(Align.CENTER, Align.CENTER, Align.MIN))
fins = [Pos(0,0,Z0+i*PITCH) * Cylinder(R_F, T_F, align=(Align.CENTER,Align.CENTER,Align.MIN)) for i in range(N_F)]
dome = Sphere(R_B) & Box(3*R_B, 3*R_B, R_B, align=(Align.CENTER,Align.CENTER,Align.MIN))
head = Pos(0,0,H_B) * dome

solid = barrel
for f in fins: solid = solid + f
solid = solid + head

bb = solid.bounding_box()
print("stack height", STACK, "first fin z", Z0, "last fin top", Z0+(N_F-1)*PITCH+T_F)
print("bbox", bb.min, bb.max, "vol", solid.volume)
print("n faces", len(solid.faces()))

# -- cell 2 -------------------------------------------------------------------------
from collections import defaultdict
import numpy as np
groups = defaultdict(list)
for f in solid.faces():
    gt = f.geom_type
    c = f.center()
    if gt == GeomType.SPHERE:
        groups["head"].append(f)
    elif gt == GeomType.CYLINDER:
        r = f.radius
        groups["barrel" if abs(r-R_B) < 1e-6 else "fins"].append(f)
    elif gt == GeomType.PLANE:
        if abs(c.Z) < 1e-9:
            groups["barrel"].append(f)   # closed bottom of barrel
        else:
            groups["fins"].append(f)
    else:
        groups["?"+str(gt)].append(f)
for k,v in groups.items():
    print(k, len(v), "area", round(sum(f.area for f in v),6))

# -- cell 3 -------------------------------------------------------------------------
# Face classification matches the analytic areas exactly (head 2πR²=0.012723, fins 12×0.019383=0.23260
D, ZMIN, ZMAX = 0.300, -0.150, 0.465     # duct side, 150 below cylinder base, 300 above head top
os.makedirs("constant/triSurface", exist_ok=True)
h = D/2
inlet  = Pos(0,0,ZMIN) * Rectangle(D, D)
outlet = Pos(0,0,ZMAX) * Rectangle(D, D)
side   = Rectangle(D, ZMAX-ZMIN)
walls  = [Pos(0,-h,(ZMIN+ZMAX)/2)*Rot(90,0,0)*side, Pos(0,h,(ZMIN+ZMAX)/2)*Rot(90,0,0)*side,
          Pos(-h,0,(ZMIN+ZMAX)/2)*Rot(90,0,90)*side, Pos(h,0,(ZMIN+ZMAX)/2)*Rot(90,0,90)*side]
patches = dict(groups); patches["inlet"]=[inlet.faces()[0]]; patches["outlet"]=[outlet.faces()[0]]
patches["ductWalls"]=[w.faces()[0] for w in walls]
for name, fl in patches.items():
    export_stl(Compound(children=[Solid(f.wrapped) if False else f for f in fl]), f"constant/triSurface/{name}.stl",
               tolerance=1e-4, angular_tolerance=0.2)
    print(name, len(fl), os.path.getsize(f"constant/triSurface/{name}.stl"))
print([f.center() for f in patches["ductWalls"]], [f.area for f in patches["ductWalls"]])

# -- cell 4 -------------------------------------------------------------------------
# Wall areas 0.3×0.615 ✓. Background hex mesh at 15 mm, with inlet/outlet/ductWalls as blockMesh patch
os.makedirs("system", exist_ok=True)
hdr = lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;\n"
 "runTimeModifiable true;\n")
nx=ny=20; nz=41
open("system/blockMeshDict","w").write(hdr("dictionary","blockMeshDict")+f"""
scale 1;
vertices ( (-{h} -{h} {ZMIN}) ({h} -{h} {ZMIN}) ({h} {h} {ZMIN}) (-{h} {h} {ZMIN})
           (-{h} -{h} {ZMAX}) ({h} -{h} {ZMAX}) ({h} {h} {ZMAX}) (-{h} {h} {ZMAX}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary (
  inlet {{ type patch; faces ((0 3 2 1)); }}
  outlet {{ type patch; faces ((4 5 6 7)); }}
  ductWalls {{ type wall; faces ((0 1 5 4) (2 3 7 6) (1 2 6 5) (3 0 4 7)); }}
);
mergePatchPairs ();
""")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-600:])

# -- cell 5 -------------------------------------------------------------------------
# Coarse snappy first: surface level 2 (3.75 mm) just to confirm the cut and patch naming, before refi
def snappy_dict(lvl, gapLvl, layers=0):
    geo = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in ["barrel","fins","head"])
    ref = "\n".join(f'    {n} {{ level ({lvl} {lvl}); patchInfo {{ type wall; }} }}' for n in ["barrel","fins","head"])
    return hdr("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers {"true" if layers else "false"};
geometry {{
{geo}
}};
castellatedMeshControls {{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces {{
{ref}
  }}
  refinementRegions {{
    fins {{ mode distance; levels ((0.010 {gapLvl})); }}
  }}
  locationInMesh (0.14 0.14 0.40);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.5;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #include "meshQualityDict" }}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
"""
open("system/meshQualityDict","w").write(hdr("dictionary","meshQualityDict")+
  '#includeEtc "caseDicts/meshQualityDict"\n')
open("system/snappyHexMeshDict","w").write(snappy_dict(2,2))
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:])

# -- cell 6 -------------------------------------------------------------------------
print(r.returncode); print(r.stderr[-2000:])

# -- cell 7 -------------------------------------------------------------------------
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} relaxationFactors{}\n")
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1200:], r.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
def write_snappy(lvl, gapLvl):
    s = snappy_dict(lvl, gapLvl).replace('#include "meshQualityDict"',
        '#include "meshQualityDict" nSmoothScale 4; errorReduction 0.75;')
    open("system/snappyHexMeshDict","w").write(s)
write_snappy(2,2)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
# Meshed at 46 k cells. Look at it: a y=0 slice of the mesh plus the patch surfaces.
import pyvista as pv
pv.start_xvfb() if False else None
pv.global_theme.window_size=[1200,700]
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read()
internal = m["internalMesh"]
bnd = m["boundary"]
print(list(bnd.keys()))
p = pv.Plotter(off_screen=True, shape=(1,2))
p.subplot(0,0)
p.add_mesh(internal.slice(normal='y'), show_edges=True, color='lightblue')
p.camera_position='xz'
p.subplot(0,1)
cols=dict(barrel='red',fins='orange',head='green',ductWalls='grey',inlet='blue',outlet='purple')
for k in bnd.keys():
    p.add_mesh(bnd[k], color=cols.get(k,'w'), opacity=0.35 if k in('ductWalls','inlet','outlet') else 1.0)
p.camera_position='xz'
p.screenshot("look1.png")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look1.png"))

# -- cell 11 ------------------------------------------------------------------------
# Shape is right (barrel red between fins, dome green, duct/inlet/outlet named). The 6 mm gaps are onl
import time
write_snappy(3,3)
t=time.time(); r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, round(time.time()-t,1)); print(r.stdout[-900:]); print(r.stderr[-400:])

# -- cell 12 ------------------------------------------------------------------------
# snappy ran on the already-snapped mesh (`-overwrite` replaced it) — rebuild the background mesh firs
subprocess.run(["blockMesh"],capture_output=True,text=True)
t=time.time(); r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, round(time.time()-t,1)); print(r.stdout[-1000:]); print(r.stderr[-300:])

# -- cell 13 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-2500:])

# -- cell 14 ------------------------------------------------------------------------
# Mesh OK. Volume check: 0.3²×0.615 − solid 0.00127941 = 0.0540706 vs mesh 0.054071 ✓. Now verify patc
import numpy as np
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0); m=rd.read()
bnd=m["boundary"]; internal=m["internalMesh"]
exact=dict(barrel=0.030112, fins=0.232604, head=0.012723, inlet=0.09, outlet=0.09, ductWalls=4*0.1845)
for k in bnd.keys():
    a=bnd[k].compute_cell_sizes(length=False,volume=False)["Area"].sum()
    print(f"{k:10s} area {a:.6f}  target {exact[k]:.6f}")
# radial extents of fins / barrel patches
for k in ["fins","barrel","head"]:
    pts=bnd[k].points; rr=np.hypot(pts[:,0],pts[:,1])
    print(k,"r range",rr.min().round(5),rr.max().round(5),"z range",pts[:,2].min().round(4),pts[:,2].max().round(4))
# cells across a gap: sample cell centres in a vertical line at r=0.058, x axis
cc=internal.cell_centers().points
sel=(np.abs(cc[:,1])<0.002)&(cc[:,0]>0.050)&(cc[:,0]<0.065)&(cc[:,2]>0.0)&(cc[:,2]<0.120)
z=np.sort(cc[sel,2]); print("z of cells in fin region (r~0.058):", np.unique(z.round(4))[:24])
