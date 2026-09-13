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

L = 0.900
OUT_R = [0.450, 0.430, 0.410]
CORE_R = [0.280, 0.310, 0.250]
ZS = [0.0, L/2, L]

outer_pts = [(OUT_R[i], 0, ZS[i]) for i in range(3)]
core_pts  = [(CORE_R[i], 0, ZS[i]) for i in range(3)]

e_out = Spline(*outer_pts)
e_core = Spline(*core_pts)
e_in  = Line(core_pts[0], outer_pts[0])   # fan-exit inlet
e_nzl = Line(core_pts[2], outer_pts[2])   # nozzle outlet
face = make_face([e_out, e_core, e_in, e_nzl])
solid = revolve(face, Axis.Z, 360)
print(solid.volume, solid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
for i,f in enumerate(solid.faces()):
    c = f.center()
    print(i, type(f.geometry) if hasattr(f,'geometry') else '', f.geom_type, round(f.area,4), [round(v,4) for v in c], f.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
import os
from build123d import Mesher, export_stl, Compound
faces = solid.faces()
patch_faces = {"outerCowl": faces[0], "coreCowl": faces[2], "nozzleOutlet": faces[1], "fanInlet": faces[3]}
os.makedirs("constant/triSurface", exist_ok=True)
for name, f in patch_faces.items():
    export_stl(Compound(children=[f]), f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.1, ascii_format=True)
    # rename solid label inside ascii stl to patch name
    p = f"constant/triSurface/{name}.stl"
    txt = open(p).read().replace("solid ", f"solid {name} ",1) if False else open(p).read()
    lines = txt.splitlines()
    lines[0] = f"solid {name}"; lines[-1] = f"endsolid {name}"
    open(p,"w").write("\n".join(lines)+"\n")
    print(name, f.area, os.path.getsize(p))

# -- cell 4 -------------------------------------------------------------------------
import numpy as np
def r_at(edge, z):
    ts = np.linspace(0,1,2001)
    pts = np.array([tuple(edge@t) for t in ts])
    i = np.argmin(abs(pts[:,2]-z))
    return np.hypot(pts[i,0],pts[i,1]), pts[i,2]
for nm,e,targ in [("outer",e_out,OUT_R),("core",e_core,CORE_R)]:
    for z,t in zip(ZS,targ):
        r,zz = r_at(e,z); print(f"{nm} z={z:.3f} r={r*1000:.2f} mm (asked {t*1000:.0f})")
    pts = np.array([tuple(e@t) for t in np.linspace(0,1,2001)])
    print(nm,"r range mm", 1000*np.hypot(pts[:,0],pts[:,1]).min(), 1000*np.hypot(pts[:,0],pts[:,1]).max())

# -- cell 5 -------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
head = lambda cls,obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
open("system/controlDict","w").write(head("dictionary","controlDict")+
"application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes")+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution")+"solvers{} SIMPLE{}\n")
N=(24,24,26)
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ((-0.47 -0.47 -0.06)(0.47 -0.47 -0.06)(0.47 0.47 -0.06)(-0.47 0.47 -0.06)(-0.47 -0.47 0.96)(0.47 -0.47 0.96)(0.47 0.47 0.96)(-0.47 0.47 0.96));
blocks (hex (0 1 2 3 4 5 6 7) ({N[0]} {N[1]} {N[2]}) simpleGrading (1 1 1));
edges (); boundary (); mergePatchPairs ();
""")
import subprocess
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 6 -------------------------------------------------------------------------
import subprocess
names=["outerCowl","coreCowl","fanInlet","nozzleOutlet"]
sfe = "".join(f"{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}\n" for n in names)
open("system/surfaceFeatureExtractDict","w").write(head("dictionary","surfaceFeatureExtractDict")+sfe)
geom = "".join(f"    {n}.stl {{ type triSurfaceMesh; name {n}; }}\n" for n in names)
feats = "".join(f"            {{ file \"{n}.eMesh\"; level 1; }}\n" for n in names)
refs = "".join(f"            {n} {{ level (1 1); patchInfo {{ type {'patch' if n in ('fanInlet','nozzleOutlet') else 'wall'}; }} }}\n" for n in names)
open("system/snappyHexMeshDict","w").write(head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}}}
castellatedMeshControls {{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features (
{feats}    );
    refinementSurfaces {{
{refs}    }}
    refinementRegions {{}}
    locationInMesh (0.37 0.0 0.45);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
""")
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.returncode, r.stdout[-800:])

# -- cell 7 -------------------------------------------------------------------------
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2500:])

# -- cell 9 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
subprocess.run(["foamToVTK","-latestTime","-ascii"],capture_output=True,text=True)
import glob; print(glob.glob("VTK/*"))

# -- cell 10 ------------------------------------------------------------------------
m = pv.read("VTK/t16_0.vtm")
internal = m[0]
clip = internal.clip(normal='y', origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(1100,700))
p.add_mesh(clip, show_edges=True, color='lightblue')
p.camera_position='xz'; p.show(screenshot="cut.png")

# -- cell 11 ------------------------------------------------------------------------
sl = internal.slice(normal='y', origin=(0,0,0.45))
p = pv.Plotter(off_screen=True, window_size=(1100,600))
p.add_mesh(sl, show_edges=True, color='lightblue')
p.view_xz(); p.camera.up=(0,0,1); p.show(screenshot="slice.png")

# -- cell 12 ------------------------------------------------------------------------
N=(48,48,52)
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ((-0.47 -0.47 -0.06)(0.47 -0.47 -0.06)(0.47 0.47 -0.06)(-0.47 0.47 -0.06)(-0.47 -0.47 0.96)(0.47 -0.47 0.96)(0.47 0.47 0.96)(-0.47 0.47 0.96));
blocks (hex (0 1 2 3 4 5 6 7) ({N[0]} {N[1]} {N[2]}) simpleGrading (1 1 1));
edges (); boundary (); mergePatchPairs ();
""")
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-600:])

# -- cell 13 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-1400:])

# -- cell 14 ------------------------------------------------------------------------
subprocess.run(["foamToVTK","-latestTime","-ascii"],capture_output=True,text=True)
m = pv.read("VTK/t16_0.vtm"); 
def walk(b,acc):
    for i in range(b.n_blocks):
        x=b[i]
        if isinstance(x,pv.MultiBlock): walk(x,acc)
        elif x is not None: acc.append((b.get_block_name(i),x))
    return acc
blocks = walk(m,[])
cad_area = {"outerCowl":2.4340,"coreCowl":1.6795,"fanInlet":0.38987,"nozzleOutlet":0.33175}
for nm,x in blocks:
    a = x.extract_surface().compute_cell_sizes(length=False,volume=False)["Area"].sum() if nm!="internalMesh" else None
    print(nm, x.n_cells, round(a,5) if a else "", "CAD", cad_area.get(nm,""))

# -- cell 15 ------------------------------------------------------------------------
import numpy as np
bd = dict(blocks)
for nm,targ in [("outerCowl",OUT_R),("coreCowl",CORE_R)]:
    P = bd[nm].points; r = np.hypot(P[:,0],P[:,1])
    for z,t in zip(ZS,targ):
        sel = abs(P[:,2]-z)<0.004
        print(f"{nm} z={z:.2f} mesh r={1000*r[sel].mean():.2f} mm (asked {1000*t:.0f})")
for nm in ["fanInlet","nozzleOutlet"]:
    P=bd[nm].points; r=np.hypot(P[:,0],P[:,1]); print(nm,"z",P[:,2].min(),P[:,2].max(),"r mm",round(1000*r.min(),1),round(1000*r.max(),1))

# -- cell 16 ------------------------------------------------------------------------
sl = bd["internal"].slice(normal='y', origin=(0,0,0.45))
p = pv.Plotter(off_screen=True, window_size=(1000,600)); p.add_mesh(sl, show_edges=True, color='lightblue')
p.view_xz(); p.camera.up=(0,0,1); p.show(screenshot="slice_final.png")
