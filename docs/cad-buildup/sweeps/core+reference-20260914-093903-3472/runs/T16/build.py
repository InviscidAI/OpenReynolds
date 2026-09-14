"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, os, numpy as np, inspect
import build123d as bd
print(os.getcwd(), os.listdir('.'))
print(inspect.signature(bd.revolve))
print(inspect.signature(bd.Spline))

# -- cell 2 -------------------------------------------------------------------------
print([m for m in dir(bd.Edge) if 'rev' in m.lower()])
print([m for m in dir(bd.Face) if 'rev' in m.lower() or 'sweep' in m.lower()])
print(inspect.signature(bd.Face.revolve) if hasattr(bd.Face,'revolve') else '')

# -- cell 3 -------------------------------------------------------------------------
import build123d as bd
from build123d import Vector, Axis

L = 0.900               # duct length (m), given
# outer cowl inner surface radii (m), given
Ro = [0.450, 0.430, 0.410]
# core cowl outer surface radii (m), given
Ri = [0.280, 0.310, 0.250]
zs = [0.0, L/2, L]

outer_edge = bd.Spline(*[(Ro[i], 0, zs[i]) for i in range(3)]).edge()
inner_edge = bd.Spline(*[(Ri[i], 0, zs[i]) for i in range(3)]).edge()
inlet_edge = bd.Line((Ri[0], 0, 0.0), (Ro[0], 0, 0.0)).edge()
outlet_edge = bd.Line((Ri[2], 0, L), (Ro[2], 0, L)).edge()

faces = {
    "outerCowl": bd.Face.revolve(outer_edge, 360, Axis.Z),
    "coreCowl":  bd.Face.revolve(inner_edge, 360, Axis.Z),
    "inlet":     bd.Face.revolve(inlet_edge, 360, Axis.Z),
    "outlet":    bd.Face.revolve(outlet_edge, 360, Axis.Z),
}
for k, f in faces.items():
    print(k, "area=%.5f" % f.area, "bbox", [round(v,4) for v in f.bounding_box().size])

# -- cell 4 -------------------------------------------------------------------------
import numpy as np
def radii(edge, n=10):
    out=[]
    for t in np.linspace(0,1,n):
        p = edge.position_at(t)
        out.append((round(p.Z,4), round((p.X**2+p.Y**2)**0.5,5)))
    return out
print("outer:", radii(outer_edge))
print("inner:", radii(inner_edge))
for e,name,tgt in ((outer_edge,'outer',Ro),(inner_edge,'inner',Ri)):
    for z,ti in zip([0,L/2,L],tgt):
        # find param with that Z
        ts=np.linspace(0,1,2001); zz=np.array([e.position_at(t).Z for t in ts])
        t=ts[np.argmin(abs(zz-z))]; p=e.position_at(t)
        print(name,"z=%.3f r=%.5f target %.3f"%(p.Z,(p.X**2+p.Y**2)**0.5,ti))

# -- cell 5 -------------------------------------------------------------------------
import os
profile = bd.make_face(bd.Wire([outer_edge, inner_edge, inlet_edge, outlet_edge]))
solid = bd.revolve(profile, Axis.Z, 360)
print("solid volume %.6f m^3" % solid.volume, "valid:", solid.is_valid)
os.makedirs("constant/triSurface", exist_ok=True)
for k, f in faces.items():
    bd.export_stl(f, f"constant/triSurface/{k}.stl", tolerance=5e-5, angular_tolerance=0.1)
print(sorted(os.listdir("constant/triSurface")))

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
ms = {k: pv.read(f"constant/triSurface/{k}.stl") for k in faces}
for k,m in ms.items(): print(k, m.n_points, m.n_cells, np.round(m.bounds,4))
union = ms['outerCowl'].merge(list(ms.values())[1:]).clean(tolerance=1e-6)
fe = union.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges of union:", fe.n_cells)

# -- cell 7 -------------------------------------------------------------------------
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_REVERSED
import numpy as np, pyvista as pv

# match solid faces to the named construction faces by area (areas are all distinct)
sfaces = solid.faces()
name_by_area = {round(f.area,6): k for k, f in faces.items()}
assert len(name_by_area) == 4
BRepMesh_IncrementalMesh(solid.wrapped, 2e-4, False, 0.2, True)

def face_poly(f):
    loc = __import__("OCP.TopLoc", fromlist=["TopLoc_Location"]).TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    trsf = loc.Transformation()
    pts = np.array([[(lambda p: (p.X(),p.Y(),p.Z()))(tri.Node(i+1).Transformed(trsf))] for i in range(tri.NbNodes())]).reshape(-1,3)
    tris=[]
    rev = f.wrapped.Orientation() == TopAbs_REVERSED
    for i in range(tri.NbTriangles()):
        a,b,c = tri.Triangle(i+1).Get()
        tris.append([a-1,c-1,b-1] if rev else [a-1,b-1,c-1])
    tris=np.array(tris)
    return pv.PolyData(pts, np.hstack([np.full((len(tris),1),3), tris]).ravel())

polys={}
for f in sfaces:
    k = name_by_area[round(f.area,6)]
    p = face_poly(f); polys[k]=p
    p.save(f"constant/triSurface/{k}.stl", binary=True)
    print(k, p.n_points, p.n_cells)

# -- cell 8 -------------------------------------------------------------------------
union = polys['outerCowl'].merge([polys[k] for k in ['coreCowl','inlet','outlet']]).clean(tolerance=1e-8)
fe = union.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("union free edges:", fe.n_cells, "volume %.6f"%union.volume, "vs solid %.6f"%solid.volume)
pl = pv.Plotter(off_screen=True, window_size=(900,600))
cols={'outerCowl':'lightblue','coreCowl':'salmon','inlet':'green','outlet':'gold'}
clip = {k: polys[k].clip(normal='y', origin=(0,0,0)) for k in polys}
for k,p in clip.items(): pl.add_mesh(p, color=cols[k], show_edges=True, line_width=0.3)
pl.camera_position='xz'; pl.add_axes()
pl.screenshot("geom.png")
print("ok")

# -- cell 9 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 10 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(900,600))
for k,p in polys.items():
    pl.add_mesh(p.clip(normal=(0,1,0), origin=(0,0,0), invert=False), color=cols[k], show_edges=True, line_width=0.3)
pl.camera_position='xz'; pl.add_axes(); pl.screenshot("geom_cut.png")
display(Image("geom_cut.png"))

# -- cell 11 ------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True)
def w(p,s): open(p,"w").write(s)
head = lambda cls,obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
w("system/controlDict", head("dictionary","controlDict")+"""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;
writePrecision 6; runTimeModifiable true;
""")
# background block: cubic-ish cells of ~0.046 m
XY, Z0, Z1 = 0.46, -0.06, 0.96
NX, NZ = 20, 22
w("system/blockMeshDict", head("dictionary","blockMeshDict")+f"""
scale 1;
vertices
(
 (-{XY} -{XY} {Z0}) ({XY} -{XY} {Z0}) ({XY} {XY} {Z0}) (-{XY} {XY} {Z0})
 (-{XY} -{XY} {Z1}) ({XY} -{XY} {Z1}) ({XY} {XY} {Z1}) (-{XY} {XY} {Z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({NX} {NX} {NZ}) simpleGrading (1 1 1));
edges ();
boundary ( background {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""")
import subprocess
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.returncode, r.stdout[-400:], r.stderr[-300:])

# -- cell 12 ------------------------------------------------------------------------
names = ["outerCowl","coreCowl","inlet","outlet"]
w("system/surfaceFeatureExtractDict", head("dictionary","surfaceFeatureExtractDict")+
  "".join(f"""{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}\n""" for n in names))
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.returncode, r.stdout[-600:], r.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
LVL = 1
geo = "".join(f"""    {n}.stl {{ type triSurfaceMesh; name {n}; }}\n""" for n in names)
refs = "".join(f"""        {n} {{ level ({LVL} {LVL}); patchInfo {{ type {'patch' if n in ('inlet','outlet') else 'wall'}; }} }}\n""" for n in names)
feats = "".join(f"""            {{ file "{n}.eMesh"; level {LVL}; }}\n""" for n in names)
w("system/snappyHexMeshDict", head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geo}}}
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features (
{feats}    );
    refinementSurfaces
    {{
{refs}    }}
    refinementRegions {{}}
    locationInMesh (0.37 0.0 0.45);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-1500:], r.stderr[-500:])

# -- cell 14 ------------------------------------------------------------------------
w("system/fvSchemes", head("dictionary","fvSchemes")+"""
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}
laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}
snGradSchemes{default corrected;}
""")
w("system/fvSolution", head("dictionary","fvSolution")+"solvers{} relaxationFactors{}\n")
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-1200:], r.stderr[-400:])

# -- cell 15 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True); print(r.stdout[-2500:])

# -- cell 16 ------------------------------------------------------------------------
LVL = 2   # refinement level on all four patch surfaces (final)
refs = "".join(f"""        {n} {{ level ({LVL} {LVL}); patchInfo {{ type {'patch' if n in ('inlet','outlet') else 'wall'}; }} }}\n""" for n in names)
feats = "".join(f"""            {{ file "{n}.eMesh"; level {LVL}; }}\n""" for n in names)
dict_txt = open("system/snappyHexMeshDict").read()
import re
dict_txt = re.sub(r"    features \(.*?\);", "    features (\n"+feats+"    );", dict_txt, flags=re.S)
dict_txt = re.sub(r"    refinementSurfaces\n    \{.*?\n    \}", "    refinementSurfaces\n    {\n"+refs+"    }", dict_txt, flags=re.S)
w("system/snappyHexMeshDict", dict_txt)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-500:])

# -- cell 17 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
out=r.stdout
print(out[out.index("Mesh stats"):out.index("Checking topology")])
print(out[out.index("Checking patch topology"):out.index("Checking faceZone")])
print(out[out.index("Overall domain"):])

# -- cell 18 ------------------------------------------------------------------------
from OCP.BRepTools import BRepTools
BRepTools.Clean_s(solid.wrapped)
BRepMesh_IncrementalMesh(solid.wrapped, 2e-5, False, 0.05, True)   # finer STL faceting
polys = {}
for f in solid.faces():
    k = name_by_area[round(f.area,6)]
    p = face_poly(f); polys[k]=p
    p.save(f"constant/triSurface/{k}.stl", binary=True)
union = polys['outerCowl'].merge([polys[k] for k in ['coreCowl','inlet','outlet']]).clean(tolerance=1e-9)
fe = union.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print({k:p.n_cells for k,p in polys.items()}, "free edges", fe.n_cells, "STL volume %.6f vs exact %.6f"%(union.volume, solid.volume))

# -- cell 19 ------------------------------------------------------------------------
subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:])
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True); out=r.stdout
print(out[out.index("Overall domain"):])

# -- cell 20 ------------------------------------------------------------------------
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.time_values else None
rd.enable_all_patch_arrays()
mb = rd.read()
bnd = mb["boundary"]
for k in names:
    p = bnd[k]; c = p.cell_centers().points; r = np.hypot(c[:,0],c[:,1])
    print(f"{k:10s} z:[{p.bounds[4]:.3f},{p.bounds[5]:.3f}] r:[{r.min():.4f},{r.max():.4f}] faces={p.n_cells}")
inner = mb["internalMesh"]
sl = inner.slice(normal=(0,1,0))
pl = pv.Plotter(off_screen=True, window_size=(1000,600))
pl.add_mesh(sl, show_edges=True, color='white', line_width=0.5)
pl.camera_position='xz'; pl.screenshot("mesh_cut.png"); display(Image("mesh_cut.png"))

# -- cell 21 ------------------------------------------------------------------------
# Mesh is built and checks out. Cut plane shows the annulus with the outer wall tapering 450→410 mm an
print("final: cells=", inner.n_cells, " exact solid volume %.6f"%solid.volume)
