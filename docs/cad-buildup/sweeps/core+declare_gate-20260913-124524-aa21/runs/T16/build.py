"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, os
from build123d import *

L = 0.900          # duct length, m
R_OUT = (0.450, 0.430, 0.410)   # outer cowl inner surface: fan exit, mid, nozzle
R_CORE = (0.280, 0.310, 0.250)  # core cowl outer surface

zs = (0.0, L/2, L)
outer_edge = Spline([(R_OUT[i], 0, zs[i]) for i in range(3)])
core_edge  = Spline([(R_CORE[i], 0, zs[i]) for i in range(3)])

for name, e, target in (("outer", outer_edge, R_OUT), ("core", core_edge, R_CORE)):
    got = [e @ t for t in (0.0, 0.5, 1.0)]
    print(name, [(round(p.X,4), round(p.Z,4)) for p in got], "target", target)
print("len outer", outer_edge.length, "len core", core_edge.length)

# -- cell 2 -------------------------------------------------------------------------
print([m for m in dir(Face) if 'make' in m or 'revolve' in m])
print([m for m in dir(Wire) if 'circle' in m])

# -- cell 3 -------------------------------------------------------------------------
from build123d import *
import math

srf_outer = Face.revolve(outer_edge, 360, Axis.Z)
srf_core  = Face.revolve(core_edge, 360, Axis.Z)
srf_inlet  = Face(Wire.make_circle(R_OUT[0], Plane.XY)).make_holes(
                  [Wire.make_circle(R_CORE[0], Plane.XY)])
srf_outlet = Face(Wire.make_circle(R_OUT[2], Plane(origin=(0,0,L)))).make_holes(
                  [Wire.make_circle(R_CORE[2], Plane(origin=(0,0,L)))])

profile = Face(Wire([outer_edge,
                     Edge.make_line((R_OUT[2],0,L), (R_CORE[2],0,L)),
                     core_edge,
                     Edge.make_line((R_CORE[0],0,0), (R_OUT[0],0,0))]))
solid = revolve(profile, Axis.Z, 360)

print("areas", [round(f.area,5) for f in (srf_outer, srf_core, srf_inlet, srf_outlet)])
print("inlet area check", round(math.pi*(R_OUT[0]**2-R_CORE[0]**2),5))
print("outlet area check", round(math.pi*(R_OUT[2]**2-R_CORE[2]**2),5))
print("solid volume", solid.volume, "bbox", solid.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
from scipy.optimize import brentq
def r_at_z(edge, zt):
    f = lambda t: (edge @ t).Z - zt
    t = brentq(f, 0, 1)
    return (edge @ t).X
for nm, e, tgt in (("outer", outer_edge, 0.430), ("core", core_edge, 0.310)):
    print(nm, "r(z=0.45) =", round(r_at_z(e, 0.45), 6), " target", tgt)
# monotone / smoothness sample
for z in [0,0.15,0.3,0.45,0.6,0.75,0.9]:
    print(round(z,2), round(r_at_z(outer_edge,z),4), round(r_at_z(core_edge,z),4))

# -- cell 5 -------------------------------------------------------------------------
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)
patches = {"outerCowl": srf_outer, "coreCowl": srf_core, "inlet": srf_inlet, "outlet": srf_outlet}
for nm, f in patches.items():
    p = f"constant/triSurface/{nm}.stl"
    export_stl(f, p, tolerance=2e-4, angular_tolerance=0.05, ascii_format=True)
    txt = open(p).read()
    n = txt.count("facet normal")
    txt = txt.replace("solid \n", f"solid {nm}\n", 1).replace("endsolid \n", f"endsolid {nm}\n", 1)
    open(p, "w").write(txt)
    print(nm, "tris", n, "| first line:", txt.split("\n")[0])

# -- cell 6 -------------------------------------------------------------------------
# The four patch files are separately tessellated, so their shared rim vertices may not match, which w
for i, f in enumerate(solid.faces()):
    c = f.center()
    print(i, "area", round(f.area,5), "type", f.geom_type, "center", (round(c.X,3),round(c.Y,3),round(c.Z,3)))

# -- cell 7 -------------------------------------------------------------------------
# Four faces, areas matching the four constructed patch surfaces exactly. I map each by area equality 
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
import numpy as np

name_by_area = {round(srf_outer.area,6):"outerCowl", round(srf_core.area,6):"coreCowl",
                round(srf_inlet.area,6):"inlet", round(srf_outlet.area,6):"outlet"}
BRepMesh_IncrementalMesh(solid.wrapped, 2e-4, False, 0.05, True)

def write_face_stl(face, name):
    loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    trsf = loc.Transformation()
    pts = [tri.Node(i+1).Transformed(trsf) for i in range(tri.NbNodes())]
    rev = face.wrapped.Orientation() == TopAbs_REVERSED
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for k in range(tri.NbTriangles()):
            a,b,c = tri.Triangle(k+1).Get()
            if rev: a,c = c,a
            P = [np.array([pts[i-1].X(),pts[i-1].Y(),pts[i-1].Z()]) for i in (a,b,c)]
            n = np.cross(P[1]-P[0], P[2]-P[0]); L2=np.linalg.norm(n)
            n = n/L2 if L2>0 else n
            fh.write("facet normal %g %g %g\n outer loop\n"%tuple(n))
            for p in P: fh.write("  vertex %.8f %.8f %.8f\n"%tuple(p))
            fh.write(" endloop\nendfacet\n")
        fh.write(f"endsolid {name}\n")
    return tri.NbTriangles()

for f in solid.faces():
    nm = name_by_area[round(f.area,6)]
    print(nm, write_face_stl(f, nm))

# -- cell 8 -------------------------------------------------------------------------
import subprocess
open("/tmp/union.stl","w").write("".join(open(f"constant/triSurface/{n}.stl").read()
        for n in ["outerCowl","coreCowl","inlet","outlet"]))
r = subprocess.run(["surfaceCheck","/tmp/union.stl"],capture_output=True,text=True)
print("\n".join(l for l in r.stdout.split("\n") if any(k in l for k in
      ["Surface","edges","triangles","closed","region","normal","illegal","Number"])))

# -- cell 9 -------------------------------------------------------------------------
# Union is closed, single zone, consistent normals, 4 regions. Now the case files: a background box an
import os, textwrap
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
hdr = lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"

open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 8;\n"
 "runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} relaxationFactors{}\n")

XY, Z0, Z1, N = 0.47, -0.06, 0.96, 32
open("system/blockMeshDict","w").write(hdr("dictionary","blockMeshDict")+f"""
scale 1;
vertices
(
 (-{XY} -{XY} {Z0}) ({XY} -{XY} {Z0}) ({XY} {XY} {Z0}) (-{XY} {XY} {Z0})
 (-{XY} -{XY} {Z1}) ({XY} -{XY} {Z1}) ({XY} {XY} {Z1}) (-{XY} {XY} {Z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({N} {N} {int(round((Z1-Z0)/(2*XY/N)))}) simpleGrading (1 1 1) );
edges (); boundary (); mergePatchPairs ();
""")
import subprocess
r = subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.stdout[-500:], r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
names = ["outerCowl","coreCowl","inlet","outlet"]
open("system/surfaceFeatureExtractDict","w").write(hdr("dictionary","surfaceFeatureExtractDict")+
 "".join(f'{n}.stl {{ extractionMethod extractFromSurface; includedAngle 150; writeObj no; }}\n' for n in names))

geom = "".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}\n' for n in names)
refs = "".join(f'      {n} {{ level (1 1); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}\n' for n in names)
feats = "".join(f'        {{ file "{n}.eMesh"; level 1; }}\n' for n in names)

open("system/snappyHexMeshDict","w").write(hdr("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}}}
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
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.4;
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
import subprocess
r = subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True)
print(r.stdout[-800:], r.stderr[-300:])

# -- cell 11 ------------------------------------------------------------------------
import subprocess
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:]); print("ERR", r.stderr[-400:])

# -- cell 12 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-allTopology"],capture_output=True,text=True)
print(r.stdout[-2600:])

# -- cell 13 ------------------------------------------------------------------------
# Mesh OK, 55.8k cells, volume 0.2749 vs exact 0.27639 (0.5% low, as expected from coarse faceting). N
import pyvista as pv, numpy as np
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read()
internal = m["internalMesh"]
bnd = m["boundary"]
cols = {"outerCowl":"steelblue","coreCowl":"indianred","inlet":"green","outlet":"orange"}
p = pv.Plotter(off_screen=True, window_size=(1100,500), shape=(1,2))
p.subplot(0,0)
for k,c in cols.items(): p.add_mesh(bnd[k].clip('y'), color=c, show_edges=True, line_width=0.3)
p.camera_position='xz'
p.subplot(0,1)
p.add_mesh(internal.slice(normal='y'), show_edges=True, color='lightgray', line_width=0.3)
p.camera_position='xz'
p.screenshot("look.png")
for k in cols:
    pts = bnd[k].points; r = np.hypot(pts[:,0],pts[:,1])
    print(k, "r range %.4f..%.4f"%(r.min(),r.max()), "z range %.4f..%.4f"%(pts[:,2].min(),pts[:,2].max()))

# -- cell 14 ------------------------------------------------------------------------
import os
from IPython.display import Image, display
print(os.path.getsize("look.png"))
display(Image("look.png"))

# -- cell 15 ------------------------------------------------------------------------
# Cross-section shows the converging outer wall and the mid-duct bulge of the core cowl correctly. The
d = open("system/snappyHexMeshDict").read().replace("level (1 1)","level (2 2)").replace('"; level 1','"; level 2')
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-400:]); print("ERR",r.stderr[-300:])

# -- cell 16 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-1800:])

# -- cell 17 ------------------------------------------------------------------------
# Volume is unchanged at 0.2749 vs the exact 0.27639, so the 0.5% is not surface refinement. Check whe
import numpy as np
def stl_volume(path):
    v=[]; 
    for line in open(path):
        if line.strip().startswith("vertex"): v.append([float(x) for x in line.split()[1:]])
    v=np.array(v).reshape(-1,3,3)
    return np.abs(np.einsum('ij,ij->i', v[:,0], np.cross(v[:,1],v[:,2])).sum()/6.0)
print("union STL volume", stl_volume("/tmp/union.stl"), "exact", solid.volume)

# -- cell 18 ------------------------------------------------------------------------
# The mesh matches the STL to 1e-5; the 0.54% is STL faceting. Tighten the tessellation and check conv
from OCP.BRepMesh import BRepMesh_IncrementalMesh
BRepMesh_IncrementalMesh(solid.wrapped, 1e-5, False, 0.01, True)
for f in solid.faces():
    nm = name_by_area[round(f.area,6)]
    print(nm, write_face_stl(f, nm))
open("/tmp/union.stl","w").write("".join(open(f"constant/triSurface/{n}.stl").read() for n in names))
print("union STL volume", stl_volume("/tmp/union.stl"), "exact", solid.volume)
r = subprocess.run(["surfaceCheck","/tmp/union.stl"],capture_output=True,text=True)
print([l for l in r.stdout.split("\n") if "closed" in l or "zones" in l])

# -- cell 19 ------------------------------------------------------------------------
# Faceting error now 0.03%. Re-extract features and re-mesh on the tighter surfaces.
print(subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True).returncode)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-300:]); print("ERR",r.stderr[-200:])

# -- cell 20 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh"],capture_output=True,text=True)
print("\n".join(l for l in r.stdout.split("\n") if any(k in l for k in
  ["cells:","Patch","outerCowl","coreCowl","inlet","outlet","Total volume","non-orthogonality","skewness","Mesh OK","aspect","regions"])))

# -- cell 21 ------------------------------------------------------------------------
# Mesh OK, 262k cells, volume within 0.03% of exact. Final visual and a measurement of the meshed wall
import pyvista as pv, numpy as np
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0); m = rd.read()
bnd = m["boundary"]; internal = m["internalMesh"]
def r_at(patch, z, tol=0.004):
    c = bnd[patch].cell_centers().points
    s = c[np.abs(c[:,2]-z) < tol]
    return np.hypot(s[:,0],s[:,1]).mean()
print("outerCowl  z=0: %.4f (0.450)  z=0.45: %.4f (0.430)  z=0.9: %.4f (0.410)"
      %(r_at("outerCowl",0.002),r_at("outerCowl",0.45),r_at("outerCowl",0.898)))
print("coreCowl   z=0: %.4f (0.280)  z=0.45: %.4f (0.310)  z=0.9: %.4f (0.250)"
      %(r_at("coreCowl",0.002),r_at("coreCowl",0.45),r_at("coreCowl",0.898)))
print("axial extent of duct: %.4f (0.900)"%(bnd["outlet"].points[:,2].mean()-bnd["inlet"].points[:,2].mean()))
p = pv.Plotter(off_screen=True, window_size=(1100,500), shape=(1,2))
p.subplot(0,0)
for k,c in {"outerCowl":"steelblue","coreCowl":"indianred","inlet":"green","outlet":"orange"}.items():
    p.add_mesh(bnd[k].clip('y',invert=False), color=c, show_edges=False)
p.camera_position='xz'; p.add_axes()
p.subplot(0,1); p.add_mesh(internal.slice(normal='y'), show_edges=True, color='w', line_width=0.3)
p.camera_position='xz'
p.screenshot("final.png")
from IPython.display import Image, display; display(Image("final.png"))
