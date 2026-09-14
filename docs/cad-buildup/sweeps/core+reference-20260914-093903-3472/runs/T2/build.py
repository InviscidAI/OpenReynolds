"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os, subprocess
print(bd.__version__ if hasattr(bd,'__version__') else 'ok', os.getcwd(), os.listdir('.'))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd
from build123d import Pos, Vector
D = 0.040            # sphere diameter, from request
R = D/2
XU, XD, YS = 5*D, 15*D, 5*D    # 5D up, 15D down, 5D lateral/vert
x0, x1 = -XU, XD
y0, y1, z0, z1 = -YS, YS, -YS, YS

sphere = bd.Sphere(R)
print(sphere.bounding_box(), sphere.volume, 4/3*3.141592653589793*R**3)
box = bd.Box(x1-x0, y1-y0, z1-z0).moved(bd.Location(Vector((x0+x1)/2,0,0)))
print(box.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
import os
from build123d import Plane, Rectangle, export_stl
os.makedirs("constant/triSurface", exist_ok=True)
L = 2*YS
inlet    = Plane(origin=(x0,0,0), z_dir=(1,0,0)) * Rectangle(L, L)
outlet   = Plane(origin=(x1,0,0), z_dir=(1,0,0)) * Rectangle(L, L)
side_ym  = Plane(origin=(0,y0,0), z_dir=(0,1,0)) * Rectangle(x1-x0, L)
side_yp  = Plane(origin=(0,y1,0), z_dir=(0,1,0)) * Rectangle(x1-x0, L)
side_zm  = Plane(origin=(0,0,z0), z_dir=(0,0,1)) * Rectangle(x1-x0, L)
side_zp  = Plane(origin=(0,0,z1), z_dir=(0,0,1)) * Rectangle(x1-x0, L)
farfield = side_ym + side_yp + side_zm + side_zp
for name, sh in [("sphere", sphere), ("inlet", inlet), ("outlet", outlet), ("farfield", farfield)]:
    export_stl(sh, f"constant/triSurface/{name}.stl", tolerance=1e-5, angular_tolerance=0.1)
    print(name, sh.bounding_box(), sh.area)

# -- cell 4 -------------------------------------------------------------------------
xc = (x0+x1)/2
side_ym  = Plane(origin=(xc,y0,0), z_dir=(0,1,0)) * Rectangle(x1-x0, L)
side_yp  = Plane(origin=(xc,y1,0), z_dir=(0,1,0)) * Rectangle(x1-x0, L)
side_zm  = Plane(origin=(xc,0,z0), z_dir=(0,0,1)) * Rectangle(x1-x0, L)
side_zp  = Plane(origin=(xc,0,z1), z_dir=(0,0,1)) * Rectangle(x1-x0, L)
for n,f in [("ym",side_ym),("yp",side_yp),("zm",side_zm),("zp",side_zp)]:
    print(n, f.bounding_box(), f.area)

# -- cell 5 -------------------------------------------------------------------------
side_ym  = Plane(origin=(xc,y0,0), z_dir=(0,1,0)) * Rectangle(L, x1-x0)
side_yp  = Plane(origin=(xc,y1,0), z_dir=(0,1,0)) * Rectangle(L, x1-x0)
farfield = side_ym + side_yp + side_zm + side_zp
print(farfield.bounding_box(), farfield.area, 4*(x1-x0)*L)
export_stl(farfield, "constant/triSurface/farfield.stl", tolerance=1e-5, angular_tolerance=0.1)

# -- cell 6 -------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
head = lambda cls, obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
NX, NY, NZ = 40, 20, 20   # background cell ~0.02 m = D/2 (coarse first)
verts = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
vtxt = "\n".join(f"({a} {b} {c})" for a,b,c in verts)
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ({vtxt});
blocks (hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1));
edges ();
boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5)); }}
    farfield {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
open("system/controlDict","w").write(head("dictionary","controlDict")+"""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;
runTimeModifiable true;
""")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes")+"""
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;}
divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;} snGradSchemes{default corrected;}
""")
open("system/fvSolution","w").write(head("dictionary","fvSolution")+"""
solvers{} 
""")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-600:])

# -- cell 7 -------------------------------------------------------------------------
SNAPPY = lambda layers: head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers {'true' if layers else 'false'};
geometry {{ sphere.stl {{ type triSurfaceMesh; name sphere; }} }};
castellatedMeshControls
{{
    maxLocalCells 1000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 3; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces {{ sphere {{ level (3 3); patchInfo {{ type wall; }} }} }}
    refinementRegions {{ }}
    locationInMesh (-0.15 0.013 0.017);
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
   nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{
    relativeSizes true;
    layers {{ sphere {{ nSurfaceLayers 4; }} }}
    expansionRatio 1.2; finalLayerThickness 0.4; minThickness 0.1;
    nGrow 0; featureAngle 120; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
   minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
   minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(SNAPPY(False))
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.stdout[-1200:])

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read()
internal = m["internalMesh"]; bnd = m["boundary"]
print(list(bnd.keys()))
sph = bnd["sphere"]
print("sphere patch area", sph.extract_surface().compute_cell_sizes()["Area"].sum(), "analytic", 4*np.pi*R**2)
print("bbox sphere patch", sph.bounds)

# -- cell 9 -------------------------------------------------------------------------
sl = internal.slice(normal="z", origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(1300,650))
p.add_mesh(sl, show_edges=True, color="lightgray", line_width=1)
p.add_mesh(sph, color="red")
p.view_xy(); p.camera.zoom(1.3)
p.screenshot("slice_coarse.png"); print("ok")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("slice_coarse.png"))

# -- cell 11 ------------------------------------------------------------------------
SNAPPY2 = head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers true;
geometry
{{
    sphere.stl {{ type triSurfaceMesh; name sphere; }}
    wake {{ type searchableBox; min (-0.06 -0.06 -0.06); max (0.32 0.06 0.06); }}
}};
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 3; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces {{ sphere {{ level (3 3); patchInfo {{ type wall; }} }} }}
    refinementRegions {{ wake {{ mode inside; levels ((1e15 2)); }} }}
    locationInMesh (-0.15 0.013 0.017);
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 100; nRelaxIter 5;
   nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{
    relativeSizes true;
    layers {{ sphere {{ nSurfaceLayers 5; }} }}
    expansionRatio 1.2; finalLayerThickness 0.4; minThickness 0.05;
    nGrow 0; featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; nRelaxedIter 20;
}}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
   minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
   minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
   relaxed {{ maxNonOrtho 75; }} }}
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(SNAPPY2)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-1500:])

# -- cell 12 ------------------------------------------------------------------------
open("system/snappyHexMeshDict","w").write(SNAPPY2.replace("level (3 3)","level (4 4)"))
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.stdout[-900:])

# -- cell 13 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2500:])

# -- cell 14 ------------------------------------------------------------------------
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0); m = rd.read()
sph = m["boundary"]["sphere"].extract_surface(); internal=m["internalMesh"]
pts = sph.points; rr = np.linalg.norm(pts,axis=1)
print("patch radius min/max/mean", rr.min(), rr.max(), rr.mean(), " target", R)
print("patch area", sph.compute_cell_sizes()["Area"].sum(), "analytic", 4*np.pi*R**2)
sl = internal.slice(normal="z", origin=(0,0,0))
p = pv.Plotter(off_screen=True, window_size=(900,900))
p.add_mesh(sl, show_edges=True, color="white", line_width=1)
p.view_xy(); p.camera.focal_point=(0,0,0); p.camera.position=(0,0,0.12); p.camera.parallel_projection=True; p.camera.parallel_scale=0.045
p.screenshot("layers.png"); display(Image("layers.png"))

# -- cell 15 ------------------------------------------------------------------------
cc = internal.cell_centers().points
sel = cc[(np.abs(cc[:,1])<0.0008)&(np.abs(cc[:,2])<0.0008)&(cc[:,0]<-R)&(cc[:,0]>-R-0.008)]
rad = np.sort(-sel[:,0]-R)
print(np.round(rad*1000,4), "mm from wall (upstream stagnation line cell centres)")
