"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, os, numpy as np, build123d as bd
print(os.getcwd(), os.listdir('.'))
print(subprocess.run(["grep","-n","-i","spline\\|revolve",".reference/b123d_api.md"],capture_output=True,text=True).stdout[:2000])

# -- cell 2 -------------------------------------------------------------------------
print(os.listdir('.'), os.path.exists('.reference'))
import build123d as bd, inspect
print(inspect.signature(bd.Spline.__init__))
print(inspect.signature(bd.revolve))

# -- cell 3 -------------------------------------------------------------------------
import build123d as bd
from build123d import Vector, Axis

L = 0.900  # duct length, m (given)
R_out = [0.450, 0.430, 0.410]   # outer cowl inner surface radii at z=0, L/2, L (given)
R_in  = [0.280, 0.310, 0.250]   # core cowl outer surface radii (given)
Z = [0.0, L/2, L]

outer = bd.Spline(*[(R_out[i], 0, Z[i]) for i in range(3)])
inner = bd.Spline(*[(R_in[i], 0, Z[i]) for i in range(3)])
outlet_e = bd.Line((R_in[2],0,L), (R_out[2],0,L))
inlet_e  = bd.Line((R_in[0],0,0), (R_out[0],0,0))
wire = bd.Wire([outer.edge(), inner.edge(), outlet_e.edge(), inlet_e.edge()])
prof = bd.make_face(wire)
print(prof.area, wire.is_closed)
duct = bd.revolve(prof, Axis.Z, 360)
print(duct.volume, duct.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
import math
faces = duct.faces()
for f in faces:
    c = f.center()
    print(f.geom_type, round(f.area,5), (round(c.X,3),round(c.Y,3),round(c.Z,3)), round(math.hypot(c.X,c.Y),4))

# -- cell 5 -------------------------------------------------------------------------
import os, inspect, math
print(inspect.signature(bd.export_stl))
os.makedirs("constant/triSurface", exist_ok=True)
patches = {}
for f in duct.faces():
    c = f.center(); r = math.hypot(c.X, c.Y)
    if f.geom_type == bd.GeomType.PLANE:
        patches["inlet" if abs(c.Z) < 1e-6 else "outlet"] = f
    else:
        patches["outerCowl" if r > 0.37 else "coreCowl"] = f
print({k: round(v.area,4) for k,v in patches.items()})
for name, f in patches.items():
    bd.export_stl(f, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.1, ascii_format=True)
    # rename solid to patch name
    p = f"constant/triSurface/{name}.stl"
    txt = open(p).read().replace("solid ", f"solid {name} ",1) if False else open(p).read()
    lines = txt.splitlines()
    lines[0] = f"solid {name}"; lines[-1] = f"endsolid {name}"
    open(p,"w").write("\n".join(lines)+"\n")
print(subprocess.run(["ls","-la","constant/triSurface"],capture_output=True,text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
import pyvista as pv, numpy as np
for name, want in [("outerCowl",R_out),("coreCowl",R_in)]:
    m = pv.read(f"constant/triSurface/{name}.stl"); p = m.points
    r = np.hypot(p[:,0],p[:,1]); z = p[:,2]
    for zt, w in zip([0,0.45,0.9], want):
        sel = np.abs(z-zt) < 2e-3
        print(name, f"z={zt}: measured r={r[sel].mean():.4f} (min {r[sel].min():.4f} max {r[sel].max():.4f}) asked {w}")
print("inlet area", pv.read("constant/triSurface/inlet.stl").area, "expect", math.pi*(0.45**2-0.28**2))
print("outlet area", pv.read("constant/triSurface/outlet.stl").area, "expect", math.pi*(0.41**2-0.25**2))

# -- cell 7 -------------------------------------------------------------------------
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(900,600))
cols = {"outerCowl":"lightblue","coreCowl":"orange","inlet":"green","outlet":"red"}
for n,c in cols.items():
    m = pv.read(f"constant/triSurface/{n}.stl")
    pl.add_mesh(m.clip(normal='y', origin=(0,0,0)), color=c, show_edges=False, label=n)
pl.add_legend(); pl.view_xz(); pl.screenshot("geom.png")
print("ok")

# -- cell 8 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 9 -------------------------------------------------------------------------
import os, textwrap
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
head = lambda cls,obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
open("system/controlDict","w").write(head("dictionary","controlDict")+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes")+"gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution")+"solvers{}\n")
NC = 20
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ((-0.5 -0.5 -0.05)(0.5 -0.5 -0.05)(0.5 0.5 -0.05)(-0.5 0.5 -0.05)(-0.5 -0.5 0.95)(0.5 -0.5 0.95)(0.5 0.5 0.95)(-0.5 0.5 0.95));
blocks (hex (0 1 2 3 4 5 6 7) ({NC} {NC} {NC}) simpleGrading (1 1 1));
edges ();
boundary (bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }});
""")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-600:])

# -- cell 10 ------------------------------------------------------------------------
names = ["outerCowl","coreCowl","inlet","outlet"]
sfe = head("dictionary","surfaceFeatureExtractDict")
for n in names:
    sfe += f'{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}\n'
open("system/surfaceFeatureExtractDict","w").write(sfe)
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.stdout[-400:], r.returncode)
print(os.listdir("constant/triSurface"))

# -- cell 11 ------------------------------------------------------------------------
geom = "\n".join(f'{n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
feats = "\n".join(f'{{ file "{n}.eMesh"; level 1; }}' for n in names)
surfs = "\n".join(f'{n} {{ level (1 1); patchInfo {{ type {"wall" if "Cowl" in n else "patch"}; }} }}' for n in names)
snap = head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{ {geom} }}
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2;
    features ( {feats} );
    refinementSurfaces {{ {surfs} }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh (0.37 0 0.45);
    allowFreeStandingZoneFaces false;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
debug 0; mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snap)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-1500:])

# -- cell 12 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2200:])

# -- cell 13 ------------------------------------------------------------------------
snap_ref = snap.replace("level (1 1)","level (2 2)").replace('level 1; }','level 2; }')
open("system/snappyHexMeshDict","w").write(snap_ref)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-500:])

# -- cell 14 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
out=r.stdout; import re
print("\n".join(l for l in out.splitlines() if re.search("cells:|Patch|outerCowl|coreCowl|inlet|outlet|Total volume|non-orthogonality|skewness|Mesh OK|\\*\\*\\*|aspect",l)))

# -- cell 15 ------------------------------------------------------------------------
r=subprocess.run(["foamToVTK","-constant","-excludePatches","(bg)"],capture_output=True,text=True); print(r.returncode)
import glob; print(glob.glob("VTK/**/*.vt*",recursive=True)[:10])

# -- cell 16 ------------------------------------------------------------------------
grid = pv.read("VTK/t16_0/internal.vtu")
sl = grid.clip(normal='y', origin=(0,0,0))
pl = pv.Plotter(off_screen=True, window_size=(1000,650))
pl.add_mesh(sl, show_edges=True, color='lightgray', edge_color='k', line_width=0.3)
for n,c in [("outerCowl","blue"),("coreCowl","orange"),("inlet","green"),("outlet","red")]:
    pl.add_mesh(pv.read(f"VTK/t16_0/boundary/{n}.vtp").clip(normal='y',origin=(0,0,0)), color=c, opacity=0.6)
pl.view_xz(); pl.camera.zoom(1.3); pl.screenshot("mesh.png")
display(Image("mesh.png"))

# -- cell 17 ------------------------------------------------------------------------
sl = grid.slice(normal='y', origin=(0,0,0.0))
pl = pv.Plotter(off_screen=True, window_size=(1000,500))
pl.add_mesh(sl, show_edges=True, color='lightsteelblue', edge_color='k', line_width=0.4)
pl.view_xz(); pl.camera.zoom(1.4); pl.screenshot("slice.png")
display(Image("slice.png"))
p=sl.points; print("slice x range", p[:,0].min(), p[:,0].max())

# -- cell 18 ------------------------------------------------------------------------
u = pv.read("constant/triSurface/outerCowl.stl")
for n in ["coreCowl","inlet","outlet"]: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(tolerance=1e-6).extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("free edges after weld:", uc.n_cells, "| n_tris", u.n_cells)

# -- cell 19 ------------------------------------------------------------------------
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
import numpy as np

def write_patch_stls(solid, patch_faces, outdir, lin=2e-4, ang=0.2):
    BRepMesh_IncrementalMesh(solid.wrapped, lin, False, ang, True)
    for name, f in patch_faces.items():
        loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
        trsf = loc.Transformation()
        pts = []
        for i in range(1, tri.NbNodes()+1):
            p = tri.Node(i).Transformed(trsf); pts.append((p.X(), p.Y(), p.Z()))
        pts = np.array(pts)
        rev = f.wrapped.Orientation() == TopAbs_REVERSED
        with open(f"{outdir}/{name}.stl","w") as fh:
            fh.write(f"solid {name}\n")
            for i in range(1, tri.NbTriangles()+1):
                a,b,c = tri.Triangle(i).Get()
                v = pts[[a-1,b-1,c-1]]
                if rev: v = v[[0,2,1]]
                n = np.cross(v[1]-v[0], v[2]-v[0]); ln = np.linalg.norm(n); n = n/ln if ln>0 else n
                fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
                for q in v: fh.write(f"   vertex {q[0]:.8e} {q[1]:.8e} {q[2]:.8e}\n")
                fh.write("  endloop\n endfacet\n")
            fh.write(f"endsolid {name}\n")

write_patch_stls(duct, patches, "constant/triSurface")
u = pv.read("constant/triSurface/outerCowl.stl")
for n in ["coreCowl","inlet","outlet"]: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(tolerance=1e-8)
print("tris", uc.n_cells, "free edges", uc.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False).n_cells)

# -- cell 20 ------------------------------------------------------------------------
from OCP.BRepTools import BRepTools
BRepTools.Clean_s(duct.wrapped)
write_patch_stls(duct, patches, "constant/triSurface")
u = pv.read("constant/triSurface/outerCowl.stl")
for n in ["coreCowl","inlet","outlet"]: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(tolerance=1e-8)
fe = uc.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
print("tris", uc.n_cells, "free edges", fe.n_cells)
if fe.n_cells: print(fe.points[:6])

# -- cell 21 ------------------------------------------------------------------------
nrm = uc.compute_normals(cell_normals=True, point_normals=False, consistent_normals=False)
c = nrm.cell_centers().points; nv = nrm.cell_data['Normals']
# outward test: on outer cowl surface (r>0.37, 0<z<0.9) normals should point outward (+r) for a solid-outward convention
selo = (np.hypot(c[:,0],c[:,1])>0.37)&(c[:,2]>0.02)&(c[:,2]<0.88)
radial = (c[selo,0]*nv[selo,0]+c[selo,1]*nv[selo,1])/np.hypot(c[selo,0],c[selo,1])
print("outer cowl mean radial component:", radial.mean())
for name,want in [("outerCowl",R_out),("coreCowl",R_in)]:
    p=pv.read(f"constant/triSurface/{name}.stl").points; r=np.hypot(p[:,0],p[:,1])
    print(name, [round(r[np.abs(p[:,2]-zt)<2e-3].mean(),4) for zt in [0,0.45,0.9]], "asked", want)

# -- cell 22 ------------------------------------------------------------------------
subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:])

# -- cell 23 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); out=r.stdout
print("\n".join(l for l in out.splitlines() if re.search("cells:|Patch |outerCowl|coreCowl|inlet|outlet|Total volume|non-orthogonality|skewness|Mesh OK|\\*\\*\\*",l)))
print("exact volume", duct.volume)

# -- cell 24 ------------------------------------------------------------------------
BRepTools.Clean_s(duct.wrapped)
write_patch_stls(duct, patches, "constant/triSurface", lin=5e-5, ang=0.05)
u = pv.read("constant/triSurface/outerCowl.stl")
for n in ["coreCowl","inlet","outlet"]: u = u.merge(pv.read(f"constant/triSurface/{n}.stl"))
uc = u.clean(tolerance=1e-8)
print("tris", uc.n_cells, "free edges", uc.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False).n_cells, "vol", uc.volume)

# -- cell 25 ------------------------------------------------------------------------
subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True); print(r.returncode)
r=subprocess.run(["checkMesh"],capture_output=True,text=True); out=r.stdout
print("\n".join(l for l in out.splitlines() if re.search("cells:|Patch |outerCowl|coreCowl|inlet|outlet|Total volume|non-orthogonality|skewness|Mesh OK|\\*\\*\\*",l)))

# -- cell 26 ------------------------------------------------------------------------
subprocess.run(["foamToVTK","-constant","-overwrite"],capture_output=True,text=True)
grid = pv.read("VTK/t16_0/internal.vtu"); sl = grid.slice(normal='y')
pl = pv.Plotter(off_screen=True, window_size=(1000,500))
pl.add_mesh(sl, show_edges=True, color='lightsteelblue', line_width=0.3)
pl.view_xz(); pl.camera.zoom(1.4); pl.screenshot("final_slice.png"); display(Image("final_slice.png"))
