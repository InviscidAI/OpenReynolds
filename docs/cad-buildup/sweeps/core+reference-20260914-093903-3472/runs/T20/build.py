"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, os
os.makedirs("constant/triSurface", exist_ok=True)

R = 0.030      # bore radius, m (60 mm dia, given)
T_FL = 0.020   # flange thickness, given
T_GK = 0.002   # gasket thickness, given
L = 2*T_FL + T_GK

segA = bd.Pos(0,0,0)            * bd.Cylinder(R, T_FL, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
segG = bd.Pos(0,0,T_FL)         * bd.Cylinder(R, T_GK, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
segB = bd.Pos(0,0,T_FL+T_GK)    * bd.Cylinder(R, T_FL, align=(bd.Align.CENTER,)*2+(bd.Align.MIN,))
fluid = segA + segG + segB
print(L, fluid.volume, 3.14159265*R*R*L, fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
fs = fluid.faces()
for f in fs:
    print(f.geom_type, round(f.area,6), [round(c,4) for c in f.center()], f.normal_at(f.center()))

# -- cell 3 -------------------------------------------------------------------------
import numpy as np
def lateral(seg):
    return seg.faces().filter_by(bd.GeomType.CYLINDER)[0]
def planar_norm(seg, nz):
    return [f for f in seg.faces().filter_by(bd.GeomType.PLANE)
            if abs(f.normal_at(f.center()).Z - nz) < 1e-9][0]

patches = {
    "inlet":            bd.Compound([planar_norm(segA, -1.0)]),   # -Z end of flange A bore
    "outlet":           bd.Compound([planar_norm(segB, +1.0)]),   # +Z end of flange B bore
    "flangeA_bore":     bd.Compound([lateral(segA)]),
    "gasket_inner":     bd.Compound([lateral(segG)]),
    "flangeB_bore":     bd.Compound([lateral(segB)]),
}
for n,c in patches.items():
    bd.export_stl(c, f"constant/triSurface/{n}.stl", tolerance=2e-5, angular_tolerance=0.1)
    print(n, round(c.area,7))
print("areas check: lat A/B =", round(2*np.pi*R*T_FL,7), " gasket =", round(2*np.pi*R*T_GK,7), " disc =", round(np.pi*R*R,7))

# -- cell 4 -------------------------------------------------------------------------
import pyvista as pv, glob
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
meshes = [pv.read(p) for p in sorted(glob.glob("constant/triSurface/*.stl"))]
union = meshes[0].merge(meshes[1:]).clean(tolerance=1e-6)
fe = union.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("cells", union.n_cells, "pts", union.n_points, "free edges", fe.n_cells)
p = pv.Plotter(off_screen=True)
for m,c in zip(meshes, ["r","g","b","y","m"]): p.add_mesh(m, color=c, show_edges=True)
p.camera_position='iso'; p.screenshot("view.png")

# -- cell 5 -------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
pl = pv.Plotter(off_screen=True, window_size=(700,600))
for m,c in zip(meshes, ["red","green","blue","orange","magenta"]):
    pl.add_mesh(m, color=c, show_edges=True, opacity=1.0)
pl.camera.position=(0.12,0.10,0.09); pl.camera.focal_point=(0,0,0.021)
img = pl.screenshot(return_img=True)
plt.figure(figsize=(6,5)); plt.imshow(img); plt.axis("off"); plt.title("patch STLs: "+", ".join(sorted(p.split('/')[-1][:-4] for p in glob.glob('constant/triSurface/*.stl'))))
plt.show()

# -- cell 6 -------------------------------------------------------------------------
import importlib, matplotlib
importlib.reload(matplotlib); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,5))
ax.imshow(img); ax.axis("off"); ax.set_title("bore fluid: 5 patch STLs")
fig

# -- cell 7 -------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True)
head = lambda cls, obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
open("system/controlDict","w").write(head("dictionary","controlDict")+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;\n"
 "writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(head("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution")+"solvers{} relaxationFactors{}\n")

# background block: 4 mm cells, slightly larger than the bore
N = [16,16,11]
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+f"""
scale 1;
vertices ((-0.032 -0.032 -0.002) (0.032 -0.032 -0.002) (0.032 0.032 -0.002) (-0.032 0.032 -0.002)
          (-0.032 -0.032 0.044) (0.032 -0.032 0.044) (0.032 0.032 0.044) (-0.032 0.032 0.044));
blocks (hex (0 1 2 3 4 5 6 7) ({N[0]} {N[1]} {N[2]}) simpleGrading (1 1 1));
edges (); boundary (
 (bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }})
); mergePatchPairs ();
""")
os.rename("system","system_tmp"); os.rename("system_tmp","system")
print(sorted(os.listdir("system")))

# -- cell 8 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-800:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
print(r.stdout[:3000])

# -- cell 10 ------------------------------------------------------------------------
env = dict(os.environ); env["PWD"] = os.getcwd()
r = subprocess.run(["blockMesh"], capture_output=True, text=True, env=env, cwd=os.getcwd())
print(r.returncode); print("\n".join(l for l in r.stdout.splitlines() if "Warning" not in l and "PWD" not in l and "cwd" not in l and "POSIX" not in l)[-1500:])

# -- cell 11 ------------------------------------------------------------------------
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+"""
scale 1;
vertices
(
    (-0.032 -0.032 -0.002) (0.032 -0.032 -0.002) (0.032 0.032 -0.002) (-0.032 0.032 -0.002)
    (-0.032 -0.032  0.044) (0.032 -0.032  0.044) (0.032 0.032  0.044) (-0.032 0.032  0.044)
);
blocks ( hex (0 1 2 3 4 5 6 7) (16 16 11) simpleGrading (1 1 1) );
edges ();
boundary
(
    bg
    {
        type patch;
        faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3) );
    }
);
mergePatchPairs ();
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True, env=env)
print(r.returncode); print("\n".join(l for l in r.stdout.splitlines() if "PWD" not in l and "POSIX" not in l and "Warning" not in l and "cwd_L" not in l)[-1200:])

# -- cell 12 ------------------------------------------------------------------------
names = ["inlet","outlet","flangeA_bore","gasket_inner","flangeB_bore"]
geom = "\n".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs = "\n".join(f'        {n} {{ level (1 1); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in names)
open("system/snappyHexMeshDict","w").write(head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 200000; maxGlobalCells 2000000; minRefinementCells 5; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{refs}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh (0.001 0.001 0.021);
    allowFreeStandingZoneFaces false;
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
  implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{ relativeSizes true; layers {{}} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
  nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
debug 0;
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, env=env)
print(r.returncode); print("\n".join(l for l in r.stdout.splitlines() if "PWD" not in l and "POSIX" not in l and "cwd_L" not in l)[-1500:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True, env=env)
out = "\n".join(l for l in r.stdout.splitlines() if "PWD" not in l and "POSIX" not in l and "cwd_L" not in l and "Warning" not in l)
print(out[out.find("Mesh stats"):])

# -- cell 14 ------------------------------------------------------------------------
lev = {"inlet":(1,1),"outlet":(1,1),"flangeA_bore":(1,2),"flangeB_bore":(1,2),"gasket_inner":(3,3)}
refs = "\n".join(f'        {n} {{ level ({lev[n][0]} {lev[n][1]}); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in names)
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"    refinementSurfaces\n    \{.*?\n    \}", "    refinementSurfaces\n    {\n"+refs+"\n    }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"], capture_output=True, text=True, env=env)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True, env=env)
print(r.returncode); print(r.stdout[-600:])

# -- cell 15 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True, env=env)
out = "\n".join(l for l in r.stdout.splitlines() if "PWD" not in l and "POSIX" not in l and "cwd_L" not in l)
i=out.find("Checking patch topology"); print(out[out.find("Mesh stats"):out.find("Overall number")])
print(out[i:i+900]); print(out[out.find("Checking geometry"):])

# -- cell 16 ------------------------------------------------------------------------
print(open("system/snappyHexMeshDict").read()[:1400])

# -- cell 17 ------------------------------------------------------------------------
print([l for l in r.stdout.splitlines() if "refinement level" in l or l.strip().startswith(("0\t","1\t","2\t","3\t")) or "Snapped mesh" in l][-12:])
import glob, time
print([(p, time.ctime(os.path.getmtime(p))) for p in glob.glob("constant/polyMesh/*")])

# -- cell 18 ------------------------------------------------------------------------
import re
print(re.findall(r"Cells per refinement level:\n(?:\s+\d+\t\d+\n)+", r.stdout)[-1:] )
print(subprocess.run(["grep","-n","-A2","-E","^\\s+(inlet|outlet|gasket_inner|flangeA_bore|flangeB_bore)","constant/polyMesh/boundary"],capture_output=True,text=True).stdout)

# -- cell 19 ------------------------------------------------------------------------
subprocess.run(["blockMesh"], capture_output=True, text=True, env=env)
with open("log.snappy","w") as f:
    rc = subprocess.run(["snappyHexMesh","-overwrite"], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
log = open("log.snappy").read()
print(rc, len(log))
print("\n".join(re.findall(r".*(?:Cells per refinement|^\s+\d+\t\d+$|Snapped mesh|patch.*gasket|Removing).*", log, flags=re.M))[:2000])

# -- cell 20 ------------------------------------------------------------------------
print("\n".join(l for l in log.splitlines() if "gasket" in l.lower() or "Refinement level" in l or "level" in l.lower() and "Cells" not in l)[:3000])

# -- cell 21 ------------------------------------------------------------------------
i = log.find("Refinement phase"); print(log[log.find("Surfaces")-200: log.find("Surfaces")+1500] if "Surfaces" in log else "no")

# -- cell 22 ------------------------------------------------------------------------
j = log.find("Refinement phase"); print(log[j:j+2500])

# -- cell 23 ------------------------------------------------------------------------
k = log.find("Reading refinement surfaces"); print(log[max(0,k-500):k+2000])

# -- cell 24 ------------------------------------------------------------------------
# The gasket band is only 2 mm wide and snappy never resolved it, so it got no faces. I'll align the b
open("system/blockMeshDict","w").write(head("dictionary","blockMeshDict")+"""
scale 1;
vertices
(
    (-0.032 -0.032 -0.002) (0.032 -0.032 -0.002) (0.032 0.032 -0.002) (-0.032 0.032 -0.002)
    (-0.032 -0.032  0.044) (0.032 -0.032  0.044) (0.032 0.032  0.044) (-0.032 0.032  0.044)
);
blocks ( hex (0 1 2 3 4 5 6 7) (32 32 46) simpleGrading (1 1 1) );
edges ();
boundary
(
    bg
    {
        type patch;
        faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3) );
    }
);
mergePatchPairs ();
""")
lev = {n:(1,1) for n in names}
refs = "\n".join(f'        {n} {{ level (1 1); patchInfo {{ type {"patch" if n in ("inlet","outlet") else "wall"}; }} }}' for n in names)
d = open("system/snappyHexMeshDict").read()
d = re.sub(r"    refinementSurfaces\n    \{.*?\n    \}", "    refinementSurfaces\n    {\n"+refs+"\n    }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"], capture_output=True, text=True, env=env)
with open("log.snappy","w") as f:
    rc = subprocess.run(["snappyHexMesh","-overwrite"], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
log = open("log.snappy").read(); print(rc, log[-400:])

# -- cell 25 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True, env=env)
out = "\n".join(l for l in r.stdout.splitlines() if "PWD" not in l and "POSIX" not in l and "cwd_L" not in l)
print(out[out.find("Mesh stats"):out.find("Overall number")])
print(out[out.find("Checking patch topology"):out.find("Checking faceZone")])
print(out[out.find("Checking geometry"):])

# -- cell 26 ------------------------------------------------------------------------
rc = subprocess.run(["foamToVTK","-constant","-ascii"], capture_output=True, text=True, env=env).returncode
import glob
print(rc, glob.glob("VTK/**/*.vtp", recursive=True)[:10], glob.glob("VTK/*"))

# -- cell 27 ------------------------------------------------------------------------
exp = {"inlet":(np.pi*R*R,(0,0)),"outlet":(np.pi*R*R,(0.042,0.042)),
       "flangeA_bore":(2*np.pi*R*T_FL,(0,0.020)),"gasket_inner":(2*np.pi*R*T_GK,(0.020,0.022)),
       "flangeB_bore":(2*np.pi*R*T_FL,(0.022,0.042))}
for n in names:
    m = pv.read(f"VTK/t20_0/boundary/{n}.vtp")
    a = m.compute_cell_sizes(length=False, volume=False)["Area"].sum()
    z = m.points[:,2]
    print(f"{n:13s} area {a:.6f} (exact {exp[n][0]:.6f})  z {z.min():.4f}..{z.max():.4f} (exact {exp[n][1]})")

# -- cell 28 ------------------------------------------------------------------------
# Patch areas and axial bands match. The end discs ripple ±0.5 mm at the rim, though — I'll snap the t
union.save("constant/triSurface/bore_union.stl")   # welded copy, only for feature extraction
open("system/surfaceFeatureExtractDict","w").write(head("dictionary","surfaceFeatureExtractDict")+"""
bore_union.stl { extractionMethod extractFromSurface; includedAngle 150; subsetFeatures { nonManifoldEdges no; openEdges yes; } writeObj no; }
""")
print(subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True,env=env).returncode, os.listdir("constant/triSurface"))

# -- cell 29 ------------------------------------------------------------------------
os.remove("constant/triSurface/bore_union.stl")   # keep only the per-patch STLs + the eMesh
d = open("system/snappyHexMeshDict").read()
d = d.replace("    features ();", '    features ( { file "bore_union.eMesh"; level 1; } );')
d = d.replace("implicitFeatureSnap false; explicitFeatureSnap true", "implicitFeatureSnap false; explicitFeatureSnap true")
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"], capture_output=True, text=True, env=env)
with open("log.snappy","w") as f:
    rc = subprocess.run(["snappyHexMesh","-overwrite"], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
print(rc, open("log.snappy").read()[-300:])
