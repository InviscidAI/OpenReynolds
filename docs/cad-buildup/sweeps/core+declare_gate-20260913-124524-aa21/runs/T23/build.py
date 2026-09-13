"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess, math
from build123d import *
os.makedirs("constant/triSurface", exist_ok=True)

MM = 1e-3
L      = 40*MM
R_HOLL = 16*MM
R_BALL = 15*MM
R_HOLE = 12*MM

cube   = Box(L, L, L)
hollow = Sphere(R_HOLL)
ball   = Sphere(R_BALL)
cylz   = Cylinder(R_HOLE, L*1.2)
holes  = cylz + Rot(0, 90, 0)*cylz + Rot(90, 0, 0)*cylz

water = ((hollow + holes) & cube) - ball
print("solids:", len(water.solids()), " faces:", len(water.faces()))
print("volume m^3:", water.volume)
print("shell-only volume:", 4/3*math.pi*(R_HOLL**3 - R_BALL**3))
print("bbox:", water.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_SurfaceType as GST

AXN = {0:"x", 1:"y", 2:"z"}
def classify(f):
    """Analytic patch name from the face's own surface (radius/axis), not bbox position."""
    a = BRepAdaptor_Surface(f.wrapped); t = a.GetType(); c = f.center()
    if t == GST.GeomAbs_Sphere:
        r = a.Sphere().Radius()
        return "ball" if abs(r - R_BALL) < 1e-9 else "hollowWall"
    if t == GST.GeomAbs_Cylinder:
        d = a.Cylinder().Axis().Direction(); v = [d.X(), d.Y(), d.Z()]
        i = max(range(3), key=lambda k: abs(v[k]))
        s = "max" if [c.X, c.Y, c.Z][i] > 0 else "min"
        return f"holeWall_{AXN[i]}{s}"
    if t == GST.GeomAbs_Plane:
        n = a.Plane().Axis().Direction(); v = [n.X(), n.Y(), n.Z()]
        i = max(range(3), key=lambda k: abs(v[k]))
        s = "max" if [c.X, c.Y, c.Z][i] > 0 else "min"
        return f"opening_{AXN[i]}{s}"
    raise RuntimeError("unexpected surface")

patches = {}
for f in water.faces():
    patches.setdefault(classify(f), []).append(f)
for k in sorted(patches):
    print(k, len(patches[k]), f"area={sum(f.area for f in patches[k]):.4e}")
print("n patches:", len(patches))

# -- cell 3 -------------------------------------------------------------------------
import numpy as np
rng = np.random.default_rng(0)
p = rng.normal(size=(400000,3)); p /= np.linalg.norm(p,axis=1)[:,None]
frac = (np.abs(p*R_HOLL).max(axis=1) < math.sqrt(R_HOLL**2-R_HOLE**2)).mean()
print("MC hollow-wall area:", frac*4*math.pi*R_HOLL**2, " measured:", sum(f.area for f in patches["hollowWall"]))
print("water volume (m^3):", water.volume, " -> mm^3:", water.volume/MM**3)

# -- cell 4 -------------------------------------------------------------------------
from build123d import export_stl
TOL, ATOL = 2e-5, 0.15
for name, fs in sorted(patches.items()):
    shp = fs[0] if len(fs) == 1 else Compound(children=[Face(f.wrapped) for f in fs])
    ok = export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=TOL, angular_tolerance=ATOL, ascii_format=True)
    n = sum(1 for l in open(f"constant/triSurface/{name}.stl") if l.startswith("  facet") or l.startswith("facet"))
    print(name, ok, "tris", n)

# -- cell 5 -------------------------------------------------------------------------
print(open("constant/triSurface/ball.stl").read()[:300])
print(subprocess.run("ls -la constant/triSurface | head -20", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run("grep -c facet constant/triSurface/*.stl", shell=True, capture_output=True, text=True).stdout)

# -- cell 6 -------------------------------------------------------------------------
import time
for tol in (5e-5, 1e-4):
    t0=time.time(); export_stl(patches["ball"][0], "/tmp/b.stl", tolerance=tol, angular_tolerance=0.3, ascii_format=True)
    n=sum(1 for l in open("/tmp/b.stl") if " facet" in l[:7])
    print(tol, n, f"{time.time()-t0:.1f}s")

# -- cell 7 -------------------------------------------------------------------------
import re, time
TOL2, ATOL2 = 5e-5, 0.3   # 0.05 mm chord deflection, small vs the 1 mm water gap
t0 = time.time()
for name, fs in sorted(patches.items()):
    shp = fs[0] if len(fs) == 1 else Compound(children=[Face(f.wrapped) for f in fs])
    path = f"constant/triSurface/{name}.stl"
    export_stl(shp, path, tolerance=TOL2, angular_tolerance=ATOL2, ascii_format=True)
    txt = open(path).read()
    txt = re.sub(r"^solid.*$", f"solid {name}", txt, count=1, flags=re.M)
    txt = re.sub(r"^endsolid.*$", f"endsolid {name}", txt, count=1, flags=re.M)
    open(path, "w").write(txt)
print("export done", f"{time.time()-t0:.0f}s")
print(subprocess.run("grep -c facet constant/triSurface/*.stl; du -sh constant/triSurface", shell=True, capture_output=True, text=True).stdout)

# -- cell 8 -------------------------------------------------------------------------
import os
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
HEAD = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(HEAD % "controlDict" + """
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;
writePrecision 8; runTimeModifiable true;
""")
open("system/fvSchemes","w").write(HEAD % "fvSchemes" + "gradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(HEAD % "fvSolution" + "solvers{}\n")

BOX = 0.022   # background box half-size, just outside the 20 mm cube faces
NB  = 22      # 2 mm base cells
open("system/blockMeshDict","w").write(HEAD % "blockMeshDict" + f"""
scale 1;
vertices ((-{BOX} -{BOX} -{BOX}) ({BOX} -{BOX} -{BOX}) ({BOX} {BOX} -{BOX}) (-{BOX} {BOX} -{BOX})
          (-{BOX} -{BOX} {BOX}) ({BOX} -{BOX} {BOX}) ({BOX} {BOX} {BOX}) (-{BOX} {BOX} {BOX}));
blocks (hex (0 1 2 3 4 5 6 7) ({NB} {NB} {NB}) simpleGrading (1 1 1));
edges ();
boundary (bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }});
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-500:], r.returncode)

# -- cell 9 -------------------------------------------------------------------------
NAMES = sorted(patches)
LOC = (0.00895, 0.00895, 0.00895)   # in the 1 mm shell, on the (1,1,1) diagonal, clear of every hole

def snappy_dict(level, nlayers=0):
    geom = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in NAMES)
    refs = "\n".join(f'    {n} {{ level ({level} {level}); patchInfo {{ type wall; }} }}' for n in NAMES)
    return HEAD % "snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls {{
  maxLocalCells 4000000; maxGlobalCells 12000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces {{
{refs}
  }}
  refinementRegions {{}}
  locationInMesh ({LOC[0]} {LOC[1]} {LOC[2]});
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.4;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-14; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy_dict(2))
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 10 ------------------------------------------------------------------------
subprocess.run(["blockMesh"], capture_output=True, text=True)
open("system/snappyHexMeshDict","w").write(snappy_dict(3))
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode)
import re as _re
print("\n".join(l for l in r.stdout.splitlines() if "cells:" in l or "Cells per" in l or _re.match(r"^\s+\d\s+\d+$", l)))

# -- cell 11 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[-3000:])

# -- cell 12 ------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
open("case.foam","w").write("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read()["internalMesh"]
sl = m.slice(normal=(0,0,1), origin=(0,0,0.0005))
p = pv.Plotter(off_screen=True, window_size=(900,900))
p.add_mesh(sl, show_edges=True, line_width=0.4, color="lightsteelblue")
p.view_xy(); p.add_text("z=0.5mm slice", font_size=10)
p.screenshot("slice_z.png")
print(m.n_cells, sl.bounds)

# -- cell 13 ------------------------------------------------------------------------
import matplotlib.pyplot as plt, numpy as np
sl = sl.compute_cell_sizes()
c = np.array(sl.cell_centers().points)
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(c[:,0]*1000, c[:,1]*1000, s=0.6, c=np.sqrt(sl.cell_data["Area"]) * 1000)
ax.set_aspect("equal"); ax.set_title("water cells, z=0.5 mm slice (colour = cell size, mm)")
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
plt.colorbar(ax.collections[0], ax=ax, shrink=0.8); plt.show()

# -- cell 14 ------------------------------------------------------------------------
d = np.array([1,1,1])/np.sqrt(3)
rs = np.linspace(0.01500, 0.01600, 400)
pts = pv.PolyData(np.outer(rs, d))
ids = pts.sample(m, locator="cell")["vtkOriginalCellIds"] if False else None
res = pts.select_enclosed_points(m.extract_surface(), tolerance=0.0, check_surface=False)
ins = np.array(res["SelectedPoints"])
print("fraction of the 15->16 mm diagonal ray inside the mesh:", ins.mean())
fc = m.find_containing_cell(np.outer(rs, d))
print("unique cells across the 1 mm gap:", len(set(fc[fc >= 0])), " misses:", int((fc < 0).sum()))

# -- cell 15 ------------------------------------------------------------------------
rd2 = pv.OpenFOAMReader("case.foam"); rd2.set_active_time_value(0.0)
rd2.enable_all_patch_arrays(); blk = rd2.read()
bnd = blk["boundary"]
analytic = {"ball": 4*math.pi*R_BALL**2, "opening_zmax": math.pi*R_HOLE**2,
            "hollowWall": sum(f.area for f in patches["hollowWall"]),
            "holeWall_zmax": patches["holeWall_zmax"][0].area}
for k in bnd.keys():
    a = bnd[k].compute_cell_sizes()["Area"].sum()
    ref = analytic.get(k)
    print(f"{k:15s} mesh {a:.4e}" + (f"   CAD {ref:.4e}  err {100*(a-ref)/ref:+.2f}%" if ref else ""))

# -- cell 16 ------------------------------------------------------------------------
import glob
tris = []
for fn in glob.glob("constant/triSurface/*.stl"):
    v = [tuple(map(float, l.split()[1:4])) for l in open(fn) if l.strip().startswith("vertex")]
    tris += [v[i:i+3] for i in range(0, len(v), 3)]
V = np.round(np.array(tris).reshape(-1,3), 9)
uniq, inv = np.unique(V, axis=0, return_inverse=True)
T = inv.reshape(-1,3)
from collections import Counter
E = Counter(tuple(sorted((T[i,a], T[i,(a+1)%3]))) for i in range(len(T)) for a in range(3))
print("triangles:", len(T), " free edges (used once):", sum(1 for k,c in E.items() if c == 1))

# -- cell 17 ------------------------------------------------------------------------
bad = [k for k,c in E.items() if c == 1]
for k in bad:
    print(uniq[k[0]]*1000, uniq[k[1]]*1000, "mm")
print([ (k,c) for k,c in E.items() if c > 2 ][:5], "edges used >2:", sum(1 for c in E.values() if c>2))

# -- cell 18 ------------------------------------------------------------------------
V2 = np.round(np.array(tris).reshape(-1,3), 9) + 0.0
V2[V2 == 0] = 0.0
uniq2, inv2 = np.unique(V2, axis=0, return_inverse=True)
T2 = inv2.reshape(-1,3)
E2 = Counter(tuple(sorted((T2[i,a], T2[i,(a+1)%3]))) for i in range(len(T2)) for a in range(3))
print("free edges:", sum(1 for c in E2.values() if c == 1), " non-manifold(>2):", sum(1 for c in E2.values() if c > 2))

# -- cell 19 ------------------------------------------------------------------------
bad2 = [k for k,c in E2.items() if c == 1]
for k in bad2:
    print(repr(uniq2[k[0]]), repr(uniq2[k[1]]), np.linalg.norm(uniq2[k[0]]-uniq2[k[1]]))

# -- cell 20 ------------------------------------------------------------------------
def strip_degenerate(path):
    L = open(path).read().split("\n")
    out, i, dropped = [], 0, 0
    while i < len(L):
        if L[i].strip().startswith("facet"):
            blk = L[i:i+7]
            vs = np.array([[float(x) for x in b.split()[1:4]] for b in blk if b.strip().startswith("vertex")])
            a = 0.5*np.linalg.norm(np.cross(vs[1]-vs[0], vs[2]-vs[0]))
            if a > 1e-16: out += blk
            else: dropped += 1
            i += 7
        else:
            out.append(L[i]); i += 1
    open(path, "w").write("\n".join(out))
    return dropped
for fn in sorted(glob.glob("constant/triSurface/*.stl")):
    d = strip_degenerate(fn)
    if d: print(fn, "dropped", d)
print("done")

# -- cell 21 ------------------------------------------------------------------------
tris3 = []
for fn in glob.glob("constant/triSurface/*.stl"):
    v = [tuple(map(float, l.split()[1:4])) for l in open(fn) if l.strip().startswith("vertex")]
    tris3 += [v[i:i+3] for i in range(0, len(v), 3)]
V3 = np.round(np.array(tris3).reshape(-1,3), 9) + 0.0
u3, i3 = np.unique(V3, axis=0, return_inverse=True); T3 = i3.reshape(-1,3)
E3 = Counter(tuple(sorted((T3[i,a], T3[i,(a+1)%3]))) for i in range(len(T3)) for a in range(3))
print("tris:", len(T3), "free edges:", sum(1 for c in E3.values() if c==1), "non-manifold:", sum(1 for c in E3.values() if c>2))

# -- cell 22 ------------------------------------------------------------------------
def normal_sense(fn):
    ns, cs = [], []
    L = open(fn).read().split("\n")
    for i,l in enumerate(L):
        if l.strip().startswith("facet"):
            ns.append([float(x) for x in l.split()[2:5]])
            cs.append(np.mean([[float(x) for x in L[i+k].split()[1:4]] for k in (2,3,4)], axis=0))
    ns, cs = np.array(ns), np.array(cs)
    return (np.einsum("ij,ij->i", ns, cs/np.linalg.norm(cs,axis=1)[:,None])).mean()
for fn in ["ball", "opening_zmax", "hollowWall", "holeWall_zmax"]:
    print(fn, "mean n.r_hat =", round(normal_sense(f"constant/triSurface/{fn}.stl"), 3))

# -- cell 23 ------------------------------------------------------------------------
subprocess.run(["blockMesh"], capture_output=True, text=True)
open("system/snappyHexMeshDict","w").write(snappy_dict(3))
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, [l for l in r.stdout.splitlines() if "Finished meshing" in l])
c = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True).stdout
print("\n".join(l for l in c.splitlines() if any(k in l for k in ("cells:","Total volume","non-orthogonality","Max skewness","Mesh OK","***","Number of regions"))))
