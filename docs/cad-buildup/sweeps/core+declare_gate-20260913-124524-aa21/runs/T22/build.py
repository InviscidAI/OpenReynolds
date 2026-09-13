"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run that cell.
import numpy as np, os
from build123d import *

R_out, R_in, GAP, N_VANE, W_VANE = 0.160, 0.095, 0.012, 36, 0.008

annulus = Cylinder(R_out, GAP) - Cylinder(R_in, GAP)
vane1 = Box(0.090, W_VANE, GAP*1.2).moved(Pos((R_in+R_out)/2, 0, 0))
vanes = [vane1.moved(Rot(0, 0, 360.0*i/N_VANE)) for i in range(N_VANE)]
fluid = annulus - vanes
print(fluid.volume, len(fluid.solids()), fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Now classify faces by their own geometry (surface type, normal, radius) into the five named patches.
groups = {"frictionTop": [], "frictionBottom": [], "vanes": [], "inlet": [], "outlet": []}
for f in fluid.faces():
    c = f.center()
    if f.geom_type == GeomType.PLANE:
        n = f.normal_at(c)
        if abs(n.Z) > 0.99:
            groups["frictionTop" if c.Z > 0 else "frictionBottom"].append(f)
        else:
            groups["vanes"].append(f)
    elif f.geom_type == GeomType.CYLINDER:
        r = (c.X**2 + c.Y**2)**0.5
        groups["outlet" if r > (R_in+R_out)/2 else "inlet"].append(f)
    else:
        raise RuntimeError(f.geom_type)
for k, v in groups.items():
    print(k, len(v), round(sum(f.area for f in v), 6))

# -- cell 3 -------------------------------------------------------------------------
# Areas match analytics (friction 0.03335 m² = annulus − 36×8×65 mm; vane 72×65×12 mm; inlet/outlet ar
import os
os.makedirs("constant/triSurface", exist_ok=True)
TOL = 0.0005

def tri_of(faces):
    tris = []
    for f in faces:
        v, t = f.tessellate(TOL)
        v = [(p.X, p.Y, p.Z) for p in v]
        for a, b, c in t:
            tris.append((v[a], v[b], v[c]))
    return tris

def write_solid(fh, name, tris):
    fh.write(f"solid {name}\n")
    for p0, p1, p2 in tris:
        u = np.subtract(p1, p0); w = np.subtract(p2, p0)
        n = np.cross(u, w); ln = np.linalg.norm(n)
        n = n/ln if ln > 0 else n
        fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
        for p in (p0, p1, p2):
            fh.write(f"   vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")
        fh.write("  endloop\n endfacet\n")
    fh.write(f"endsolid {name}\n")

alltris = {k: tri_of(v) for k, v in groups.items()}
for k, t in alltris.items():
    with open(f"constant/triSurface/{k}.stl", "w") as fh:
        write_solid(fh, k, t)
with open("constant/triSurface/brakeDisc.stl", "w") as fh:
    for k, t in alltris.items():
        write_solid(fh, k, t)
print({k: len(t) for k, t in alltris.items()}, os.listdir("constant/triSurface"))

# -- cell 4 -------------------------------------------------------------------------
from OCP.TopLoc import TopLoc_Location
alltris = {k: tri_of(v) for k, v in groups.items()}
for k, t in alltris.items():
    with open(f"constant/triSurface/{k}.stl", "w") as fh:
        write_solid(fh, k, t)
with open("constant/triSurface/brakeDisc.stl", "w") as fh:
    for k, t in alltris.items():
        write_solid(fh, k, t)
print({k: len(t) for k, t in alltris.items()})

# -- cell 5 -------------------------------------------------------------------------
# Check the union is watertight and normals consistent, by counting edge multiplicities across all pat
from collections import Counter
def key(p): return tuple(round(c, 9) for c in p)
ec = Counter(); dc = Counter()
for t in [tt for v in alltris.values() for tt in v]:
    k = [key(p) for p in t]
    for i in range(3):
        e = (k[i], k[(i+1) % 3])
        ec[frozenset(e)] += 1; dc[e] += 1
print("edges", len(ec), "free", sum(1 for v in ec.values() if v == 1), "nonmanifold", sum(1 for v in ec.values() if v > 2))
print("bad-oriented(dup dir)", sum(1 for v in dc.values() if v > 1))

# -- cell 6 -------------------------------------------------------------------------
for nd in (7, 6, 5):
    ec2 = Counter()
    for t in [tt for v in alltris.values() for tt in v]:
        k = [tuple(round(c, nd) for c in p) for p in t]
        for i in range(3):
            ec2[frozenset((k[i], k[(i+1) % 3]))] += 1
    print(nd, "edges", len(ec2), "free", sum(1 for v in ec2.values() if v == 1))
free = [e for e, v in ec.items() if v == 1][:3]
print(free)

# -- cell 7 -------------------------------------------------------------------------
def nodes(name, cond):
    s = set()
    for t in alltris[name]:
        for p in t:
            if cond(p): s.add(key(p))
    return s
c = lambda p: abs(p[2]-0.006) < 1e-9 and abs((p[0]**2+p[1]**2)**0.5-0.095) < 1e-6 and p[1] < 0 and p[0] > 0.09
a = nodes("frictionTop", c); b = nodes("inlet", c)
print(len(a), len(b)); print(sorted(a)[:5]); print(sorted(b)[:5])

# -- cell 8 -------------------------------------------------------------------------
# The OCC tessellation leaves mismatched nodes on shared edges. Since each passage is a simple prism, 
NU, NV, NZ = 14, 6, 3
half = W_VANE/2
xA_i, xA_o = (R_in**2-half**2)**0.5, (R_out**2-half**2)**0.5
pitch = 2*np.pi/N_VANE

def rot(p, a): return np.array([p[0]*np.cos(a)-p[1]*np.sin(a), p[0]*np.sin(a)+p[1]*np.cos(a)])
P00, P10 = np.array([xA_i, half]), np.array([xA_o, half])
P01, P11 = rot([xA_i, -half], pitch), rot([xA_o, -half], pitch)
thI = (np.arctan2(P00[1], P00[0]), np.arctan2(P01[1], P01[0]))
thO = (np.arctan2(P10[1], P10[0]), np.arctan2(P11[1], P11[0]))

def grid2d():
    g = np.zeros((NU+1, NV+1, 2))
    for i in range(NU+1):
        u = i/NU
        A, B = P00+u*(P10-P00), P01+u*(P11-P01)
        for j in range(NV+1):
            v = j/NV
            tI, tO = thI[0]+v*(thI[1]-thI[0]), thO[0]+v*(thO[1]-thO[0])
            I = R_in*np.array([np.cos(tI), np.sin(tI)]); O = R_out*np.array([np.cos(tO), np.sin(tO)])
            g[i, j] = ((1-v)*A + v*B + (1-u)*I + u*O
                       - ((1-u)*(1-v)*P00 + u*(1-v)*P10 + (1-u)*v*P01 + u*v*P11))
    return g

G = grid2d()
print(np.linalg.norm(G[0,0]-P00), np.linalg.norm(G[NU,NV]-P11), np.linalg.norm(G[0,3]), np.linalg.norm(G[NU,3]))

# -- cell 9 -------------------------------------------------------------------------
zs = np.linspace(-GAP/2, GAP/2, NZ+1)
Nd = np.zeros((NU+1, NV+1, NZ+1, 3))
Nd[:, :, :, :2] = G[:, :, None, :]
Nd[:, :, :, 2] = zs[None, None, :]

def quad(p0, p1, p2, p3, ref):
    n = np.cross(np.array(p1)-p0, np.array(p2)-p0)
    if np.dot(n, ref) < 0: p0, p1, p2, p3 = p0, p3, p2, p1
    return [(tuple(p0), tuple(p1), tuple(p2)), (tuple(p0), tuple(p2), tuple(p3))]

base = {k: [] for k in groups}
for i in range(NU):
    for j in range(NV):
        base["frictionTop"] += quad(Nd[i,j,NZ], Nd[i+1,j,NZ], Nd[i+1,j+1,NZ], Nd[i,j+1,NZ], [0,0,1])
        base["frictionBottom"] += quad(Nd[i,j,0], Nd[i+1,j,0], Nd[i+1,j+1,0], Nd[i,j+1,0], [0,0,-1])
for i in range(NU):
    for k in range(NZ):
        base["vanes"] += quad(Nd[i,0,k], Nd[i+1,0,k], Nd[i+1,0,k+1], Nd[i,0,k+1], [0,-1,0])
        base["vanes"] += quad(Nd[i,NV,k], Nd[i+1,NV,k], Nd[i+1,NV,k+1], Nd[i,NV,k+1], rot([0,1,0], pitch).tolist()+[0])
for j in range(NV):
    for k in range(NZ):
        cin = -np.array([*Nd[0,j,k][:2], 0]); cout = np.array([*Nd[NU,j,k][:2], 0])
        base["inlet"] += quad(Nd[0,j,k], Nd[0,j+1,k], Nd[0,j+1,k+1], Nd[0,j,k+1], cin)
        base["outlet"] += quad(Nd[NU,j,k], Nd[NU,j+1,k], Nd[NU,j+1,k+1], Nd[NU,j,k+1], cout)

alltris = {k: [] for k in base}
for s in range(N_VANE):
    a = s*pitch; ca, sa = np.cos(a), np.sin(a)
    for k, tl in base.items():
        for t in tl:
            alltris[k].append(tuple((p[0]*ca-p[1]*sa, p[0]*sa+p[1]*ca, p[2]) for p in t))
print({k: len(v) for k, v in alltris.items()})

# -- cell 10 ------------------------------------------------------------------------
ec = Counter(); dc = Counter(); vol = 0.0
for t in [tt for v in alltris.values() for tt in v]:
    k = [key(p) for p in t]
    for i in range(3):
        ec[frozenset((k[i], k[(i+1) % 3]))] += 1; dc[(k[i], k[(i+1) % 3])] += 1
    p0, p1, p2 = (np.array(p) for p in t)
    vol += np.dot(p0, np.cross(p1, p2))/6
print("free", sum(1 for v in ec.values() if v == 1), "nonmanifold", sum(1 for v in ec.values() if v > 2),
      "same-dir-dup", sum(1 for v in dc.values() if v > 1))
print("STL volume", vol, "CAD volume", fluid.volume)

# -- cell 11 ------------------------------------------------------------------------
# Watertight, outward-oriented, volume within 0.01 % of the CAD solid. Write the per-patch STLs and th
for k, t in alltris.items():
    with open(f"constant/triSurface/{k}.stl", "w") as fh:
        write_solid(fh, k, t)
with open("constant/triSurface/brakeDisc.stl", "w") as fh:
    for k in ["frictionTop", "frictionBottom", "vanes", "inlet", "outlet"]:
        write_solid(fh, k, alltris[k])
print(subprocess.run(["ls", "-la", "constant/triSurface"], capture_output=True, text=True).stdout if 'subprocess' in dir() else os.listdir("constant/triSurface"))

# -- cell 12 ------------------------------------------------------------------------
# Now the mesher. cfMesh's `cartesianMesh` fills all 36 disconnected passages without needing a seed p
import subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = lambda cls, obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict", "w").write(hdr("dictionary", "controlDict") + textwrap.dedent("""
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
"""))
open("system/fvSchemes", "w").write(hdr("dictionary", "fvSchemes") + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution", "w").write(hdr("dictionary", "fvSolution") + "solvers{}\n")
open("system/meshDict", "w").write(hdr("dictionary", "meshDict") + textwrap.dedent("""
surfaceFile "constant/triSurface/brakeDisc.stl";
maxCellSize 0.003;
boundaryCellSize 0.003;
"""))
r = subprocess.run(["cartesianMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh", "-constant"], capture_output=True, text=True)
print(r.stdout[-2500:])
print(open("constant/polyMesh/boundary").read()[-1200:])

# -- cell 14 ------------------------------------------------------------------------
# Only one passage survived — cfMesh discards the disconnected regions. Switching to snappyHexMesh wit
CELL = 0.003
nx = int(round(2*0.165/CELL))
open("system/blockMeshDict", "w").write(hdr("dictionary", "blockMeshDict") + f"""
scale 1;
vertices ((-0.165 -0.165 -0.009)(0.165 -0.165 -0.009)(0.165 0.165 -0.009)(-0.165 0.165 -0.009)
          (-0.165 -0.165 0.009)(0.165 -0.165 0.009)(0.165 0.165 0.009)(-0.165 0.165 0.009));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {nx} 6) simpleGrading (1 1 1));
edges (); boundary ((box {{type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3));}}));
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-400:], r.stderr[-400:])

# -- cell 15 ------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[:3000])

# -- cell 16 ------------------------------------------------------------------------
print("\n".join(l for l in r.stdout.splitlines() if "cwd" not in l and "POSIX" not in l and "Warning" not in l)[-2000:])

# -- cell 17 ------------------------------------------------------------------------
# The `boundary` entry had spurious parentheses. Fixing the dict.
open("system/blockMeshDict", "w").write(hdr("dictionary", "blockMeshDict") + f"""
scale 1;
vertices ((-0.165 -0.165 -0.009)(0.165 -0.165 -0.009)(0.165 0.165 -0.009)(-0.165 0.165 -0.009)
          (-0.165 -0.165 0.009)(0.165 -0.165 0.009)(0.165 0.165 0.009)(-0.165 0.165 0.009));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {nx} 6) simpleGrading (1 1 1));
edges ();
boundary
(
    box {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }}
);
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, [l for l in r.stdout.splitlines() if "cells" in l or "nCells" in l][-3:])

# -- cell 18 ------------------------------------------------------------------------
seeds = "\n".join(f"    ((%.6f %.6f 0.0) fluid)" % (0.1275*np.cos(pitch*(i+0.5)), 0.1275*np.sin(pitch*(i+0.5))) for i in range(N_VANE))
patches = {"frictionTop": "wall", "frictionBottom": "wall", "vanes": "wall", "inlet": "patch", "outlet": "patch"}
geom = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in patches)
refs = "\n".join(f'    {n} {{ level (1 1); patchInfo {{ type {t}; }} }}' for n, t in patches.items())
open("system/snappyHexMeshDict", "w").write(hdr("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}}
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces
  {{
{refs}
  }}
  refinementRegions {{}}
  locationsInMesh
  (
{seeds}
  );
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10;
  implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print(r.returncode, "\n".join(l for l in r.stdout.splitlines() if "cwd" not in l)[-1500:])

# -- cell 19 ------------------------------------------------------------------------
print(r.stderr[-1500:]); print("----"); print("\n".join(l for l in r.stdout.splitlines()[-40:] if "cwd" not in l))

# -- cell 20 ------------------------------------------------------------------------
# Seed points at exactly z = 0 land on a cell face plane. Offsetting them off the grid planes.
seeds = "\n".join("    ((%.6f %.6f 0.000731) fluid)" % (0.12731*np.cos(pitch*(i+0.5)), 0.12731*np.sin(pitch*(i+0.5))) for i in range(N_VANE))
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"locationsInMesh\s*\(.*?\);", "locationsInMesh\n  (\n" + seeds + "\n  );", d, flags=re.S)
open("system/snappyHexMeshDict", "w").write(d)
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stderr[-600:])
print("\n".join(l for l in r.stdout.splitlines()[-25:] if "cwd" not in l))

# -- cell 21 ------------------------------------------------------------------------
# Meshed. Check volume, patch counts, region count.
r = subprocess.run(["checkMesh", "-constant"], capture_output=True, text=True)
o = "\n".join(l for l in r.stdout.splitlines() if "cwd" not in l)
i = o.find("Mesh stats")
print(o[i:i+2600])

# -- cell 22 ------------------------------------------------------------------------
j = o.find("Checking patch topology")
print(o[j:])

# -- cell 23 ------------------------------------------------------------------------
# Mesh OK, 36 passages all present, volume 3.961e-4 vs CAD 4.002e-4 (1 % low from snapping). Now look 
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
os.makedirs("VTK", exist_ok=True)
subprocess.run(["foamToVTK", "-constant", "-latestTime"], capture_output=True, text=True)
import glob; print(glob.glob("VTK/**/*.vt*", recursive=True)[:10])

# -- cell 24 ------------------------------------------------------------------------
pv.global_theme.allow_empty_mesh = True
pl = pv.Plotter(off_screen=True, window_size=(1100, 900))
cols = {"frictionTop": "lightsteelblue", "frictionBottom": "lightgray", "vanes": "orange", "inlet": "green", "outlet": "red"}
for n, c in cols.items():
    m = pv.read(f"VTK/t22_0/boundary/{n}.vtp")
    if n == "frictionTop":
        m = m.clip(normal=(0, 1, 0), origin=(0, 0, 0))
    pl.add_mesh(m, color=c, show_edges=(n != "frictionTop"), edge_color="k", line_width=0.3, label=n)
pl.add_legend(); pl.camera_position = "iso"; pl.camera.azimuth = 20
pl.screenshot("view_iso.png"); print("ok")

# -- cell 25 ------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
img = plt.imread("view_iso.png"); print(img.shape, img.mean())
plt.figure(figsize=(9, 7)); plt.imshow(img); plt.axis("off"); plt.show()

# -- cell 26 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view_iso.png"))

# -- cell 27 ------------------------------------------------------------------------
# Measure the requested dimensions back out of the mesh boundary points.
P = {n: pv.read(f"VTK/t22_0/boundary/{n}.vtp").points for n in cols}
rad = lambda p: np.hypot(p[:, 0], p[:, 1])
print("inlet  r  min/max  %.4f %.4f  (target 0.095)" % (rad(P["inlet"]).min(), rad(P["inlet"]).max()))
print("outlet r  min/max  %.4f %.4f  (target 0.160)" % (rad(P["outlet"]).min(), rad(P["outlet"]).max()))
print("frictionTop z %.4f..%.4f ; Bottom z %.4f..%.4f (gap target 0.012)" %
      (P["frictionTop"][:,2].min(), P["frictionTop"][:,2].max(), P["frictionBottom"][:,2].min(), P["frictionBottom"][:,2].max()))
# vane width: tangential gap between two adjacent passages at mid radius, from vane patch points
v = P["vanes"]; sel = v[(np.abs(v[:,2]) < 1e-4) & (np.abs(rad(v)-0.1275) < 1e-3)]
th = np.sort(np.unique(np.round(np.arctan2(sel[:,1], sel[:,0]), 4)))
print("distinct vane-face angles near r=127.5mm:", len(th), " chordal width samples (mm):",
      np.round(np.diff(th)[np.diff(th) < 0.08]*0.1275*1000, 2)[:6])
