"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Blocky, orthogonal geometry — the right tool is a multi-block `blockMesh` with two cell zones split 
# Interpretation: "set into its floor" = flush-mounted, the block occupies a recess below the duct flo
from build123d import *
import os, subprocess

L, W, H = 0.200, 0.030, 0.030      # duct length, width, height (m)
BL, BW, BT = 0.040, 0.030, 0.010   # block length, width, thickness (m)
XC = L/2                            # block centred halfway along

fluid = Pos(L/2, W/2, H/2) * Box(L, W, H)
solid = Pos(XC, W/2, -BT/2) * Box(BL, BW, BT)

for nm, s, want in [("fluid", fluid, (L, W, H)), ("solid", solid, (BL, BW, BT))]:
    bb = s.bounding_box()
    print(nm, "bbox m:", [round(v,4) for v in (bb.min.X, bb.max.X, bb.min.Y, bb.max.Y, bb.min.Z, bb.max.Z)],
          "size:", [round(v,4) for v in (bb.size.X, bb.size.Y, bb.size.Z)],
          "want:", want, "vol:", round(s.volume,9))
print("interface area m^2:", round(BL*BW, 6), " (block top at z=0, flush with duct floor)")
export_stl(fluid, "fluid.stl"); export_stl(solid, "solid.stl")
print(os.listdir("."))

# -- cell 2 -------------------------------------------------------------------------
# Dimensions match. Look at the two bodies together before meshing.
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
p = pv.Plotter(off_screen=True, window_size=(1000, 650))
p.add_mesh(pv.read("fluid.stl"), color="lightblue", opacity=0.35, show_edges=True)
p.add_mesh(pv.read("solid.stl"), color="orange", show_edges=True)
p.add_axes()
p.camera_position = [(0.30, -0.25, 0.18), (0.10, 0.015, 0.005), (0, 0, 1)]
p.screenshot("geom.png")
print("ok")

# -- cell 3 -------------------------------------------------------------------------
# The screenshot didn't come back attached — display it explicitly.
# [图片]
from IPython.display import Image, display
import os
print(os.path.getsize("geom.png"))
display(Image(filename="geom.png"))

# -- cell 4 -------------------------------------------------------------------------
# Now the mesh: one `blockMesh` with four blocks — three fluid blocks along x and one solid block unde
import subprocess, os, textwrap

XS = [0.0, 0.080, 0.120, 0.200]   # x stations (m)
YS = [0.0, 0.030]                 # duct width
ZS = [-0.010, 0.0, 0.030]         # solid bottom, floor/interface, duct top
NX = [16, 8, 16]                  # cells per x segment  (5 mm)
NY = 6                            # across width          (5 mm)
NZF, NZS = 6, 2                   # fluid height / solid thickness (5 mm)

def vid(i, j, k): return i + 4*j + 8*k

verts = []
for k, z in enumerate(ZS):
    for j, y in enumerate(YS):
        for i, x in enumerate(XS):
            verts.append(f"    ({x:.6f} {y:.6f} {z:.6f})  // {vid(i,j,k)}")

def hexb(i0, i1, k0, k1, zone, n):
    v = [vid(i0,0,k0), vid(i1,0,k0), vid(i1,1,k0), vid(i0,1,k0),
         vid(i0,0,k1), vid(i1,0,k1), vid(i1,1,k1), vid(i0,1,k1)]
    return f"    hex ({' '.join(map(str,v))}) {zone} ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)"

blocks = [hexb(s, s+1, 1, 2, "fluid", (NX[s], NY, NZF)) for s in range(3)]
blocks.append(hexb(1, 2, 0, 1, "solid", (NX[1], NY, NZS)))

def fx(i, k0, k1, sign):  # constant-x face
    a = [vid(i,0,k0), vid(i,0,k1), vid(i,1,k1), vid(i,1,k0)]
    return a if sign < 0 else list(reversed(a))
def fy(i0, i1, j, k0, k1, sign):
    a = [vid(i0,j,k0), vid(i1,j,k0), vid(i1,j,k1), vid(i0,j,k1)]
    return a if sign < 0 else list(reversed(a))
def fz(i0, i1, k, sign):
    a = [vid(i0,0,k), vid(i1,0,k), vid(i1,1,k), vid(i0,1,k)]
    return a if sign > 0 else list(reversed(a))

faces = {"inlet": [fx(0,1,2,-1)], "outlet": [fx(3,1,2,+1)], "duct_wall": [], "block_wall": []}
for s in range(3):
    faces["duct_wall"] += [fz(s,s+1,2,+1), fy(s,s+1,0,1,2,-1), fy(s,s+1,1,1,2,+1)]
faces["duct_wall"] += [fz(0,1,1,-1), fz(2,3,1,-1)]          # floor outside the block
faces["block_wall"] += [fz(1,2,0,-1), fx(1,0,1,-1), fx(2,0,1,+1),
                        fy(1,2,0,0,1,-1), fy(1,2,1,0,1,+1)]

def pblock(name, typ, fl):
    s = f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n"
    s += "".join(f"            ({' '.join(map(str,f))})\n" for f in fl)
    return s + "        );\n    }\n"

d = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n\n"
d += "scale 1;\n\nvertices\n(\n" + "\n".join(verts) + "\n);\n\nblocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges();\n\nboundary\n(\n"
d += pblock("inlet", "patch", faces["inlet"]) + pblock("outlet", "patch", faces["outlet"])
d += pblock("duct_wall", "wall", faces["duct_wall"]) + pblock("block_wall", "wall", faces["block_wall"])
d += ");\n\nmergePatchPairs();\n"

os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
open("system/blockMeshDict", "w").write(d)
open("system/controlDict", "w").write(
    "FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}\n"
    "application chtMultiRegionFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
    "endTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
open("system/fvSchemes", "w").write(
    "FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\n"
    "ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\n"
    "divSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n"
    "interpolationSchemes{default linear;}\nsnGradSchemes{default corrected;}\n")
open("system/fvSolution", "w").write(
    "FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}\n")
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-1500:])

# -- cell 5 -------------------------------------------------------------------------
# Zones as expected (1440 fluid, 96 solid). Check the single mesh, then split into regions.
import subprocess
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1200:])
r2 = subprocess.run(["splitMeshRegions", "-cellZones", "-overwrite"], capture_output=True, text=True)
print(r2.stdout[-1800:], r2.returncode)

# -- cell 6 -------------------------------------------------------------------------
# Interface is 48 faces = 8×6 at 5 mm over the 40×30 mm block top. Inspect the two region boundary fil
import subprocess
for reg in ("fluid", "solid"):
    print("="*20, reg)
    print(open(f"constant/{reg}/polyMesh/boundary").read()[400:2600])

# -- cell 7 -------------------------------------------------------------------------
# Shape and interface are right at 5 mm. Refine to 2.5 mm cells (12 cells across the duct, 4 through t
import subprocess
NX = [32, 16, 32]; NY = 12; NZF = 12; NZS = 4     # 2.5 mm cells

verts = []
for k, z in enumerate(ZS):
    for j, y in enumerate(YS):
        for i, x in enumerate(XS):
            verts.append(f"    ({x:.6f} {y:.6f} {z:.6f})  // {vid(i,j,k)}")
blocks = [hexb(s, s+1, 1, 2, "fluid", (NX[s], NY, NZF)) for s in range(3)]
blocks.append(hexb(1, 2, 0, 1, "solid", (NX[1], NY, NZS)))

d = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n\n"
d += "scale 1;\n\nvertices\n(\n" + "\n".join(verts) + "\n);\n\nblocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges();\n\nboundary\n(\n"
d += pblock("inlet", "patch", faces["inlet"]) + pblock("outlet", "patch", faces["outlet"])
d += pblock("duct_wall", "wall", faces["duct_wall"]) + pblock("block_wall", "wall", faces["block_wall"])
d += ");\n\nmergePatchPairs();\n"
open("system/blockMeshDict", "w").write(d)

print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-600:])
print(subprocess.run(["splitMeshRegions", "-cellZones", "-overwrite"], capture_output=True, text=True).stdout[-700:])

# -- cell 8 -------------------------------------------------------------------------
# Check each region and verify the patch areas against the requested dimensions.
import subprocess, re
for reg in ("fluid", "solid"):
    r = subprocess.run(["checkMesh", "-region", reg], capture_output=True, text=True)
    out = r.stdout
    print("="*15, reg, "cells:", re.search(r"cells:\s+(\d+)", out).group(1),
          "| verdict:", [l for l in out.splitlines() if "Mesh OK" in l or "FAILED" in l or "***" in l])
    print("  bbox:", re.search(r"Overall domain bounding box.*", out).group(0))
    print("  maxNonOrtho:", re.search(r"non-orthogonality Max: \S+", out).group(0),
          "| maxSkew:", re.search(r"Max skewness = \S+", out).group(0))
    print(subprocess.run(["checkMesh", "-region", reg, "-writeFields", "no"], capture_output=True, text=True).stdout.count("FAILED"))

# -- cell 9 -------------------------------------------------------------------------
# Verify the interface numerically: face counts and total area on both sides, parsed straight from the
import re, numpy as np

def read_points(p):
    t = open(p).read(); body = t[t.index("(", t.index("\n(") ):]
    nums = np.array(re.findall(r"\(([-\d.eE+ ]+)\)", body))
    return np.array([list(map(float, s.split())) for s in nums])

def read_faces(p):
    t = open(p).read(); body = t[t.rindex("\n(\n"):]
    return [list(map(int, s.split())) for s in re.findall(r"\d+\(([\d ]+)\)", body)]

def patch_area(reg, patch):
    b = open(f"constant/{reg}/polyMesh/boundary").read()
    m = re.search(patch + r"\s*\{[^}]*nFaces\s+(\d+);[^}]*startFace\s+(\d+);", b, re.S)
    n, s = int(m.group(1)), int(m.group(2))
    P = read_points(f"constant/{reg}/polyMesh/points"); F = read_faces(f"constant/{reg}/polyMesh/faces")
    A = 0.0
    for f in F[s:s+n]:
        v = P[f]; A += 0.5*np.linalg.norm(np.cross(v[2]-v[0], v[3]-v[1]))
    return n, A

for reg, pat, want in [("fluid","fluid_to_solid",0.0012), ("solid","solid_to_fluid",0.0012),
                       ("fluid","inlet",0.0009), ("fluid","outlet",0.0009)]:
    n, A = patch_area(reg, pat)
    print(f"{reg}/{pat}: {n} faces, area {A:.6g} m^2 (want {want})")

# -- cell 10 ------------------------------------------------------------------------
# See what foamToVTK actually wrote.
import subprocess, os
r = subprocess.run(["foamToVTK", "-region", "fluid", "-constant"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-500:])
for root, dirs, files in os.walk("VTK"):
    print(root, dirs[:5], files[:5])

# -- cell 11 ------------------------------------------------------------------------
import pyvista as pv
from IPython.display import Image, display
p = pv.Plotter(off_screen=True, window_size=(1100, 700))
p.add_mesh(pv.read("VTK/fluid/t8_0/internal.vtu"), color="steelblue", opacity=0.18, show_edges=True, edge_color="lightgray")
p.add_mesh(pv.read("VTK/fluid/t8_0/boundary/inlet.vtp"), color="green", show_edges=True)
p.add_mesh(pv.read("VTK/fluid/t8_0/boundary/outlet.vtp"), color="red", show_edges=True)
p.add_mesh(pv.read("VTK/solid/t8_0/internal.vtu"), color="orange", show_edges=True)
p.add_mesh(pv.read("VTK/solid/t8_0/boundary/solid_to_fluid.vtp"), color="purple", show_edges=True)
p.camera_position = [(0.30, -0.24, 0.17), (0.10, 0.015, 0.0), (0, 0, 1)]
p.add_axes(); p.screenshot("regions.png"); display(Image("regions.png"))

# -- cell 12 ------------------------------------------------------------------------
# Clean up the VTK dump and the `cellToRegion` fields left by the split, keeping the two region meshes
import shutil, os, subprocess
shutil.rmtree("VTK", ignore_errors=True)
shutil.rmtree("0", ignore_errors=True)
for f in ("case.foam", "constant/cellToRegion"):
    if os.path.exists(f): os.remove(f)
print(sorted(os.listdir(".")), sorted(os.listdir("constant")))
print(subprocess.run(["checkMesh","-region","fluid"],capture_output=True,text=True).stdout.strip()[-30:],
      subprocess.run(["checkMesh","-region","solid"],capture_output=True,text=True).stdout.strip()[-30:])
