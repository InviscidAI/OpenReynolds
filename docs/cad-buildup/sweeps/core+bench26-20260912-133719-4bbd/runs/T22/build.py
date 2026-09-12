"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: the air volume is the 12 mm gap between the two friction faces, an annulus from r=95 mm to r=1
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build123d import *

RO, RI, H, N, W = 0.160, 0.095, 0.012, 36, 0.008   # metres
OVER = 0.004                                       # radial overshoot for clean cuts

annulus = Circle(RO) - Circle(RI)
vane_blank = Pos((RI + RO) / 2, 0) * Rectangle(RO - RI + 2 * OVER, W)
vanes = [Rot(0, 0, k * 360.0 / N) * vane_blank for k in range(N)]
section = annulus - vanes

faces = section.faces()
print("passages:", len(faces), " expected", N)
print("area of one passage [mm^2]:", faces[0].area * 1e6)
print("total fluid area [mm^2]  :", section.area * 1e6)

fig, ax = plt.subplots(figsize=(6, 6))
for e in section.edges():
    p = np.array([tuple(e @ t) for t in np.linspace(0, 1, 40)])
    ax.plot(p[:, 0] * 1e3, p[:, 1] * 1e3, "k-", lw=0.7)
ax.set_aspect("equal"); ax.set_title("fluid cross-section [mm]")
fig.savefig("section.png", dpi=90)
print("bbox mm:", [round(v * 1e3, 2) for v in (section.bounding_box().min.X, section.bounding_box().max.X)])

# -- cell 2 -------------------------------------------------------------------------
# `display` was an IPython freebie — noted; figures will be the last expression from here on. Now the 
from collections import defaultdict
solid = extrude(section, amount=H)
print("solids:", len(solid.solids()), " volume [mm^3]:", solid.volume*1e9, " expected:", section.area*H*1e9)

groups = defaultdict(list)
for f in solid.faces():
    c = f.center()
    if f.geom_type == GeomType.PLANE:
        n = f.normal_at(c)
        if abs(n.Z) > 0.99:
            groups["friction_bottom" if c.Z < H/2 else "friction_top"].append(f)
        else:
            groups["vanes"].append(f)
    elif f.geom_type == GeomType.CYLINDER:
        r = (c.X**2 + c.Y**2)**0.5
        groups["inlet" if r < (RI+RO)/2 else "outlet"].append(f)
    else:
        groups["UNKNOWN"].append(f)
for k, v in groups.items():
    print(f"{k:16s} nfaces={len(v):4d}  area[mm^2]={sum(f.area for f in v)*1e6:10.1f}")

# -- cell 3 -------------------------------------------------------------------------
# All five groups measure right (vane flanks 72×65×12 ≈ 56.2e3 mm²; inlet 12·(2π·95 − 36·8) ≈ 3707 mm²
import os
from build123d import Compound, export_stl
os.makedirs("constant/triSurface", exist_ok=True)
PATCHES = ["friction_top", "friction_bottom", "vanes", "inlet", "outlet"]
for name in PATCHES:
    shp = Compound(children=[f for f in groups[name]])
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.2)
    print(name, os.path.getsize(f"constant/triSurface/{name}.stl"), "bytes")

# -- cell 4 -------------------------------------------------------------------------
# STEP out the section, then extrude it in gmsh with recombination for hexes — coarse first (≈4 mm in-
import gmsh, numpy as np
from build123d import export_step
export_step(section, "section.step")

NZ, LC = 3, 0.004
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("vented_disc")
base = gmsh.model.occ.importShapes("section.step")
base = [d for d in base if d[0] == 2]
ext = gmsh.model.occ.extrude(base, 0, 0, H, numElements=[NZ], recombine=True)
gmsh.model.occ.synchronize()
vols = [d for d in ext if d[0] == 3]
print("base faces:", len(base), " volumes:", len(vols), " total surfaces:", len(gmsh.model.getEntities(2)))

# -- cell 5 -------------------------------------------------------------------------
# Now tag physical groups from each surface's own type and location, and mesh.
from collections import defaultdict as dd
tags = dd(list)
for dim, tag in gmsh.model.getEntities(2):
    typ = gmsh.model.getType(2, tag)
    x, y, z = gmsh.model.occ.getCenterOfMass(2, tag)
    if typ == "Plane":
        n = gmsh.model.getNormal(tag, [0.5, 0.5])
        if abs(n[2]) > 0.99:
            tags["friction_bottom" if z < H/2 else "friction_top"].append(tag)
        else:
            tags["vanes"].append(tag)
    elif typ == "Cylinder":
        tags["inlet" if (x*x + y*y)**0.5 < (RI+RO)/2 else "outlet"].append(tag)
    else:
        tags["UNKNOWN_" + typ].append(tag)
for k, v in tags.items():
    print(k, len(v))

# -- cell 6 -------------------------------------------------------------------------
# Physical groups (volume + the five patches), then generate the coarse hex mesh.
for name, tl in tags.items():
    gmsh.model.addPhysicalGroup(2, tl, name=name)
gmsh.model.addPhysicalGroup(3, [t for d, t in vols], name="fluid")

for d, t in base:
    gmsh.model.mesh.setRecombine(2, t)
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.MeshSizeMin", LC)
gmsh.option.setNumber("Mesh.MeshSizeMax", LC)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3)
print("hexes:", len(gmsh.model.mesh.getElementsByType(5)[0]), "prisms:", len(gmsh.model.mesh.getElementsByType(6)[0]))
gmsh.write("disc.msh")
print("written")

# -- cell 7 -------------------------------------------------------------------------
# Now the OpenFOAM case skeleton and conversion.
import subprocess, textwrap, os
os.makedirs("system", exist_ok=True)
head = lambda cls, obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(head("dictionary","controlDict") + textwrap.dedent("""
    application simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
    endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
    """))
open("system/fvSchemes","w").write(head("dictionary","fvSchemes") + "gradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\nddtSchemes{default steadyState;}\n")
open("system/fvSolution","w").write(head("dictionary","fvSolution") + "solvers{}\n")
r = subprocess.run(["gmshToFoam","disc.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read()[-1600:])

# -- cell 9 -------------------------------------------------------------------------
# All 10656 boundary faces landed in the five named patches (no `defaultFaces` left). Set the solid on
import subprocess
for p in ["friction_bottom", "friction_top", "vanes"]:
    subprocess.run(["foamDictionary", "constant/polyMesh/boundary",
                    "-entry", f"entry0/{p}/type", "-set", "wall"], check=True, capture_output=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 10 ------------------------------------------------------------------------
# checkMesh passes on the coarse version (max non-orth 44.2, skewness 1.33, volume 400.15 cm³ vs 400.1
def build_gmsh_mesh(lc, nz, out="disc.msh"):
    gmsh.clear(); gmsh.model.add("vented_disc_f")
    b = [d for d in gmsh.model.occ.importShapes("section.step") if d[0] == 2]
    e = gmsh.model.occ.extrude(b, 0, 0, H, numElements=[nz], recombine=True)
    gmsh.model.occ.synchronize()
    g = dd(list)
    for dim, tag in gmsh.model.getEntities(2):
        typ = gmsh.model.getType(2, tag)
        x, y, z = gmsh.model.occ.getCenterOfMass(2, tag)
        if typ == "Plane":
            n = gmsh.model.getNormal(tag, [0.5, 0.5])
            g["friction_bottom" if z < H/2 else "friction_top"].append(tag) if abs(n[2]) > 0.99 else g["vanes"].append(tag)
        elif typ == "Cylinder":
            g["inlet" if (x*x + y*y)**0.5 < (RI+RO)/2 else "outlet"].append(tag)
        else:
            raise RuntimeError("unexpected surface type " + typ)
    for nm, tl in g.items():
        gmsh.model.addPhysicalGroup(2, tl, name=nm)
    gmsh.model.addPhysicalGroup(3, [t for d, t in e if d == 3], name="fluid")
    for d, t in b:
        gmsh.model.mesh.setRecombine(2, t)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc); gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(out)
    return {k: len(v) for k, v in g.items()}, len(gmsh.model.mesh.getElementsByType(5)[0]), len(gmsh.model.mesh.getElementsByType(6)[0])

print(build_gmsh_mesh(0.002, 6))

# -- cell 11 ------------------------------------------------------------------------
# Same 216 surfaces classified identically. Convert and check the fine mesh.
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
subprocess.run(["gmshToFoam", "disc.msh"], check=True, capture_output=True)
for p in ["friction_bottom", "friction_top", "vanes"]:
    subprocess.run(["foamDictionary", "constant/polyMesh/boundary",
                    "-entry", f"entry0/{p}/type", "-set", "wall"], check=True, capture_output=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Checking patch topology"):][:900])
print(r.stdout[r.stdout.find("Checking geometry"):][:1600])

# -- cell 12 ------------------------------------------------------------------------
# Mesh OK: 57450 cells, max non-orth 41.1, max skewness 0.95, volume 400.17 cm³ (CAD: 400.18). Final v
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
pl = pv.Plotter(off_screen=True, window_size=(900, 800))
cols = {"friction_bottom": "lightgray", "friction_top": "peachpuff", "vanes": "steelblue",
        "inlet": "green", "outlet": "red"}
for nm, c in cols.items():
    m = pv.read(f"constant/triSurface/{nm}.stl")
    if nm == "friction_top":
        m = m.clip(normal="x", origin=(0, 0, 0))
    pl.add_mesh(m, color=c, show_edges=False, label=nm)
pl.add_legend()
pl.camera_position = [(0.30, -0.30, 0.28), (0, 0, 0.006), (0, 0, 1)]
img = pl.screenshot("patches.png")
from IPython.display import Image
Image("patches.png")

# -- cell 13 ------------------------------------------------------------------------
# The render confirms the topology. One last numeric audit of the dimensions the request named, measur
import numpy as np
rad = lambda fs: sorted({round((f.center().X**2 + f.center().Y**2)**0.5, 6) for f in fs})
ci = [f for f in groups["inlet"]]; co = [f for f in groups["outlet"]]
print("inlet  cyl radius [mm]:", round(max(np.hypot(*np.array([v.to_tuple()[:2] for v in ci[0].vertices()]).T)) * 1e3, 3), "vs 95")
print("outlet cyl radius [mm]:", round(max(np.hypot(*np.array([v.to_tuple()[:2] for v in co[0].vertices()]).T)) * 1e3, 3), "vs 160")
bb = solid.bounding_box()
print("gap height [mm]:", round((bb.max.Z - bb.min.Z) * 1e3, 3), "vs 12")
f0 = [f for f in groups["vanes"] if abs(f.center().Y) < 0.01 and f.center().X > 0]
print("flanks near +x axis:", len(f0), " y of centres [mm]:", sorted(round(f.center().Y * 1e3, 3) for f in f0))
print("=> vane width [mm]:", round(abs(f0[0].center().Y - f0[1].center().Y) * 1e3, 3), "vs 8")
print("passages:", len(solid.solids()), "vs 36   pitch [deg]:", 360 / 36)
