"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: a 2D planar fluid region built in build123d (mm, scaled to metres at the end), meshed with gms
import numpy as np, matplotlib.pyplot as plt
from build123d import *

W = 6.0        # channel width mm
L = 120.0      # overall length mm
ALPHA = 20.0   # branch take-off angle, deg
L1 = 22.9      # straight forward leg length
R1 = 8.0       # radius of first turn (up to vertical)
R2 = 5.0       # radius of the over-the-top return turn
PITCH = 27.0
X0 = 6.0
NLOOP = 4

def loop_path(xa):
    A = Vector(xa, 0, 0)
    d0 = Vector(np.cos(np.radians(ALPHA)), np.sin(np.radians(ALPHA)), 0)
    P1 = A + d0 * L1
    l1 = Line(A, P1)
    a1 = JernArc(start=P1, tangent=d0, radius=R1, arc_size=90 - ALPHA)
    a2 = JernArc(start=a1 @ 1, tangent=a1 % 1, radius=R2, arc_size=170)
    p2, t2 = a2 @ 1, a2 % 1
    l2len = -p2.Y / t2.Y
    B = p2 + t2 * l2len
    return (l1 + a1 + a2 + Line(p2, B)), A, B, t2

path, A, B, t2 = loop_path(X0)
pts = np.array([(path @ (i / 300)).to_tuple()[:2] for i in range(301)])
print("span A->B  =", round(B.X - A.X, 3), "mm   B =", (round(B.X,3), round(B.Y,4)))
print("x-extent   =", round(pts[:,0].max() - A.X, 3), "mm  (+3 for wall)")
print("max height =", round(pts[:,1].max(), 3), "mm")
print("re-entry dir angle =", round(np.degrees(np.arctan2(t2.Y, t2.X)), 2), "deg (180=straight upstream)")
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(pts[:,0], pts[:,1]); ax.plot([0,L],[0,0],'k--',lw=0.5); ax.set_aspect('equal'); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The centrelines cross at ~(20, 5.5) — that crossing is actually the splitter tip of the teardrop isl
GAP = 27.0     # loop pitch
XS = [3.0, 30.0, 57.0, 84.0]   # left-most x of each loop cell

def loop_mirrored(xs):
    """loop built at origin, mirrored in x, then shifted so its left-most centreline x = xs"""
    p, A, B, t2 = loop_path(0.0)
    pm = mirror(p, Plane.YZ)
    xmin = min((pm @ (i/400)).X for i in range(401))
    sh = xs - xmin
    return pm.moved(Location((sh, 0, 0))), Vector(A.X*-1+sh, 0, 0), Vector(B.X*-1+sh, 0, 0)

loops = [loop_mirrored(x) for x in XS]
fig, ax = plt.subplots(figsize=(12, 3))
for pm, Aj, Bj in loops:
    q = np.array([tuple(pm @ (i/400))[:2] for i in range(401)])
    ax.plot(q[:,0], q[:,1], lw=1)
    ax.plot([Aj.X, Bj.X], [0, 0], 'r.')
print("junctions  take-off A (shallow, faces -x):", [round(l[1].X,2) for l in loops])
print("           re-entry B (steep, fires +x)  :", [round(l[2].X,2) for l in loops])
print("loop span  :", round(loops[0][1].X - loops[0][2].X, 2), "mm ; pitch", GAP, "; last loop ends", round(loops[-1][1].X+3,2))
ax.plot([0, L], [3, 3], 'k--', lw=0.5); ax.plot([0, L], [-3, -3], 'k--', lw=0.5)
ax.set_aspect('equal'); plt.show()

# -- cell 3 -------------------------------------------------------------------------
# `trace` gives one face per loop, but `+` only compounded (area was exactly the sum). I'll fuse at th
from functools import reduce
main = Rectangle(L, W).moved(Location((L/2, 0, 0)))
bands = [trace(pm, line_width=W).faces()[0] for pm, _, _ in loops]
fused = reduce(lambda a, b: a.fuse(b), [main.faces()[0]] + bands).clean()
fl = fused.faces()
print("faces after fuse:", len(fl), " area:", round(sum(f.area for f in fl), 1),
      "mm^2  (sum of parts was", round(main.area + sum(b.area for b in bands), 1), ")")
face2d = fl[0]
print("wires on face:", len(face2d.wires()), "(1 outer + 4 islands expected)")
bb = face2d.bounding_box()
print("bbox mm: x", round(bb.min.X,2), round(bb.max.X,2), " y", round(bb.min.Y,2), round(bb.max.Y,2))

# -- cell 4 -------------------------------------------------------------------------
# 3D booleans on the OCC kernel are more reliable than sketch fusion, so I'll extrude each piece 1 mm 
TH = 1.0   # mm, single-cell thickness in z
main3 = extrude(main, amount=TH)
band3 = [extrude(Sketch() + b, amount=TH) for b in bands]
solid = reduce(lambda a, b: a + b, [main3] + band3).clean()
print("solids:", len(solid.solids()), " volume:", round(solid.volume, 1),
      "mm^3  (sum of parts:", round(main3.volume + sum(b.volume for b in band3), 1), ")")
s = solid.solids()[0]
bot = [f for f in s.faces() if abs(f.center().Z) < 1e-6 and abs(f.normal_at(f.center()).Z) > 0.99]
print("bottom faces:", len(bot), " area:", round(bot[0].area, 1), " wires:", len(bot[0].wires()))

# -- cell 5 -------------------------------------------------------------------------
# Good: one solid, 1364.3 mm³ over 1 mm → fluid area 1364.3 mm². But its z=0 boundary is 6 unmerged co
import os, subprocess, gmsh
faces_mm = Compound(children=[main.faces()[0]] + bands)
export_step(faces_mm, "faces_mm.step")
print("wrote", os.path.getsize("faces_mm.step"), "bytes")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("tesla")
gmsh.model.occ.importShapes("faces_mm.step")
ents = gmsh.model.occ.getEntities(2)
print("imported surfaces:", len(ents))
out, _ = gmsh.model.occ.fuse([ents[0]], ents[1:])
gmsh.model.occ.synchronize()
print("after fuse:", out)
for d, t in out:
    print("  surf", t, "area mm^2 =", round(gmsh.model.occ.getMass(d, t), 1),
          "curves:", len(gmsh.model.getBoundary([(d, t)], oriented=False)))

# -- cell 6 -------------------------------------------------------------------------
# The 13-surface import suggests the overlapping-face fuse is fighting me. Better source: the solid's 
bot_all = [f for f in s.faces() if abs(f.center().Z) < 1e-6 and abs(f.normal_at(f.center()).Z) > 0.99]
print("bottom pieces:", len(bot_all), "areas:", [round(f.area,1) for f in bot_all], "sum:", round(sum(f.area for f in bot_all),1))
merged = reduce(lambda a, b: a.fuse(b), bot_all).clean()
mf = merged.faces()
print("after fuse:", len(mf), "faces, area", round(sum(f.area for f in mf),1))
for f in mf:
    print("   area", round(f.area,1), " wires", len(f.wires()))

# -- cell 7 -------------------------------------------------------------------------
# Fusing filled the island holes, so I'll stop trying to make one face: gmsh can mesh the six coplanar
solid_m = s.scale(0.001)
export_step(solid_m, "tesla_m.step")
bbm = solid_m.bounding_box()
print("EXPORTED IN METRES  bbox x: %.4f .. %.4f   y: %.4f .. %.4f   z: %.4f .. %.4f"
      % (bbm.min.X, bbm.max.X, bbm.min.Y, bbm.max.Y, bbm.min.Z, bbm.max.Z))
print("requested: length 0.120 m, channel width 0.006 m, thickness %.3f m" % (TH/1000))
print("measured  : length %.4f m, loop stack height %.4f m" % (bbm.max.X-bbm.min.X, bbm.max.Y-bbm.min.Y))
print("main-channel walls at y = %.4f / %.4f -> width %.4f m" % (bbm.min.Y, 0.003, 0.006))
print("fluid area = %.2f mm^2, volume = %.3e m^3" % (s.volume/TH, solid_m.volume))

# -- cell 8 -------------------------------------------------------------------------
# Now the mesh model: re-import the metre-scale solid into a fresh gmsh model, strip everything except
gmsh.clear()
gmsh.model.add("tesla2d")
gmsh.model.occ.importShapes("tesla_m.step")
gmsh.model.occ.synchronize()
bottoms = []
for d, t in gmsh.model.getEntities(2):
    com = gmsh.model.occ.getCenterOfMass(d, t)
    if abs(com[2]) < 1e-9:
        bottoms.append(t)
print("bottom faces:", bottoms, "total area m^2 =",
      round(sum(gmsh.model.occ.getMass(2, t) for t in bottoms), 8))
gmsh.model.occ.remove(gmsh.model.getEntities(3))
gmsh.model.occ.remove([(2, t) for d, t in gmsh.model.getEntities(2) if t not in bottoms])
gmsh.model.occ.synchronize()
keep1 = {abs(c) for d, t in gmsh.model.getEntities(2)
         for c in gmsh.model.getBoundary([(2, t)], oriented=False, combined=False)[0:0]}
kept_curves = {abs(t) for st in bottoms for (d, t) in gmsh.model.getBoundary([(2, st)], oriented=False)}
gmsh.model.occ.remove([(1, t) for d, t in gmsh.model.getEntities(1) if t not in kept_curves])
gmsh.model.occ.synchronize()
print("remaining: surfaces", len(gmsh.model.getEntities(2)), " curves", len(gmsh.model.getEntities(1)),
      " volumes", len(gmsh.model.getEntities(3)))

# -- cell 9 -------------------------------------------------------------------------
# 13 z=0 faces totalling 1551.9 mm² instead of 6 totalling 1364.3 — the STEP round-trip changed the fa
gmsh.clear()
gmsh.model.add("chk")
gmsh.model.occ.importShapes("tesla_m.step")
gmsh.model.occ.synchronize()
for d, t in gmsh.model.getEntities(2):
    com = gmsh.model.occ.getCenterOfMass(d, t)
    if abs(com[2]) < 1e-9:
        print("surf %3d area_mm2 %8.2f  com=(%.4f, %.4f, %.4f)" %
              (t, gmsh.model.occ.getMass(d, t) * 1e6, *com))
print("volumes:", gmsh.model.getEntities(3), "total vol m^3",
      sum(gmsh.model.occ.getMass(3, t) for d, t in gmsh.model.getEntities(3)))

# -- cell 10 ------------------------------------------------------------------------
# The imported solid's z=0 faces overlap (they sum to more than the cross-section), so I'll take a pro
cut = Rectangle(400, 400).moved(Location((0, 0, TH/2))).faces()[0]
sec = s.intersect(cut)
sfs = sec.faces()
print("section faces:", len(sfs), " area:", round(sum(f.area for f in sfs), 2),
      "mm^2  (expected", round(s.volume/TH, 2), ")")
for f in sfs:
    print("   area %.2f  wires %d" % (f.area, len(f.wires())))

# -- cell 11 ------------------------------------------------------------------------
# Re-emitting the valid build self-contained, plus the metre-scale 2D section face export and a look a
pieces = [main3]
for pm, _, _ in loops:
    for e in pm.edges():
        pieces.append(extrude(trace(Curve() + e, line_width=W).faces()[0], amount=TH))
blob = reduce(lambda a, b: a + b, pieces).clean()
sol = blob.solids()[0]
secl = sol.intersect(Rectangle(400, 400).moved(Location((0, 0, TH/2))).faces()[0])
face_mm = [f for x in (secl if isinstance(secl, list) else [secl]) for f in x.faces()][0]
print("solids:", len(blob.solids()), " fluid area mm^2:", round(face_mm.area, 2),
      " wires:", len(face_mm.wires()), "(outer + 4 islands)")
face_m = face_mm.moved(Location((0, 0, -TH/2))).scale(0.001)   # to z=0 plane, mm -> m
export_step(face_m, "fluid_face_m.step")
fig, ax = plt.subplots(figsize=(13, 3.4))
for w in face_mm.wires():
    q = np.array([tuple(w @ (i/2000))[:2] for i in range(2001)])
    ax.plot(q[:, 0], q[:, 1], lw=0.9)
ax.set_aspect('equal'); ax.set_title("Tesla valve fluid region (mm)"); plt.show()

# -- cell 12 ------------------------------------------------------------------------
# Shape is right: four teardrop islands, shallow take-off funnel facing −x, steep re-entry firing +x. 
gmsh.clear()
gmsh.model.add("tv")
gmsh.model.occ.importShapes("fluid_face_m.step")
gmsh.model.occ.synchronize()
surf = gmsh.model.getEntities(2)
print("surfaces:", surf, "area m^2", gmsh.model.occ.getMass(2, surf[0][1]))
bnd = gmsh.model.getBoundary(surf, oriented=False)
print("boundary curves:", len(bnd))
for d, t in bnd:
    com = gmsh.model.occ.getCenterOfMass(d, t)
    if com[0] < 1e-6 or com[0] > 0.1199:
        print("  end curve", t, "com", [round(c, 5) for c in com],
              "len", round(gmsh.model.occ.getMass(d, t), 5))

# -- cell 13 ------------------------------------------------------------------------
# The exported face sits at z≈−0.4995 (build123d's scale about the location), so I'll drop it to z=0 i
LC = 1.5e-3
gmsh.model.occ.translate(surf, 0, 0, -gmsh.model.occ.getCenterOfMass(2, surf[0][1])[2])
gmsh.model.occ.synchronize()
print("z after translate:", gmsh.model.occ.getCenterOfMass(2, 1)[2])
gmsh.option.setNumber("Mesh.MeshSizeMin", LC)
gmsh.option.setNumber("Mesh.MeshSizeMax", LC)
gmsh.option.setNumber("Mesh.Algorithm", 8)          # frontal-delaunay for quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
ext = gmsh.model.occ.extrude([(2, 1)], 0, 0, 0.001, numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
print("extrude result:", ext[:3], "... total", len(ext))
print("volumes:", gmsh.model.getEntities(3), " surfaces:", len(gmsh.model.getEntities(2)))

# -- cell 14 ------------------------------------------------------------------------
# `RecombinationAlgorithm 2` (blossom full-quad) needs an even boundary count. Switching to plain blos
gmsh.model.removePhysicalGroups()
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
back, front, inlet, outlet, wall = [], [], [], [], []
for d, t in gmsh.model.getEntities(2):
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(d, t)
    if abs(cz) < 1e-9: back.append(t)
    elif abs(cz - 0.001) < 1e-9: front.append(t)
    elif cx < 1e-6: inlet.append(t)
    elif cx > 0.11999: outlet.append(t)
    else: wall.append(t)
print("inlet area m^2 =", sum(gmsh.model.occ.getMass(2, t) for t in inlet),
      " outlet area m^2 =", sum(gmsh.model.occ.getMass(2, t) for t in outlet), " expect 6e-06 each")
gmsh.model.addPhysicalGroup(3, [1], name="fluid")
gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
gmsh.model.addPhysicalGroup(2, wall, name="walls")
gmsh.model.addPhysicalGroup(2, back + front, name="frontAndBack")
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("tesla.msh")
print("hexes:", len(gmsh.model.mesh.getElementsByType(5)[0]),
      " prisms:", len(gmsh.model.mesh.getElementsByType(6)[0]))

# -- cell 15 ------------------------------------------------------------------------
# Mesh exists (1059 cells). Now a minimal case skeleton and `gmshToFoam`.
import textwrap, pathlib
for d in ["system", "constant", "0"]:
    pathlib.Path(d).mkdir(exist_ok=True)
head = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object %s;\n}\n"
pathlib.Path("system/controlDict").write_text(
    head % "controlDict" + textwrap.dedent("""
    application     simpleFoam;
    startFrom       startTime;
    startTime       0;
    stopAt          endTime;
    endTime         1;
    deltaT          1;
    writeControl    timeStep;
    writeInterval   100;
    """))
r = subprocess.run(["gmshToFoam", "tesla.msh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1500:])
print(r.stderr[-500:])

# -- cell 16 ------------------------------------------------------------------------
# `gmshToFoam` didn't match my physical names (2429 faces went to `defaultFaces`). Let me look at what
txt = open("tesla.msh").read()
i = txt.index("$PhysicalNames"); j = txt.index("$EndPhysicalNames")
print(txt[i:j+20])
import re
print(open("constant/polyMesh/boundary").read()[-1200:])

# -- cell 17 ------------------------------------------------------------------------
# The physical *names* are tagged 2–5 but the elements carry tags 6–10 (left over from my earlier abor
def build_mesh(lc, fname="tesla.msh"):
    gmsh.clear(); gmsh.model.add("tv")
    gmsh.model.occ.importShapes("fluid_face_m.step")
    gmsh.model.occ.synchronize()
    s0 = gmsh.model.getEntities(2)[0][1]
    gmsh.model.occ.translate([(2, s0)], 0, 0, -gmsh.model.occ.getCenterOfMass(2, s0)[2])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc); gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.model.occ.extrude([(2, s0)], 0, 0, 0.001, numElements=[1], recombine=True)
    gmsh.model.occ.synchronize()
    bk, fr, inl, outl, wl = [], [], [], [], []
    for d, t in gmsh.model.getEntities(2):
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(d, t)
        if abs(cz) < 1e-9: bk.append(t)
        elif abs(cz - 0.001) < 1e-9: fr.append(t)
        elif cx < 1e-6: inl.append(t)
        elif cx > 0.11999: outl.append(t)
        else: wl.append(t)
    assert abs(sum(gmsh.model.occ.getMass(2, t) for t in inl) - 6e-6) < 1e-12
    assert abs(sum(gmsh.model.occ.getMass(2, t) for t in outl) - 6e-6) < 1e-12
    gmsh.model.addPhysicalGroup(2, inl, tag=1, name="inlet")
    gmsh.model.addPhysicalGroup(2, outl, tag=2, name="outlet")
    gmsh.model.addPhysicalGroup(2, wl, tag=3, name="walls")
    gmsh.model.addPhysicalGroup(2, bk + fr, tag=4, name="frontAndBack")
    gmsh.model.addPhysicalGroup(3, [1], tag=5, name="fluid")
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(fname)
    return len(gmsh.model.mesh.getElementsByType(5)[0]), len(gmsh.model.mesh.getElementsByType(6)[0])

print("hex, prism =", build_mesh(1.5e-3))
r = subprocess.run(["gmshToFoam", "tesla.msh"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "name" in l or "Mapping" in l or "undefined" in l])

# -- cell 18 ------------------------------------------------------------------------
# Named patches now. Set `frontAndBack` to `empty` and `walls` to `wall`, then run `checkMesh`.
# exit 0 (0 s)
# Selecting decompositionMethod none
# lease and be aware of its limitations.
# // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
# Create time
# Create polyMesh for time = 0
# Time = 0
# Mesh stats
# points 2778
# cells 1141
# faces 5046
bf = "constant/polyMesh/boundary"
t = open(bf).read()
def settype(txt, name, ftype):
    k = txt.index(name + "\n")
    seg = txt[k:k + 400]
    return txt[:k] + seg.replace("type            patch;", "type            %s;" % ftype, 1) + txt[k + 400:]
t = settype(t, "frontAndBack", "empty")
t = settype(t, "walls", "wall")
open(bf, "w").write(t)
print(open(bf).read()[t.index("4\n("):][:900])
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 19 ------------------------------------------------------------------------
# Boundary types set. Let me see the actual checkMesh verdict.
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(r.stdout.splitlines()[-45:]))

# -- cell 20 ------------------------------------------------------------------------
print("rc =", r.returncode)
print("STDERR:", r.stderr[-2000:])

# -- cell 21 ------------------------------------------------------------------------
# One block: convert to VTK so I can render the mesh and colour the patches.
# sub­process exit 0 (2 s)
# rc 0
# l
# patch inlet;
# patch outlet;
# patch walls;
# End
# ['t3_0.vtm', 't3_boundary']
import pyvista as pv
r = subprocess.run(["foamToVTK", "-ascii", "-latestTime"], capture_output=True, text=True)
print("rc", r.returncode, r.stdout[-300:])
print(sorted(os.listdir("VTK")) if os.path.isdir("VTK") else sorted(os.listdir(".")))

# -- cell 22 ------------------------------------------------------------------------
# Parser bugs (point count short, bad dict build). Redoing the parse with regex over the whole file.
praw = open("constant/polyMesh/points").read()
pts = np.array(re.findall(r"\(([-\d.eE+ ]+)\)", praw.split("// * * *")[-1]))
pts = np.array([[float(v) for v in s.split()] for s in pts])
fraw = open("constant/polyMesh/faces").read().split("// * * *")[-1]
faces = [[int(v) for v in s.split()] for s in re.findall(r"\(([\d ]+)\)", fraw)]
bnd = open("constant/polyMesh/boundary").read()
pat = {m[0]: (int(m[1]), int(m[2])) for m in
       re.findall(r"(\w+)\s*\n\s*\{.*?nFaces\s+(\d+);\s*startFace\s+(\d+);", bnd, re.S)}
print("points", pts.shape, "faces", len(faces), "patches", pat)
fig, ax = plt.subplots(figsize=(13, 3.4))
for nm, (nf, sf) in pat.items():
    if nm == "frontAndBack": continue
    c = np.array([pts[faces[i]].mean(axis=0) for i in range(sf, sf+nf)])
    ax.plot(c[:, 0]*1e3, c[:, 1]*1e3, '.', ms=3, label="%s (%d)" % (nm, nf))
    print(nm, "x range mm: %.2f .. %.2f" % (c[:,0].min()*1e3, c[:,0].max()*1e3))
ax.legend(loc="upper right"); ax.set_aspect("equal"); plt.show()

# -- cell 23 ------------------------------------------------------------------------
# Patches verified geometrically: inlet only at x=0 (4 faces, 6 mm² × 1 mm), outlet only at x=120 mm, 
# sub­process exit 0 (10 s)
# hex, prism = (9884, 0)
# ['Patch 0 gets name frontAndBack', 'Patch 1 gets name inlet', 'Patch 2 gets name walls', 'Patch 3 ge
# Checking topology...
# Patch topology OK.
# Checking geometry...
# Overall domain bounding box (0 -0.003 0) (0.12 0.0233 0.001)
# Mesh OK.
# ...
# Checking topology...
# Patch topology OK.
print("hex, prism =", build_mesh(0.5e-3))
r = subprocess.run(["gmshToFoam", "tesla.msh"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "gets name" in l])
t = open(bf).read()
t = settype(t, "frontAndBack", "empty")
t = settype(t, "walls", "wall")
open(bf, "w").write(t)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("rc", r.returncode)
print("\n".join(r.stdout.splitlines()[-32:]))
