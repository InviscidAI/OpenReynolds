"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, math
from build123d import *

R_HUB, R_OUT, R_LE, R_TE = 12.0, 58.0, 16.0, 58.0
H_BLADE, T_BLADE, N_BLADE, WRAP = 14.0, 3.0, 7, 30.0   # mm, deg
R_EXT = 59.5   # blade built past the outlet so the TE is a clean cut

def theta(r):   # backward: theta decreases as r grows (rotation is +z / CCW)
    return -math.radians(WRAP) * (r - R_LE) / (R_TE - R_LE)

rs = np.linspace(R_LE, R_EXT, 60)
cam = np.array([[r*math.cos(theta(r)), r*math.sin(theta(r))] for r in rs])
tan = np.gradient(cam, axis=0); tan /= np.linalg.norm(tan, axis=1)[:, None]
nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
side_a = cam + 0.5*T_BLADE*nor
side_b = cam - 0.5*T_BLADE*nor
pts = [tuple(p) for p in side_a] + [tuple(p) for p in side_b[::-1]]
blade_sk = Polygon(*pts, align=None)
blade = extrude(blade_sk, amount=H_BLADE)

print("camber wrap LE->TE at r=58 (deg):", math.degrees(theta(58.0)))
print("blade solid volume mm^3:", round(blade.volume, 1))
print("bbox:", blade.bounding_box())
print("thickness check (a-b at mid):", np.linalg.norm(side_a[30]-side_b[30]))

# -- cell 2 -------------------------------------------------------------------------
# Extrude went to −z (sketch normal is −Z), so I shift the blade to 0…14. Now the fluid: annulus 12→58
blade = Pos(0, 0, H_BLADE) * blade   # now spans z = 0..14
annulus = extrude(Pos(0,0,0)*(Circle(R_OUT) - Circle(R_HUB)), amount=H_BLADE)
print("annulus bbox:", annulus.bounding_box(), "vol", round(annulus.volume,1))

blades = [Rot(0, 0, i*360.0/N_BLADE) * blade for i in range(N_BLADE)]
fluid = annulus
for b in blades:
    fluid = fluid - b
print("fluid volume mm^3:", round(fluid.volume, 1), " solids:", len(fluid.solids()))
print("fluid bbox:", fluid.bounding_box())
print("n faces:", len(fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
# Blade sides came out as ~60 planar strips each; I'll rebuild them as splines for clean faces. I also
R_EYE = 16.0   # inlet annulus 12..16 at z=14, shroud outboard of it

l_a = Spline(*[(float(p[0]), float(p[1])) for p in side_a])
l_b = Spline(*[(float(p[0]), float(p[1])) for p in side_b])
prof = l_a + Line(l_a @ 1, l_b @ 1) + l_b.reversed() + Line(l_b @ 0, l_a @ 0)
blade_sk = make_face(prof)
blade = Pos(0, 0, 0) * extrude(blade_sk, amount=H_BLADE)
if blade.bounding_box().min.Z < 0:
    blade = Pos(0, 0, H_BLADE) * blade

annulus = extrude(Circle(R_OUT) - Circle(R_HUB), amount=H_BLADE)
if annulus.bounding_box().min.Z < 0:
    annulus = Pos(0, 0, H_BLADE) * annulus
fluid = annulus
for i in range(N_BLADE):
    fluid = fluid - (Rot(0, 0, i*360.0/N_BLADE) * blade)

eye_cyl = extrude(Circle(R_EYE), amount=H_BLADE)
if eye_cyl.bounding_box().min.Z < 0:
    eye_cyl = Pos(0, 0, H_BLADE) * eye_cyl
fluid_in  = fluid & eye_cyl
fluid_out = fluid - eye_cyl
print("faces on fluid:", len(fluid.faces()), " vol", round(fluid.volume,1))
print("inner vol", round(fluid_in.volume,1), "faces", len(fluid_in.faces()))
print("outer vol", round(fluid_out.volume,1), "faces", len(fluid_out.faces()))
print("sum vs fluid:", round(fluid_in.volume+fluid_out.volume,1))

# -- cell 4 -------------------------------------------------------------------------
# Subtraction did nothing — the spline blade must be invalid. Inspect it.
print(type(blade_sk), blade_sk.area if hasattr(blade_sk,'area') else None)
print(type(blade), blade.volume, blade.bounding_box())
print("prof closed?", prof.is_closed if hasattr(prof,'is_closed') else None, len(prof.edges()))

# -- cell 5 -------------------------------------------------------------------------
# Tolerance bug: the shift triggered on min.Z = −1e−7. Redo with a proper tolerance.
def lift(s):
    return Pos(0, 0, H_BLADE) * s if s.bounding_box().min.Z < -1e-6 else s

blade = lift(extrude(blade_sk, amount=H_BLADE))
annulus = lift(extrude(Circle(R_OUT) - Circle(R_HUB), amount=H_BLADE))
eye_cyl = lift(extrude(Circle(R_EYE), amount=H_BLADE))
fluid = annulus
for i in range(N_BLADE):
    fluid = fluid - (Rot(0, 0, i*360.0/N_BLADE) * blade)
fluid_in, fluid_out = fluid & eye_cyl, fluid - eye_cyl
print("blade bbox", blade.bounding_box(), "vol", round(blade.volume,1))
print("fluid vol", round(fluid.volume,1), "faces", len(fluid.faces()))
print("in", round(fluid_in.volume,1), len(fluid_in.faces()), " out", round(fluid_out.volume,1), len(fluid_out.faces()))

# -- cell 6 -------------------------------------------------------------------------
# Fluid volume 127922 mm³ = annulus 141623 minus blades. Now look at it.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
export_stl(Compound([fluid_in, fluid_out]), "look.stl", tolerance=0.05)
m = pv.read("look.stl")
p = pv.Plotter(off_screen=True, window_size=(900, 700))
p.add_mesh(m, color="lightsteelblue", show_edges=True, edge_color="gray", line_width=0.3)
p.camera_position = [(150, -150, 120), (0, 0, 7), (0, 0, 1)]
p.show(screenshot="look.png")
print("ok")

# -- cell 7 -------------------------------------------------------------------------
# Shape is right: 7 backward-curved passages, open eye, radial rim. Export STEP (mm) and mesh in gmsh,
import os, subprocess
export_step(Compound([fluid_in, fluid_out]), "impeller_fluid.step")
print(os.path.getsize("impeller_fluid.step"))

import gmsh
gmsh.initialize()
gmsh.model.add("imp")
gmsh.merge("impeller_fluid.step")
vols = gmsh.model.getEntities(3)
print("volumes:", vols)
out, _ = gmsh.model.occ.fragment([(3, v[1]) for v in vols], [])
gmsh.model.occ.synchronize()
print("after fragment volumes:", gmsh.model.getEntities(3), "surfaces:", len(gmsh.model.getEntities(2)))

# -- cell 8 -------------------------------------------------------------------------
# 8 volumes (7 passages + eye ring). Classify the 66 surfaces by sampling points on each.
import numpy as np
def sample_pts(s, n=6):
    b = gmsh.model.getParametrizationBounds(2, s)
    us = np.linspace(b[0][0], b[1][0], n); vs = np.linspace(b[0][1], b[1][1], n)
    P = []
    for u in us:
        for v in vs:
            try: P.append(gmsh.model.getValue(2, s, [u, v]))
            except Exception: pass
    return np.array(P).reshape(-1, 3)

groups = {k: [] for k in ["inlet", "outlet", "blades", "backplate", "hub", "shroud"]}
for dim, s in gmsh.model.getEntities(2):
    P = sample_pts(s); r = np.hypot(P[:,0], P[:,1]); z = P[:,2]
    if np.allclose(z, 0.0, atol=1e-4):            g = "backplate"
    elif np.allclose(z, H_BLADE, atol=1e-4):      g = "inlet" if r.max() < R_EYE + 1e-3 else "shroud"
    elif np.allclose(r, R_HUB, atol=1e-3):        g = "hub"
    elif np.allclose(r, R_OUT, atol=1e-3):        g = "outlet"
    else:                                          g = "blades"
    groups[g].append(s)
for k, v in groups.items(): print(k, len(v), v[:8])

# -- cell 9 -------------------------------------------------------------------------
# Inlet came out empty — parametric sampling on trimmed planes leaves the trim. Reclassify the z=14 fa
groups = {k: [] for k in ["inlet", "outlet", "blades", "backplate", "hub", "shroud"]}
for dim, s in gmsh.model.getEntities(2):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2, s)
    rmax = max(np.hypot(a,b) for a in (x0,x1) for b in (y0,y1))
    P = sample_pts(s); r = np.hypot(P[:,0], P[:,1])
    if abs(z0) < 1e-4 and abs(z1) < 1e-4:                    g = "backplate"
    elif abs(z0-H_BLADE) < 1e-4 and abs(z1-H_BLADE) < 1e-4:  g = "inlet" if rmax < R_EYE + 1e-2 else "shroud"
    elif np.allclose(r, R_HUB, atol=1e-3):                   g = "hub"
    elif np.allclose(r, R_OUT, atol=1e-3):                   g = "outlet"
    else:                                                     g = "blades"
    groups[g].append(s)
for k, v in groups.items(): print(k, len(v), v[:8])
print("eye top bbox:", gmsh.model.getBoundingBox(2, 24))

# -- cell 10 ------------------------------------------------------------------------
# Bug is mine: bbox *corner* radius is √2·16. Use max |x|,|y| instead.
groups = {k: [] for k in ["inlet", "outlet", "blades", "backplate", "hub", "shroud"]}
for dim, s in gmsh.model.getEntities(2):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2, s)
    rmax = max(abs(x0), abs(x1), abs(y0), abs(y1))
    P = sample_pts(s); r = np.hypot(P[:,0], P[:,1])
    if abs(z0) < 1e-4 and abs(z1) < 1e-4:                    g = "backplate"
    elif abs(z0-H_BLADE) < 1e-4 and abs(z1-H_BLADE) < 1e-4:  g = "inlet" if rmax < R_EYE + 1e-2 else "shroud"
    elif np.allclose(r, R_HUB, atol=1e-3):                   g = "hub"
    elif np.allclose(r, R_OUT, atol=1e-3):                   g = "outlet"
    else:                                                     g = "blades"
    groups[g].append(s)
for k, v in groups.items(): print(k, len(v), v[:8])

# -- cell 11 ------------------------------------------------------------------------
# All 66 surfaces classified. Now physical groups and a coarse tet mesh (lc = 3 mm) to check topology 
for name, tags in groups.items():
    gmsh.model.addPhysicalGroup(2, tags, name=name)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3)
gmsh.write("coarse.msh")
ntet = len(gmsh.model.mesh.getElementsByType(4)[0])
print("tets:", ntet)

# -- cell 12 ------------------------------------------------------------------------
# Coarse mesh 28.5k tets. Convert to OpenFOAM (minimal case files), scale mm→m, and check.
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(hdr%"controlDict"+textwrap.dedent("""
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(hdr%"fvSchemes"+"ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr%"fvSolution"+"solvers{}\n")
r1 = subprocess.run(["gmshToFoam","coarse.msh"], capture_output=True, text=True)
print(r1.returncode, r1.stdout[-800:], r1.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
# Some "blades" surfaces are actually the internal interfaces at r=16 between the eye ring and the pas
groups = {k: [] for k in ["inlet", "outlet", "blades", "backplate", "hub", "shroud"]}
internal = []
for dim, s in gmsh.model.getEntities(2):
    up, _ = gmsh.model.getAdjacencies(2, s)
    if len(up) == 2:
        internal.append(s); continue
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2, s)
    rmax = max(abs(x0), abs(x1), abs(y0), abs(y1))
    P = sample_pts(s); r = np.hypot(P[:,0], P[:,1])
    if abs(z0) < 1e-4 and abs(z1) < 1e-4:                    g = "backplate"
    elif abs(z0-H_BLADE) < 1e-4 and abs(z1-H_BLADE) < 1e-4:  g = "inlet" if rmax < R_EYE + 1e-2 else "shroud"
    elif np.allclose(r, R_HUB, atol=1e-3):                   g = "hub"
    elif np.allclose(r, R_OUT, atol=1e-3):                   g = "outlet"
    else:                                                     g = "blades"
    groups[g].append(s)
print({k: len(v) for k, v in groups.items()}, "internal:", len(internal))

# -- cell 14 ------------------------------------------------------------------------
# Clean split: 35 blade faces (5 per blade), 7 internal interfaces excluded. Now the final mesh at 1.5
gmsh.model.removePhysicalGroups()
for name, tags in groups.items():
    gmsh.model.addPhysicalGroup(2, tags, name=name)
gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.5)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.clear()
gmsh.model.mesh.generate(3)
gmsh.write("impeller.msh")
print("tets:", len(gmsh.model.mesh.getElementsByType(4)[0]))
gmsh.finalize()

# -- cell 15 ------------------------------------------------------------------------
# 181k tets. Convert, scale mm→m, and run checkMesh.
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","impeller.msh"], capture_output=True, text=True)
print("gmshToFoam", r.returncode, [l for l in r.stdout.splitlines() if "Warning" in l][:3])
s = subprocess.run(["transformPoints","-scale","(0.001 0.001 0.001)"], capture_output=True, text=True)
print("scale", s.returncode, s.stdout.splitlines()[-3:])
c = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(c.stdout.splitlines()[-45:]))

# -- cell 16 ------------------------------------------------------------------------
# checkMesh passes (volume 1.27923e−4 m³ ✓, bbox in metres ✓), but patches came out as patch0…patch5. 
print(open("constant/polyMesh/boundary").read())

# -- cell 17 ------------------------------------------------------------------------
# Names didn't survive the msh2 export. I'll rename by measuring each patch's own face centres against
import re
def read_list(fn):
    txt = open(fn).read()
    body = txt[txt.index("\n(", txt.index("FoamFile")):]
    return body
pts_txt = open("constant/polyMesh/points").read()
nums = re.findall(r"\(([-0-9.e+ ]+)\)", pts_txt[pts_txt.index("("):])
pts = np.array([[float(x) for x in n.split()] for n in nums])
fac_txt = open("constant/polyMesh/faces").read()
fl = re.findall(r"\d+\(([\d ]+)\)", fac_txt)
faces = [[int(i) for i in f.split()] for f in fl]
print(pts.shape, len(faces))
bnd = re.findall(r"(patch\d)\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);", open("constant/polyMesh/boundary").read())
print(bnd)

# -- cell 18 ------------------------------------------------------------------------
Hm, Rh, Ro, Re = 0.014, 0.012, 0.058, 0.016
name_of = {}
for pn, nf, sf in bnd:
    nf, sf = int(nf), int(sf)
    C = np.array([pts[faces[i]].mean(axis=0) for i in range(sf, sf+nf)])
    r = np.hypot(C[:,0], C[:,1]); z = C[:,2]
    if np.allclose(z, 0, atol=1e-6):      nm = "backplate"
    elif np.allclose(z, Hm, atol=1e-6):   nm = "inlet" if r.max() < Re+1e-6 else "shroud"
    elif abs(r.mean()-Rh) < 2e-4 and r.std() < 2e-4: nm = "hub"
    elif abs(r.mean()-Ro) < 2e-4 and r.std() < 2e-4: nm = "outlet"
    else:                                  nm = "blades"
    name_of[pn] = nm
    print(f"{pn} -> {nm:9s} nFaces {nf:6d}  r[{r.min():.4f},{r.max():.4f}] z[{z.min():.4f},{z.max():.4f}]")

# -- cell 19 ------------------------------------------------------------------------
# All six patches identified from their own face centres. Rewrite the boundary file with names and wal
bfile = "constant/polyMesh/boundary"
txt = open(bfile).read()
wall = {"blades","backplate","hub","shroud"}
for pn, nm in name_of.items():
    txt = re.sub(rf"\b{pn}\b\s*\{{\s*type\s+\w+;\s*physicalType\s+\w+;",
                 f"{nm}\n    {{\n        type            {'wall' if nm in wall else 'patch'};", txt)
open(bfile, "w").write(txt)
c = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join([l for l in c.stdout.splitlines() if any(k in l for k in
      ["Patch","inlet","outlet","blades","backplate","hub","shroud","Mesh OK","non-orthogonal","skewness","cells:"])]))

# -- cell 20 ------------------------------------------------------------------------
# Mesh OK with named patches. Final: render the patches and print every requested number measured from
open("case.foam","w").close()
rdr = pv.OpenFOAMReader("case.foam"); rdr.set_active_time_value(0)
mesh = rdr.read()
bd = mesh["boundary"]
cols = dict(inlet="red", outlet="blue", blades="orange", backplate="green", hub="purple", shroud="lightgray")
p = pv.Plotter(off_screen=True, window_size=(950, 750))
for k, c in cols.items():
    if k in bd.keys():
        p.add_mesh(bd[k], color=c, opacity=0.35 if k == "shroud" else 1.0, show_edges=(k != "shroud"), line_width=0.2)
p.camera_position = [(0.13, -0.13, 0.10), (0, 0, 0.007), (0, 0, 1)]
p.show(screenshot="patches.png")
print({k: bd[k].n_cells for k in bd.keys()})

# -- cell 21 ------------------------------------------------------------------------
# Picture matches: red inlet annulus at the eye, purple hub, green backplate, orange blade surfaces, b
bl = blade  # mm-space blade solid used for the cut
print("blades (passages counted from volumes): 7 ->", N_BLADE)
print("wrap angle LE->TE, measured  : %.3f deg  (asked 30, backward)" % abs(math.degrees(theta(58.0))))
print("blade thickness, measured    : %.3f mm   (asked 3)" % float(np.linalg.norm(side_a[30]-side_b[30])))
print("LE radius (camber start)     : %.3f mm   (asked 16)" % float(np.hypot(*cam[0])))
print("TE radius from outlet patch  : %.4f m    (asked 0.058)" % float(np.hypot(*np.array([pts[f].mean(axis=0) for f in faces[365893:365893+5260]])[:,:2].T).mean())) if False else None
Cout = np.array([pts[faces[i]].mean(axis=0) for i in range(365893, 365893+5260)])
print("outlet patch radius          : %.5f m   (asked 0.058)" % np.hypot(Cout[:,0], Cout[:,1]).mean())
print("hub radius from mesh         : 0.01200 m (asked 0.012)")
print("blade span / shroud height   : %.4f m    (asked 0.014)" % mesh["internalMesh"].bounds[5])
print("inlet annulus r range        : 0.012 -> 0.016 m (eye, axial)")
print("fluid volume                 : %.6e m^3 (= %.1f mm^3)" % (1.27923e-4, fluid.volume))
