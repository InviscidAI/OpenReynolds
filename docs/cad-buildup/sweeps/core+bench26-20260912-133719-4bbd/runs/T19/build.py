"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, os, math
from build123d import *

# --- all dimensions in METRES ---
R_SHAFT = 0.020        # 40 mm dia sealed length
R_BORE  = 0.020050     # 40.1 mm dia bore -> 0.05 mm radial clearance
L_SEAL  = 0.060        # 60 mm sealed length
KW_W    = 0.012        # keyway width
KW_D    = 0.005        # keyway depth (radial, from shaft OD)
R_FLOOR = R_SHAFT - KW_D

bore_cyl  = Pos(0, 0, L_SEAL/2) * Cylinder(R_BORE,  L_SEAL)
shaft_cyl = Pos(0, 0, L_SEAL/2) * Cylinder(R_SHAFT, L_SEAL)
keybox    = Pos(0, R_FLOOR + 0.010, L_SEAL/2) * Box(KW_W, 0.020, L_SEAL)
shaft_solid = shaft_cyl - keybox
fluid = bore_cyl - shaft_solid

ann = math.pi*(R_BORE**2 - R_SHAFT**2)*L_SEAL
y_corner = math.sqrt(R_SHAFT**2 - (KW_W/2)**2)
th = math.asin((KW_W/2)/R_SHAFT)
seg = R_SHAFT**2*th - (KW_W/2)*y_corner
pocket = (KW_W*(y_corner-R_FLOOR) + seg)*L_SEAL
print("fluid volume  [m^3] =", fluid.volume)
print("expected ann+pocket =", ann+pocket, "(ann=", ann, "pocket=", pocket, ")")
print("clearance [mm]=", (R_BORE-R_SHAFT)*1e3, "kw depth [mm]=", KW_D*1e3, "kw width [mm]=", KW_W*1e3)
print("flank/OD corner y [mm]=", y_corner*1e3)
print("bbox (m):", fluid.bounding_box())
print("n faces:", len(fluid.faces()))

# -- cell 2 -------------------------------------------------------------------------
# Volume matches the analytic annulus+pocket to 13 digits. Now classify the 7 faces by geometric prope
from build123d import *
import math
def face_info(f):
    s = f.geom_type
    c = f.center()
    n = f.normal_at(c) if s == GeomType.PLANE else None
    return s, (c.X, c.Y, c.Z), f.area, n
for i, f in enumerate(fluid.faces()):
    s, c, a, n = face_info(f)
    print(i, s, "area=%.6e"%a, "c=(%.5f,%.5f,%.5f)"%c, "n=", n)

# -- cell 3 -------------------------------------------------------------------------
# `Face.radius` is None here; take the radius from the OCC adaptor instead.
import os, math
from build123d import *
from OCP.BRepAdaptor import BRepAdaptor_Surface
os.makedirs("constant/triSurface", exist_ok=True)

def cyl_radius(f):
    return BRepAdaptor_Surface(f.wrapped).Cylinder().Radius()

groups = {"shaft":[], "bore":[], "keywayFloor":[], "keywayFlanks":[],
          "highPressure":[], "lowPressure":[]}
for f in fluid.faces():
    c = f.center()
    if f.geom_type == GeomType.CYLINDER:
        groups["bore" if abs(cyl_radius(f)-R_BORE) < 1e-9 else "shaft"].append(f)
    else:
        n = f.normal_at(c)
        if abs(n.Z) > 0.9:      # ends: HP = 50 mm-step end at z=0, LP = 30 mm-step end at z=L
            groups["highPressure" if c.Z < L_SEAL/2 else "lowPressure"].append(f)
        elif abs(n.X) > 0.9:
            groups["keywayFlanks"].append(f)
        else:
            groups["keywayFloor"].append(f)

for name, fs in groups.items():
    shp = Compound(children=fs) if len(fs) > 1 else fs[0]
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=5e-6, angular_tolerance=0.05)
    print(name, "nfaces=", len(fs), "area=%.6e"%sum(f.area for f in fs),
          "bytes=", os.path.getsize(f"constant/triSurface/{name}.stl"))

# -- cell 4 -------------------------------------------------------------------------
# Now look at it: the high-pressure end-face STL *is* the cross-section, so plot its triangles in 2D (
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pyvista as pv
pv.OFF_SCREEN = True
hp = pv.read("constant/triSurface/highPressure.stl")
tri = hp.points[hp.faces.reshape(-1,4)[:,1:]]*1e3   # mm
fig, ax = plt.subplots(1,2, figsize=(11,5))
for a in ax:
    a.add_collection(matplotlib.collections.PolyCollection(tri[:,:,:2], fc="steelblue", ec="k", lw=0.2))
    a.set_aspect("equal")
ax[0].set_xlim(-21,21); ax[0].set_ylim(-21,21); ax[0].set_title("HP end face (mm)")
ax[1].set_xlim(-9,9);   ax[1].set_ylim(13,21);  ax[1].set_title("zoom: keyway + 0.05 mm gap")
fig.savefig("section.png", dpi=110)

p = pv.Plotter(window_size=(800,600))
for n,c in [("shaft","grey"),("bore","tan"),("keywayFloor","red"),("keywayFlanks","orange"),
            ("highPressure","blue"),("lowPressure","green")]:
    p.add_mesh(pv.read(f"constant/triSurface/{n}.stl"), color=c, show_edges=False, label=n)
p.add_legend(); p.camera_position="iso"; p.show(screenshot="patches.png")
print("ok")

# -- cell 5 -------------------------------------------------------------------------
# 3D patches look right. Now display the cross-section figure itself.
from IPython.display import Image, display
display(Image("section.png"))

# -- cell 6 -------------------------------------------------------------------------
# Pocket loop traversal: after the right flank I need `+inner[0]` (pin[0]→pin[1]), not its reverse.
import gmsh, math, numpy as np
if gmsh.isInitialized(): gmsh.finalize()
NZ, NR, NKW, NS, NPOCK = 20, 3, 6, 10, 5     # axial, radial(gap), keyway-arc, sector-arc, pocket-depth
A1 = math.degrees(math.acos((KW_W/2)/R_SHAFT))   # flank/OD corner angle from +x
A2 = 180.0 - A1
angs = [A1] + [A2 + i*(360.0-(A2-A1))/5 for i in range(5)]
gmsh.initialize(); gmsh.model.add("seal")
g = gmsh.model.geo
O = g.addPoint(0,0,0)
def pt(r,a): return g.addPoint(r*math.cos(math.radians(a)), r*math.sin(math.radians(a)), 0)
pin  = [pt(R_SHAFT,a) for a in angs]
pout = [pt(R_BORE ,a) for a in angs]
inner = [g.addCircleArc(pin[i],  O, pin[(i+1)%6]) for i in range(6)]
outer = [g.addCircleArc(pout[i], O, pout[(i+1)%6]) for i in range(6)]
radial= [g.addLine(pin[i], pout[i]) for i in range(6)]
nc = [NKW] + [NS]*5
sects = []
for i in range(6):
    s = g.addPlaneSurface([g.addCurveLoop([inner[i], radial[(i+1)%6], -outer[i], -radial[i]])])
    sects.append(s)
    g.mesh.setTransfiniteCurve(inner[i], nc[i]+1); g.mesh.setTransfiniteCurve(outer[i], nc[i]+1)
    g.mesh.setTransfiniteCurve(radial[i], NR+1)
    g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2, s)
pfr = g.addPoint( KW_W/2, R_FLOOR, 0); pfl = g.addPoint(-KW_W/2, R_FLOOR, 0)
floor  = g.addLine(pfl, pfr)
flankR = g.addLine(pfr, pin[0]); flankL = g.addLine(pin[1], pfl)
pock = g.addPlaneSurface([g.addCurveLoop([floor, flankR, inner[0], flankL])])
g.mesh.setTransfiniteCurve(floor, NKW+1)
g.mesh.setTransfiniteCurve(flankR, NPOCK+1); g.mesh.setTransfiniteCurve(flankL, NPOCK+1)
g.mesh.setTransfiniteSurface(pock, cornerTags=[pfl,pfr,pin[0],pin[1]]); g.mesh.setRecombine(2, pock)
g.synchronize()
allsurf = sects + [pock]
out = g.extrude([(2,s) for s in allsurf], 0,0,L_SEAL, numElements=[NZ], recombine=True)
g.synchronize()
tops = [out[i-1][1] for i,e in enumerate(out) if e[0]==3]
print("extruded entities:", len(out), "tops:", tops, "vols:", [e[1] for e in out if e[0]==3])

# -- cell 7 -------------------------------------------------------------------------
# 28 laterals returned — need to check whether shared curves (the keyway arc, the radial lines) produc
lat = [e[1] for e in out if e[0]==2 and e[1] not in tops]
print(len(lat), "lateral entries,", len(set(lat)), "unique")
# map each named curve -> its lateral surface(s), by curve identity
def lat_of(c):
    r = [s for s in set(lat) if c in [abs(t) for _,t in gmsh.model.getBoundary([(2,s)], oriented=False)]]
    return r
print("keyway arc inner[0] ->", lat_of(inner[0]))
print("radial[1] ->", lat_of(radial[1]), " floor ->", lat_of(floor), " outer[2] ->", lat_of(outer[2]))

# -- cell 8 -------------------------------------------------------------------------
# Shared curves gave single lateral surfaces (21 unique = 6+6+6+3), so the mesh is conformal. Now tag 
def one(c):
    r = lat_of(c); assert len(r)==1, (c,r); return r[0]
shaft_s  = [one(inner[i]) for i in range(1,6)]     # inner[0] is the keyway opening -> internal
bore_s   = [one(outer[i]) for i in range(6)]
floor_s  = [one(floor)]
flank_s  = [one(flankR), one(flankL)]
gmsh.model.addPhysicalGroup(2, shaft_s, name="shaft")
gmsh.model.addPhysicalGroup(2, bore_s,  name="bore")
gmsh.model.addPhysicalGroup(2, floor_s, name="keywayFloor")
gmsh.model.addPhysicalGroup(2, flank_s, name="keywayFlanks")
gmsh.model.addPhysicalGroup(2, allsurf, name="highPressure")   # z=0, 50 mm-step end
gmsh.model.addPhysicalGroup(2, tops,    name="lowPressure")    # z=L, 30 mm-step end
gmsh.model.addPhysicalGroup(3, [e[1] for e in out if e[0]==3], name="oil")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3)
gmsh.write("seal.msh")
import numpy as np
et, ec, _ = gmsh.model.mesh.getElements(3)
print("3D element types:", et, "counts:", [len(c) for c in ec])

# -- cell 9 -------------------------------------------------------------------------
# 3960 hexes (3360 annulus + 600 pocket), all hexahedra. Now convert to OpenFOAM with a minimal `syste
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True)
open("system/controlDict","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
    application     icoFoam;  startFrom startTime; startTime 0; stopAt endTime;
    endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
    """))
r = subprocess.run(["gmshToFoam","seal.msh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:]); print(r.stderr[-800:])

# -- cell 10 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 11 ------------------------------------------------------------------------
# All six patches present with exactly the expected face counts (shaft 50×20, bore 56×20, floor 6×20, 
import subprocess
for p in ["shaft","bore","keywayFloor","keywayFlanks"]:
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/type",
                    "-set","wall"], check=True, capture_output=True)
    subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/physicalType",
                    "-set","wall"], check=True, capture_output=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 12 ------------------------------------------------------------------------
print("rc:", r.returncode, "len stdout:", len(r.stdout))
print(r.stderr[-2000:])

# -- cell 13 ------------------------------------------------------------------------
# Needs `fvSchemes`/`fvSolution` to construct the mesh; add minimal ones and re-run.
import textwrap, subprocess
open("system/fvSchemes","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
    ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; }
    divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; }
    interpolationSchemes { default linear; } snGradSchemes { default corrected; }
    """))
open("system/fvSolution","w").write(textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
    solvers {} 
    """))
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-2400:])

# -- cell 14 ------------------------------------------------------------------------
# Mesh OK at 3960 cells; volume is 0.36 % under the exact 3.75845e-06 m³ from the coarse facetting of 
import gmsh, math
def build_mesh(NZ, NR, NKW, NS, NPOCK, fname="seal.msh"):
    if gmsh.isInitialized(): gmsh.finalize()
    A1 = math.degrees(math.acos((KW_W/2)/R_SHAFT)); A2 = 180.0 - A1
    angs = [A1] + [A2 + i*(360.0-(A2-A1))/5 for i in range(5)]
    gmsh.initialize(); gmsh.model.add("seal"); g = gmsh.model.geo
    O = g.addPoint(0,0,0)
    P = lambda r,a: g.addPoint(r*math.cos(math.radians(a)), r*math.sin(math.radians(a)), 0)
    pin  = [P(R_SHAFT,a) for a in angs]; pout = [P(R_BORE,a) for a in angs]
    inner = [g.addCircleArc(pin[i],  O, pin[(i+1)%6]) for i in range(6)]
    outer = [g.addCircleArc(pout[i], O, pout[(i+1)%6]) for i in range(6)]
    radial= [g.addLine(pin[i], pout[i]) for i in range(6)]
    nc = [NKW] + [NS]*5; sects = []
    for i in range(6):
        s = g.addPlaneSurface([g.addCurveLoop([inner[i], radial[(i+1)%6], -outer[i], -radial[i]])])
        sects.append(s)
        g.mesh.setTransfiniteCurve(inner[i], nc[i]+1); g.mesh.setTransfiniteCurve(outer[i], nc[i]+1)
        g.mesh.setTransfiniteCurve(radial[i], NR+1)
        g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2, s)
    pfr = g.addPoint(KW_W/2, R_FLOOR, 0); pfl = g.addPoint(-KW_W/2, R_FLOOR, 0)
    floor = g.addLine(pfl, pfr); flankR = g.addLine(pfr, pin[0]); flankL = g.addLine(pin[1], pfl)
    pock = g.addPlaneSurface([g.addCurveLoop([floor, flankR, inner[0], flankL])])
    g.mesh.setTransfiniteCurve(floor, NKW+1)
    g.mesh.setTransfiniteCurve(flankR, NPOCK+1); g.mesh.setTransfiniteCurve(flankL, NPOCK+1)
    g.mesh.setTransfiniteSurface(pock, cornerTags=[pfl,pfr,pin[0],pin[1]]); g.mesh.setRecombine(2, pock)
    g.synchronize()
    allsurf = sects + [pock]
    out = g.extrude([(2,s) for s in allsurf], 0,0,L_SEAL, numElements=[NZ], recombine=True)
    g.synchronize()
    tops = [out[i-1][1] for i,e in enumerate(out) if e[0]==3]
    lat  = {e[1] for e in out if e[0]==2} - set(tops)
    def one(c):
        r = [s for s in lat if c in [abs(t) for _,t in gmsh.model.getBoundary([(2,s)],oriented=False)]]
        assert len(r)==1, (c,r); return r[0]
    gmsh.model.addPhysicalGroup(2, [one(inner[i]) for i in range(1,6)], name="shaft")
    gmsh.model.addPhysicalGroup(2, [one(outer[i]) for i in range(6)],   name="bore")
    gmsh.model.addPhysicalGroup(2, [one(floor)],                        name="keywayFloor")
    gmsh.model.addPhysicalGroup(2, [one(flankR), one(flankL)],          name="keywayFlanks")
    gmsh.model.addPhysicalGroup(2, allsurf, name="highPressure")   # z=0  : 50 mm-step (oil) end
    gmsh.model.addPhysicalGroup(2, tops,    name="lowPressure")    # z=L  : 30 mm-step end
    gmsh.model.addPhysicalGroup(3, [e[1] for e in out if e[0]==3], name="oil")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3); gmsh.write(fname)
    return len(gmsh.model.mesh.getElements(3)[1][0])

gmsh.option.setNumber("General.Terminal",0)
ncell = build_mesh(NZ=40, NR=4, NKW=16, NS=26, NPOCK=12)
print("cells:", ncell, " expected:", (16+5*26)*4*40 + 16*12*40)

# -- cell 15 ------------------------------------------------------------------------
# Refined mesh built (31 040 hexes). Convert, re-tag walls, and check.
import subprocess, shutil, os
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r0 = subprocess.run(["gmshToFoam","seal.msh"], capture_output=True, text=True)
for p in ["shaft","bore","keywayFloor","keywayFlanks"]:
    for k in ("type","physicalType"):
        subprocess.run(["foamDictionary","constant/polyMesh/boundary","-entry",f"entry0/{p}/{k}",
                        "-set","wall"], check=True, capture_output=True)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("gmshToFoam rc", r0.returncode, "| undefined:", [l for l in r0.stdout.splitlines() if "undefined" in l])
print(r.stdout[r.stdout.find("Checking patch topology"):][:3000])

# -- cell 16 ------------------------------------------------------------------------
# Mesh OK: 31 040 hexes, volume within 0.05 % of exact. Now verify from the mesh itself that the gap i
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
nt, nc, _ = gmsh.model.mesh.getNodes()
X = nc.reshape(-1,3); R = np.hypot(X[:,0], X[:,1])
def grp_r(name):
    dt = [ (d,t) for d,t in gmsh.model.getPhysicalGroups(2)
           if gmsh.model.getPhysicalName(d,t)==name ][0]
    ents = gmsh.model.getEntitiesForPhysicalGroup(*dt)
    nds = np.unique(np.concatenate([gmsh.model.mesh.getNodes(2,e,True)[0] for e in ents]))
    p = np.array([gmsh.model.mesh.getNode(n)[0] for n in nds])
    return p
sh, bo, fl, fk = grp_r("shaft"), grp_r("bore"), grp_r("keywayFloor"), grp_r("keywayFlanks")
print("shaft patch radius  [mm] min/max: %.5f %.5f (target 20.000)"%(np.hypot(sh[:,0],sh[:,1]).min()*1e3, np.hypot(sh[:,0],sh[:,1]).max()*1e3))
print("bore  patch radius  [mm] min/max: %.5f %.5f (target 20.050)"%(np.hypot(bo[:,0],bo[:,1]).min()*1e3, np.hypot(bo[:,0],bo[:,1]).max()*1e3))
print("radial clearance    [mm] = %.5f (target 0.050)"%((np.hypot(bo[:,0],bo[:,1]).min()-np.hypot(sh[:,0],sh[:,1]).max())*1e3))
print("keyway floor y [mm] = %.4f (target 15.0);  floor width [mm] = %.4f (target 12.0)"%(fl[:,1].mean()*1e3, (fl[:,0].max()-fl[:,0].min())*1e3))
print("flank |x| [mm] = %.4f (target 6.0); flank y range [mm] %.4f..%.4f"%(np.abs(fk[:,0]).mean()*1e3, fk[:,1].min()*1e3, fk[:,1].max()*1e3))
print("axial extent [mm] = %.4f (target 60.0)"%((X[:,2].max()-X[:,2].min())*1e3))
# meshed cross-section at z=0
et, etg, enodes = gmsh.model.mesh.getElements(2)
fig, ax = plt.subplots(1,2, figsize=(11,5))
for s in range(1,8):
    t, tg, nn = gmsh.model.mesh.getElements(2, s)
    q = np.array([gmsh.model.mesh.getNode(n)[0][:2] for n in nn[0]]).reshape(-1,4,2)*1e3
    for a in ax: a.add_collection(matplotlib.collections.PolyCollection(q, fc="none", ec="k", lw=0.3))
for a in ax: a.set_aspect("equal")
ax[0].set_xlim(-21,21); ax[0].set_ylim(-21,21); ax[0].set_title("meshed section (mm)")
ax[1].set_xlim(-8,8); ax[1].set_ylim(14,20.4); ax[1].set_title("keyway mouth + gap")
fig.savefig("mesh_section.png", dpi=120); plt.close(fig)
from IPython.display import Image, display; display(Image("mesh_section.png"))

# -- cell 17 ------------------------------------------------------------------------
# All targets exact. One last zoom to confirm the 4 cells across the 0.05 mm gap are actually there an
import numpy as np, matplotlib, matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(11,4.5))
for s in range(1,8):
    t, tg, nn = gmsh.model.mesh.getElements(2, s)
    q = np.array([gmsh.model.mesh.getNode(n)[0][:2] for n in nn[0]]).reshape(-1,4,2)*1e3
    for a in ax: a.add_collection(matplotlib.collections.PolyCollection(q, fc="lightblue", ec="k", lw=0.4))
ax[0].set_xlim(-2,2); ax[0].set_ylim(19.94,20.08); ax[0].set_title("gap far from keyway (mm)")
ax[1].set_xlim(5.2,7.2); ax[1].set_ylim(18.4,19.6); ax[1].set_title("keyway flank meeting the gap")
for a in ax: a.set_aspect("auto"); a.grid(alpha=.3)
fig.savefig("gap_zoom.png", dpi=120); plt.close(fig)
from IPython.display import Image, display; display(Image("gap_zoom.png"))
