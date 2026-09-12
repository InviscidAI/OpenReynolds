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

R, H = 0.060, 0.200            # vessel inner radius, height
R_OUT = 0.020                  # outlet radius (40 mm dia)
PW, PH = 0.020, 0.030          # port width (tangential-ish), height (vertical)
ZP = 0.040                     # port axis height above floor
LD = 0.050                     # duct length along own axis
ANG = math.radians(15.0)       # off local tangent

vessel = Cylinder(radius=R, height=H, align=(Align.CENTER, Align.CENTER, Align.MIN))

ducts, frames = [], []
for i in range(4):
    th = i * math.pi / 2
    rhat = Vector(math.cos(th), math.sin(th), 0)
    that = Vector(-math.sin(th), math.cos(th), 0)
    din = (that * math.cos(ANG) - rhat * math.sin(ANG)).normalized()   # inflow dir
    P0 = rhat * R + Vector(0, 0, ZP)                                   # axis crosses wall here
    O = P0 - din * LD                                                  # outer (inlet) face centre
    pl = Plane(origin=O, z_dir=din, x_dir=Vector(0, 0, 1))
    tube = extrude(pl * Rectangle(PH, PW), amount=LD + 2 * R + 0.02)
    pieces = (tube - vessel).solids()
    keep = min(pieces, key=lambda s: (Vector(*s.center()) - O).length)
    ducts.append(keep); frames.append((din, O, P0))

fluid = vessel + ducts[0] + ducts[1] + ducts[2] + ducts[3]
print(fluid.is_valid, len(fluid.solids()), fluid.volume)
print(fluid.bounding_box())
print("vessel vol", math.pi*R**2*H)

# -- cell 2 -------------------------------------------------------------------------
# The union kept the whole 190 mm strip: with a 15° port the duct's outboard wall never re-enters the 
import numpy as np, math
from build123d import *

R, H = 0.060, 0.200            # vessel inner radius, height
R_OUT = 0.020                  # outlet radius (40 mm dia)
PW, PH = 0.020, 0.030          # port width (horizontal), height (vertical)
ZP = 0.040                     # port axis height above floor
LD = 0.050                     # duct length along own axis
ANG = math.radians(15.0)       # duct axis off local tangent

vessel = Cylinder(radius=R, height=H, align=(Align.CENTER, Align.CENTER, Align.MIN))

ducts, frames = [], []
for i in range(4):
    th = i * math.pi / 2
    rhat = Vector(math.cos(th), math.sin(th), 0)
    that = Vector(-math.sin(th), math.cos(th), 0)
    din = (that * math.cos(ANG) - rhat * math.sin(ANG)).normalized()   # inflow direction
    P0 = rhat * R + Vector(0, 0, ZP)                                   # axis meets wall here
    O = P0 - din * LD                                                  # inlet face centre
    pl = Plane(origin=O, z_dir=din, x_dir=Vector(0, 0, 1))
    ducts.append(extrude(pl * Rectangle(PH, PW), amount=LD))
    frames.append((din, O, P0))

fluid = vessel + ducts[0] + ducts[1] + ducts[2] + ducts[3]
print("valid", fluid.is_valid, "solids", len(fluid.solids()), "vol", fluid.volume)
print("extra over bare vessel per duct:", (fluid.volume - math.pi*R**2*H)/4, "vs full tube", LD*PW*PH)
print(fluid.bounding_box())
print("faces", len(fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
for i, f in enumerate(fluid.faces()):
    c = f.center(); n = f.normal_at(c)
    print(f"{i:2d} {str(f.geom_type):18s} area={f.area*1e6:8.1f}mm2 c=({c.X*1000:7.1f},{c.Y*1000:7.1f},{c.Z*1000:7.1f}) n=({n.X:5.2f},{n.Y:5.2f},{n.Z:5.2f})")

# -- cell 4 -------------------------------------------------------------------------
# Face inventory matches the design (inlet 600 mm², outboard duct wall 50×30=1500 mm², tongue face 300
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.TopTools import TopTools_ListOfShape

tool = Cylinder(radius=R_OUT, height=0.02, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((0,0,H)))
tool_face = [f for f in tool.faces() if f.geom_type == GeomType.CYLINDER][0]

args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
args.Append(fluid.wrapped); tools.Append(tool_face.wrapped)
sp = BRepAlgoAPI_Splitter(); sp.SetArguments(args); sp.SetTools(tools); sp.Build()
fluid_s = Solid(sp.Shape()) if sp.Shape().ShapeType() == 2 else Compound(sp.Shape()).solids()[0]
print("valid", fluid_s.is_valid, "vol", fluid_s.volume, "faces", len(fluid_s.faces()))
for f in fluid_s.faces():
    c = f.center()
    if abs(c.Z - H) < 1e-9 and f.geom_type == GeomType.PLANE:
        print("top-plane face area mm2:", f.area*1e6)

# -- cell 5 -------------------------------------------------------------------------
# The classifier cell died because `duct_local` came from an errored cell — and the face list also sho
sh = sp.Shape()
print("result type", sh.ShapeType(), "fluid_s type", type(fluid_s), fluid_s.wrapped.ShapeType())
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL
ex = TopExp_Explorer(sh, TopAbs_SOLID); n=0
while ex.More(): n+=1; ex.Next()
print("solids in result:", n, " shells in fluid_s:", len(fluid_s.shells()))
for s in fluid_s.shells():
    print("  shell faces", len(s.faces()), "closed", s.wrapped.Closed())

# -- cell 6 -------------------------------------------------------------------------
# The solid picked up the tool face as a second, open internal shell. I'll patch from the closed 28-fa
import math
from build123d import GeomType, Vector

outer_shell = max(fluid_s.shells(), key=lambda s: len(s.faces()))
print("outer shell faces", len(outer_shell.faces()), "closed", outer_shell.wrapped.Closed())

def duct_local(c, i):
    """(s along duct axis from inlet face, o across it, z) in duct i's own frame."""
    din, O, P0 = frames[i]
    n = din.cross(Vector(0, 0, 1)).normalized()
    d = Vector(c.X, c.Y, c.Z) - O
    return d.dot(din), d.dot(n), c.Z

patch_of = {}
for f in outer_shell.faces():
    c = f.center(); nrm = f.normal_at(c); a = f.area
    if f.geom_type == GeomType.CYLINDER and abs(math.hypot(c.X, c.Y) - R) < 1e-6:
        p = "cylwall"
    elif abs(c.Z) < 1e-9 and abs(nrm.Z) > 0.99:
        p = "floor"
    elif abs(c.Z - H) < 1e-9 and abs(nrm.Z) > 0.99:
        p = "top" if a > 5e-3 else "outlet"
    else:
        i, (s, o, z) = min(((i, duct_local(c, i)) for i in range(4)),
                           key=lambda t: abs(t[1][1]) + abs(t[1][0] - LD / 2))
        assert -1e-6 <= s <= LD + 1e-6 and abs(o) <= PW / 2 + 1e-6, (s, o)
        p = f"inlet{i}" if (s < 1e-6 and nrm.dot(frames[i][0]) < -0.99) else f"ductwall{i}"
    patch_of.setdefault(p, []).append(f)

for p in sorted(patch_of):
    print(f"{p:10s} nf={len(patch_of[p]):2d} area={sum(f.area for f in patch_of[p])*1e6:9.1f} mm2")

# -- cell 7 -------------------------------------------------------------------------
# All twelve patches resolved. Now measure the requested numbers back off the built solid, and draw it
print("vessel bore dia  :", 2*R*1000, "mm  (asked 120)")
print("vessel height    :", (fluid_s.bounding_box().max.Z - fluid_s.bounding_box().min.Z)*1000, "mm (asked 200)")
print("outlet dia       :", 2*math.sqrt(sum(f.area for f in patch_of['outlet'])/math.pi)*1000, "mm (asked 40)")
for i in range(4):
    din, O, P0 = frames[i]
    rhat = Vector(P0.X, P0.Y, 0).normalized(); that = Vector(-rhat.Y, rhat.X, 0)
    ang = math.degrees(math.acos(max(-1, min(1, din.dot(that)))))
    inl = patch_of[f"inlet{i}"][0]
    bb = inl.bounding_box()
    L = (Vector(P0.X,P0.Y,P0.Z) - Vector(inl.center().X, inl.center().Y, inl.center().Z)).length
    w = math.hypot(bb.max.X-bb.min.X, bb.max.Y-bb.min.Y)
    print(f"duct{i}: axis-vs-tangent {ang:5.2f} deg (asked 15) | len along axis {L*1000:5.1f} mm (asked 50) | "
          f"port {w*1000:4.1f} x {(bb.max.Z-bb.min.Z)*1000:4.1f} mm (asked 20x30) | port axis z {P0.Z*1000:4.1f} mm (asked 40) | "
          f"wall crossing r {math.hypot(P0.X,P0.Y)*1000:5.1f} mm (asked 60)")
print("swirl sense (din x rhat)_z, all same sign:", [round(frames[i][0].cross(Vector(math.cos(i*math.pi/2), math.sin(i*math.pi/2),0)).Z,3) for i in range(4)])

# -- cell 8 -------------------------------------------------------------------------
# All twelve requested numbers check out. Now tessellate the whole shell once (so shared edges are con
import os, numpy as np
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED

os.makedirs("constant/triSurface", exist_ok=True)
BRepMesh_IncrementalMesh(fluid_s.wrapped, 0.0004, False, 0.3, True)

def face_tris(f):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    trsf = loc.Transformation()
    pts = []
    for k in range(1, tri.NbNodes() + 1):
        p = tri.Node(k).Transformed(trsf); pts.append((p.X(), p.Y(), p.Z()))
    pts = np.array(pts)
    rev = f.wrapped.Orientation() == TopAbs_REVERSED
    tris = []
    for k in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(k).Get()
        tris.append((a-1, c-1, b-1) if rev else (a-1, b-1, c-1))
    return pts, np.array(tris)

def write_stl(name, faces):
    V, T = [], []
    off = 0
    for f in faces:
        p, t = face_tris(f)
        V.append(p); T.append(t + off); off += len(p)
    V = np.vstack(V); T = np.vstack(T)
    with open(f"constant/triSurface/{name}.stl", "w") as fh:
        fh.write(f"solid {name}\n")
        for a, b, c in T:
            n = np.cross(V[b]-V[a], V[c]-V[a]); ln = np.linalg.norm(n)
            n = n/ln if ln > 0 else np.array([0., 0., 1.])
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in (a, b, c):
                fh.write(f"   vertex {V[v][0]:.7e} {V[v][1]:.7e} {V[v][2]:.7e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
    return len(T), V

tot = 0
for p in sorted(patch_of):
    ntri, V = write_stl(p, patch_of[p]); tot += ntri
    print(f"{p:10s} {ntri:5d} tris")
print("total tris", tot)

# -- cell 9 -------------------------------------------------------------------------
# Enough analysis — get a mesh on disk. Background block ~8 mm cells, snappy with each STL as its own 
import os, textwrap, subprocess
os.makedirs("system", exist_ok=True)
PATCHES = sorted(patch_of)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"

open("system/controlDict","w").write(HDR%"controlDict"+
 "application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;\n"
 "writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(HDR%"fvSchemes"+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(HDR%"fvSolution"+"solvers{} relaxationFactors{}\n")

XY, Z0, Z1, NB = 0.092, -0.006, 0.206, 0.008
nx = int(round(2*XY/NB)); nz = int(round((Z1-Z0)/NB))
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+f"""
scale 1;
vertices
(
 (-{XY} -{XY} {Z0}) ({XY} -{XY} {Z0}) ({XY} {XY} {Z0}) (-{XY} {XY} {Z0})
 (-{XY} -{XY} {Z1}) ({XY} -{XY} {Z1}) ({XY} {XY} {Z1}) (-{XY} {XY} {Z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1));
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(3 0 4 7)); }} );
""")
print(nx, nx, nz, nx*nx*nz)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-400:])

# -- cell 10 ------------------------------------------------------------------------
# Background mesh is in. Now snappy with all twelve STLs as separate named patches — coarse levels fir
geom = "\n".join(f'  {p}.stl {{ type triSurfaceMesh; name {p}; }}' for p in PATCHES)
refs = "\n".join(f'    {p} {{ level (1 1); patchInfo {{ type {"patch" if (p.startswith("inlet") or p=="outlet") else "wall"}; }} }}' for p in PATCHES)
open("system/snappyHexMeshDict","w").write(HDR%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}};
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 2;
  maxLoadUnbalance 0.1; resolveFeatureAngle 40; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces
  {{
{refs}
  }}
  refinementRegions {{}};
  locationInMesh (0.003 0.002 0.120);
}}
snapControls
{{
  nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls
{{
  relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
  maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6; debug 0;
""")
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 11 ------------------------------------------------------------------------
# A snapped mesh exists. Look at it: patch inventory, checkMesh, and a rendering.
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print(r.stdout[r.stdout.find("Mesh stats"):r.stdout.find("End")][:3000])

# -- cell 12 ------------------------------------------------------------------------
# checkMesh is clean and all 12 patches are populated. Let me look at it.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pv.OFF_SCREEN = True
m = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.number_time_points else None
mesh = rd.read()
internal = mesh["internalMesh"]
pl = pv.Plotter(off_screen=True, window_size=(900,700))
bnd = mesh["boundary"]
for nm in bnd.keys():
    col = "red" if nm.startswith("inlet") else ("blue" if nm=="outlet" else ("green" if nm.startswith("ductwall") else "lightgrey"))
    pl.add_mesh(bnd[nm], color=col, show_edges=True, opacity=1.0 if nm!="cylwall" else 0.35, line_width=0.5)
pl.camera_position = [(0.35,-0.30,0.30),(0,0,0.09),(0,0,1)]
pl.screenshot("view_iso.png")
print("ok", list(bnd.keys()))

# -- cell 13 ------------------------------------------------------------------------
# The screenshot was written but not shown — display it explicitly.
from IPython.display import Image, display
print(os.path.getsize("view_iso.png"))
display(Image("view_iso.png"))

# -- cell 14 ------------------------------------------------------------------------
# Shape is right: four near-tangential ducts, red inlet faces at the outer ends, blue outlet disk in t
NB2 = 0.006
nx2 = int(round(2*XY/NB2)); nz2 = int(round((Z1-Z0)/NB2))
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+f"""
scale 1;
vertices
(
 (-{XY} -{XY} {Z0}) ({XY} -{XY} {Z0}) ({XY} {XY} {Z0}) (-{XY} {XY} {Z0})
 (-{XY} -{XY} {Z1}) ({XY} -{XY} {Z1}) ({XY} {XY} {Z1}) (-{XY} {XY} {Z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx2} {nx2} {nz2}) simpleGrading (1 1 1));
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(3 0 4 7)); }} );
""")
lev = lambda p: 2 if (p.startswith("duct") or p.startswith("inlet")) else 1
refs2 = "\n".join(f'    {p} {{ level ({lev(p)} {lev(p)}); patchInfo {{ type {"patch" if (p.startswith("inlet") or p=="outlet") else "wall"}; }} }}' for p in PATCHES)
d = open("system/snappyHexMeshDict").read()
import re
d = re.sub(r"refinementSurfaces\s*\{.*?\n  \}", "refinementSurfaces\n  {\n"+refs2+"\n  }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).returncode, nx2, nz2, nx2*nx2*nz2)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:])

# -- cell 15 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
s = r.stdout
print(s[s.find("Mesh stats"):s.find("Checking topology")])
import re as _re
print("\n".join(l for l in s.splitlines() if _re.search(r"(non-orthogonality|skewness|Total volume|Mesh OK|\*\*\*|inlet|outlet|cylwall|floor|top|ductwall)", l)))

# -- cell 16 ------------------------------------------------------------------------
# One skewness flag (max 4.06, 36 faces). Tighten snappy's quality target below checkMesh's threshold 
d = open("system/snappyHexMeshDict").read()
d = d.replace("maxInternalSkewness 4;", "maxInternalSkewness 3.2;").replace("maxBoundarySkewness 20;", "maxBoundarySkewness 10;")
d = d.replace("nSolveIter 50; nRelaxIter 5;", "nSolveIter 100; nRelaxIter 8;").replace("errorReduction 0.75;", "errorReduction 0.9;")
open("system/snappyHexMeshDict","w").write(d)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).returncode)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True); print(r.returncode)
c = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True).stdout
print("\n".join(l for l in c.splitlines() if _re.search(r"(cells:|non-orthogonality|skewness|Total volume|Mesh OK|\*\*\*)", l)))

# -- cell 17 ------------------------------------------------------------------------
# Skewness is stuck at 4.06 — likely at the tangential cusp where each duct end plane grazes the wall.
r = subprocess.run(["checkMesh","-constant","-writeSets","vtk"], capture_output=True, text=True)
print([l for l in r.stdout.splitlines() if "skew" in l.lower()])
import glob
print(glob.glob("**/*skew*", recursive=True)[:10])

# -- cell 18 ------------------------------------------------------------------------
sk = pv.read("postProcessing/constant/skewFaces/skewFaces.vtp")
c = sk.cell_centers().points
import numpy as np
print(sk.n_cells)
for p in c:
    s, o, z = duct_local(Vector(*p), min(range(4), key=lambda i: abs(duct_local(Vector(*p), i)[1])))
    print(f"({p[0]*1e3:7.1f},{p[1]*1e3:7.1f},{p[2]*1e3:6.1f})  r={np.hypot(p[0],p[1])*1e3:6.1f}  s={s*1e3:6.1f} o={o*1e3:6.1f}")

# -- cell 19 ------------------------------------------------------------------------
# The 38 skew faces are all at one place per duct: the shallow wedge where the duct's inboard sidewall
wedge_pts = []
for i in range(4):
    din, O, P0 = frames[i]
    n = din.cross(Vector(0,0,1)).normalized()
    wedge_pts.append(O + din*0.0292 - n*0.010)      # upstream edge of port opening
HW = 0.012
gb = "\n".join(f'  wedge{i} {{ type searchableBox; min ({p.X-HW} {p.Y-HW} 0.020); max ({p.X+HW} {p.Y+HW} 0.060); }}'
               for i, p in enumerate(wedge_pts))
rr = "\n".join(f'    wedge{i} {{ mode inside; levels ((1e15 3)); }}' for i in range(4))
d = open("system/snappyHexMeshDict").read()
d = d.replace("};\ncastellatedMeshControls", gb + "\n};\ncastellatedMeshControls")
d = d.replace("refinementRegions {};", "refinementRegions\n  {\n" + rr + "\n  };")
open("system/snappyHexMeshDict","w").write(d)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).returncode)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True); print(r.returncode)
c = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True).stdout
print("\n".join(l for l in c.splitlines() if _re.search(r"(cells:|non-orthogonality|skewness|Total volume|Mesh OK|\*\*\*)", l)))

# -- cell 20 ------------------------------------------------------------------------
# `Mesh OK` at 259k cells. Final verification: patch areas measured off the mesh against the CAD, and 
rd2 = pv.OpenFOAMReader("case.foam"); msh = rd2.read(); bnd = msh["boundary"]
print(f"{'patch':10s} {'mesh mm2':>10s} {'CAD mm2':>10s}")
for nm in bnd.keys():
    a_mesh = bnd[nm].compute_cell_sizes(length=False, volume=False)["Area"].sum()*1e6
    a_cad = sum(f.area for f in patch_of[nm])*1e6
    print(f"{nm:10s} {a_mesh:10.1f} {a_cad:10.1f}")
cl = msh["internalMesh"].slice(normal="z", origin=(0,0,0.040))
pl = pv.Plotter(off_screen=True, window_size=(800,760))
pl.add_mesh(cl, color="lightsteelblue", show_edges=True, line_width=0.4)
pl.view_xy(); pl.screenshot("slice_z40.png")
display(Image("slice_z40.png"))
