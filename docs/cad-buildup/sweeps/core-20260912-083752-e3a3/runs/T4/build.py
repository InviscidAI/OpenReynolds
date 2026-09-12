"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: the fluid volume is the serpentine channel itself — 4 mm wide, 5 mm deep, milled 5 mm down fro
from build123d import *
import numpy as np

# --- parameters (mm) ---
PL, PW, PH = 60.0, 40.0, 8.0      # plate length, width, height
CW, CD = 4.0, 5.0                  # channel width, depth (milled from top)
RIB = 3.0                          # metal between passes
PITCH = CW + RIB                   # 7 mm centre-to-centre
ZC = PH - CD/2                     # channel z centre -> spans z = 3..8

YS = PW/2 - PITCH                  # 13.0 : first pass centreline
Y1, Y2, Y3 = YS, YS + PITCH, YS + 2*PITCH   # 13, 20, 27
XA = PL - 4.0                      # 56 : U-turn A centreline x
XB = 4.0                           # 4  : U-turn B centreline x

pass1 = Pos(XA/2, Y1, ZC) * Box(XA, CW, CD)                 # x 0..56
turnA = Pos(XA - CW/2, (Y1+Y2)/2, ZC) * Box(CW, PITCH+CW, CD)
pass2 = Pos((XB+XA)/2, Y2, ZC) * Box(XA-XB, CW, CD)          # x 4..56
turnB = Pos(XB + CW/2, (Y2+Y3)/2, ZC) * Box(CW, PITCH+CW, CD)
pass3 = Pos((XB+PL)/2, Y3, ZC) * Box(PL-XB, CW, CD)          # x 4..60

fluid_mm = pass1 + turnA + pass2 + turnB + pass3
fluid = fluid_mm.scale(0.001)      # mm -> m, once, here

bb = fluid.bounding_box()
print("solids:", len(fluid.solids()), " volume (m^3):", fluid.volume)
print("bbox min (m):", tuple(round(v,5) for v in bb.min.to_tuple()))
print("bbox max (m):", tuple(round(v,5) for v in bb.max.to_tuple()))
print("pass centrelines y (mm):", Y1, Y2, Y3, " rib gap (mm):", Y2-Y1-CW)
print("channel z span (mm):", ZC-CD/2, "to", ZC+CD/2)
# hand-computed centreline length check
L = (XA-0) + (Y2-Y1) + (XA-XB) + (Y3-Y2) + (PL-XB)
print("centreline length (mm):", L, " x-section area (mm^2):", CW*CD,
      " vol if no corner double-count (mm^3):", L*CW*CD, " actual:", fluid_mm.volume)

# -- cell 2 -------------------------------------------------------------------------
# `export_stl` is a module-level function in this version.
import pyvista as pv
from build123d import export_stl
pv.global_theme.background = "white"

export_stl(fluid, "fluid_check.stl")
m = pv.read("fluid_check.stl")
p = pv.Plotter(off_screen=True, window_size=(1100, 700))
p.add_mesh(m, color="deepskyblue", show_edges=True, edge_color="navy")
p.add_mesh(pv.Box(bounds=(0,0.06,0,0.04,0,0.008)), style="wireframe", color="gray")
p.show_axes()
p.camera_position = [(0.10, -0.08, 0.09), (0.03, 0.02, 0.004), (0, 0, 1)]
p.screenshot("view_iso.png")
print("bounds", m.bounds)

# -- cell 3 -------------------------------------------------------------------------
# Screenshot didn't come back; check whether the file was written and display it.
import os
from IPython.display import Image, display
print(os.path.exists("view_iso.png"), os.path.getsize("view_iso.png") if os.path.exists("view_iso.png") else 0)
if os.path.exists("view_iso.png"):
    display(Image("view_iso.png"))

# -- cell 4 -------------------------------------------------------------------------
# Serpentine reads correctly: three passes, U-turn at far x, channel occupying the top 5 mm of the pla
from build123d import Face

def face_at(shape, centre_mm, normal):
    c = Vector(*centre_mm) * 0.001
    hits = [f for f in shape.faces()
            if (f.center() - c).length < 1e-9
            and abs(f.normal_at(f.center()).dot(Vector(*normal))) > 1 - 1e-9]
    assert len(hits) == 1, (centre_mm, len(hits))
    return hits[0]

inlet_f  = face_at(fluid, (0.0, Y1, ZC), (1, 0, 0))
outlet_f = face_at(fluid, (PL,  Y3, ZC), (1, 0, 0))
wall_fs  = [f for f in fluid.faces() if f is not inlet_f and f is not outlet_f]

A_nom = CW * CD * 1e-6
print("n faces total:", len(fluid.faces()), " walls:", len(wall_fs))
print("inlet  area m^2: %.3e (nominal %.3e)  centre %s" % (inlet_f.area, A_nom, tuple(inlet_f.center()*1000)))
print("outlet area m^2: %.3e (nominal %.3e)  centre %s" % (outlet_f.area, A_nom, tuple(outlet_f.center()*1000)))
print("wall area m^2: %.4e" % sum(f.area for f in wall_fs))

# -- cell 5 -------------------------------------------------------------------------
# `faces()` returns fresh wrappers, so identity filtering failed. Cache one face list and filter by ce
all_faces = fluid.faces()

def pick(centre_mm, normal):
    c = Vector(*centre_mm) * 0.001
    hits = [i for i, f in enumerate(all_faces)
            if (f.center() - c).length < 1e-9
            and abs(f.normal_at(f.center()).dot(Vector(*normal))) > 1 - 1e-9]
    assert len(hits) == 1, (centre_mm, len(hits))
    return hits[0]

i_in  = pick((0.0, Y1, ZC), (1, 0, 0))
i_out = pick((PL,  Y3, ZC), (1, 0, 0))
inlet_f, outlet_f = all_faces[i_in], all_faces[i_out]
wall_fs = [f for i, f in enumerate(all_faces) if i not in (i_in, i_out)]

print("walls:", len(wall_fs), " wall area m^2: %.4e" % sum(f.area for f in wall_fs))
print("total face area check: %.4e vs %.4e" % (sum(f.area for f in all_faces),
                                               sum(f.area for f in wall_fs) + inlet_f.area + outlet_f.area))

# -- cell 6 -------------------------------------------------------------------------
# Export each named group to its own STL and concatenate into one multi-solid surface for cfMesh.
import os, subprocess
from build123d import Compound, export_stl

os.makedirs("constant/triSurface", exist_ok=True)
groups = {"inlet": [inlet_f], "outlet": [outlet_f], "walls": wall_fs}
for name, fl in groups.items():
    export_stl(Compound(children=[f for f in fl]), f"constant/triSurface/{name}.stl",
               tolerance=1e-5, angular_tolerance=0.1, ascii_format=True)

parts = []
for name in ("inlet", "outlet", "walls"):
    txt = open(f"constant/triSurface/{name}.stl").read().splitlines()
    txt[0] = f"solid {name}"
    txt[-1] = f"endsolid {name}"
    parts.append("\n".join(txt))
open("constant/triSurface/fluid.stl", "w").write("\n".join(parts) + "\n")

out = subprocess.run(["surfaceCheck", "constant/triSurface/fluid.stl"],
                     capture_output=True, text=True)
print(out.stdout[-1500:])

# -- cell 7 -------------------------------------------------------------------------
# cfMesh `cartesianMesh` on the closed multi-region surface; all faces are axis-aligned so a 1 mm coar
import textwrap, subprocess, os
os.makedirs("system", exist_ok=True)

def foam(path, cls, obj, body):
    hdr = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
           f"    class {cls};\n    object {obj};\n}}\n")
    open(path, "w").write("FoamFile\n" if False else hdr + body)

foam("system/controlDict", "dictionary", "controlDict", textwrap.dedent("""
    application     icoFoam;
    startFrom       startTime;  startTime 0;
    stopAt          endTime;    endTime 1;
    deltaT          1;          writeControl timeStep;  writeInterval 1;
"""))
foam("system/fvSchemes", "dictionary", "fvSchemes",
     "ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\n"
     "divSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
foam("system/fvSolution", "dictionary", "fvSolution", "solvers{}\n")
foam("system/meshDict", "dictionary", "meshDict", textwrap.dedent("""
    surfaceFile "constant/triSurface/fluid.stl";
    maxCellSize 0.001;
"""))

r = subprocess.run(["cartesianMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1200:])
print(r.stderr[-600:])

# -- cell 8 -------------------------------------------------------------------------
# Mesh built. Check it and the patch names/areas.
import subprocess
print(subprocess.run(["checkMesh"], capture_output=True, text=True).stdout[-2500:])
print(open("constant/polyMesh/boundary").read()[-900:])

# -- cell 9 -------------------------------------------------------------------------
# `checkMesh` passes but total volume is 3.035e-06 m³ against the exact 3.400e-06 — cfMesh rounded ~11
import numpy as np, textwrap

XN = [0.0, XB, 2*XB, XA-CW, XA, PL]              # 0,4,8,52,56,60
YN = [Y1-CW/2, Y1+CW/2, Y2-CW/2, Y2+CW/2, Y3-CW/2, Y3+CW/2]   # 11,15,18,22,25,29
ZN = [ZC-CD/2, ZC+CD/2]                          # 3,8
H = 1.0                                          # coarse target cell size (mm)
NX = [max(1, round((XN[i+1]-XN[i])/H)) for i in range(5)]
NY = [max(1, round((YN[j+1]-YN[j])/H)) for j in range(5)]
NZ = [max(1, round((ZN[1]-ZN[0])/H))]

# fluid footprint = union of the five rectangles (x0,x1,y0,y1) in mm
RECTS = [(0.0, XA, Y1-CW/2, Y1+CW/2),            # pass 1
         (XA-CW, XA, Y1-CW/2, Y2+CW/2),          # U-turn A
         (XB, XA, Y2-CW/2, Y2+CW/2),             # pass 2
         (XB, XB+CW, Y2-CW/2, Y3+CW/2),          # U-turn B
         (XB, PL, Y3-CW/2, Y3+CW/2)]             # pass 3
def inside(xc, yc):
    return any(x0 < xc < x1 and y0 < yc < y1 for x0, x1, y0, y1 in RECTS)

cells = [(i, j) for j in range(5) for i in range(5)
         if inside(0.5*(XN[i]+XN[i+1]), 0.5*(YN[j]+YN[j+1]))]
print("blocks:", len(cells), cells)
print("NX", NX, "NY", NY, "NZ", NZ)
print("block volume sum (mm^3):",
      sum((XN[i+1]-XN[i])*(YN[j+1]-YN[j])*(ZN[1]-ZN[0]) for i, j in cells))

# -- cell 10 ------------------------------------------------------------------------
# 13 blocks summing to exactly 3400 mm³ — the decomposition is exact. Write the dict (points in metres
pid, pts = {}, []
def P(i, j, k):
    key = (i, j, k)
    if key not in pid:
        pid[key] = len(pts)
        pts.append((XN[i]*1e-3, YN[j]*1e-3, ZN[k]*1e-3))
    return pid[key]

blocks = []
for i, j in cells:
    v = [P(i,j,0), P(i+1,j,0), P(i+1,j+1,0), P(i,j+1,0),
         P(i,j,1), P(i+1,j,1), P(i+1,j+1,1), P(i,j+1,1)]
    blocks.append(f"    hex ({' '.join(map(str,v))}) ({NX[i]} {NY[j]} {NZ[0]}) simpleGrading (1 1 1)")

# inlet: x=0 end of pass 1 (grid column i=0, row j=0); outlet: x=PL end of pass 3 (i=4, j=4)
inlet_face  = f"({P(0,0,0)} {P(0,0,1)} {P(0,1,1)} {P(0,1,0)})"      # -x outward
outlet_face = f"({P(5,4,0)} {P(5,5,0)} {P(5,5,1)} {P(5,4,1)})"      # +x outward

dict_txt = f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object blockMeshDict;
}}
scale 1;

vertices
(
{chr(10).join('    (%.6g %.6g %.6g)' % p for p in pts)}
);

blocks
(
{chr(10).join(blocks)}
);

edges ();

defaultPatch {{ name walls; type wall; }}

boundary
(
    inlet  {{ type patch; faces ( {inlet_face} ); }}
    outlet {{ type patch; faces ( {outlet_face} ); }}
);
"""
open("system/blockMeshDict", "w").write(dict_txt)
print("points:", len(pts), " blocks:", len(blocks))
print("expected cells:", sum(NX[i]*NY[j]*NZ[0] for i, j in cells))

# -- cell 11 ------------------------------------------------------------------------
import subprocess, shutil, os
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1800:])
print(r.stderr[-800:])

# -- cell 12 ------------------------------------------------------------------------
# Exact: 3400 cells, inlet and outlet each 20 faces (4×5 mm at 1 mm). Check the mesh and view the patc
import subprocess
print(subprocess.run(["checkMesh"], capture_output=True, text=True).stdout[-1600:])

# -- cell 13 ------------------------------------------------------------------------
# Perfect orthogonal hex mesh: non-orthogonality 0, skewness ~0, total volume exactly 3.4e-06 m³. Now 
import pyvista as pv
open("t4.foam", "w").close()
rd = pv.OpenFOAMReader("t4.foam"); rd.set_active_time_value(0.0)
mesh = rd.read()
internal = mesh["internalMesh"]
pb = mesh["boundary"]
p = pv.Plotter(off_screen=True, window_size=(1100, 700))
p.add_mesh(internal.extract_surface(), color="lightsteelblue", show_edges=True,
           edge_color="gray", opacity=0.35)
for nm, col in (("inlet", "red"), ("outlet", "green")):
    p.add_mesh(pb[nm], color=col)
p.show_axes()
p.camera_position = [(0.11, -0.09, 0.09), (0.03, 0.02, 0.005), (0, 0, 1)]
p.screenshot("view_mesh.png")
from IPython.display import Image, display
display(Image("view_mesh.png"))
print([nm for nm in pb.keys()], "inlet centre:", pb["inlet"].center, "outlet centre:", pb["outlet"].center)

# -- cell 14 ------------------------------------------------------------------------
# Inlet (red) at x=0 on the first pass, outlet (green) at x=60 on the third pass — matching the printe
import subprocess, shutil

H = 0.5                                   # final cell size (mm)
NX = [max(1, round((XN[i+1]-XN[i])/H)) for i in range(5)]
NY = [max(1, round((YN[j+1]-YN[j])/H)) for j in range(5)]
NZ = [max(1, round((ZN[1]-ZN[0])/H))]

blocks = [f"    hex ({P(i,j,0)} {P(i+1,j,0)} {P(i+1,j+1,0)} {P(i,j+1,0)} "
          f"{P(i,j,1)} {P(i+1,j,1)} {P(i+1,j+1,1)} {P(i,j+1,1)}) "
          f"({NX[i]} {NY[j]} {NZ[0]}) simpleGrading (1 1 1)" for i, j in cells]

dict_txt = f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object blockMeshDict;
}}
scale 1;

vertices
(
{chr(10).join('    (%.6g %.6g %.6g)' % p for p in pts)}
);

blocks
(
{chr(10).join(blocks)}
);

edges ();

defaultPatch {{ name walls; type wall; }}

boundary
(
    inlet  {{ type patch; faces ( {inlet_face} ); }}
    outlet {{ type patch; faces ( {outlet_face} ); }}
);
"""
open("system/blockMeshDict", "w").write(dict_txt)
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-500:], r.stderr[-400:])

# -- cell 15 ------------------------------------------------------------------------
# Final check plus measurements taken from the mesh itself against the request.
import subprocess, pyvista as pv
print(subprocess.run(["checkMesh"], capture_output=True, text=True).stdout[-900:])

rd = pv.OpenFOAMReader("t4.foam"); rd.set_active_time_value(0.0)
m = rd.read(); pb = m["boundary"]
ib, ob = pb["inlet"].bounds, pb["outlet"].bounds
print("inlet  y-extent %.1f mm, z-extent %.1f mm, area %.2f mm^2" %
      ((ib[3]-ib[2])*1e3, (ib[5]-ib[4])*1e3, pb["inlet"].area*1e6))
print("outlet y-extent %.1f mm, z-extent %.1f mm, area %.2f mm^2" %
      ((ob[3]-ob[2])*1e3, (ob[5]-ob[4])*1e3, pb["outlet"].area*1e6))
print("fluid volume %.1f mm^3 (exact 3400.0); passes at y = %.0f, %.0f, %.0f mm; rib %.0f mm" %
      (m["internalMesh"].volume*1e9, Y1, Y2, Y3, Y2-Y1-CW))
print("plate envelope used: %g x %g x %g mm; channel depth from top face: %g mm" %
      (PL, PW, PH, PH - (ZC-CD/2)))
