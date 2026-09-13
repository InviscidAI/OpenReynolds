"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np
from build123d import *

# All dimensions in METRES
L   = 0.120   # leg length
W   = 0.010   # passage width
Rc  = 0.015   # bend centreline radius
Ri  = Rc - W/2
Ro  = Rc + W/2
XB  = L       # x of bend centre
TH  = 0.001   # 2D extrusion thickness (empty direction)

face_outline = (
    Polyline((0,-Ro),(XB,-Ro))
    + JernArc(start=(XB,-Ro), tangent=(1,0), radius=Ro, arc_size=180)
    + Polyline((XB,Ro),(0,Ro),(0,Ri))
    + Polyline((0,Ri),(XB,Ri))
    + JernArc(start=(XB,Ri), tangent=(1,0), radius=Ri, arc_size=-180)
    + Polyline((XB,-Ri),(0,-Ri),(0,-Ro))
)
face = make_face(face_outline)
print("area  :", face.area, " expected:", 2*L*W + np.pi/2*(Ro**2-Ri**2))
print("bbox  :", face.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Face area matches analytic to 1e-16. Now the structured mesh: four transfinite quad blocks (two legs
import gmsh, math
gmsh.initialize()
gmsh.model.add("ubend")
g = gmsh.model.geo

NW, NL, NA = 6, 41, 11   # nodes across width, along leg, per 90 deg arc (coarse)

P = lambda x,y: g.addPoint(x,y,0.0)
A,B,G_,E = P(0,-Ro), P(XB,-Ro), P(0,Ro), P(XB,Ro)
D,C,H,F  = P(0,-Ri), P(XB,-Ri), P(0,Ri), P(XB,Ri)
Mo, Mi, O = P(XB+Ro,0), P(XB+Ri,0), P(XB,0)

l_bot  = g.addLine(A,B)          # wall
l_BC   = g.addLine(B,C)          # internal
l_CD   = g.addLine(C,D)          # wall (inner, lower leg)
l_DA   = g.addLine(D,A)          # INLET
arc_o1 = g.addCircleArc(B,O,Mo)  # wall
arc_i1 = g.addCircleArc(Mi,O,C)  # wall
l_MoMi = g.addLine(Mo,Mi)        # internal
arc_o2 = g.addCircleArc(Mo,O,E)  # wall
arc_i2 = g.addCircleArc(F,O,Mi)  # wall
l_EF   = g.addLine(E,F)          # internal
l_EG   = g.addLine(E,G_)         # wall
l_GH   = g.addLine(G_,H)         # OUTLET
l_HF   = g.addLine(H,F)          # wall

S1 = g.addPlaneSurface([g.addCurveLoop([l_bot,l_BC,l_CD,l_DA])])
S2 = g.addPlaneSurface([g.addCurveLoop([arc_o1,l_MoMi,arc_i1,-l_BC])])
S3 = g.addPlaneSurface([g.addCurveLoop([arc_o2,l_EF,arc_i2,-l_MoMi])])
S4 = g.addPlaneSurface([g.addCurveLoop([l_EG,l_GH,l_HF,-l_EF])])

for c,n in [(l_bot,NL),(l_CD,NL),(l_EG,NL),(l_HF,NL),
            (l_BC,NW),(l_MoMi,NW),(l_EF,NW),(l_DA,NW),(l_GH,NW),
            (arc_o1,NA),(arc_i1,NA),(arc_o2,NA),(arc_i2,NA)]:
    g.mesh.setTransfiniteCurve(c,n)
for s in (S1,S2,S3,S4):
    g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2,s)
g.synchronize()
print("surfaces:",[S1,S2,S3,S4])

# -- cell 3 -------------------------------------------------------------------------
side = {}   # curve tag -> extruded side surface tag
tops, vols = [], []
for s in (S1,S2,S3,S4):
    bnd = [abs(c) for (d,c) in gmsh.model.getBoundary([(2,s)], oriented=True, combined=False)]
    out = g.extrude([(2,s)], 0,0,TH, numElements=[1], recombine=True)
    tops.append(out[0][1]); vols.append(out[1][1])
    for c, ent in zip(bnd, out[2:]):
        side[c] = ent[1]
g.synchronize()

walls   = [side[c] for c in (l_bot,l_CD,l_EG,l_HF,arc_o1,arc_i1,arc_o2,arc_i2)]
inlet   = [side[l_DA]]
outlet  = [side[l_GH]]
fb      = [S1,S2,S3,S4] + tops

for name, tags in [("inlet",inlet),("outlet",outlet),("walls",walls),("frontAndBack",fb)]:
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), name)
gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, vols), "internal")

gmsh.model.mesh.generate(3)
gmsh.write("ubend.msh")
n = gmsh.model.mesh.getNodes()[0].size
print("nodes:", n, "hexes:", len(gmsh.model.mesh.getElements(3)[1][0]))

# -- cell 4 -------------------------------------------------------------------------
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)
hdr = lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1e-3; writeControl timeStep; writeInterval 100; writeFormat ascii; writePrecision 6;\n"
 "runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+
 "ddtSchemes{default Euler;} gradSchemes{default Gauss linear;} divSchemes{default none; div(phi,U) Gauss linear;}\n"
 "laplacianSchemes{default Gauss linear orthogonal;} interpolationSchemes{default linear;} snGradSchemes{default orthogonal;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+
 "solvers{p{solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0;} U{solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;}}\nPISO{nCorrectors 2; nNonOrthogonalCorrectors 0;}\n")
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-700:], r.stderr[-500:])

# -- cell 5 -------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 6 -------------------------------------------------------------------------
import re
b = open("constant/polyMesh/boundary").read()
def settype(txt, name, t):
    return re.sub(r"(\b%s\s*\{\s*type\s+)\w+;"%name, r"\g<1>%s;"%t, txt)
b = settype(b,"frontAndBack","empty"); b = settype(b,"walls","wall")
b = re.sub(r"physicalType\s+\w+;\n\s*","",b)
open("constant/polyMesh/boundary","w").write(b)
print(re.findall(r"\n    (\w+)\n    \{\n        type\s+(\w+);", b))
r = subprocess.run(["checkMesh"], capture_output=True, text=True); print(r.stdout[-1800:])

# -- cell 7 -------------------------------------------------------------------------
# Coarse mesh is clean. Now refine to ~1 mm cells (10 across the passage) and re-check, then verify th
gmsh.model.mesh.clear()
NW2, NL2, NA2 = 11, 121, 21     # 10 cells across the 10 mm passage -> 1 mm cells
for c,n in [(l_bot,NL2),(l_CD,NL2),(l_EG,NL2),(l_HF,NL2),
            (l_BC,NW2),(l_MoMi,NW2),(l_EF,NW2),(l_DA,NW2),(l_GH,NW2),
            (arc_o1,NA2),(arc_i1,NA2),(arc_o2,NA2),(arc_i2,NA2)]:
    g.mesh.setTransfiniteCurve(c,n)
g.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("ubend.msh")
print("hexes:", len(gmsh.model.mesh.getElements(3)[1][0]))
gmsh.finalize()

# -- cell 8 -------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True); print(r.returncode)
b = open("constant/polyMesh/boundary").read()
b = settype(b,"frontAndBack","empty"); b = settype(b,"walls","wall")
b = re.sub(r"physicalType\s+\w+;\n\s*","",b)
open("constant/polyMesh/boundary","w").write(b)
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in
  ("cells","faces","Patch","frontAndBack","walls","inlet","outlet","non-orth","skew","aspect","Total volume","bounding box","Mesh OK","***","solution (non-empty)"))))

# -- cell 9 -------------------------------------------------------------------------
r = subprocess.run(["foamToVTK","-latestTime","-no-boundary"], capture_output=True, text=True); print(r.returncode)
import pyvista as pv, glob
pv.start_xvfb() if False else None
f = sorted(glob.glob("VTK/*.vtm")) + sorted(glob.glob("VTK/*.vtk"))
print(f)

# -- cell 10 ------------------------------------------------------------------------
m = pv.read("VTK/t1_0.vtm").combine()
p = np.asarray(m.points)
xy = p[np.isclose(p[:,2],0.0)]
inl = xy[np.isclose(xy[:,0],0.0) & (xy[:,1]<0)]
out = xy[np.isclose(xy[:,0],0.0) & (xy[:,1]>0)]
print("inlet  y %.4f..%.4f  width %.4f m (asked 0.0100)" % (inl[:,1].min(), inl[:,1].max(), inl[:,1].max()-inl[:,1].min()))
print("outlet y %.4f..%.4f  width %.4f m" % (out[:,1].min(), out[:,1].max(), out[:,1].max()-out[:,1].min()))
bend = xy[xy[:,0] > XB+1e-9]
rb = np.hypot(bend[:,0]-XB, bend[:,1])
print("bend inner R %.4f outer R %.4f centreline R %.4f (asked 0.0150)" % (rb.min(), rb.max(), 0.5*(rb.min()+rb.max())))
print("bend passage width %.4f m; leg straight length %.4f m (asked 0.1200)" % (rb.max()-rb.min(), XB))
print("cells %d, volume %.4e vs exact %.4e" % (m.n_cells, m.volume, face.area*TH))
pl = pv.Plotter(off_screen=True, window_size=(900,400))
pl.add_mesh(m, show_edges=True, color="lightsteelblue"); pl.view_xy(); pl.screenshot("mesh.png")

# -- cell 11 ------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
surf = m.extract_surface().faces.reshape(-1,5)[:,1:]
pts = np.asarray(m.extract_surface().points)
polys = [pts[f][:,:2] for f in surf if np.allclose(pts[f][:,2],0.0)]
fig,ax = plt.subplots(figsize=(9,4))
ax.add_collection(PolyCollection(polys, facecolors="lightsteelblue", edgecolors="k", linewidths=0.3))
ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("U-bend, %d cells (z=0 plane)"%len(polys))
plt.show()

# -- cell 12 ------------------------------------------------------------------------
fig,ax = plt.subplots(figsize=(9,4))
ax.add_collection(PolyCollection(polys, facecolors="lightsteelblue", edgecolors="k", linewidths=0.3))
ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("U-bend, %d cells (z=0 plane)"%len(polys))
fig
