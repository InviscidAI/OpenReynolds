"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, math
from build123d import *

mm = 1e-3
W   = 10*mm      # passage width
L   = 120*mm     # leg length
Rc  = 15*mm      # centreline bend radius
Ri, Ro = Rc-W/2, Rc+W/2
TH  = 1*mm       # 2D extrusion thickness

p_outer = Polyline((0,-W/2),(L,-W/2))
arc_o   = CenterArc((L,Rc), Ro, -90, 180)
p_top   = Polyline((L, Rc+Ro),(0, Rc+Ro))
p_in    = Polyline((0, Rc+Ri),(L, Rc+Ri))
arc_i   = CenterArc((L,Rc), Ri, 90, -180)
p_bot   = Polyline((L, W/2),(0, W/2))
outlet  = Polyline((0, Rc+Ro),(0, Rc+Ri))
inlet   = Polyline((0, W/2),(0,-W/2))
wire = Wire([*p_outer.edges(),*arc_o.edges(),*p_top.edges(),outlet.edges()[0],
             *p_in.edges(),*arc_i.edges(),*p_bot.edges(),inlet.edges()[0]])
face = Face(wire)
bb = face.bounding_box()
print("closed:", wire.is_closed, "area mm^2:", face.area/mm**2,
      " expected:", 2*120*10 + math.pi*15*10)
print("bbox mm:", [round(v/mm,3) for v in (bb.min.X,bb.max.X,bb.min.Y,bb.max.Y)])
print("inlet len mm:", inlet.edges()[0].length/mm, " outlet len mm:", outlet.edges()[0].length/mm)
print("arc radii mm:", arc_i.edges()[0].radius/mm, arc_o.edges()[0].radius/mm)

# -- cell 2 -------------------------------------------------------------------------
import gmsh, numpy as np
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize()
gmsh.model.add("ubend")
g = gmsh.model.geo
Nw, Nl, Na = 4, 48, 12          # coarse: across width, along leg, per 90deg arc
XC, YC = L, Rc                   # bend centre

P = lambda x,y: g.addPoint(x,y,0)
a0,a1,a2,a3 = P(0,-W/2), P(L,-W/2), P(L,W/2), P(0,W/2)          # lower leg
c0,c1,c2,c3 = P(L,Rc+Ri), P(0,Rc+Ri), P(0,Rc+Ro), P(L,Rc+Ro)    # upper leg
bi,bo = P(XC+Ri,YC), P(XC+Ro,YC)                                 # bend mid radial
ctr = P(XC,YC)

wall_bot  = g.addLine(a0,a1)
iface_in  = g.addLine(a1,a2)
wall_lowi = g.addLine(a2,a3)
inlet_l   = g.addLine(a3,a0)
arc_o1 = g.addCircleArc(a1,ctr,bo); arc_i1 = g.addCircleArc(bi,ctr,a2)
rad    = g.addLine(bo,bi)
arc_o2 = g.addCircleArc(bo,ctr,c3); arc_i2 = g.addCircleArc(c0,ctr,bi)
iface_out = g.addLine(c3,c0)
wall_upi  = g.addLine(c0,c1)
outlet_l  = g.addLine(c1,c2)
wall_top  = g.addLine(c2,c3)

sA = g.addPlaneSurface([g.addCurveLoop([wall_bot,iface_in,wall_lowi,inlet_l])])
sB1= g.addPlaneSurface([g.addCurveLoop([arc_o1,rad,arc_i1,-iface_in])])
sB2= g.addPlaneSurface([g.addCurveLoop([arc_o2,iface_out,arc_i2,-rad])])
sC = g.addPlaneSurface([g.addCurveLoop([wall_upi,outlet_l,wall_top,iface_out])])

for c in (wall_bot,wall_lowi,wall_upi,wall_top): g.mesh.setTransfiniteCurve(c,Nl+1)
for c in (iface_in,inlet_l,rad,iface_out,outlet_l): g.mesh.setTransfiniteCurve(c,Nw+1)
for c in (arc_o1,arc_i1,arc_o2,arc_i2): g.mesh.setTransfiniteCurve(c,Na+1)
for s in (sA,sB1,sB2,sC):
    g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2,s)
g.synchronize()
print("surfaces", sA,sB1,sB2,sC)

# -- cell 3 -------------------------------------------------------------------------
loops = {sA:[wall_bot,iface_in,wall_lowi,inlet_l],
         sB1:[arc_o1,rad,arc_i1,-iface_in],
         sB2:[arc_o2,iface_out,arc_i2,-rad],
         sC :[wall_upi,outlet_l,wall_top,iface_out]}
side = {}   # curve tag (abs) -> list of side surfaces
tops, bots, vols = [], [], []
for s, curves in loops.items():
    out = g.extrude([(2,s)],0,0,TH,numElements=[1],recombine=True)
    tops.append(out[0][1]); vols.append(out[1][1]); bots.append(s)
    for c, ent in zip(curves, out[2:]):
        side.setdefault(abs(c),[]).append(ent[1])
g.synchronize()

def grp(dim, tags, name):
    t = gmsh.model.addPhysicalGroup(dim, tags); gmsh.model.setPhysicalName(dim, t, name)
walls = [x for c in (wall_bot,wall_lowi,wall_upi,wall_top,arc_o1,arc_i1,arc_o2,arc_i2) for x in side[c]]
grp(2, walls, "walls")
grp(2, side[inlet_l],  "inlet")
grp(2, side[outlet_l], "outlet")
grp(2, tops+bots, "frontAndBack")
grp(3, vols, "internal")
gmsh.model.mesh.generate(3)
print("nodes", len(gmsh.model.mesh.getNodes()[0]))
gmsh.write("ubend.msh")
print(sorted(os.listdir(".")))

# -- cell 4 -------------------------------------------------------------------------
import subprocess, textwrap, os
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("ubend.msh")
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
head = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(head%"controlDict"+textwrap.dedent("""
 application     icoFoam; startFrom startTime; startTime 0; stopAt endTime;
 endTime 1; deltaT 1e-3; writeControl timeStep; writeInterval 100;
 purgeWrite 0; writeFormat ascii; writePrecision 6; writeCompression off;
 timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(head%"fvSchemes"+textwrap.dedent("""
 ddtSchemes{default Euler;} gradSchemes{default Gauss linear;}
 divSchemes{default none; div(phi,U) Gauss linear;}
 laplacianSchemes{default Gauss linear orthogonal;}
 interpolationSchemes{default linear;} snGradSchemes{default orthogonal;}
"""))
open("system/fvSolution","w").write(head%"fvSolution"+textwrap.dedent("""
 solvers{p{solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0;}
         U{solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;}}
 PISO{nCorrectors 2; nNonOrthogonalCorrectors 0;}
"""))
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 5 -------------------------------------------------------------------------
import re
bf = "constant/polyMesh/boundary"
txt = open(bf).read()
for name, typ in [("frontAndBack","empty"),("walls","wall"),("inlet","patch"),("outlet","patch")]:
    txt = re.sub(r"(\b%s\s*\{\s*type\s+)\w+;"%name, r"\g<1>%s;"%typ, txt)
    if typ=="wall": txt = re.sub(r"(\bwalls\s*\{\s*type\s+wall;)", r"\1\n        inGroups        1(wall);", txt)
open(bf,"w").write(txt)
print(txt[txt.index("("):])

# -- cell 6 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2200:])

# -- cell 7 -------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
nt, nc, _ = gmsh.model.mesh.getNodes()
xyz = {int(t): nc[3*i:3*i+3] for i,t in enumerate(nt)}
fig, ax = plt.subplots(figsize=(9,4))
for s in (sA,sB1,sB2,sC):
    et, ets, enodes = gmsh.model.mesh.getElements(2, s)
    for typ, conn in zip(et, enodes):
        if typ==3:
            q = np.array(conn).reshape(-1,4)
            for e in q:
                p = np.array([xyz[n][:2] for n in e]+[xyz[e[0]][:2]])*1000
                ax.plot(p[:,0], p[:,1], 'k-', lw=0.4)
ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title("U-bend section, coarse (480 cells)")
plt.tight_layout(); plt.show()

# -- cell 8 -------------------------------------------------------------------------
fig

# -- cell 9 -------------------------------------------------------------------------
# Shape is right. Now the production resolution: 0.5 mm cells (20 across the 10 mm passage), rebuilt f
def build(Nw, Nl, Na, fn="ubend.msh"):
    if gmsh.isInitialized(): gmsh.finalize()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("ubend"); G = gmsh.model.geo
    XC, YC = L, Rc
    P = lambda x,y: G.addPoint(x,y,0)
    a0,a1,a2,a3 = P(0,-W/2), P(L,-W/2), P(L,W/2), P(0,W/2)
    c0,c1,c2,c3 = P(L,Rc+Ri), P(0,Rc+Ri), P(0,Rc+Ro), P(L,Rc+Ro)
    bi,bo,ctr = P(XC+Ri,YC), P(XC+Ro,YC), P(XC,YC)
    wb=G.addLine(a0,a1); ii=G.addLine(a1,a2); wli=G.addLine(a2,a3); inl=G.addLine(a3,a0)
    ao1=G.addCircleArc(a1,ctr,bo); ai1=G.addCircleArc(bi,ctr,a2); rd=G.addLine(bo,bi)
    ao2=G.addCircleArc(bo,ctr,c3); ai2=G.addCircleArc(c0,ctr,bi); io=G.addLine(c3,c0)
    wui=G.addLine(c0,c1); out=G.addLine(c1,c2); wt=G.addLine(c2,c3)
    lps = {}
    for cs in ([wb,ii,wli,inl],[ao1,rd,ai1,-ii],[ao2,io,ai2,-rd],[wui,out,wt,io]):
        lps[G.addPlaneSurface([G.addCurveLoop(cs)])] = cs
    for c in (wb,wli,wui,wt): G.mesh.setTransfiniteCurve(c,Nl+1)
    for c in (ii,inl,rd,io,out): G.mesh.setTransfiniteCurve(c,Nw+1)
    for c in (ao1,ai1,ao2,ai2): G.mesh.setTransfiniteCurve(c,Na+1)
    for s in lps: G.mesh.setTransfiniteSurface(s); G.mesh.setRecombine(2,s)
    G.synchronize()
    sides, tops, vols, bots = {}, [], [], list(lps)
    for s, cs in lps.items():
        o = G.extrude([(2,s)],0,0,TH,numElements=[1],recombine=True)
        tops.append(o[0][1]); vols.append(o[1][1])
        for c,e in zip(cs,o[2:]): sides.setdefault(abs(c),[]).append(e[1])
    G.synchronize()
    def grp(d,t,n):
        gmsh.model.setPhysicalName(d, gmsh.model.addPhysicalGroup(d,t), n)
    grp(2,[x for c in (wb,wli,wui,wt,ao1,ai1,ao2,ai2) for x in sides[c]],"walls")
    grp(2,sides[inl],"inlet"); grp(2,sides[out],"outlet")
    grp(2,tops+bots,"frontAndBack"); grp(3,vols,"internal")
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2); gmsh.write(fn)
    return sum(len(gmsh.model.mesh.getElements(3,v)[1][0]) for v in vols)

ncells = build(20, 240, 48)
print("cells:", ncells)

# -- cell 10 ------------------------------------------------------------------------
r = subprocess.run(["gmshToFoam","ubend.msh"], capture_output=True, text=True); print(r.returncode)
txt = open(bf).read()
for name, typ in [("frontAndBack","empty"),("walls","wall"),("inlet","patch"),("outlet","patch")]:
    txt = re.sub(r"(\b%s\s*\{\s*type\s+)\w+;"%name, r"\g<1>%s;"%typ, txt)
txt = re.sub(r"(\bwalls\s*\{\s*type\s+wall;)", r"\1\n        inGroups        1(wall);", txt)
open(bf,"w").write(txt)
c = subprocess.run(["checkMesh"], capture_output=True, text=True).stdout
print("\n".join(l for l in c.splitlines() if any(k in l for k in
      ("cells:","hexahedra","nFaces","Mesh OK","non-orthogonality","skewness","aspect","Total volume","non-empty) directions"))))
print(txt[txt.index("("):txt.index("//")])

# -- cell 11 ------------------------------------------------------------------------
print(open(bf).read().split("boundaryField")[0][-900:])

# -- cell 12 ------------------------------------------------------------------------
for dim,tag in gmsh.model.getPhysicalGroups(2):
    name = gmsh.model.getPhysicalName(dim,tag)
    nd, co = gmsh.model.mesh.getNodesForPhysicalGroup(dim,tag)[:2]
    c = np.array(co).reshape(-1,3)*1000
    print(f"{name:13s} n={len(nd):5d} x[{c[:,0].min():7.2f},{c[:,0].max():7.2f}] "
          f"y[{c[:,1].min():7.2f},{c[:,1].max():7.2f}] z[{c[:,2].min():.2f},{c[:,2].max():.2f}]  (mm)")

# -- cell 13 ------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9,4))
nt2, nc2, _ = gmsh.model.mesh.getNodes()
xyz2 = {int(t): nc2[3*i:3*i+3] for i,t in enumerate(nt2)}
for s in [t for d,t in gmsh.model.getEntities(2) if t<=4]:
    et, _, en = gmsh.model.mesh.getElements(2, s)
    for typ, conn in zip(et, en):
        if typ==3:
            for e in np.array(conn).reshape(-1,4)[::3]:
                p = np.array([xyz2[n][:2] for n in e]+[xyz2[e[0]][:2]])*1000
                ax2.plot(p[:,0],p[:,1],'k-',lw=0.25)
ax2.set_aspect("equal"); ax2.set_title("final 2D mesh, 11520 cells (every 3rd face drawn)")
ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]"); plt.tight_layout(); fig2
