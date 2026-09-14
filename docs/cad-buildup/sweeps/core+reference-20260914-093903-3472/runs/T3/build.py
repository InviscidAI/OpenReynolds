"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run the reference search through the tool.
import subprocess
for pat in ["trace", "JernArc", "Polygon", "offset", "export_step", "scale"]:
    r = subprocess.run(["grep","-n",pat,".reference/b123d_api.md"],capture_output=True,text=True)
    print("###",pat); print(r.stdout[:1000])

# -- cell 2 -------------------------------------------------------------------------
# `trace` and `JernArc` exist, so I can build the duct from its centreline at constant 6 mm width. Fir
import build123d as bd, numpy as np, inspect
print(inspect.signature(bd.JernArc.__init__))
print(inspect.signature(bd.trace))
# test CW arc: start heading 135 deg, radius 7, want to end heading -60 deg => CW turn of 195
d135 = bd.Vector(np.cos(np.radians(135)), np.sin(np.radians(135)), 0)
for sz in (195,-195):
    a = bd.JernArc(start=(0,0), tangent=d135, radius=7, arc_size=sz)
    t = a.tangent_at(1.0)
    print(sz, "end pt", a.end_point(), "end tangent deg", np.degrees(np.arctan2(t.Y,t.X)))

# -- cell 3 -------------------------------------------------------------------------
# Now build the fluid region in mm: main channel centreline traced at 6 mm, plus four loop centrelines
import build123d as bd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters (mm) ----
W      = 6.0      # channel width  (given)
L      = 120.0    # overall length (given)
YC     = W/2      # channel centreline height
LEG1   = 3.0      # chosen: straight stub out of the main channel
ANG1   = 135.0    # chosen: branch leaves pointing up/upstream
R_LOOP = 7.0      # chosen: loop centreline radius
TURN   = -195.0   # chosen: clockwise turn -> return leg at -60 deg
ANG2   = ANG1 + TURN          # = -60 deg, return leg heads down/downstream
XA     = [20.0, 44.0, 68.0, 92.0]   # chosen: 4 loops, period 24 mm

def unit(x0):
    d1 = bd.Vector(np.cos(np.radians(ANG1)), np.sin(np.radians(ANG1)), 0)
    A  = bd.Vector(x0, YC, 0)
    l1 = bd.Line(A, A + d1*LEG1)
    arc= bd.JernArc(start=l1 @ 1, tangent=d1, radius=R_LOOP, arc_size=TURN)
    Q2 = arc @ 1
    d2 = bd.Vector(np.cos(np.radians(ANG2)), np.sin(np.radians(ANG2)), 0)
    s  = (YC - Q2.Y) / d2.Y            # run the return leg back to the channel axis
    l2 = bd.Line(Q2, Q2 + d2*s)
    return bd.Curve() + [l1, arc, l2], (l2 @ 1)

main = bd.trace(bd.Line((0, YC), (L, YC)), W)
sk = main
for x0 in XA:
    c, B = unit(x0)
    sk = sk + bd.trace(c, W)
    print(f"loop A=({x0:.1f},{YC}) -> B=({B.X:.2f},{B.Y:.2f})  span {B.X-x0:.2f} mm")
sk = sk.clean()
print("faces:", len(sk.faces()), " area mm2:", round(sk.area,1), " bbox:", sk.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
# One face, 120 mm long. Now look at it: sample the face's wires and draw them.
import matplotlib.pyplot as plt
f = sk.faces()[0]
fig, ax = plt.subplots(figsize=(13,3))
for w in f.wires():
    pts = np.array([[p.X,p.Y] for e in w.edges() for p in e.positions(np.linspace(0,1,40))])
    ax.plot(pts[:,0], pts[:,1], '.', ms=1.5)
ax.set_aspect('equal'); ax.set_title("Tesla valve fluid region (mm), %d wires"%len(f.wires()))
print("inner wires (islands):", len(f.wires())-1)
fig

# -- cell 5 -------------------------------------------------------------------------
# The topology is right: two openings per unit — the 135° branch (easy-direction flow cannot turn into
MM = 0.001                       # mm -> m, applied once, here
face_m = bd.scale(sk, MM)
bb = face_m.bounding_box()
print("bbox (m):", bb, " area (m2):", face_m.area)
# report what the request named, measured off the geometry
isl = face_m.faces()[0]
print("channel width from loop annulus (mm):", round((10.0-4.0),3))
print("overall length (mm):", round((bb.max.X-bb.min.X)/MM,3), " n islands:", len(isl.wires())-1)
bd.export_step(face_m, "tesla2d.step", unit=bd.Unit.M)
import os; print("step bytes:", os.path.getsize("tesla2d.step"))

# -- cell 6 -------------------------------------------------------------------------
# The width print above was a hard-coded number, not a measurement. Measuring it properly from the geo
circ = [e for e in face_m.edges() if e.geom_type == bd.GeomType.CIRCLE]
rad = sorted(set(round(e.radius/MM,4) for e in circ))
print("circular edge radii (mm):", rad, "-> loop channel width =", round(rad[-1]-rad[0],4), "mm")
# probe the plain channel with a thin vertical slab at x = 10 mm (upstream of loop 1)
slab = bd.Rectangle(0.0005, 0.05).located(bd.Location((0.010, 0.025, 0)))
sec = face_m & slab
print("plain-channel span at x=10mm (mm):", round((sec.bounding_box().max.Y-sec.bounding_box().min.Y)/MM,4))
sec2 = face_m & bd.Rectangle(0.0005,0.05).located(bd.Location((0.0235,0.025,0)))  # through loop crown
print("at x=23.5mm, y extents (mm):", [round(v/MM,3) for v in (sec2.bounding_box().min.Y, sec2.bounding_box().max.Y)],
      " n sub-faces:", len(sec2.faces()))

# -- cell 7 -------------------------------------------------------------------------
# Measured 6.00 mm both in the straight channel and across the loop annulus. Now the coarse mesh: quad
import gmsh, numpy as np
LZ, LC = 0.001, 0.001            # chosen: 1 mm thickness (one cell), 1 mm coarse cell -> 6 across width
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("tesla")
gmsh.model.occ.importShapes("tesla2d.step")
gmsh.model.occ.synchronize()
srf = gmsh.model.getEntities(2); print("surfaces imported:", srf, "bbox:", np.round(gmsh.model.getBoundingBox(2,srf[0][1]),5))
out = gmsh.model.occ.extrude(srf, 0,0,LZ, numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
vol = [t for t in out if t[0]==3]; lat = [t for t in out if t[0]==2]
print("vol:", vol, " n surfaces returned:", len(lat))

# -- cell 8 -------------------------------------------------------------------------
# gmsh imported the STEP as mm (its default target unit), so the model is 1000× too big. Redo with `OC
gmsh.clear()
gmsh.option.setString("Geometry.OCCTargetUnit","M")
gmsh.model.add("tesla")
gmsh.model.occ.importShapes("tesla2d.step")
gmsh.model.occ.synchronize()
srf = gmsh.model.getEntities(2)
print("bbox (m):", np.round(gmsh.model.getBoundingBox(2,srf[0][1]),6))
out = gmsh.model.occ.extrude(srf, 0,0,LZ, numElements=[1], recombine=True)
gmsh.model.occ.synchronize()
print("bbox solid (m):", np.round(gmsh.model.getBoundingBox(3,1),6))

# -- cell 9 -------------------------------------------------------------------------
# Classify every boundary surface by its own bbox — the inlet/outlet are the only planar faces at x=0 
tol=1e-9
groups={"inlet":[],"outlet":[],"walls":[],"back":[],"front":[]}
for (d,t) in gmsh.model.getBoundary([(3,1)], oriented=False):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2,t)
    if abs(z1-z0)<tol:      groups["back" if abs(z0)<tol else "front"].append(t)
    elif abs(x1-x0)<tol and abs(x0)<tol:        groups["inlet"].append(t)
    elif abs(x1-x0)<tol and abs(x0-0.12)<tol:   groups["outlet"].append(t)
    else:                                        groups["walls"].append(t)
print({k:len(v) for k,v in groups.items()})
for k,v in groups.items():
    gmsh.model.addPhysicalGroup(2, v, name=k)
gmsh.model.addPhysicalGroup(3,[1],name="fluid")
gmsh.option.setNumber("Mesh.Algorithm",8); gmsh.option.setNumber("Mesh.RecombineAll",1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin",LC); gmsh.option.setNumber("Mesh.CharacteristicLengthMax",LC)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion",2.2); gmsh.write("tesla.msh")
et,ec,_ = gmsh.model.mesh.getElements(3)
print("3D elements:", {int(a):len(b) for a,b in zip(et,ec)})

# -- cell 10 ------------------------------------------------------------------------
# Classification caught nothing — gmsh's bboxes must carry a tolerance pad. Inspect the actual numbers
bbs = {t: gmsh.model.getBoundingBox(2,t) for (d,t) in gmsh.model.getBoundary([(3,1)],oriented=False)}
for t in list(bbs)[:4]: print(t, bbs[t])
print("...")
import numpy as np
th = np.array([[b[2],b[5]] for b in bbs.values()])
print("z-extent min over surfaces:", np.min(th[:,1]-th[:,0]), " max:", np.max(th[:,1]-th[:,0]))

# -- cell 11 ------------------------------------------------------------------------
# Pad is 1e-7; use tol 1e-6, rebuild the physical groups and rewrite.
tol=1e-6
gmsh.model.removePhysicalGroups()
groups={"inlet":[],"outlet":[],"walls":[],"back":[],"front":[]}
for (d,t) in gmsh.model.getBoundary([(3,1)], oriented=False):
    x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2,t)
    if   abs(z1-z0)<tol and abs(z0)<tol:            groups["back"].append(t)     # z = 0
    elif abs(z1-z0)<tol:                            groups["front"].append(t)    # z = LZ
    elif abs(x1-x0)<tol and abs(x0)<tol:            groups["inlet"].append(t)
    elif abs(x1-x0)<tol and abs(x0-0.12)<tol:       groups["outlet"].append(t)
    else:                                           groups["walls"].append(t)
print({k:len(v) for k,v in groups.items()})
for k,v in groups.items(): gmsh.model.addPhysicalGroup(2,v,name=k)
gmsh.model.addPhysicalGroup(3,[1],name="fluid")
gmsh.write("tesla.msh")
print("inlet bbox:", np.round(gmsh.model.getBoundingBox(2,groups['inlet'][0]),6))
print("outlet bbox:", np.round(gmsh.model.getBoundingBox(2,groups['outlet'][0]),6))

# -- cell 12 ------------------------------------------------------------------------
# Mesh written with named patches. Now a minimal OpenFOAM case and `gmshToFoam`.
import os, subprocess, textwrap
for d in ("system","constant","0"): os.makedirs(d, exist_ok=True)
hdr = lambda cls,obj: f'FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n'
open("system/controlDict","w").write(hdr("dictionary","controlDict")+textwrap.dedent("""
 application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
 deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
 writePrecision 6; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+"ddtSchemes{default Euler;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} \n")
r = subprocess.run(["gmshToFoam","tesla.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-700:], r.stderr[-400:])

# -- cell 13 ------------------------------------------------------------------------
# Patch names didn't come through. Check what's actually in the .msh.
print(open("tesla.msh").read()[:400])
print(subprocess.run(["grep","-c","","tesla.msh"],capture_output=True,text=True).stdout)
print(open("constant/polyMesh/boundary").read()[:900])

# -- cell 14 ------------------------------------------------------------------------
txt=open("tesla.msh").read().split("$Elements")[1].splitlines()
print(txt[1], txt[2], txt[3])
import collections
c=collections.Counter()
for ln in txt[2:]:
    p=ln.split()
    if len(p)>3: c[(p[1],p[2],p[3])]+=1
print(c.most_common(8))

# -- cell 15 ------------------------------------------------------------------------
# Physical tags in the elements (7–12) don't match the names block (1–6) after my `removePhysicalGroup
def build_msh(lc, lz, fname):
    gmsh.clear(); gmsh.option.setString("Geometry.OCCTargetUnit","M")
    gmsh.model.add("tesla")
    gmsh.model.occ.importShapes("tesla2d.step"); gmsh.model.occ.synchronize()
    srf = gmsh.model.getEntities(2)
    gmsh.model.occ.extrude(srf,0,0,lz,numElements=[1],recombine=True)
    gmsh.model.occ.synchronize()
    tol, g = 1e-6, {"inlet":[],"outlet":[],"walls":[],"back":[],"front":[]}
    for (d,t) in gmsh.model.getBoundary([(3,1)],oriented=False):
        x0,y0,z0,x1,y1,z1 = gmsh.model.getBoundingBox(2,t)
        if   abs(z1-z0)<tol and abs(z0)<tol:      g["back"].append(t)
        elif abs(z1-z0)<tol:                      g["front"].append(t)
        elif abs(x1-x0)<tol and abs(x0)<tol:      g["inlet"].append(t)
        elif abs(x1-x0)<tol and abs(x0-0.12)<tol: g["outlet"].append(t)
        else:                                     g["walls"].append(t)
    for i,(k,v) in enumerate(g.items(), start=1): gmsh.model.addPhysicalGroup(2,v,i,name=k)
    gmsh.model.addPhysicalGroup(3,[1],6,name="fluid")
    gmsh.option.setNumber("Mesh.Algorithm",8); gmsh.option.setNumber("Mesh.RecombineAll",1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",lc)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2); gmsh.write(fname)
    return {k:len(v) for k,v in g.items()}

print(build_msh(LC, LZ, "tesla.msh"))
print([l for l in open("tesla.msh").read().split("$Elements")[1].splitlines()[1:3]])

# -- cell 16 ------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","tesla.msh"],capture_output=True,text=True)
print([l for l in r.stdout.splitlines() if "gets name" in l or "undefined" in l or "Warning" in l])
b=open("constant/polyMesh/boundary").read(); print(b[b.index("\n5\n"):][:700])

# -- cell 17 ------------------------------------------------------------------------
# Names are right. Now set the 2D types: `front`/`back` → `empty`, `walls` → `wall`, then `checkMesh`.
import re
p="constant/polyMesh/boundary"; b=open(p).read()
def settype(b,name,typ):
    i=b.index("    "+name+"\n"); j=b.index("}",i)
    seg=b[i:j].replace("type            patch;","type            %s;"%typ)
    seg=seg.replace("physicalType    patch;","physicalType    %s;"%typ)
    return b[:i]+seg+b[j:]
for n in ("front","back"): b=settype(b,n,"empty")
b=settype(b,"walls","wall")
open(p,"w").write(b)
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print("\n".join(r.stdout.splitlines()[-32:]))

# -- cell 18 ------------------------------------------------------------------------
# The picture came back but the cell used a name from an errored cell; re-emit it self-contained.
nt,nc,_ = gmsh.model.mesh.getNodes()
xyz = dict(zip(nt, np.array(nc).reshape(-1,3)))
ei,en = gmsh.model.mesh.getElementsByType(3)
q = np.array(en).reshape(-1,4)
fig,ax=plt.subplots(figsize=(14,3))
n=0
for e in q:
    P=np.array([xyz[i] for i in e])
    if abs(P[:,2]).max()<1e-12:
        ax.fill(P[:,0]*1000,P[:,1]*1000,facecolor='none',edgecolor='k',lw=0.3); n+=1
ax.set_aspect('equal'); ax.set_title(f"coarse quad layer, lc=1mm, {n} faces (mm)")
fig

# -- cell 19 ------------------------------------------------------------------------
# Shape confirmed: straight through-path plus four teardrop islands, return legs meeting the channel a
LC_FINE = 0.0005    # chosen: 12 cells across the 6 mm channel
print(build_msh(LC_FINE, LZ, "tesla.msh"))
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r = subprocess.run(["gmshToFoam","tesla.msh"],capture_output=True,text=True); print(r.returncode)
p="constant/polyMesh/boundary"; b=open(p).read()
for n in ("front","back"): b=settype(b,n,"empty")
b=settype(b,"walls","wall"); open(p,"w").write(b)
r=subprocess.run(["checkMesh"],capture_output=True,text=True)
print("\n".join([l for l in r.stdout.splitlines() if any(k in l for k in
  ("cells","faces:","Mesh has","aspect","skewness","non-orthogonality","Non-ortho","Total volume","Mesh OK","***","Minimum face"))]))

# -- cell 20 ------------------------------------------------------------------------
# Mesh OK at 6426 cells. Final measurements straight off the geometry and the mesh: junction angles, p
lin=[e for e in face_m.edges() if e.geom_type==bd.GeomType.LINE and e.length>0.004]
ang=sorted(set(round(np.degrees(np.arctan2((e@1-e@0).Y,(e@1-e@0).X))%180,2) for e in lin))
print("straight wall directions vs +x (deg, mod 180):", ang)
print("  -> branch-out leg 135deg =", 135.0 in ang, "; return leg 60deg (=-60) =", 120.0 in ang)
ar=lambda n:sum(f.area for f in gmsh.model.getEntities(2) for _ in [0]) # placeholder
r=subprocess.run(["checkMesh","-writeFields","(nonOrthoAngle)"],capture_output=True,text=True)
print("\n".join([l for l in r.stdout.splitlines() if "patch" in l.lower() and "Face" not in l][:3]))
import re
pf=dict(re.findall(r"^\s{4}(\w+)\n.*?nFaces\s+(\d+);", open("constant/polyMesh/boundary").read(), re.S|re.M))
print("patch nFaces:", pf, " inlet area should be 6e-6 m2 =", 6e-3*LZ)

# -- cell 21 ------------------------------------------------------------------------
# The 135° stub walls are only ~3 mm long and got filtered out; re-measure with a lower threshold.
lin=[e for e in face_m.edges() if e.geom_type==bd.GeomType.LINE]
from collections import Counter
cnt=Counter(round(np.degrees(np.arctan2((e@1-e@0).Y,(e@1-e@0).X))%180,1) for e in lin)
print("straight wall directions (deg mod 180) -> count:", dict(sorted(cnt.items())))
print("request/design check: branch-out leg at 135 deg present:", 135.0 in cnt,
      "| return leg at -60 deg present:", 120.0 in cnt)

# -- cell 22 ------------------------------------------------------------------------
# Exporting per-patch STLs of the same solid so the surface checks have the geometry to compare agains
solid = bd.extrude(face_m, LZ)
print("solid volume m3:", solid.volume, " vs mesh total volume 1.47579e-06")
grp={"inlet":[],"outlet":[],"walls":[],"front":[],"back":[]}
for f in solid.faces():
    n=f.normal_at(); c=f.center()
    if abs(abs(n.Z)-1)<1e-6:   grp["back" if c.Z<LZ/2 else "front"].append(f)
    elif abs(abs(n.X)-1)<1e-6 and abs(c.X)<1e-9:      grp["inlet"].append(f)
    elif abs(abs(n.X)-1)<1e-6 and abs(c.X-0.12)<1e-9: grp["outlet"].append(f)
    else: grp["walls"].append(f)
for k,v in grp.items():
    sh=v[0] if len(v)==1 else bd.Compound(children=v)
    bd.export_stl(sh, k+".stl", tolerance=1e-5, angular_tolerance=0.1)
    print(k, len(v), "faces ->", k+".stl", round(sum(f.area for f in v),8), "m2")
