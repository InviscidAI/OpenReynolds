"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, math, numpy as np
from build123d import *
print(subprocess.run(["bash","-lc","pwd; ls -la"],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
# The geometry cell errored on `is_valid` (a property, not a method), so it never entered the script. 
import subprocess, math, numpy as np
from build123d import *

R    = 0.060      # internal radius (120 mm dia)
H    = 0.200      # vessel height
R_OUT= 0.020      # outlet radius (40 mm dia)
Z_P  = 0.040      # port centre height above floor
PW   = 0.020      # port width (horizontal, across the duct)
PH   = 0.030      # port height
LD   = 0.050      # duct length outside, along its own axis from the wall point
OVER = 0.018      # overlap into the vessel (absorbed by the union)
ALPHA= 15.0       # degrees off the local tangent
NP   = 4

vessel = Cylinder(radius=R, height=H).moved(Location((0,0,H/2)))

thetas = [i*2*math.pi/NP for i in range(NP)]
ducts, dirs, Ps = [], [], []
for th in thetas:
    rhat = Vector(math.cos(th), math.sin(th), 0)
    that = Vector(-math.sin(th), math.cos(th), 0)      # local tangent (CCW)
    d = that*math.cos(math.radians(ALPHA)) - rhat*math.sin(math.radians(ALPHA))  # inward flow dir
    P = rhat*R + Vector(0,0,Z_P)                       # duct axis crosses the wall here
    C = P + d*((OVER - LD)/2)
    phi = math.degrees(math.atan2(d.Y, d.X))
    ducts.append(Location(C, (0,0,phi)) * Box(LD+OVER, PW, PH))
    dirs.append(d); Ps.append(P)

fluid = vessel + ducts[0] + ducts[1] + ducts[2] + ducts[3]
print(fluid.is_valid, len(fluid.solids()), "V =", fluid.volume, fluid.bounding_box())
print("angle off tangent (deg):", math.degrees(math.acos(dirs[0].dot(Vector(0,1,0)))))
print("n faces:", len(fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
for i,f in enumerate(fluid.faces()):
    c=f.center(); print(i, f.geom_type, "area=%.5f"%f.area, "c=(%.3f,%.3f,%.3f)"%(c.X,c.Y,c.Z))

# -- cell 4 -------------------------------------------------------------------------
# The union is right; now the top disk must be imprinted with the 40 mm outlet circle so both share te
from OCP.BRepFeat import BRepFeat_SplitShape
top_face = [f for f in fluid.faces() if f.geom_type==GeomType.PLANE and abs(f.center().Z-H)<1e-9][0]
circ = Wire(Pos(0,0,H)*Circle(R_OUT).wire())
sp = BRepFeat_SplitShape(fluid.wrapped)
sp.Add(circ.wrapped, top_face.wrapped)
sp.Build()
fluid_s = Solid(sp.Shape())
print(fluid_s.is_valid, len(fluid_s.faces()), abs(fluid_s.volume-fluid.volume))
for f in fluid_s.faces():
    if abs(f.center().Z-H)<1e-9: print("  top-plane face area=%.6f"%f.area)

# -- cell 5 -------------------------------------------------------------------------
# Split is exact (outlet 1.257e-3 m² = π·0.02²). Now tessellate the whole solid once with BRepMesh so 
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

BRepMesh_IncrementalMesh(fluid_s.wrapped, 0.0004, False, 0.25, True)

def face_tris(f):
    loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    trsf = loc.Transformation()
    pts = [tri.Node(i+1).Transformed(trsf) for i in range(tri.NbNodes())]
    V = np.array([[p.X(),p.Y(),p.Z()] for p in pts])
    rev = f.wrapped.Orientation()==TopAbs_REVERSED
    T=[]
    for i in range(tri.NbTriangles()):
        a,b,c = tri.Triangle(i+1).Get()
        T.append((a-1,c-1,b-1) if rev else (a-1,b-1,c-1))
    return V, np.array(T)

patches = {}   # name -> list of (V,T)
unassigned = []
ns = [Vector(0,0,1).cross(d) for d in dirs]
for f in fluid_s.faces():
    c = f.center(); name=None
    if f.geom_type==GeomType.PLANE and abs(c.Z)<1e-9: name="floor"
    elif f.geom_type==GeomType.PLANE and abs(c.Z-H)<1e-9:
        name = "outlet" if f.area < 0.005 else "top"
    elif f.geom_type==GeomType.CYLINDER and abs(Vector(c.X,c.Y,0).length-R)<1e-6: name="cylinderWall"
    else:
        for i,(P,d,n) in enumerate(zip(Ps,dirs,ns)):
            v = c-P; s,y,z = v.dot(d), v.dot(n), c.Z-Z_P
            if abs(y)<=PW/2+1e-6 and abs(z)<=PH/2+1e-6 and -LD-1e-6<=s<=OVER+1e-6:
                name = f"inlet{i+1}" if abs(s+LD)<1e-6 else f"ductWall{i+1}"
                break
    if name is None: unassigned.append(f)
    else: patches.setdefault(name,[]).append(face_tris(f))

print("unassigned:", len(unassigned))
for k in sorted(patches): print(k, "faces=%d"%len(patches[k]), "tris=%d"%sum(len(t) for _,t in patches[k]),
                                "area=%.6f"%sum(f.area for f in fluid_s.faces() if False) or "")

# -- cell 6 -------------------------------------------------------------------------
# All 28 faces classified, none left over. Now write one ASCII STL per patch into `constant/triSurface
import os
os.makedirs("constant/triSurface", exist_ok=True)

def write_stl(name, blocks, path):
    with open(path,"w") as fh:
        fh.write(f"solid {name}\n")
        for V,T in blocks:
            for t in T:
                p0,p1,p2 = V[t[0]],V[t[1]],V[t[2]]
                n = np.cross(p1-p0,p2-p0); L=np.linalg.norm(n); n = n/L if L>0 else n
                fh.write("facet normal %e %e %e\n outer loop\n"%tuple(n))
                for p in (p0,p1,p2): fh.write("  vertex %e %e %e\n"%tuple(p))
                fh.write(" endloop\nendfacet\n")
        fh.write(f"endsolid {name}\n")

def tri_area(blocks):
    a=0.0
    for V,T in blocks:
        for t in T: a += 0.5*np.linalg.norm(np.cross(V[t[1]]-V[t[0]],V[t[2]]-V[t[0]]))
    return a

for name, blocks in patches.items():
    write_stl(name, blocks, f"constant/triSurface/{name}.stl")
for k in sorted(patches): print("%-13s area=%.6f m2"%(k, tri_area(patches[k])))
print("inlet target area = %.6f"%(PW*PH), " outlet target = %.6f"%(math.pi*R_OUT**2))

# -- cell 7 -------------------------------------------------------------------------
# Weld all patch triangles together and count free edges — this is the closure the check will measure.
allV=[]; allT=[]; off=0
for name in sorted(patches):
    for V,T in patches[name]:
        allV.append(V); allT.append(np.array(T)+off); off+=len(V)
AV=np.vstack(allV); AT=np.vstack(allT)
key=np.round(AV/1e-9).astype(np.int64)
uniq,inv=np.unique(key,axis=0,return_inverse=True)
WT=inv[AT]
from collections import Counter
ec=Counter()
for t in WT:
    for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])): ec[(min(a,b),max(a,b))]+=1
print("tris",len(WT),"nodes",len(uniq),"free edges",sum(1 for v in ec.values() if v==1),
      "non-manifold",sum(1 for v in ec.values() if v>2))
with open("constant/triSurface/swirl.stl","w") as out:
    for name in sorted(patches):
        out.write(open(f"constant/triSurface/{name}.stl").read())
print(subprocess.run(["surfaceCheck","constant/triSurface/swirl.stl"],capture_output=True,text=True).stdout[-700:])

# -- cell 8 -------------------------------------------------------------------------
# Chord error at 0.4 mm is coarse for a 60 mm wall; re-tessellate at 0.1 mm deviation and re-export, r
BRepMesh_IncrementalMesh(fluid_s.wrapped, 0.0001, False, 0.2, True)
patches = {}
for f in fluid_s.faces():
    c = f.center(); name=None
    if f.geom_type==GeomType.PLANE and abs(c.Z)<1e-9: name="floor"
    elif f.geom_type==GeomType.PLANE and abs(c.Z-H)<1e-9:
        name = "outlet" if f.area < 0.005 else "top"
    elif f.geom_type==GeomType.CYLINDER and abs(Vector(c.X,c.Y,0).length-R)<1e-6: name="cylinderWall"
    else:
        for i,(P,d,n) in enumerate(zip(Ps,dirs,ns)):
            v = c-P; s,y,z = v.dot(d), v.dot(n), c.Z-Z_P
            if abs(y)<=PW/2+1e-6 and abs(z)<=PH/2+1e-6 and -LD-1e-6<=s<=OVER+1e-6:
                name = f"inlet{i+1}" if abs(s+LD)<1e-6 else f"ductWall{i+1}"
                break
    patches.setdefault(name,[]).append(face_tris(f))
for name, blocks in patches.items(): write_stl(name, blocks, f"constant/triSurface/{name}.stl")
with open("constant/triSurface/swirl.stl","w") as out:
    for name in sorted(patches): out.write(open(f"constant/triSurface/{name}.stl").read())
r=subprocess.run(["surfaceCheck","constant/triSurface/swirl.stl"],capture_output=True,text=True).stdout
print(sum(len(t) for b in patches.values() for _,t in b),"tris;",
      [l for l in r.splitlines() if "closed" in l or "unconnected" in l])
print("cylWall area %.6f vs analytic %.6f"%(tri_area(patches["cylinderWall"]), 2*math.pi*R*H - 0))

# -- cell 9 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(900,700))
cols = {"cylinderWall":"lightgrey","floor":"tan","top":"wheat","outlet":"red",
        **{f"inlet{i}":"blue" for i in range(1,5)}, **{f"ductWall{i}":"green" for i in range(1,5)}}
for n,c in cols.items():
    pl.add_mesh(pv.read(f"constant/triSurface/{n}.stl"), color=c, opacity=0.45 if n=="cylinderWall" else 1.0, show_edges=True)
pl.camera_position=[(0.35,-0.30,0.28),(0,0,0.08),(0,0,1)]
pl.screenshot("geom.png"); print("ok")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 11 ------------------------------------------------------------------------
# Enough shaping — mesh it. cfMesh `cartesianMesh` takes the multi-solid STL directly and names patche
import os, textwrap
os.makedirs("system",exist_ok=True); os.makedirs("constant",exist_ok=True)
hdr=lambda cls,obj:f"FoamFile{{version 2.0;format ascii;class {cls};object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application simpleFoam;startFrom startTime;startTime 0;stopAt endTime;endTime 1;deltaT 1;writeControl timeStep;writeInterval 1;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+"ddtSchemes{default steadyState;}gradSchemes{default Gauss linear;}divSchemes{default none;}laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{}\n")
open("system/meshDict","w").write(hdr("dictionary","meshDict")+textwrap.dedent("""
surfaceFile "constant/triSurface/swirl.stl";
maxCellSize 0.008;
boundaryCellSize 0.008;
"""))
r=subprocess.run(["cartesianMesh"],capture_output=True,text=True)
print(r.stdout[-1500:], r.stderr[-800:])

# -- cell 12 ------------------------------------------------------------------------
print(subprocess.run(["bash","-lc","checkMesh 2>&1 | tail -40; echo ---; foamListTimes -withZero 2>/dev/null; ls constant/polyMesh"],capture_output=True,text=True).stdout)

# -- cell 13 ------------------------------------------------------------------------
# `checkMesh` passes at 8 mm, but inlets get only 7 faces each. Refine: 5 mm global, 2.5 mm on the duc
loc = "\n".join(f"  {p} {{ cellSize 0.0025; }}" for p in
      [f"inlet{i}" for i in range(1,5)]+[f"ductWall{i}" for i in range(1,5)]+["outlet"])
open("system/meshDict","w").write(hdr("dictionary","meshDict")+f"""
surfaceFile "constant/triSurface/swirl.stl";
maxCellSize 0.005;
boundaryCellSize 0.005;
localRefinement
{{
{loc}
}}
""")
r=subprocess.run(["cartesianMesh"],capture_output=True,text=True)
print(r.stdout[-300:]); print(r.stderr[-300:])

# -- cell 14 ------------------------------------------------------------------------
out=subprocess.run(["bash","-lc","checkMesh 2>&1"],capture_output=True,text=True).stdout
print("\n".join(out.splitlines()[-52:]))

# -- cell 15 ------------------------------------------------------------------------
# Mesh OK. Now verify the numbers the request named *from the mesh itself*: inlet area, inlet centroid
open("case.foam","w").write("")
m = pv.OpenFOAMReader("case.foam"); m.set_active_time_value(0)
mb = m.read(); bnd = mb["boundary"]
for nm in ["inlet1","inlet2","inlet3","inlet4","outlet"]:
    s = bnd[nm].compute_normals(); s = s.compute_cell_sizes(length=False,volume=False)
    A = s["Area"].sum(); c = np.array(s.cell_centers().points).mean(axis=0)
    n = np.average(s.cell_normals, axis=0, weights=s["Area"]); n/=np.linalg.norm(n)
    th = math.atan2(c[1],c[0]); tan=np.array([-math.sin(th),math.cos(th),0])
    ang = math.degrees(math.asin(abs(np.dot(n,tan)))) if nm.startswith("inlet") else float('nan')
    print("%-7s A=%.6f m2  c=(%.3f,%.3f,%.3f)  angle_off_tangent=%.2f deg"%(nm,A,*c,90-ang if False else ang))

# -- cell 16 ------------------------------------------------------------------------
# The angle must be measured against the tangent at the *port on the wall*, not at the inlet's own azi
for i,nm in enumerate(["inlet1","inlet2","inlet3","inlet4"]):
    s = bnd[nm].compute_cell_sizes(length=False,volume=False)
    c = np.array(s.cell_centers().points).mean(axis=0)
    P = np.array([Ps[i].X,Ps[i].Y,Ps[i].Z]); axis = P-c; axis/=np.linalg.norm(axis)
    th = math.atan2(P[1],P[0]); tan=np.array([-math.sin(th),math.cos(th),0])
    rad=np.array([math.cos(th),math.sin(th),0.])
    print("%-7s axis-to-tangent=%.2f deg  radial(inward) comp=%.3f  c_z=%.4f (target 0.0400)"
          %(nm, math.degrees(math.acos(np.clip(np.dot(axis,tan),-1,1))), -np.dot(axis,rad), c[2]))
print("duct axis length inlet->wall: %.4f m (target %.3f)"%(np.linalg.norm(P-c), LD))

# -- cell 17 ------------------------------------------------------------------------
# Inlet area and centroid height carry a ~2 % staircase bias. One refinement pass on the duct patches 
loc = "\n".join(f"  {p} {{ cellSize 0.00125; }}" for p in
      [f"inlet{i}" for i in range(1,5)]+[f"ductWall{i}" for i in range(1,5)]) + \
      "\n  outlet { cellSize 0.0025; }"
open("system/meshDict","w").write(hdr("dictionary","meshDict")+f"""
surfaceFile "constant/triSurface/swirl.stl";
maxCellSize 0.005;
boundaryCellSize 0.005;
localRefinement
{{
{loc}
}}
""")
r=subprocess.run(["cartesianMesh"],capture_output=True,text=True)
print(r.stdout[-200:], r.stderr[-200:])
print("\n".join([l for l in subprocess.run(["checkMesh"],capture_output=True,text=True).stdout.splitlines()
                 if any(k in l for k in ["cells:","Mesh OK","FAILED","skewness","non-orthogonality","Total volume","aspect"])]))

# -- cell 18 ------------------------------------------------------------------------
mb = pv.OpenFOAMReader("case.foam").read(); bnd = mb["boundary"]
for i,nm in enumerate(["inlet1","inlet2","inlet3","inlet4","outlet"]):
    s = bnd[nm].compute_cell_sizes(length=False,volume=False)
    A=s["Area"].sum(); c=np.array(s.cell_centers().points).mean(axis=0)
    print("%-7s A=%.6f (target %.6f)  c_z=%.4f"%(nm,A,PW*PH if i<4 else math.pi*R_OUT**2,c[2]))
print("mesh volume 0.00238143 vs CAD %.6f"%fluid_s.volume)

# -- cell 19 ------------------------------------------------------------------------
# Ports are within 0.5 % now. A look at the mesh: surface patches plus a clip through the port plane.
internal = mb["internalMesh"]
pl = pv.Plotter(off_screen=True, window_size=(1000,450), shape=(1,2))
pl.subplot(0,0)
for n,c in cols.items(): pl.add_mesh(bnd[n], color=c, show_edges=True, line_width=0.3)
pl.camera_position=[(0.32,-0.28,0.26),(0,0,0.09),(0,0,1)]
pl.subplot(0,1)
pl.add_mesh(internal.clip("z", origin=(0,0,Z_P), invert=False).slice(normal="z", origin=(0,0,Z_P)),
            show_edges=True, color="lightblue")
pl.camera_position="xy"
pl.screenshot("mesh.png"); display(Image("mesh.png"))

# -- cell 20 ------------------------------------------------------------------------
# Geometry and mesh both check out. Final cell: drop the combined multi-solid STL (it was only cfMesh'
os.remove("constant/triSurface/swirl.stl")
print(sorted(os.listdir("constant/triSurface")))
print(subprocess.run(["bash","-lc","checkMesh 2>&1 | tail -3"],capture_output=True,text=True).stdout)
