"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll plan: cavity 0.16×0.14×0.12 m block; pinion (z=20) axis along one horizontal axis, wheel (z=30)
# Start with basics and check the loft function availability.
import build123d as bd, numpy as np, inspect
print(bd.__version__)
print(inspect.signature(bd.loft))
print(hasattr(bd,'Cone'))

# -- cell 2 -------------------------------------------------------------------------
# Re-emitting the loft test self-contained, with `wire_at` defined inside:
import build123d as bd, math
def wire_at(s):
    pts=[(0,0,s),(2,0,s),(2,2,s),(0,2,s)]
    return bd.Wire.make_polygon(pts,close=True)
t=bd.loft([bd.Face(wire_at(0)),bd.Face(wire_at(3))],ruled=True)
print(type(t), t.volume)

# -- cell 3 -------------------------------------------------------------------------
# Loft works with Faces. Now gear builder and both gears:
import math
def cyl_pt(r, th, z):
    return (r*math.cos(th), r*math.sin(th), z)

def bevel_gear(z, m, delta, Rin, Rout, backlash_mm=0.0, add_f=1.0, ded_f=1.25):
    r_in = Rin*math.sin(delta); r_out = Rout*math.sin(delta)
    blank = bd.Pos(0,0,Rin)*bd.Cone(r_out, r_in, Rout-Rin)
    eta = math.pi/(2*z)/2 + (backlash_mm/2)/(Rout*math.sin(delta))  # half angular slot width
    cuts=[]
    for k in range(z):
        th = 2*math.pi*k/z
        faces=[]
        for s in (Rin-2, Rout+2):
            r=s*math.sin(delta)
            r_lo=r-(ded_f+0.3)*m; r_hi=r+(add_f+0.3)*m
            pts=[cyl_pt(r_lo,th-eta,s),cyl_pt(r_lo,th+eta,s),
                 cyl_pt(r_hi,th+eta,s),cyl_pt(r_hi,th-eta,s)]
            faces.append(bd.Face(bd.Wire.make_polygon(pts,close=True)))
        cuts.append(bd.loft(faces,ruled=True))
    gear=blank
    for c in cuts:
        gear=gear-c
    return gear

m=3; z1,z2=20,30
d1=m*z1; d2=m*z2
delta1=math.atan2(z1,z2); delta2=math.pi/2-delta1
Rcone=d1/(2*math.sin(delta1))
b=12.0
Rin=Rcone-b; Rout=Rcone
pinion=bevel_gear(z1,m,delta1,Rin,Rout)
wheel =bevel_gear(z2,m,delta2,Rin,Rout)
print("d1,d2 mm",d1,d2,"| deltas",180*delta1/math.pi,180*delta2/math.pi,
      "| Rcone",Rcone,"| vols",pinion.volume,wheel.volume)
import matplotlib_inline; import matplotlib.pyplot as plt
bd.export_stl(pinion,"/tmp/pinion.stl"); bd.export_stl(wheel,"/tmp/wheel.stl")
print("exported test")

# -- cell 4 -------------------------------------------------------------------------
import pyvista as pv
p=pv.Plotter(off_screen=True)
p.add_mesh(pv.read("/tmp/pinion.stl"),color="gold")
p.add_mesh(pv.read("/tmp/wheel.stl"),color="silver")
p.camera_position='iso'
p.show()

# -- cell 5 -------------------------------------------------------------------------
# Gears look right: conically tapered slots, correct pitch cones. Now assemble: pinion axis → +X, whee
def rotZ_to_X():   # maps local +Z axis onto global +X
    return bd.Rot(0,90,0)
def rotZ_to_Y():   # local +Z onto global +Y
    return bd.Rot(-90,0,0)

pin_l = rotZ_to_X() * pinion
best=None
for ph in np.linspace(0, 2*np.pi/z2, 9):
    w = rotZ_to_Y() * bd.Rot(0,0,ph*180/np.pi) * wheel
    ov = (pin_l & w)
    v = ov.volume if hasattr(ov,'volume') else 0.0
    if best is None or v<best[0]: best=(v,ph)
print("min overlap mm^3, phase deg:",best[0],best[1]*180/np.pi)

# -- cell 6 -------------------------------------------------------------------------
test=bd.Rot(90,0,0)*bd.Pos(0,0,10)*bd.Cylinder(1,1)
print("center:", test.center().to_tuple())
test2=bd.Rot(-90,0,0)*bd.Pos(0,0,10)*bd.Cylinder(1,1)
print("center:", test2.center().to_tuple())

# -- cell 7 -------------------------------------------------------------------------
# Re-emitting the assembly definition and the classification/export as two dependent cells:
HX,HY,HZ=80,70,60
cav = bd.Box(2*HX,2*HY,2*HZ)
pin_g = rotZ_to_X() * pinion
whl_g = rotZ_to_Y() * wheel
rsh1, rsh2 = 8.0, 11.0
def shaft(axis,r,c0,c1):
    c=bd.Cylinder(r, c1-c0)
    if axis=='X': c=bd.Rot(0,90,0)*c
    if axis=='Y': c=bd.Rot(-90,0,0)*c
    return (bd.Pos(c0,0,0) if axis=='X' else bd.Pos(0,c0,0))*c
shaftP = shaft('X', rsh1, -HX, Rin)
shaftW = shaft('Y', rsh2, -HY, Rin)
oil = cav - pin_g - whl_g - shaftP - shaftW
print("solids:", len(oil.solids()), "vol m^3:", oil.volume/1e9)
print("gear overlap:", (pin_g & whl_g).volume)

# -- cell 8 -------------------------------------------------------------------------
import os, math
os.makedirs("constant/triSurface",exist_ok=True)
groups={"pinion":[], "wheel":[], "cavity_walls":[], "shaft_bore_pinion":[], "shaft_bore_wheel":[]}
tol=0.5
for f in oil.faces():
    cx,cy,cz = tuple(f.center())
    if abs(abs(cx)-HX)<tol or abs(abs(cy)-HY)<tol or abs(abs(cz)-HZ)<tol:
        groups["cavity_walls"].append(f)
    elif abs(math.hypot(cy,cz)-rsh1)<tol and cx<Rin+tol:
        groups["shaft_bore_pinion"].append(f)
    elif abs(math.hypot(cx,cz)-rsh2)<tol and cy<Rin+tol:
        groups["shaft_bore_wheel"].append(f)
    else:
        dP=pin_g.distance(f); dW=whl_g.distance(f)
        groups["pinion" if dP<=dW else "wheel"].append(f)
for k,v in groups.items(): print(k, len(v), round(sum(fi.area for fi in v)/100,2), "cm2")
for name,fs in groups.items():
    if fs:
        bd.export_stl(bd.Compound(children=fs).scale(0.001),
                      f"constant/triSurface/{name}.stl", tolerance=1e-6, angular_tolerance=0.25)
print(sorted(os.listdir('constant/triSurface')))

# -- cell 9 -------------------------------------------------------------------------
# STLs done. Now the OpenFOAM meshing set-up — coarse first.
import os, textwrap, subprocess
os.makedirs("system",exist_ok=True)
blockmesh = """
convertToMeters 1;
vertices
(
    (-0.085 -0.075 -0.065)
    ( 0.085 -0.075 -0.065)
    ( 0.085  0.075 -0.065)
    (-0.085  0.075 -0.065)
    (-0.085 -0.075  0.065)
    ( 0.085 -0.075  0.065)
    ( 0.085  0.075  0.065)
    (-0.085  0.075  0.065)
);
blocks ( hex (0 1 2 3 4 5 6 7) (26 22 20) simpleGrading (1 1 1) );
edges (); boundary ( w { type wall; faces ((0 3 7 4)(1 2 6 5)(0 1 5 4)(3 2 6 7)(4 5 6 7)(0 1 2 3)); } );
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(blockmesh)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).returncode)

# -- cell 10 ------------------------------------------------------------------------
r=subprocess.run(["blockMesh"],capture_output=True,text=True)
print(r.stderr[-800:] or r.stdout[-800:])

# -- cell 11 ------------------------------------------------------------------------
open("system/controlDict","w").write("""FoamFile { class dictionary; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat ascii; writeCompression off;
runTimeModifiable true;
""")
open("system/fvSchemes","w").write("FoamFile { class dictionary; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default none; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
open("system/fvSolution","w").write("FoamFile { class dictionary; }\nsolvers {}\nSIMPLE {}\nrelaxationFactors {}\n")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-200:])
