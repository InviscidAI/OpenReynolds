"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, math, os
os.makedirs("constant/triSurface", exist_ok=True)

MODULE = 3.0; ALPHA = math.radians(20.0); Z_P = 20; Z_W = 30
BACKLASH = 0.3
SIGMA = math.radians(90)
GAM_P = math.atan2(Z_P, Z_W); GAM_W = SIGMA - GAM_P
R_P = MODULE*Z_P/2; R_W = MODULE*Z_W/2
CONE = math.hypot(R_P, R_W)
FACEW = 18.0

def inv(a): return math.tan(a)-a

def gear_profile(z, gam, r_pitch, npf=10):
    cg = math.cos(gam); r_pv = r_pitch/cg
    r_b = r_pv*math.cos(ALPHA); r_a = r_pv+MODULE; r_f = r_pv-1.25*MODULE
    t_p = math.pi*MODULE/2 - BACKLASH/2
    psi = t_p/(2*r_pv)
    rho_lo = max(r_b, r_f)
    def d(rho):
        ar = math.acos(min(1.0, r_b/rho)); return (psi + inv(ALPHA) - inv(ar))/cg
    def R(rho): return r_pitch + (rho-r_pv)*cg
    rhos = np.linspace(rho_lo, r_a, npf)
    half = []
    if r_f < r_b-1e-9: half.append((R(r_f), d(rho_lo)))
    half += [(R(r), d(r)) for r in rhos]
    tip = [(R(r_a), t) for t in np.linspace(d(r_a), -d(r_a), 5)[1:-1]]
    tooth = half + tip + [(rr,-dd) for rr,dd in reversed(half)]
    P=[]
    for i in range(z):
        phi = 2*math.pi*i/z
        for rr,dd in tooth: P.append((rr, phi+dd))
        # root arc to next tooth start
        a0 = phi + tooth[-1][1]; a1 = phi + 2*math.pi/z + tooth[0][1]
        for aa in np.linspace(a0,a1,4)[1:-1]: P.append((R(r_f), aa))
    return [(rr*math.cos(t), rr*math.sin(t)) for rr,t in P], dict(r_pv=r_pv,r_b=r_b,r_a=r_a,r_f=r_f,psi=psi,tip=R(r_a),root=R(r_f))

pp, ip = gear_profile(Z_P, GAM_P, R_P)
pw, iw = gear_profile(Z_W, GAM_W, R_W)
print("gamma_p %.3f deg  gamma_w %.3f  A=%.3f  r_p=%.1f r_w=%.1f"%(math.degrees(GAM_P),math.degrees(GAM_W),CONE,R_P,R_W))
print("pinion",{k:round(v,3) for k,v in ip.items()}); print("wheel ",{k:round(v,3) for k,v in iw.items()})
fig,ax=plt.subplots(figsize=(6,6)); a=np.array(pp); b=np.array(pw)
ax.plot(a[:,0],a[:,1],'-',lw=.7); ax.plot(b[:,0],b[:,1],'-',lw=.7); ax.set_aspect(1); plt.savefig("prof.png",dpi=90)
print(len(pp),len(pw))

# -- cell 2 -------------------------------------------------------------------------
# The previous cell errored so its bindings aren't in the script; I re-emit the imports, loft constant
import time
from build123d import *
import build123d as bd

K_IN = (CONE-FACEW)/CONE               # inner section scale toward the common apex
ZO_P = CONE*math.cos(GAM_P); ZO_W = CONE*math.cos(GAM_W)
print("K=%.4f  z_outer pinion=%.2f wheel=%.2f"%(K_IN,ZO_P,ZO_W))

def tooth_pts(z, gam, r_pitch, npf=10):
    cg=math.cos(gam); r_pv=r_pitch/cg
    r_b=r_pv*math.cos(ALPHA); r_a=r_pv+MODULE; r_f=r_pv-1.25*MODULE
    psi=(math.pi*MODULE/2-BACKLASH/2)/(2*r_pv)     # half tooth angle, thinned by j/2
    rho_lo=max(r_b,r_f)
    d=lambda rho:(psi+inv(ALPHA)-inv(math.acos(min(1.0,r_b/rho))))/cg
    R=lambda rho:r_pitch+(rho-r_pv)*cg
    half=[]
    if r_f<r_b-1e-9: half.append((R(r_f),d(rho_lo)))
    half+=[(R(r),d(r)) for r in np.linspace(rho_lo,r_a,npf)]
    tip=[(R(r_a),t) for t in np.linspace(d(r_a),-d(r_a),5)[1:-1]]
    poly=half+tip+[(rr,-dd) for rr,dd in reversed(half)]
    rc=0.90*R(r_f)
    return [(rc,poly[0][1])]+poly+[(rc,poly[-1][1])], R(r_f), R(r_a)

tp_p, RF_P, RA_P = tooth_pts(Z_P, GAM_P, R_P)
tp_w, RF_W, RA_W = tooth_pts(Z_W, GAM_W, R_W)
print("pinion root/tip radius %.3f/%.3f  wheel %.3f/%.3f  npts=%d"%(RF_P,RA_P,RF_W,RA_W,len(tp_p)))

def tooth_solid(poly, z_out, phase):
    ring=lambda s:[(s*rr*math.cos(dd+phase), s*rr*math.sin(dd+phase), s*z_out) for rr,dd in poly]
    return loft([make_face(Polyline(*ring(1.0),close=True)), make_face(Polyline(*ring(K_IN),close=True))], ruled=True)

t=time.time(); t0=tooth_solid(tp_p, ZO_P, 0.0); print("one tooth %.2fs vol=%.2f mm3"%(time.time()-t, t0.volume))

# -- cell 3 -------------------------------------------------------------------------
# Now the full pinion and wheel: frustum blank + teeth + shaft, phased so the pinion tooth centre and 
# cavity (mm): apex of both pitch cones at origin, pinion axis +z, wheel axis +x
XL,XH = -50.0,110.0; YL,YH = -70.0,70.0; ZL,ZH = -50.0,70.0
RS_P, RS_W = 10.0, 14.0            # shaft radii
SLEEVE, CLR = 12.0, 3.0            # bore sleeve length, shaft-to-bore clearance

def gear_assembly(poly, z_out, rf, z_teeth, phase0, r_shaft, shaft_to):
    blank = loft([make_face(Polyline(*[(rf*math.cos(a),rf*math.sin(a),z_out) for a in np.linspace(0,2*math.pi,181)[:-1]],close=True)),
                  make_face(Polyline(*[(K_IN*rf*math.cos(a),K_IN*rf*math.sin(a),K_IN*z_out) for a in np.linspace(0,2*math.pi,181)[:-1]],close=True))], ruled=True)
    g = blank
    for i in range(z_teeth):
        g = g + tooth_solid(poly, z_out, phase0 + 2*math.pi*i/z_teeth)
    shaft = Pos(0,0,(z_out-5+shaft_to)/2) * Cylinder(r_shaft, shaft_to-(z_out-5))
    return g + shaft

t=time.time()
pinion = gear_assembly(tp_p, ZO_P, RF_P, Z_P, 0.0, RS_P, ZH+SLEEVE+5)
print("pinion %.1fs vol=%.0f mm3 bbox=%s"%(time.time()-t, pinion.volume, pinion.bounding_box()))
t=time.time()
wheel_loc = gear_assembly(tp_w, ZO_W, RF_W, Z_W, math.pi/Z_W, RS_W, (XH+SLEEVE+5))
wheel = Rot(0,90,0) * wheel_loc
print("wheel  %.1fs vol=%.0f mm3 bbox=%s"%(time.time()-t, wheel.volume, wheel.bounding_box()))

# -- cell 4 -------------------------------------------------------------------------
# Interference check: the two gear solids must not touch, and the backlash gap must be ≈0.15 mm per fl
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
t=time.time()
inter = pinion.intersect(wheel)
vi = sum(s.volume for s in inter.solids()) if inter is not None else 0.0
print("intersection volume = %.4f mm3  (%.1fs)"%(vi, time.time()-t))
t=time.time()
d = BRepExtrema_DistShapeShape(pinion.wrapped, wheel.wrapped); d.Perform()
print("min distance pinion-wheel = %.4f mm  (%.1fs)"%(d.Value(), time.time()-t))

# -- cell 5 -------------------------------------------------------------------------
# Interference of 24 mm³ — I need to see where. Locating it before changing anything.
for s in inter.solids():
    bb=s.bounding_box(); c=s.center()
    print("vol %.3f  centre (%.2f,%.2f,%.2f)  r_from_zaxis %.2f  r_from_xaxis %.2f"%(
        s.volume, c.X,c.Y,c.Z, math.hypot(c.X,c.Y), math.hypot(c.Y,c.Z)))

# -- cell 6 -------------------------------------------------------------------------
# Single blob on the mesh plane (y=0) — a flank contact, so the relative phase is slightly off. Scanni
for dd in [-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]:
    w = Rot(dd,0,0)*wheel
    it = pinion.intersect(w)
    v = sum(s.volume for s in it.solids()) if it is not None else 0.0
    print(" delta=%+.2f deg  overlap=%.3f mm3"%(dd,v))

# -- cell 7 -------------------------------------------------------------------------
# The overlap is the pinion tip cone against the wheel root: my planar sections put the root cone too 
def tooth_pts_cone(z, gam, r_pitch, npf=10):
    """one tooth polygon (R,delta) at the outer section plane z_out, with the back-cone
       axial offset folded into R so tip/root cones pass through the common apex."""
    cg=math.cos(gam); sg=math.sin(gam); r_pv=r_pitch/cg; z_out=CONE*cg
    r_b=r_pv*math.cos(ALPHA); r_a=r_pv+MODULE; r_f=r_pv-1.25*MODULE
    psi=(math.pi*MODULE/2-BACKLASH/2)/(2*r_pv)
    rho_lo=max(r_b,r_f)
    d=lambda rho:(psi+inv(ALPHA)-inv(math.acos(min(1.0,r_b/rho))))/cg
    R=lambda rho:(r_pitch+(rho-r_pv)*cg)/(1.0-(rho-r_pv)*sg/z_out)
    half=[]
    if r_f<r_b-1e-9: half.append((R(r_f),d(rho_lo)))
    half+=[(R(r),d(r)) for r in np.linspace(rho_lo,r_a,npf)]
    tip=[(R(r_a),t) for t in np.linspace(d(r_a),-d(r_a),5)[1:-1]]
    poly=half+tip+[(rr,-dd) for rr,dd in reversed(half)]
    return [(0.90*R(r_f),poly[0][1])]+poly+[(0.90*R(r_f),poly[-1][1])], R(r_f), R(r_a)

tp_p, RF_P, RA_P = tooth_pts_cone(Z_P, GAM_P, R_P)
tp_w, RF_W, RA_W = tooth_pts_cone(Z_W, GAM_W, R_W)
print("outer-section root/tip: pinion %.3f/%.3f  wheel %.3f/%.3f"%(RF_P,RA_P,RF_W,RA_W))
t=time.time()
pinion = gear_assembly(tp_p, ZO_P, RF_P, Z_P, 0.0, RS_P, ZH+SLEEVE+5)
wheel  = Rot(0,90,0) * gear_assembly(tp_w, ZO_W, RF_W, Z_W, math.pi/Z_W, RS_W, XH+SLEEVE+5)
it = pinion.intersect(wheel); v = sum(s.volume for s in it.solids()) if it is not None else 0.0
print("rebuild %.1fs  overlap = %.4f mm3"%(time.time()-t, v))

# -- cell 8 -------------------------------------------------------------------------
# No interference. Now measuring the actual gap at the mesh — it should be ≈ backlash/2 on each flank.
t=time.time(); dss=BRepExtrema_DistShapeShape(pinion.wrapped, wheel.wrapped); dss.Perform()
p1=dss.PointOnShape1(1); p2=dss.PointOnShape2(1)
print("min gap pinion-wheel = %.4f mm  at (%.2f,%.2f,%.2f)  [%.0fs]"%(dss.Value(),p1.X(),p1.Y(),p1.Z(),time.time()-t))
print("target: backlash/2 per flank = %.3f mm ; tip clearance 0.25*m = %.3f mm"%(BACKLASH/2, 0.25*MODULE))

# -- cell 9 -------------------------------------------------------------------------
# Casing next: box plus the two bore sleeves, with faces named by *what they are* (box plane vs sleeve
RB_P, RB_W = RS_P+CLR, RS_W+CLR       # bore radii
box = Pos((XL+XH)/2,(YL+YH)/2,(ZL+ZH)/2) * Box(XH-XL, YH-YL, ZH-ZL)
sleeve_p = Pos(0,0,ZH+SLEEVE/2) * Cylinder(RB_P, SLEEVE)
sleeve_w = Pos(XH+SLEEVE/2,0,0) * Rot(0,90,0) * Cylinder(RB_W, SLEEVE)
casing = box + sleeve_p + sleeve_w
print("casing vol %.0f mm3, faces %d"%(casing.volume, len(casing.faces())))

planes = [("x",XL),("x",XH),("y",YL),("y",YH),("z",ZL),("z",ZH)]
groups = {"cavityWalls":[], "shaftBorePinion":[], "shaftBoreWheel":[]}
for f in casing.faces():
    c = f.center(); ax = f.normal_at(c) if f.geom_type==GeomType.PLANE else None
    onbox = any(abs(getattr(c, a.upper())-v)<1e-6 for a,v in planes) and f.geom_type==GeomType.PLANE
    if onbox: groups["cavityWalls"].append(f)
    elif c.Z > ZH-1e-6 or (f.geom_type==GeomType.CYLINDER and abs(c.X)<RB_P+1e-6): groups["shaftBorePinion"].append(f)
    else: groups["shaftBoreWheel"].append(f)
for k,v in groups.items():
    print(k, len(v), ["%s@(%.1f,%.1f,%.1f) A=%.0f"%(f.geom_type.name,f.center().X,f.center().Y,f.center().Z,f.area) for f in v])

# -- cell 10 ------------------------------------------------------------------------
# Exports worked; only my stray import failed. Re-emitting the export cleanly, then straight to blockM
MM = 1e-3
groups["pinion"] = pinion.faces(); groups["wheel"] = wheel.faces()
for name, fs in groups.items():
    shp = bd.scale(Compound(children=[Face(f.wrapped) for f in fs]), MM)
    bd.export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=2e-5, angular_tolerance=0.25, ascii_format=False)
    print(name, os.path.getsize(f"constant/triSurface/{name}.stl")//1024, "kB")

# -- cell 11 ------------------------------------------------------------------------
# Case files and a coarse castellated+snap run: base 5 mm, gears level 2.
import subprocess, textwrap, pathlib
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
head = lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
pathlib.Path("system/controlDict").write_text(head("dictionary","controlDict")+
  "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
pathlib.Path("system/fvSchemes").write_text(head("dictionary","fvSchemes")+"ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
pathlib.Path("system/fvSolution").write_text(head("dictionary","fvSolution")+"solvers{} \n")

BX=(-0.052,0.126); BY=(-0.072,0.072); BZ=(-0.052,0.086)
n=(36,29,28)
v=[(BX[i],BY[j],BZ[k]) for i,j,k in [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]]
pathlib.Path("system/blockMeshDict").write_text(head("dictionary","blockMeshDict")+"scale 1;\nvertices (\n"+
  "".join("(%g %g %g)\n"%p for p in v)+f");\nblocks (hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1));\nedges();\n"+
  "boundary (background {type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3));});\nmergePatchPairs();\n")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-500:])

# -- cell 12 ------------------------------------------------------------------------
# Now snappy, coarse: gears and bores at level 2, no layers.
PATCHES = {"pinion":(2,2), "wheel":(2,2), "cavityWalls":(0,0), "shaftBorePinion":(2,2), "shaftBoreWheel":(2,2)}
def snappy(levels, nlayer=0):
    geo = "".join(f'  {p}.stl {{type triSurfaceMesh; name {p};}}\n' for p in levels)
    rs  = "".join(f'    {p} {{level ({l[0]} {l[1]});}}\n' for p,l in levels.items())
    txt = head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers {"true" if nlayer else "false"};
geometry {{\n{geo}}}
castellatedMeshControls {{
  maxLocalCells 4000000; maxGlobalCells 12000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; features(); refinementSurfaces {{\n{rs}  }}
  resolveFeatureAngle 30; refinementRegions {{}}
  locationInMesh (-0.030 0.050 0.030); allowFreeStandingZoneFaces false;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 40; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.4; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
"""
    pathlib.Path("system/snappyHexMeshDict").write_text(txt)
snappy(PATCHES)
t=time.time(); r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print("rc",r.returncode,"%.0fs"%(time.time()-t)); print(r.stdout[-1200:])

# -- cell 13 ------------------------------------------------------------------------
print(r.stderr[-2000:])

# -- cell 14 ------------------------------------------------------------------------
def snappy(levels, nlayer=0, loc="(-0.030 0.050 0.030)"):
    geo = "".join(f'  {p}.stl {{type triSurfaceMesh; name {p};}}\n' for p in levels)
    rs  = "".join(f'    {p} {{level ({l[0]} {l[1]});}}\n' for p,l in levels.items())
    lay = "".join(f'    {p} {{nSurfaceLayers {nlayer};}}\n' for p in levels) if nlayer else ""
    txt = head("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers {"true" if nlayer else "false"};
geometry {{\n{geo}}}
castellatedMeshControls {{
  maxLocalCells 4000000; maxGlobalCells 12000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2;
  features ( );
  refinementSurfaces {{\n{rs}  }}
  resolveFeatureAngle 30;
  refinementRegions {{ }}
  locationInMesh {loc};
  allowFreeStandingZoneFaces false;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 40; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{\n{lay}  }} expansionRatio 1.2; finalLayerThickness 0.4; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
"""
    pathlib.Path("system/snappyHexMeshDict").write_text(txt)
snappy(PATCHES)
t=time.time(); r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print("rc",r.returncode,"%.0fs"%(time.time()-t)); print(r.stdout[-700:]); print(r.stderr[-600:])

# -- cell 15 ------------------------------------------------------------------------
cm=subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
keys=["cells:","points:","Mesh OK","***","pinion ","wheel ","cavityWalls","shaftBore"]
print("\n".join(l for l in cm.splitlines() if any(k in l for k in keys)))

# -- cell 16 ------------------------------------------------------------------------
# All five patches exist. Now look at it: render the gear patches from the mesh itself.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pathlib.Path("case.foam").write_text("")
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m = rd.read(); bnd = m["boundary"]
pl = pv.Plotter(off_screen=True, window_size=(1100,500), shape=(1,2))
cols={"pinion":"orange","wheel":"steelblue","cavityWalls":"lightgrey","shaftBorePinion":"red","shaftBoreWheel":"green"}
pl.subplot(0,0)
for k,c in cols.items():
    if k!="cavityWalls": pl.add_mesh(bnd[k], color=c, show_edges=True, line_width=0.3)
pl.camera_position=[(0.25,-0.25,0.20),(0.03,0,0.01),(0,0,1)]
pl.subplot(0,1)
clip = m.combine().clip(normal=(0,1,0), origin=(0,0.0005,0))
pl.add_mesh(clip, color="wheat", show_edges=True, line_width=0.2)
pl.camera_position=[(0.03,-0.35,0.03),(0.03,0,0.01),(0,0,1)]
pl.screenshot("look1.png"); print("ok")

# -- cell 17 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look1.png"))

# -- cell 18 ------------------------------------------------------------------------
# Shape is right: 20-tooth pinion on z, 30-tooth wheel on x, meshing at the pitch line, both bores nam
snappy({"pinion":(3,3),"wheel":(3,3),"cavityWalls":(0,0),"shaftBorePinion":(2,3),"shaftBoreWheel":(2,3)})
subprocess.run(["blockMesh"],capture_output=True,text=True)
log=open("log.snappy","w")
proc=subprocess.Popen(["snappyHexMesh","-overwrite"],stdout=log,stderr=subprocess.STDOUT)
time.sleep(120); print("running" if proc.poll() is None else "done rc=%s"%proc.returncode)
print(subprocess.run(["tail","-5","log.snappy"],capture_output=True,text=True).stdout)

# -- cell 19 ------------------------------------------------------------------------
time.sleep(150); print("running" if proc.poll() is None else "done rc=%s"%proc.returncode)
print(subprocess.run(["tail","-12","log.snappy"],capture_output=True,text=True).stdout)
