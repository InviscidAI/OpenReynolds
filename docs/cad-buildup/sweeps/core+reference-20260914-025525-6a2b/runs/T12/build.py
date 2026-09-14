"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by checking the environment and sketching the blade camber math before building anything.
import numpy as np, build123d as bd, subprocess, os
print(bd.__version__ if hasattr(bd,'__version__') else 'b123d', os.getcwd())
print(subprocess.run(["bash","-lc","which blockMesh snappyHexMesh surfaceFeatureExtract"],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
# Blade camber: θ(r) linear in r with total wrap 30° backward. I'll build the 2D blade section in mm a
import numpy as np, build123d as bd
# ---- parameters (mm, from the request) ----
R_BACK, T_BACK = 60.0, 4.0      # backplate radius, thickness
R_HUB, H_HUB   = 12.0, 18.0     # hub radius, height above backplate
N_BLADE        = 7
T_BLADE        = 3.0            # blade thickness
H_BLADE        = 14.0           # blade height off backplate = shroud height
R_LE, R_TE     = 16.0, 58.0     # leading/trailing edge radius
WRAP           = 30.0           # backward sweep, degrees
R_FILLET       = 2.0
R_TE_EXT       = 59.0           # blade extended past outlet cut so TE is not tangent

def camber(r):
    th = np.deg2rad(-WRAP*(r-R_LE)/(R_TE-R_LE))
    return np.c_[r*np.cos(th), r*np.sin(th)]

rs = np.linspace(R_LE, R_TE_EXT, 80)
c  = camber(rs)
t  = np.gradient(c, axis=0); t /= np.linalg.norm(t,axis=1)[:,None]
n  = np.c_[-t[:,1], t[:,0]]
sideA, sideB = c + 0.5*T_BLADE*n, c - 0.5*T_BLADE*n
pts = [bd.Vector(*p,0) for p in sideA] + [bd.Vector(*p,0) for p in sideB[::-1]]
sect = bd.make_face(bd.Polyline(*pts, close=True))
print("section area mm^2", round(sect.area,2), " expected ~", round(T_BLADE*np.linalg.norm(np.diff(c,axis=0),axis=1).sum(),2))
thLE = np.degrees(np.arctan2(*camber(R_LE)[0][::-1])); thTE = np.degrees(np.arctan2(*camber(R_TE)[0][::-1]))
print("wrap LE->TE deg:", round(thLE-thTE,3), " LE r:", round(np.linalg.norm(camber(R_LE)),3), " TE r:", round(np.linalg.norm(camber(R_TE)),3))

# -- cell 3 -------------------------------------------------------------------------
# Now the fluid volume: annulus (hub r=12 to r=58, z 0→14) minus the 7 blades, no fillets yet. Coarse 
blade0 = bd.extrude(sect, H_BLADE)
blades = [bd.Rotation(0,0,i*360/N_BLADE)*blade0 for i in range(N_BLADE)]
annulus = bd.extrude(bd.make_face(bd.Circle(R_TE).wire()) - bd.make_face(bd.Circle(R_HUB).wire()), H_BLADE)
fluid_nf = annulus - bd.Compound(blades)
bb = fluid_nf.bounding_box()
print("solids:", len(fluid_nf.solids()), "vol mm^3", round(fluid_nf.volume,1), "bbox", bb.min, bb.max)
V_ann = np.pi*(R_TE**2-R_HUB**2)*H_BLADE
print("annulus vol", round(V_ann,1), "blade displaced (each)", round((V_ann-fluid_nf.volume)/7,1))

# -- cell 4 -------------------------------------------------------------------------
# Now attempt the 2 mm root fillets on the fluid's blade/backplate edges — the request asks me to try 
root_edges = fluid_nf.edges().filter_by_position(bd.Axis.Z, -1e-6, 1e-6).filter_by(lambda e: not isinstance(e.geom_type, type(None)) and e.geom_type != bd.GeomType.CIRCLE or abs(e.center().length-R_HUB)>1e-6 and abs(e.center().length-R_TE)>1e-6)
root_edges = [e for e in fluid_nf.edges().filter_by_position(bd.Axis.Z,-1e-6,1e-6) if e.center().length > R_HUB+1e-3 and not (e.geom_type==bd.GeomType.CIRCLE and abs(e.radius-R_TE)<1e-6)]
print(len(root_edges), [ (e.geom_type, round(e.length,2)) for e in root_edges[:6]])
try:
    fluid_f = bd.fillet(bd.ShapeList(root_edges), R_FILLET)
    print("fillet ok, vol", round(fluid_f.volume,1))
except Exception as ex:
    print("fillet FAILED:", type(ex).__name__, str(ex)[:200])

# -- cell 5 -------------------------------------------------------------------------
# The 2 mm fillet is geometrically impossible here (2+2 mm > the 3 mm blade thickness at the blunt end
R_SPLIT = R_LE   # 16 mm: shroud covers the bladed annulus, inside it is the axial inlet opening
ring = lambda ro, ri: bd.extrude(bd.make_face(bd.Circle(ro).wire()) - bd.make_face(bd.Circle(ri).wire()), H_BLADE)
blade_cmp = bd.Compound(blades)
outer = ring(R_TE, R_SPLIT) - blade_cmp
inner = ring(R_SPLIT, R_HUB) - blade_cmp
fluid = outer + inner
print("solids", len(fluid.solids()), "vol", round(fluid.volume,1), "vs", round(fluid_nf.volume,1))
top = fluid.faces().filter_by_position(bd.Axis.Z, H_BLADE-1e-6, H_BLADE+1e-6)
print("top faces", len(top), [round(max(v.center().length for v in f.vertices()),2) for f in top])

# -- cell 6 -------------------------------------------------------------------------
# The fuse merged the coplanar top faces, so I'll split the top face by boolean with a r=16 disc inste
import inspect
print(inspect.signature(bd.export_stl))
print([m for m in dir(bd) if 'scale' in m.lower()])

# -- cell 7 -------------------------------------------------------------------------
MM = 0.001
fluid_m = bd.scale(fluid, MM)          # mm -> metres, once, here
Hm, R_SPLITm, R_HUBm, R_TEm = H_BLADE*MM, R_SPLIT*MM, R_HUB*MM, R_TE*MM
tol = 1e-7
rad = lambda v: (v.X**2+v.Y**2)**0.5
allrad = lambda f,r: all(abs(rad(v)-r)<1e-6 for v in f.vertices())
F = fluid_m.faces()
top  = F.filter_by_position(bd.Axis.Z, Hm-tol, Hm+tol)
bot  = F.filter_by_position(bd.Axis.Z, -tol, tol)
hub  = bd.ShapeList([f for f in F if f not in top and f not in bot and allrad(f,R_HUBm)])
out  = bd.ShapeList([f for f in F if f not in top and f not in bot and allrad(f,R_TEm)])
rest = bd.ShapeList([f for f in F if f not in top and f not in bot and f not in hub and f not in out])
disc = bd.Plane.XY.offset(Hm)*bd.make_face(bd.Circle(R_SPLITm).wire())
inlet_f  = top[0] & disc
shroud_f = top[0] - disc
print("counts top/bot/hub/out/blade", len(top),len(bot),len(hub),len(out),len(rest))
print("inlet %.6f exp %.6f | shroud %.6f | backplate %.6f | hub %.6f exp %.6f | outlet %.6f exp %.6f | blades %.6f"%(
 inlet_f.area, np.pi*(R_SPLITm**2-R_HUBm**2), shroud_f.area, sum(f.area for f in bot),
 sum(f.area for f in hub), 2*np.pi*R_HUBm*Hm, sum(f.area for f in out), 2*np.pi*R_TEm*Hm - 7*T_BLADE*MM*Hm, sum(f.area for f in rest)))

# -- cell 8 -------------------------------------------------------------------------
# Faces are faceted (1092 blade faces from the polyline). Rebuilding the blade sides as splines gives 
sideA_pts=[bd.Vector(*p,0) for p in sideA]; sideB_pts=[bd.Vector(*p,0) for p in sideB]
sect = bd.make_face(bd.Wire([bd.Spline(*sideA_pts).edge(), bd.Line(sideA_pts[-1],sideB_pts[-1]).edge(),
                             bd.Spline(*sideB_pts).reversed().edge(), bd.Line(sideB_pts[0],sideA_pts[0]).edge()]))
blade0 = bd.extrude(sect, H_BLADE)
blades = [bd.Rotation(0,0,i*360/N_BLADE)*blade0 for i in range(N_BLADE)]
blade_cmp = bd.Compound(blades)
outer = ring(R_TE, R_SPLIT) - blade_cmp; inner = ring(R_SPLIT, R_HUB) - blade_cmp
fluid = outer + inner
fluid_m = bd.scale(fluid, MM)
F = fluid_m.faces()
top  = F.filter_by_position(bd.Axis.Z, Hm-tol, Hm+tol); bot = F.filter_by_position(bd.Axis.Z,-tol,tol)
hub  = bd.ShapeList([f for f in F if f not in top and f not in bot and allrad(f,R_HUBm)])
out  = bd.ShapeList([f for f in F if f not in top and f not in bot and allrad(f,R_TEm)])
rest = bd.ShapeList([f for f in F if f not in top and f not in bot and f not in hub and f not in out])
inlet_f, shroud_f = top[0] & disc, top[0] - disc
print("solids",len(fluid.solids()),"vol mm3",round(fluid.volume,1),"faces b/h/o/blade",len(bot),len(hub),len(out),len(rest))
print("blade area m2 %.6f, section area mm2 %.2f"%(sum(f.area for f in rest), sect.area))

# -- cell 9 -------------------------------------------------------------------------
# Exporting the six patch STLs in metres, then checking the welded union's free edges myself before me
import os
os.makedirs("constant/triSurface", exist_ok=True)
groups = {"inlet":[inlet_f], "shroud":[shroud_f], "backplate":list(bot), "hub":list(hub),
          "outlet":list(out), "blades":list(rest)}
for n,fs in groups.items():
    bd.export_stl(bd.Compound(fs), f"constant/triSurface/{n}.stl", tolerance=1e-4, angular_tolerance=0.2)
    print(n, len(fs), os.path.getsize(f"constant/triSurface/{n}.stl"))

# -- cell 10 ------------------------------------------------------------------------
import numpy as np, glob
from collections import Counter
def read_stl(p):
    d=np.fromfile(p,dtype=np.uint8); n=int(np.frombuffer(d[80:84],dtype='<u4')[0])
    rec=np.frombuffer(d[84:84+50*n],dtype='<u1').reshape(n,50)
    return np.frombuffer(rec[:,12:48].tobytes(),dtype='<f4').reshape(n,3,3).astype(float)
tris=np.vstack([read_stl(p) for p in sorted(glob.glob("constant/triSurface/*.stl"))])
pts=tris.reshape(-1,3); key=np.round(pts/1e-9).astype(np.int64)
uniq,inv=np.unique(key,axis=0,return_inverse=True); f=inv.reshape(-1,3)
ec=Counter(); dc=Counter()
for a,b,c in f:
    for e in ((a,b),(b,c),(c,a)): ec[tuple(sorted(e))]+=1; dc[e]+=1
free=sum(1 for v in ec.values() if v==1)
same=sum(1 for e,v in dc.items() if v>1)
print("tris",len(f),"verts",len(uniq),"free edges",free,"edges walked twice same dir",same)

# -- cell 11 ------------------------------------------------------------------------
# 1660 free edges — but my weld tolerance (1e-9 m) is tighter than float32 STL precision. Re-weld at 1
def count(tol):
    key=np.round(pts/tol).astype(np.int64); uniq,inv=np.unique(key,axis=0,return_inverse=True); f=inv.reshape(-1,3)
    ec=Counter()
    for a,b,c in f:
        for e in ((a,b),(b,c),(c,a)): ec[tuple(sorted(e))]+=1
    return len(uniq), sum(1 for v in ec.values() if v==1)
for t in (1e-9,1e-8,1e-7,1e-6):
    print(t, count(t))

# -- cell 12 ------------------------------------------------------------------------
# Meshing first. Coarse background block (~2 mm cells) around the passage, then snappy with the six su
import os, textwrap, subprocess
for d in ("system","constant"): os.makedirs(d,exist_ok=True)
open("system/controlDict","w").write("""FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6; runTimeModifiable true;
""")
open("system/fvSchemes","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}\nddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write("FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}\nsolvers{}\n")
L=0.0605; z0,z1=-0.0015,0.0155; n=(60,60,9)
open("system/blockMeshDict","w").write(f"""FoamFile{{version 2.0;format ascii;class dictionary;object blockMeshDict;}}
scale 1;
vertices ((-{L} -{L} {z0})({L} -{L} {z0})({L} {L} {z0})(-{L} {L} {z0})(-{L} -{L} {z1})({L} -{L} {z1})({L} {L} {z1})(-{L} {L} {z1}));
blocks (hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1));
edges (); boundary (); mergePatchPairs ();
""")
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:], r.stderr[-300:])

# -- cell 13 ------------------------------------------------------------------------
th=np.deg2rad(-30*(40-16)/42 + 0.5*360/7)
P=(40*np.cos(th)*MM, 40*np.sin(th)*MM, 0.007)
print("seed", P, "inside:", fluid_m.is_inside(bd.Vector(*P)))
names=["inlet","outlet","blades","backplate","hub","shroud"]
geo="".join(f"    {n}.stl {{ type triSurfaceMesh; name {n}; }}\n" for n in names)
refs="".join(f"        {n} {{ level (1 1); patchInfo {{ type {'patch' if n in ('inlet','outlet') else 'wall'}; }} }}\n" for n in names)
open("system/snappyHexMeshDict","w").write(f"""FoamFile{{version 2.0;format ascii;class dictionary;object snappyHexMeshDict;}}
castellatedMesh true; snap true; addLayers false;
geometry {{\n{geo}}}
castellatedMeshControls {{
 maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 0; nCellsBetweenLevels 2;
 features (); refinementSurfaces {{\n{refs} }}
 resolveFeatureAngle 30; refinementRegions {{}};
 locationInMesh ({P[0]:.6f} {P[1]:.6f} {P[2]:.6f}); allowFreeStandingZoneFaces false;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
 implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
 nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
 maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
 minVol 1e-13; minTetQuality 1e-9; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
 minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; writeFlags (); 
""")
print("ok")

# -- cell 14 ------------------------------------------------------------------------
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:]); print(r.stderr[-500:])

# -- cell 15 ------------------------------------------------------------------------
# Mesh exists. Now look at it and check patch areas against the geometry.
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-2200:])

# -- cell 16 ------------------------------------------------------------------------
import pyvista as pv; pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pv.OFF_SCREEN=True
rd=pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").write("")
rd=pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0) if rd.time_values else None
m=rd.read()
b=m["boundary"]; print(list(b.keys()))
cols={"inlet":"red","outlet":"blue","blades":"orange","backplate":"lightgray","hub":"green","shroud":"purple"}
p=pv.Plotter(window_size=(1000,450),shape=(1,2))
for j,cam in enumerate([(0.12,-0.12,0.14),(0.001,-0.001,0.25)]):
    p.subplot(0,j)
    for k,c in cols.items():
        if k=="shroud" and j==0: continue
        p.add_mesh(b[k], color=c, show_edges=True, line_width=0.3)
    p.camera_position=[cam,(0,0,0.007),(0,0,1)]
p.screenshot("view.png"); print("done")

# -- cell 17 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 18 ------------------------------------------------------------------------
# Re-emitting the split self-contained so it lives in the script.
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.TopTools import TopTools_ListOfShape
import build123d.topology as bt
cyl_tool = bd.Solid.make_cylinder(R_SPLITm, 0.05, bd.Plane.XY.offset(-0.01)).faces().filter_by(bd.GeomType.CYLINDER)[0]
_args=TopTools_ListOfShape(); _args.Append(fluid_m.wrapped)
_tools=TopTools_ListOfShape(); _tools.Append(cyl_tool.wrapped)
splitter=BRepAlgoAPI_Splitter(); splitter.SetArguments(_args); splitter.SetTools(_tools); splitter.Build()
split = bt.Compound(splitter.Shape())
print(splitter.IsDone(), "solids", len(split.solids()), "vol mm3", round(sum(x.volume for x in split.solids())*1e9,1), "faces", len(split.faces()))

# -- cell 19 ------------------------------------------------------------------------
# Now classify faces per solid (so orientation stays outward), drop the internal r=16 seam faces, and 
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
BRepMesh_IncrementalMesh(split.wrapped, 1e-4, False, 0.2, True)
def classify(f):
    vs=f.vertices(); zs=[v.Z for v in vs]; rs=[rad(v) for v in vs]
    if max(abs(z) for z in zs)<1e-7: return "backplate"
    if max(abs(z-Hm) for z in zs)<1e-7: return "inlet" if max(rs)<=R_SPLITm+1e-6 else "shroud"
    if f.geom_type==bd.GeomType.CYLINDER and max(abs(r-R_HUBm) for r in rs)<1e-6: return "hub"
    if f.geom_type==bd.GeomType.CYLINDER and max(abs(r-R_TEm) for r in rs)<1e-6: return "outlet"
    if f.geom_type==bd.GeomType.CYLINDER and max(abs(r-R_SPLITm) for r in rs)<1e-6: return None  # internal seam
    return "blades"
tag={}
for sol in split.solids():
    for f in sol.faces(): tag.setdefault(id(f.wrapped), None)
from collections import defaultdict
patch_tris=defaultdict(list)
for sol in split.solids():
    for f in sol.faces():
        name=classify(f)
        if name is None: continue
        loc=TopLoc_Location(); tri=BRep_Tool.Triangulation_s(f.wrapped, loc); T=loc.Transformation()
        rev = f.wrapped.Orientation()==TopAbs_REVERSED
        nodes=[tri.Node(i+1).Transformed(T) for i in range(tri.NbNodes())]
        for i in range(tri.NbTriangles()):
            a,b,c = tri.Triangle(i+1).Get()
            idx=(a-1,c-1,b-1) if rev else (a-1,b-1,c-1)
            patch_tris[name].append([[nodes[k].X(),nodes[k].Y(),nodes[k].Z()] for k in idx])
print({k:len(v) for k,v in patch_tris.items()})

# -- cell 20 ------------------------------------------------------------------------
import glob
for p in glob.glob("constant/triSurface/*.stl"): os.remove(p)
for name,tl in patch_tris.items():
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for t in tl:
            n=np.cross(np.subtract(t[1],t[0]), np.subtract(t[2],t[0])); n=n/ (np.linalg.norm(n) or 1)
            fh.write(" facet normal %.9e %.9e %.9e\n  outer loop\n"%tuple(n))
            for v in t: fh.write("   vertex %.9e %.9e %.9e\n"%tuple(v))
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
allt=np.array([t for tl in patch_tris.values() for t in tl]); pts=allt.reshape(-1,3)
key=np.round(pts/1e-9).astype(np.int64); uniq,inv=np.unique(key,axis=0,return_inverse=True); f=inv.reshape(-1,3)
ec=Counter(); dc=Counter()
for a,b,c in f:
    for e in ((a,b),(b,c),(c,a)): ec[tuple(sorted(e))]+=1; dc[e]+=1
print("tris",len(f),"free edges",sum(1 for v in ec.values() if v==1),"same-dir dup",sum(1 for v in dc.values() if v>1))

# -- cell 21 ------------------------------------------------------------------------
# Watertight and consistently wound. Check normals point out of the fluid, then refine: feature edges 
v0,v1,v2=allt[:,0],allt[:,1],allt[:,2]
sv=np.einsum('ij,ij->i', v0, np.cross(v1,v2)).sum()/6
print("signed volume m^3 %.6e (CAD %.6e) -> outward" % (sv, fluid_m.volume))
open("system/surfaceFeatureExtractDict","w").write("FoamFile{version 2.0;format ascii;class dictionary;object surfaceFeatureExtractDict;}\n"+
 "".join(f"{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}\n" for n in patch_tris))
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.returncode, r.stdout[-200:])
print(sorted(os.listdir("constant/triSurface")))

# -- cell 22 ------------------------------------------------------------------------
lev={"blades":2,"inlet":1,"outlet":1,"backplate":1,"hub":1,"shroud":1}
geo="".join(f"    {n}.stl {{ type triSurfaceMesh; name {n}; }}\n" for n in lev)
refs="".join(f"        {n} {{ level ({lev[n]} {lev[n]}); patchInfo {{ type {'patch' if n in ('inlet','outlet') else 'wall'}; }} }}\n" for n in lev)
feats="".join(f'        {{ file "{n}.eMesh"; level {lev[n]}; }}\n' for n in lev)
d=open("system/snappyHexMeshDict").read()
d=d.replace(d[d.index("geometry {"):d.index("castellatedMeshControls")], "geometry {\n"+geo+"}\n")
import re
d=re.sub(r"features \(\);", "features (\n"+feats+" );", d)
d=re.sub(r"refinementSurfaces \{\n.*?\n \}", "refinementSurfaces {\n"+refs+" }", d, flags=re.S)
open("system/snappyHexMeshDict","w").write(d)
print(d[d.index("castellatedMeshControls"):d.index("resolveFeatureAngle")])

# -- cell 23 ------------------------------------------------------------------------
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-800:]); print(r.stderr[-300:])

# -- cell 24 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
t=r.stdout
print(t[t.index("Checking patch topology"):t.index("Checking faceZone")])
print(t[t.index("Checking geometry"):][-1200:])

# -- cell 25 ------------------------------------------------------------------------
# Final render of the refined mesh, plus the numbers I'll report.
open("case.foam","w").write(""); rd=pv.OpenFOAMReader("case.foam"); m=rd.read(); b=m["boundary"]
p=pv.Plotter(window_size=(1000,450),shape=(1,2))
p.subplot(0,0)
for k,c in cols.items():
    if k!="shroud": p.add_mesh(b[k],color=c,show_edges=True,line_width=0.3)
p.camera_position=[(0.12,-0.12,0.10),(0,0,0.007),(0,0,1)]
p.subplot(0,1)
p.add_mesh(m["internalMesh"].clip('z',(0,0,0.007)),color='w',show_edges=True,line_width=0.2)
p.camera_position=[(0.02,-0.02,0.18),(0,0,0.007),(0,0,1)]
p.screenshot("final.png")
print("cells", m["internalMesh"].n_cells, "vol", round(m["internalMesh"].volume,9), "CAD vol", round(fluid_m.volume,9))
display(Image("final.png"))
