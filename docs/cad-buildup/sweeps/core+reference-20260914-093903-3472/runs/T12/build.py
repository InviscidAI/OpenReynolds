"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll run that cell now.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- parameters (mm; all from the request except where noted) ---
R_BP, T_BP = 60.0, 4.0      # backplate radius, thickness
R_HUB, H_HUB = 12.0, 18.0   # hub radius, rise above backplate
N_BL, T_BL, H_BL = 7, 3.0, 14.0
R1, R2 = 16.0, 58.0         # blade LE / TE radius
WRAP = 30.0                 # total backward wrap, deg
R_EXT = 59.0                # my choice: extend camber past outlet so the TE is cut cleanly by r=58

def camber(r):
    th = np.deg2rad(-WRAP) * (r - R1) / (R2 - R1)
    return r*np.cos(th), r*np.sin(th)

r = np.linspace(R1, R_EXT, 60)
x, y = camber(r)
t = np.gradient(np.c_[x, y], axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
n = np.c_[-t[:, 1], t[:, 0]]
A = np.c_[x, y] + n*T_BL/2
B = np.c_[x, y] - n*T_BL/2
print("wrap at r=58 :", np.rad2deg(np.arctan2(*camber(58.0)[::-1])), "deg (want -30)")
print("blade radial span of offset outline:", np.hypot(*np.r_[A, B].T).min(), np.hypot(*np.r_[A, B].T).max())

fig, ax = plt.subplots(figsize=(6, 6))
for k in range(N_BL):
    c, s = np.cos(2*np.pi*k/N_BL), np.sin(2*np.pi*k/N_BL)
    Rt = np.array([[c, -s], [s, c]])
    for P in (A, B):
        Q = P @ Rt.T; ax.plot(Q[:, 0], Q[:, 1], 'k-', lw=1)
for rad in (12, 16, 58):
    ax.add_patch(plt.Circle((0, 0), rad, fill=False, color='C0', ls='--'))
ax.set_aspect(1); ax.set_title("blade footprints, hub/LE/outlet circles")
plt.savefig("footprint.png", dpi=90)

# -- cell 2 -------------------------------------------------------------------------
# Figure backend noted. Numbers are right: wrap = −30.0° at r=58. The offset outline dips to r=15.76, 
import build123d as bd
from build123d import Pos, Rot

# blade solid: closed outline from the two offset flanks, extruded to blade height
pts_A = [(px, py, 0.0) for px, py in A]
pts_B = [(px, py, 0.0) for px, py in B][::-1]
outline = bd.Spline(*pts_A) + bd.Line(pts_A[-1], pts_B[0]) + bd.Spline(*pts_B) + bd.Line(pts_B[-1], pts_A[0])
blade0 = bd.extrude(bd.make_face(outline), H_BL)
print("blade volume mm^3:", blade0.volume, " (camber length x 3 x 14 ~", 3*14*np.hypot(*np.diff(np.c_[x,y],axis=0).T).sum(), ")")

blades = bd.Compound([Rot(Z=360.0*k/N_BL)*blade0 for k in range(N_BL)])
annulus = (Pos(0,0,H_BL/2)*bd.Cylinder(R2, H_BL)) - (Pos(0,0,H_BL/2)*bd.Cylinder(R_HUB, H_BL))
fluid = annulus - blades
print("fluid solids:", len(fluid.solids()), " volume mm^3:", fluid.volume)
print("bbox:", fluid.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
# `bd.Face(f)` wraps into an empty shape — the faces are already `Face`s.
R_EYE = 16.0
groups = {k: [] for k in ("backplate","top","hub","outlet","blades")}
for f in fluid.faces():
    c = f.center(); n = f.normal_at(c); typ = f.geom_type
    rad = np.hypot(c.X, c.Y)
    if typ == bd.GeomType.PLANE and abs(n.Z) > 0.999 and abs(c.Z) < 1e-6:        groups["backplate"].append(f)
    elif typ == bd.GeomType.PLANE and abs(n.Z) > 0.999 and abs(c.Z-H_BL) < 1e-6: groups["top"].append(f)
    elif typ == bd.GeomType.CYLINDER and abs(rad - R_HUB) < 1e-6:                groups["hub"].append(f)
    elif typ == bd.GeomType.CYLINDER and abs(rad - R2) < 1e-6:                   groups["outlet"].append(f)
    else:                                                                        groups["blades"].append(f)
for k, v in groups.items(): print(k, len(v), "area", round(sum(f.area for f in v), 1))
print("total", round(sum(f.area for f in fluid.faces()),1))
print("hub area vs 2*pi*12*14 =", round(2*np.pi*R_HUB*H_BL,1))
print("outlet area vs 2*pi*58*14 - 7*3*14 =", round(2*np.pi*R2*H_BL - N_BL*T_BL*H_BL,1))

# -- cell 4 -------------------------------------------------------------------------
# Now split the top face at r=16 into the eye (inlet) and the shroud.
top = groups["top"][0]
disc = Pos(0, 0, H_BL) * bd.Circle(R_EYE)
inlet = top & disc
shroud = top - disc
print("inlet:", type(inlet), round(inlet.area,2), " analytic annulus 12-16 =", round(np.pi*(R_EYE**2-R_HUB**2),2))
print("shroud:", round(shroud.area,2), " sum =", round(inlet.area+shroud.area,2), "vs top", round(top.area,2))

# -- cell 5 -------------------------------------------------------------------------
# Now tessellate each patch group and write ASCII STLs in **metres** (scale 1/1000 once, here). I veri
import os
os.makedirs("constant/triSurface", exist_ok=True)
TOL = 0.15  # mm, tessellation deflection

patch_faces = {"inlet":[inlet], "shroud":shroud.faces(), "backplate":groups["backplate"],
               "hub":groups["hub"], "outlet":groups["outlet"], "blades":groups["blades"]}

def tri_soup(faces):
    T = []
    for f in faces:
        vs, ts = f.tessellate(TOL)
        P = np.array([[v.X, v.Y, v.Z] for v in vs])
        for a,b,c in ts: T.append(P[[a,b,c]])
    return np.array(T)

def write_stl(name, tris):
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for t in tris:
            n = np.cross(t[1]-t[0], t[2]-t[0]); L=np.linalg.norm(n); n = n/L if L>0 else n
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in t: fh.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")

allt, sv = [], 0.0
for nm, fs in patch_faces.items():
    T = tri_soup(fs); allt.append(T)
    write_stl(nm, T*1e-3)
    print(nm, len(T), "tris")
AT = np.concatenate(allt)
sv = np.einsum('ij,ij->i', AT[:,0], np.cross(AT[:,1], AT[:,2])).sum()/6
print("signed volume from triangles:", round(sv,1), " fluid volume:", round(fluid.volume,1))

# -- cell 6 -------------------------------------------------------------------------
# Signed volume matches the solid volume to 0.03% — the union is closed, consistently wound and outwar
import pyvista as pv
pv.start_xvfb() if hasattr(pv,"start_xvfb") else None
pv.global_theme.window_size=[900,420]
p = pv.Plotter(off_screen=True, shape=(1,2))
cols = {"inlet":"green","outlet":"red","blades":"grey","backplate":"tan","hub":"orange","shroud":"lightblue"}
for i,(view,hide) in enumerate([((1,-1,0.8),()),((0.2,0.2,1),("shroud",))]):
    p.subplot(0,i)
    for nm,c in cols.items():
        if nm in hide: continue
        p.add_mesh(pv.read(f"constant/triSurface/{nm}.stl"), color=c, show_edges=True, edge_color="k", line_width=0.3)
    p.camera_position=[view,(0,0,0.007),(0,0,1)]
p.screenshot("look.png")
from IPython.display import Image
Image("look.png")

# -- cell 7 -------------------------------------------------------------------------
# Shape is there (7 backward-curved passages, eye in the middle) but too small. Zooming in, edges off.
p = pv.Plotter(off_screen=True, shape=(1,2))
for i,hide in enumerate([(), ("shroud",)]):
    p.subplot(0,i)
    for nm,c in cols.items():
        if nm in hide: continue
        p.add_mesh(pv.read(f"constant/triSurface/{nm}.stl"), color=c, smooth_shading=False)
    p.camera_position = [(0.12,-0.12,0.09),(0,0,0.007),(0,0,1)] if i==0 else [(0.001,0.001,0.14),(0,0,0.007),(0,1,0)]
    p.camera.zoom(1.9)
p.screenshot("look2.png"); Image("look2.png")

# -- cell 8 -------------------------------------------------------------------------
# Right colours in right places, just over-zoomed. One full view, then on to meshing.
p = pv.Plotter(off_screen=True, shape=(1,2))
for i,hide in enumerate([(), ("shroud","inlet")]):
    p.subplot(0,i)
    for nm,c in cols.items():
        if nm in hide: continue
        p.add_mesh(pv.read(f"constant/triSurface/{nm}.stl"), color=c)
    p.camera_position = [(0.14,-0.14,0.10),(0,0,0.007),(0,0,1)] if i==0 else [(0.0,0.0,0.2),(0,0,0.007),(0,1,0)]
p.screenshot("look3.png"); Image("look3.png")

# -- cell 9 -------------------------------------------------------------------------
# Geometry is right. Now the case files: background `blockMesh` at a uniform 3 mm, and a `snappyHexMes
import textwrap, pathlib
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
HDR = "FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"

pathlib.Path("system/controlDict").write_text(HDR.format(cls="dictionary",obj="controlDict")+
 "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\n"
 "writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8;\n"
 "writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
pathlib.Path("system/fvSchemes").write_text(HDR.format(cls="dictionary",obj="fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;}\n"
 "laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;}\n"
 "snGradSchemes{default corrected;}\n")
pathlib.Path("system/fvSolution").write_text(HDR.format(cls="dictionary",obj="fvSolution")+
 "solvers{}\n")

# background block: uniform 3 mm, loosely bounding the fluid (r<=58, 0<=z<=14 mm)
CELL = 0.003
x0,x1 = -0.0615, 0.0615; z0,z1 = -0.0030, 0.0180
nx = round((x1-x0)/CELL); nz = round((z1-z0)/CELL)
pathlib.Path("system/blockMeshDict").write_text(HDR.format(cls="dictionary",obj="blockMeshDict")+f"""
scale 1;
vertices
(
 ({x0} {x0} {z0}) ({x1} {x0} {z0}) ({x1} {x1} {z0}) ({x0} {x1} {z0})
 ({x0} {x0} {z1}) ({x1} {x0} {z1}) ({x1} {x1} {z1}) ({x0} {x1} {z1})
);
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nz}) simpleGrading (1 1 1));
edges ();
boundary ( outer {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
mergePatchPairs ();
""")
print("block cells:", nx, nx, nz, "=", nx*nx*nz, " cell size mm:", (x1-x0)/nx*1e3, (z1-z0)/nz*1e3)

# -- cell 10 ------------------------------------------------------------------------
# `snappyHexMeshDict` — one entry per patch STL, `locationInMesh` in the blade-free eye annulus (r=14,
PATCHES = {"inlet":"patch","outlet":"patch","blades":"wall","backplate":"wall","hub":"wall","shroud":"wall"}
def snappy_dict(level, nlayers=0):
    geo = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in PATCHES)
    ref = "\n".join(f'      {n} {{ level ({level} {level}); patchInfo {{ type {t}; }} }}' for n,t in PATCHES.items())
    return HDR.format(cls="dictionary",obj="snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers {'true' if nlayers else 'false'};
geometry {{
{geo}
}}
castellatedMeshControls
{{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; nCellsBetweenLevels 2;
    maxLoadUnbalance 0.10; resolveFeatureAngle 30; allowFreeStandingZoneFaces false;
    features ();
    refinementSurfaces
    {{
{ref}
    }}
    refinementRegions {{}}
    locationInMesh (0.014 0.0 0.007);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
    nGrow 0; featureAngle 130; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50;
    layers {{ {' '.join(f'{n} {{ nSurfaceLayers {nlayers}; }}' for n,t in PATCHES.items() if t=='wall')} }} }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }} }}
debug 0; mergeTolerance 1e-6;
"""
pathlib.Path("system/snappyHexMeshDict").write_text(snappy_dict(1))
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-400:], r.returncode)

# -- cell 11 ------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1800:])

# -- cell 12 ------------------------------------------------------------------------
# Coarse mesh built, 27.9 k cells. Check patch names, mesh volume against the CAD volume, and look at 
print(open("constant/polyMesh/boundary").read()[300:1600])
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("Overall","cells","Mesh OK","***","non-orth","skewness","volume"))))

# -- cell 13 ------------------------------------------------------------------------
# All six patches present, `checkMesh` OK, mesh volume 1.27872e-4 m³ vs CAD 1.27922e-4 m³ (0.04% low).
r = subprocess.run(["foamToVTK","-constant","-no-internal","-overwrite"], capture_output=True, text=True)
print(r.returncode, r.stdout[-200:])
import glob; print(glob.glob("VTK/**/*.vt*", recursive=True)[:20])

# -- cell 14 ------------------------------------------------------------------------
def show(tag, hide=("shroud",)):
    pp = pv.Plotter(off_screen=True, shape=(1,2), window_size=[1000,470])
    for i,h in enumerate([(), hide]):
        pp.subplot(0,i)
        for nm,c in cols.items():
            if nm in h: continue
            pp.add_mesh(pv.read(f"VTK/t12_0/boundary/{nm}.vtp"), color=c, show_edges=True, edge_color="k", line_width=0.4)
        pp.camera_position=[(0.14,-0.14,0.10),(0,0,0.007),(0,0,1)] if i==0 else [(0.0,0.0,0.2),(0,0,0.007),(0,1,0)]
    pp.screenshot(f"{tag}.png"); return Image(f"{tag}.png")
show("mesh_coarse")

# -- cell 15 ------------------------------------------------------------------------
# Shape is captured cleanly at 3 mm. Now refine: level 2 on all surfaces (0.75 mm at walls), which put
pathlib.Path("system/snappyHexMeshDict").write_text(snappy_dict(2))
subprocess.run(["blockMesh"], capture_output=True, text=True)
r = subprocess.run(["snappyHexMesh","-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-700:])

# -- cell 16 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"], capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("cells:","Overall domain","Total volume","non-orth","skewness","Mesh OK","***","aspect"))))
import re
b=open("constant/polyMesh/boundary").read()
print({m[0]:int(m[1]) for m in re.findall(r"(\w+)\s*\{\s*type\s+\w+;(?:\s*inGroups[^;]*;)?\s*nFaces\s+(\d+);", b)})

# -- cell 17 ------------------------------------------------------------------------
# Mesh OK at 149 k cells. Now measure the delivered geometry from the mesh patches themselves and comp
subprocess.run(["foamToVTK","-constant","-no-internal","-overwrite"], capture_output=True, text=True)
print(f"{'patch':10s} {'r_min':>7s} {'r_max':>7s} {'z_min':>7s} {'z_max':>7s}   (mm)")
for nm in cols:
    P = pv.read(f"VTK/t12_0/boundary/{nm}.vtp").points*1e3
    rr = np.hypot(P[:,0],P[:,1])
    print(f"{nm:10s} {rr.min():7.2f} {rr.max():7.2f} {P[:,2].min():7.2f} {P[:,2].max():7.2f}")
# blade metrics from the CAD
cl = np.hypot(*np.diff(np.c_[x,y],axis=0).T).sum()
print("\nblades:", N_BL, "| thickness = A/h/L =", round(blade0.volume/H_BL/cl,3), "mm (want 3)")
print("wrap LE->TE:", round(-np.rad2deg(np.arctan2(*camber(58.0)[::-1])),2), "deg (want 30)")
print("LE radius", R1, "TE radius", R2, "| blade height", H_BL, "| mesh vol/CAD vol =", round(1.2788162e-4/(fluid.volume*1e-9),5))

# -- cell 18 ------------------------------------------------------------------------
# Every number matches. Final look at the refined mesh.
show("mesh_final")

# -- cell 19 ------------------------------------------------------------------------
# Two warnings on a 458-triangle surface — that's far smaller than my patch set, so let me find out wh
for f in sorted(glob.glob("**/*.stl", recursive=True)):
    n = sum(1 for l in open(f, errors="replace") if "facet normal" in l)
    print(f, os.path.getsize(f), n)

# -- cell 20 ------------------------------------------------------------------------
# Diagnosis: I tessellated each face independently, so vertices along shared edges don't coincide — th
eye_cyl = Pos(0,0,H_BL/2)*bd.Cylinder(R_EYE, H_BL*3)
f_in  = fluid & eye_cyl
f_out = fluid - eye_cyl
fluid2 = f_in + f_out
print("solids", len(fluid2.solids()), "vol", round(fluid2.volume,2), "vs", round(fluid.volume,2), "faces", len(fluid2.faces()))
for f in fluid2.faces():
    c=f.center(); rad=np.hypot(c.X,c.Y)
    if f.geom_type==bd.GeomType.CYLINDER and abs(rad-R_EYE)<0.5: print("  leftover r=16 face! area",f.area)
tops=[f for f in fluid2.faces() if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H_BL)<1e-6]
print("top faces:", [round(f.area,2) for f in tops])

# -- cell 21 ------------------------------------------------------------------------
# Fuse unified the top face back. Instead: mesh the whole solid **once** with OCC (shared edge nodes →
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
BRepMesh_IncrementalMesh(fluid.wrapped, TOL, False, 0.35, True)
def face_tris(f):
    loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(f.wrapped, loc)
    trsf = loc.Transformation(); rev = f.wrapped.Orientation() == 1  # TopAbs_REVERSED
    P = np.array([[p.X(),p.Y(),p.Z()] for p in (tri.Node(i).Transformed(trsf) for i in range(1, tri.NbNodes()+1))])
    T = []
    for i in range(1, tri.NbTriangles()+1):
        a,b,c = tri.Triangle(i).Get()
        T.append(P[[a-1,c-1,b-1] if rev else [a-1,b-1,c-1]])
    return np.array(T)
tt = {f: face_tris(f) for f in fluid.faces()}
print("faces", len(tt), "tris", sum(len(v) for v in tt.values()))
AT = np.concatenate(list(tt.values()))
print("signed vol:", round(np.einsum('ij,ij->i',AT[:,0],np.cross(AT[:,1],AT[:,2])).sum()/6,1), "vs", round(fluid.volume,1))

# -- cell 22 ------------------------------------------------------------------------
# Now verify conformality directly by counting free edges in the welded triangle soup, before splittin
def free_edges(T, dec=9):
    from collections import Counter
    ce = Counter()
    for t in T:
        k = [tuple(np.round(v,dec)) for v in t]
        for a,b in ((0,1),(1,2),(2,0)): ce[tuple(sorted((k[a],k[b])))] += 1
    return sum(1 for v in ce.values() if v != 2), len(ce)
print("free edges / total:", free_edges(AT))

# -- cell 23 ------------------------------------------------------------------------
# Zero free edges — conformal and closed. Now split the top-face triangles at r=16 into `inlet` and `s
def split_ring(T, R):
    a_in, a_out = [], []
    for t in T:
        s = np.hypot(t[:,0], t[:,1]) - R
        if (s <= 0).all(): a_in.append(t); continue
        if (s >= 0).all(): a_out.append(t); continue
        # one vertex alone on its side
        i = int(np.argmax([np.sign(s[k]) != np.sign(s[(k+1)%3]) and np.sign(s[k]) != np.sign(s[(k+2)%3] ) for k in range(3)]))
        j, k = (i+1)%3, (i+2)%3
        pj = t[i] + (t[j]-t[i]) * (s[i]/(s[i]-s[j]))
        pk = t[i] + (t[k]-t[i]) * (s[i]/(s[i]-s[k]))
        solo = [np.array([t[i], pj, pk])]
        pair = [np.array([pj, t[j], t[k]]), np.array([pj, t[k], pk])]
        (a_in if s[i] < 0 else a_out).extend(solo)
        (a_out if s[i] < 0 else a_in).extend(pair)
    return np.array(a_in), np.array(a_out)

top_f = [f for f in tt if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H_BL)<1e-6][0]
T_in, T_out = split_ring(tt[top_f], R_EYE)
def area(T): return 0.5*np.linalg.norm(np.cross(T[:,1]-T[:,0], T[:,2]-T[:,0]), axis=1).sum()
print("inlet", len(T_in), round(area(T_in),2), "| shroud", len(T_out), round(area(T_out),2),
      "| sum", round(area(T_in)+area(T_out),2), "vs top", round(area(tt[top_f]),2))

# -- cell 24 ------------------------------------------------------------------------
# Inlet area 317 vs the analytic 351.9 — the chordal cut across coarse triangles loses 10%. Do it prop
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.TopTools import TopTools_ListOfShape
cyl_face = [f for f in (Pos(0,0,H_BL/2)*bd.Cylinder(R_EYE, H_BL*3)).faces() if f.geom_type==bd.GeomType.CYLINDER][0]
spl = BRepAlgoAPI_Splitter()
la, lt = TopTools_ListOfShape(), TopTools_ListOfShape()
la.Append(fluid.wrapped); lt.Append(cyl_face.wrapped)
spl.SetArguments(la); spl.SetTools(lt); spl.Build()
fsplit = bd.Compound(spl.Shape())
print("ok", spl.IsDone(), "solids", len(fsplit.solids()), "vol", round(sum(s.volume for s in fsplit.solids()),2))
tops = [f for f in fsplit.faces() if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H_BL)<1e-6]
print("top faces:", [round(f.area,2) for f in tops], "(want 351.86 eye + 8785 shroud)")
print("total faces", len(fsplit.faces()))

# -- cell 25 ------------------------------------------------------------------------
# Split gives 8 solids (eye + 7 passages) with the correct top-face areas (350.47 eye, 7×1255.26 = 878
from collections import Counter
solids = fsplit.solids()
key = lambda f: (round(f.area,7), tuple(np.round([f.center().X,f.center().Y,f.center().Z],7)))
cnt = Counter(key(f) for s in solids for f in s.faces())
eye_solid = min(solids, key=lambda s: np.hypot(s.center().X, s.center().Y) + s.volume*1e-6)
eye_keys = {key(f) for f in eye_solid.faces()}
print("eye solid vol", round(eye_solid.volume,1), "| shared faces:", sum(1 for v in cnt.values() if v>1))

G = {k: [] for k in PATCHES}
for s in solids:
    for f in s.faces():
        k = key(f)
        if cnt[k] > 1: continue                       # internal interface
        c = f.center(); n = f.normal_at(c); rad = np.hypot(c.X, c.Y); t = f.geom_type
        if   t==bd.GeomType.PLANE and abs(n.Z)>0.999 and abs(c.Z)<1e-6:        G["backplate"].append(f)
        elif t==bd.GeomType.PLANE and abs(n.Z)>0.999 and abs(c.Z-H_BL)<1e-6:  G["inlet" if k in eye_keys else "shroud"].append(f)
        elif t==bd.GeomType.CYLINDER and abs(rad-R_HUB)<1e-6:                  G["hub"].append(f)
        elif t==bd.GeomType.CYLINDER and abs(rad-R2)<1e-6:                     G["outlet"].append(f)
        else:                                                                  G["blades"].append(f)
for k,v in G.items(): print(k, len(v), "area", round(sum(f.area for f in v),2))

# -- cell 26 ------------------------------------------------------------------------
# Areas match the single-solid classification exactly. Now one conformal mesh of the whole split compo
BRepMesh_IncrementalMesh(fsplit.wrapped, TOL, False, 0.3, True)
allT = []
for nm, fs in G.items():
    T = np.concatenate([face_tris(f) for f in fs])
    write_stl(nm, T*1e-3); allT.append(T); print(nm, len(T), "tris, area", round(area(T),2))
AT = np.concatenate(allT)
print("free edges / total:", free_edges(AT*1e-3, 11))
print("signed vol:", round(np.einsum('ij,ij->i',AT[:,0],np.cross(AT[:,1],AT[:,2])).sum()/6,1), "vs", round(fluid.volume,1))

# -- cell 27 ------------------------------------------------------------------------
# Zero free edges and the right signed volume. Let me pre-empt the self-intersection gate by checking 
with open("all_patches.stl","w") as o:
    for nm in G: o.write(open(f"constant/triSurface/{nm}.stl").read())
r = subprocess.run(["surfaceCheck","all_patches.stl"], capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in
      ("self-inter","Surface is","edges","closed","Number of","triangles","zones","Normal"))))
