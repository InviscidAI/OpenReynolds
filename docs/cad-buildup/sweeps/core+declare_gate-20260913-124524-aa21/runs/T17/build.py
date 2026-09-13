"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Re-emitting the full geometry cell, self-contained, with the bend as a three-point arc.
from build123d import *
import numpy as np, math, os, subprocess

L, W, H = 0.300, 0.120, 0.080          # plenum length, width, height (m)
D_IN = 0.070                            # throttle inlet diameter
D_R, R_BEND = 0.038, 0.060              # runner diameter, centreline bend radius
REACH = 0.090                           # port face distance from plenum wall
Z_AX = H/2                              # runner axis height on the plenum wall
STRAIGHT = REACH - R_BEND               # straight lead before the bend
XS = [L*(2*i+1)/8 for i in range(4)]    # four evenly spaced stations along the plenum

plenum = Pos(L/2, 0, H/2) * Box(L, W, H)

def runner_path(x):
    y0 = W/2 - 0.012                       # start 12 mm inside the wall so the union is clean
    y1 = W/2 + STRAIGHT
    p0, p1 = Vector(x, y0, Z_AX), Vector(x, y1, Z_AX)
    c = Vector(x, y1, Z_AX + R_BEND)       # bend centre, above the start of the arc
    s = math.sin(math.radians(45))
    pm = Vector(x, y1 + R_BEND*s, Z_AX + R_BEND - R_BEND*s)
    p2 = Vector(x, y1 + R_BEND, Z_AX + R_BEND)
    return Wire([Line(p0, p1).edge(), ThreePointArc(p1, pm, p2).edge()]), p0, p2

def runner(x):
    path, p0, p2 = runner_path(x)
    prof = Plane(origin=p0, z_dir=(0, 1, 0)) * Circle(D_R/2)
    return sweep(prof, path)

runners = [runner(x) for x in XS]
runners = runners + [mirror(r, Plane.XZ) for r in runners]
fluid = plenum + runners
print("volume m^3", fluid.volume, "bbox", fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Geometry closes at the right extents. Now inventory the faces of the union so I can name patches fro
fs = fluid.faces()
from collections import Counter
print(len(fs), Counter(f.geom_type for f in fs))
for f in fs:
    print(f.geom_type, np.round(np.array(f.center().to_tuple()),4), round(f.area,5))

# -- cell 3 -------------------------------------------------------------------------
# Numbers first: port discs, bend radius, inlet plane. Then build the two sheet faces that split the e
ports = [f for f in fluid.faces() if f.geom_type == GeomType.PLANE and abs(f.center().Z - 0.10) < 1e-9]
tori  = [f for f in fluid.faces() if f.geom_type == GeomType.TORUS]
cyls  = [f for f in fluid.faces() if f.geom_type == GeomType.CYLINDER]
print("ports:", len(ports), "port dia m:", 2*math.sqrt(ports[0].area/math.pi), "(asked 0.038)")
print("port normal:", tuple(np.round(np.array(tuple(ports[0].normal_at(ports[0].center()))),4)))
print("port y:", sorted(set(round(abs(f.center().Y),4) for f in ports)), "-> reach from wall", 0.15-W/2)
# torus area = 2*pi*r * (quarter of 2*pi*R) -> recover R
R_meas = tori[0].area/(2*math.pi*(D_R/2))/(math.pi/2)
print("bend centreline radius m:", round(R_meas,5), "(asked 0.060)")
print("straight lead length m:", cyls[0].area/(math.pi*D_R), "(= 0.012 inside + 0.030 outside)")
print("stations mm:", [round(x*1000,2) for x in XS], " spacing", 1000*(XS[1]-XS[0]))
inlet_face = (Plane(origin=(0,0,H/2), z_dir=(-1,0,0)) * Circle(D_IN/2)).faces()[0]
endwall    = (Plane(origin=(0,0,H/2), z_dir=(-1,0,0)) * (Rectangle(W,H) - Circle(D_IN/2))).faces()[0]
print("inlet dia m:", 2*math.sqrt(inlet_face.area/math.pi), "endwall bbox:", endwall.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
# Rectangle came in rotated; fix `x_dir`. Then group the faces into named patches.
PL_END = Plane(origin=(0,0,H/2), z_dir=(-1,0,0), x_dir=(0,1,0))
inlet_face = (PL_END * Circle(D_IN/2)).faces()[0]
endwall    = (PL_END * (Rectangle(W,H) - Circle(D_IN/2))).faces()[0]
print("endwall bbox:", endwall.bounding_box(), "area", round(endwall.area,6), "expect", round(W*H-math.pi*0.035**2,6))

allf = fluid.faces()
port_targets = [(("L" if s<0 else "R")+str(i+1), Vector(x, s*(W/2+REACH), Z_AX+R_BEND))
                for s in (-1,1) for i,x in enumerate(XS)]
port_faces = {}
for nm, pt in port_targets:
    m = [f for f in allf if f.geom_type==GeomType.PLANE and (f.center()-pt).length < 1e-6]
    assert len(m)==1, (nm, len(m)); port_faces[nm]=m[0]
runner_wall_faces = [f for f in allf if f.geom_type in (GeomType.CYLINDER, GeomType.TORUS)]
port_ids = {id(f) for f in port_faces.values()}
plenum_wall_faces = [f for f in allf if f.geom_type==GeomType.PLANE and id(f) not in port_ids
                     and not (abs(f.center().X) < 1e-9)] + [endwall]
print("ports", sorted(port_faces), "runner walls", len(runner_wall_faces), "plenum walls", len(plenum_wall_faces))
print("area check: sum all =", round(sum(f.area for f in allf)+inlet_face.area+endwall.area-0.0096,6),
      " solid area =", round(fluid.area,6))

# -- cell 5 -------------------------------------------------------------------------
# Tessellate each face and write one STL per patch, then verify the welded union myself before meshing
import struct
os.makedirs("constant/triSurface", exist_ok=True)
TOL = 0.0008

def tris_of(faces):
    V=[]; T=[]
    for f in faces:
        v,t = f.tessellate(TOL)
        off=len(V); V += [np.array(tuple(p)) for p in v]
        T += [(a+off,b+off,c+off) for a,b,c in t]
    return np.array(V), np.array(T)

def write_stl(name, faces):
    V,T = tris_of(faces)
    with open(f"constant/triSurface/{name}.stl","w") as fh:
        fh.write(f"solid {name}\n")
        for a,b,c in T:
            p,q,r = V[a],V[b],V[c]; n = np.cross(q-p,r-p); L=np.linalg.norm(n)
            n = n/L if L>0 else np.array([0,0,1.])
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for w in (p,q,r): fh.write(f"   vertex {w[0]:.6e} {w[1]:.6e} {w[2]:.6e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")
    return len(T)

groups = {"inlet":[inlet_face], "plenumWalls":plenum_wall_faces, "runnerWalls":runner_wall_faces}
for nm,f in port_faces.items(): groups[f"port{nm}"] = [f]
for nm,fl in groups.items(): print(nm, write_stl(nm,fl))

# -- cell 6 -------------------------------------------------------------------------
import glob
from collections import defaultdict
V=[]; T=[]
for p in sorted(glob.glob("constant/triSurface/*.stl")):
    v,t = tris_of([])  # placeholder
    break
def read_stl(p):
    pts=[]
    for line in open(p):
        if line.strip().startswith("vertex"):
            pts.append([float(x) for x in line.split()[1:]])
    a=np.array(pts); return a.reshape(-1,3,3)
tri = np.vstack([read_stl(p) for p in sorted(glob.glob("constant/triSurface/*.stl"))])
key = np.round(tri.reshape(-1,3), 7)
uniq, inv = np.unique(key, axis=0, return_inverse=True)
f = inv.reshape(-1,3)
ec = defaultdict(int)
for a,b,c in f:
    for e in ((a,b),(b,c),(c,a)): ec[tuple(sorted(e))]+=1
free = sum(1 for k,v in ec.items() if v!=2)
print("triangles", len(f), "verts", len(uniq), "free/non-manifold edges:", free)

# -- cell 7 -------------------------------------------------------------------------
# Closed, watertight union. Look at it before meshing.
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(1000,700))
cols = {"inlet":"red","plenumWalls":"lightgray","runnerWalls":"steelblue"}
for p in sorted(glob.glob("constant/triSurface/*.stl")):
    nm = os.path.basename(p)[:-4]
    pl.add_mesh(pv.read(p), color=cols.get(nm,"orange"), opacity=0.55 if nm=="plenumWalls" else 1.0, show_edges=False)
pl.camera_position='iso'; pl.add_axes(); pl.screenshot("geom.png")
print("ok")

# -- cell 8 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 9 -------------------------------------------------------------------------
# Geometry is right. Now the case files: background blockMesh box plus snappyHexMesh with each STL as 
for d in ("system","constant/polyMesh"): os.makedirs(d, exist_ok=True)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(HDR%"controlDict"+"""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 6; runTimeModifiable true;
""")
open("system/fvSchemes","w").write(HDR%"fvSchemes"+"\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(HDR%"fvSolution"+"\nsolvers{}\n")

BASE = 0.010
x0,x1 = -0.02, 0.32; y0,y1 = -0.19, 0.19; z0,z1 = -0.02, 0.12
nx,ny,nz = [int(round((b-a)/BASE)) for a,b in ((x0,x1),(y0,y1),(z0,z1))]
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+f"""
scale 1;
vertices ((%g %g %g)(%g %g %g)(%g %g %g)(%g %g %g)(%g %g %g)(%g %g %g)(%g %g %g)(%g %g %g));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges (); boundary (( background {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} )); mergePatchPairs ();
"""%(x0,y0,z0, x1,y0,z0, x1,y1,z0, x0,y1,z0, x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1))
print(nx,ny,nz, nx*ny*nz)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 10 ------------------------------------------------------------------------
names = [os.path.basename(p)[:-4] for p in sorted(glob.glob("constant/triSurface/*.stl"))]
geom = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs = "\n".join(f'      {{ file "{n}.eMesh"; level 1; }}' for n in names)
surfs = "\n".join(f'      {n} {{ level (1 1); patchInfo {{ type {"patch" if (n=="inlet" or n.startswith("port")) else "wall"}; }} }}' for n in names)
open("system/snappyHexMeshDict","w").write(HDR%"snappyHexMeshDict"+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}};
castellatedMeshControls
{{
  maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; nCellsBetweenLevels 2;
  maxLoadUnbalance 0.1; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features (
{refs}
  );
  refinementSurfaces
  {{
{surfs}
  }}
  refinementRegions {{}};
  locationInMesh (0.15 0.0 0.04);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
  minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6; debug 0;
""")
open("system/surfaceFeatureExtractDict","w").write(HDR%"surfaceFeatureExtractDict"+"\n".join(
 f'{n}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} writeObj no; }}' for n in names))
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:])

# -- cell 11 ------------------------------------------------------------------------
# Running snappyHexMesh now at the coarse base size.
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 12 ------------------------------------------------------------------------
print(r.stderr[-2000:])

# -- cell 13 ------------------------------------------------------------------------
print(os.listdir("constant/polyMesh") if os.path.isdir("constant/polyMesh") else "none")
r2=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r2.returncode); print(r2.stderr[-1500:])

# -- cell 14 ------------------------------------------------------------------------
# blockMeshDict boundary syntax was wrong (extra parentheses). Rewriting it.
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+f"""
scale 1;
vertices
(
 ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
 ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
 background
 {{
  type patch;
  faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3));
 }}
);
mergePatchPairs ();
""")
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.returncode, os.listdir("constant/polyMesh"))

# -- cell 15 ------------------------------------------------------------------------
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stderr[-800:]); print(r.stdout[-900:])

# -- cell 16 ------------------------------------------------------------------------
# Mesh exists. Check patches and volume against the CAD, and look at it.
r=subprocess.run(["checkMesh"],capture_output=True,text=True); out=r.stdout
print([l for l in out.splitlines() if "patch" in l.lower() or "cells:" in l][:5])
print("\n".join(out.splitlines()[-25:]))

# -- cell 17 ------------------------------------------------------------------------
# Coarse mesh is sound (Mesh OK, volume 0.00399 vs CAD 0.004007). Now refine to level 2 (≈2.5 mm, ~15 
surfs2 = "\n".join(f'      {n} {{ level (2 2); patchInfo {{ type {"patch" if (n=="inlet" or n.startswith("port")) else "wall"}; }} }}' for n in names)
d = open("system/snappyHexMeshDict").read().replace(surfs, surfs2)
assert surfs2 in d
open("system/snappyHexMeshDict","w").write(d)
subprocess.run(["blockMesh"],capture_output=True,text=True)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-500:])

# -- cell 18 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); out=r.stdout
i=out.find("Checking patch topology"); print(out[i:i+1400])
print("\n".join([l for l in out.splitlines() if "Total volume" in l or "skewness" in l or "non-orthogonality" in l.lower() or l.strip()=="Mesh OK." or "cells:" in l]))

# -- cell 19 ------------------------------------------------------------------------
rd = pv.OpenFOAMReader("case.foam")
rd.enable_all_patch_arrays(); rd.cell_to_point_creation = False
m = rd.read(); print(m.keys())
bnd = m["boundary"]
for k in bnd.keys():
    a = bnd[k].extract_surface().compute_cell_sizes(length=False, volume=False)["Area"].sum()
    print(f"{k:14s} area {a:.6f}")
print("expect inlet", round(math.pi*0.035**2,6), " port", round(math.pi*0.019**2,6))

# -- cell 20 ------------------------------------------------------------------------
# Patch areas match (within faceting). Final look at the mesh.
pl = pv.Plotter(off_screen=True, window_size=(1100,750))
cmap = {"inlet":"red","plenumWalls":"lightgray","runnerWalls":"steelblue"}
for k in bnd.keys():
    pl.add_mesh(bnd[k].extract_surface(), color=cmap.get(k,"orange"), show_edges=True, edge_color="k", line_width=0.3,
                opacity=0.35 if k=="plenumWalls" else 1.0)
pl.camera_position='iso'; pl.add_axes(); pl.screenshot("mesh.png")
display(Image("mesh.png"))

# -- cell 21 ------------------------------------------------------------------------
# Quick self-check of STL orientation (signed volume of the welded union) before declaring.
p = tri[:,0]; q = tri[:,1]; s = tri[:,2]
vol = np.einsum('ij,ij->i', p, np.cross(q-p, s-p)).sum()/6.0
print("signed volume from STL winding:", vol, " CAD volume:", fluid.volume)
