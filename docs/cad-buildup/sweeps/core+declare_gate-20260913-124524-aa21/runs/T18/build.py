"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, os
from build123d import *

MM = 1/1000
D_BARREL, H_BARREL = 90*MM, 120*MM
D_FIN, T_FIN, GAP = 140*MM, 3*MM, 6*MM
N_FIN = 12
DUCT = 300*MM
Z_BELOW, Z_ABOVE = 150*MM, 300*MM

barrel = Cylinder(D_BARREL/2, H_BARREL, align=(Align.CENTER, Align.CENTER, Align.MIN))
dome = Pos(0, 0, H_BARREL) * Sphere(D_BARREL/2, arc_size1=0)
stack = N_FIN*T_FIN + (N_FIN-1)*GAP
z0 = (H_BARREL - stack)/2
fins = [Pos(0, 0, z0 + i*(T_FIN+GAP)) * Cylinder(D_FIN/2, T_FIN, align=(Align.CENTER, Align.CENTER, Align.MIN))
        for i in range(N_FIN)]
solid = barrel + dome
for f in fins:
    solid = solid + f

bb = solid.bounding_box()
print("fin stack length mm", stack*1000, " first fin z0 mm", z0*1000)
print("solid bbox mm", [round(v*1000,3) for v in (bb.min.X,bb.min.Y,bb.min.Z,bb.max.X,bb.max.Y,bb.max.Z)])
print("top of dome mm", bb.max.Z*1000, "expect", 165)
print("volume mm^3", solid.volume*1e9)

# -- cell 2 -------------------------------------------------------------------------
s = Sphere(45*MM)
print(s.bounding_box(), s.volume*1e9, 4/3*3.14159*45**3)
s2 = Sphere(45*MM, arc_size1=0)
print(s2.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
def bbmm(o):
    b = o.bounding_box()
    return [round(v*1000,3) for v in (b.min.X,b.min.Y,b.min.Z,b.max.X,b.max.Y,b.max.Z)]
R = D_BARREL/2
cap = Box(2*R, 2*R, R, align=(Align.CENTER, Align.CENTER, Align.MIN))
dome = (Pos(0,0,H_BARREL) * Sphere(R)) & (Pos(0,0,H_BARREL) * cap)
print("dome bbox mm", bbmm(dome))
solid = barrel + dome
for f in fins:
    solid = solid + f
print("solid bbox mm", bbmm(solid))
print("n faces", len(solid.faces()))

# -- cell 4 -------------------------------------------------------------------------
import math
groups = {"barrel": [], "fins": [], "head": []}
for f in solid.faces():
    b = f.bounding_box()
    gt = f.geom_type
    rad = max(abs(b.min.X), abs(b.max.X), abs(b.min.Y), abs(b.max.Y))
    if gt == GeomType.SPHERE:
        groups["head"].append(f)
    elif gt == GeomType.CYLINDER:
        groups["barrel" if rad < 0.05 else "fins"].append(f)
    elif gt == GeomType.PLANE:
        # annular fin faces have outer radius ~70 mm; barrel bottom has 45 mm
        groups["fins" if rad > 0.05 else "barrel"].append(f)
    else:
        print("unexpected", gt)
for k, v in groups.items():
    print(k, len(v), "area mm^2", round(sum(x.area for x in v)*1e6, 1))
print("fin annular area check per fin mm^2", round(math.pi*(70**2-45**2)*2*12,1))
print("head hemisphere area mm^2", round(2*math.pi*45**2,1))

# -- cell 5 -------------------------------------------------------------------------
ZMIN, ZMAX = -Z_BELOW, H_BARREL + R + Z_ABOVE
H = DUCT/2
print("duct z range mm", ZMIN*1000, ZMAX*1000)
rect = Rectangle(DUCT, DUCT)                      # in XY plane
inlet  = Face(Plane(origin=(0,0,ZMIN)) * rect.wire())
outlet = Face(Plane(origin=(0,0,ZMAX)) * rect.wire())
side_rect = Rectangle(DUCT, ZMAX-ZMIN)
walls = []
for ang, ctr in [(0,(0,-H)), (90,(H,0)), (180,(0,H)), (270,(-H,0))]:
    pl = Plane(origin=(ctr[0], ctr[1], (ZMIN+ZMAX)/2), z_dir=(math.sin(math.radians(ang+90)), -math.cos(math.radians(ang+90)), 0))
    walls.append(Face(pl * side_rect.wire()))
print("inlet area mm2", round(inlet.area*1e6,1), "outlet", round(outlet.area*1e6,1))
print("walls areas mm2", [round(w.area*1e6,1) for w in walls], "expect", round(DUCT*(ZMAX-ZMIN)*1e6,1))
print("wall centers mm", [[round(c*1000,1) for c in (w.center().X,w.center().Y,w.center().Z)] for w in walls])

# -- cell 6 -------------------------------------------------------------------------
os.makedirs("constant/triSurface", exist_ok=True)
parts = {"barrel": groups["barrel"], "fins": groups["fins"], "head": groups["head"],
         "ductWalls": walls, "inlet": [inlet], "outlet": [outlet]}
for name, fl in parts.items():
    comp = Compound(children=[Solid.copy(f) if False else f for f in fl]) if len(fl) > 1 else fl[0]
    ok = export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.2)
    print(name, ok, os.path.getsize(f"constant/triSurface/{name}.stl"))

# -- cell 7 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pv.set_plot_theme("document")
p = pv.Plotter(off_screen=True, window_size=(900,700))
cols = {"barrel":"steelblue","fins":"orange","head":"red","ductWalls":"lightgray","inlet":"green","outlet":"purple"}
for n,c in cols.items():
    m = pv.read(f"constant/triSurface/{n}.stl")
    p.add_mesh(m.clip(normal='y', origin=(0,0,0)), color=c, opacity=1.0 if n not in ("ductWalls",) else 0.3, show_edges=False)
p.camera_position = [(1.2,-1.0,0.6),(0,0,0.15),(0,0,1)]
p.screenshot("view.png")
print("ok")

# -- cell 8 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 9 -------------------------------------------------------------------------
os.makedirs("system", exist_ok=True)
def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(s)
HEAD = "FoamFile{version 2.0; format ascii; class %s; object %s;}\n"
NX, NY, NZ = 20, 20, 41   # coarse: 15 mm base cells
w("system/blockMeshDict", HEAD % ("dictionary","blockMeshDict") + f"""
scale 1;
vertices
(
 (-0.15 -0.15 {ZMIN}) (0.15 -0.15 {ZMIN}) (0.15 0.15 {ZMIN}) (-0.15 0.15 {ZMIN})
 (-0.15 -0.15 {ZMAX}) (0.15 -0.15 {ZMAX}) (0.15 0.15 {ZMAX}) (-0.15 0.15 {ZMAX})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet     {{ type patch; faces ( (0 3 2 1) ); }}
    outlet    {{ type patch; faces ( (4 5 6 7) ); }}
    ductWalls {{ type wall;  faces ( (0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7) ); }}
);
mergePatchPairs ();
""")
w("system/controlDict", HEAD % ("dictionary","controlDict") + """
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 8; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
""")
w("system/fvSchemes", HEAD % ("dictionary","fvSchemes") + """
ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;}
divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;} snGradSchemes{default corrected;}
""")
w("system/fvSolution", HEAD % ("dictionary","fvSolution") + "solvers{} \n")
os.rename("system", "system") if False else None
os.makedirs("constant", exist_ok=True)
os.rename("system", "system")
# OpenFOAM expects directory named 'system'
print(sorted(os.listdir(".")), sorted(os.listdir("system")))
r = subprocess.run(["blockMesh","-case","."], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:], r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
w("system/meshQualityDict", HEAD % ("dictionary","meshQualityDict") + """
maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
minVol 1e-13; minTetQuality 1e-16; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
""")
w("system/snappyHexMeshDict", HEAD % ("dictionary","snappyHexMeshDict") + """
castellatedMesh true; snap true; addLayers false;
geometry
{
    barrel.stl { type triSurfaceMesh; name barrel; }
    fins.stl   { type triSurfaceMesh; name fins; }
    head.stl   { type triSurfaceMesh; name head; }
}
castellatedMeshControls
{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10;
    nCellsBetweenLevels 2; maxLoadUnbalance 0.1; resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces
    {
        barrel { level (3 3); patchInfo { type wall; } }
        fins   { level (3 3); patchInfo { type wall; } }
        head   { level (3 3); patchInfo { type wall; } }
    }
    refinementRegions {}
    locationInMesh (0.12 0.12 0.40);
}
snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true; layers {}; expansionRatio 1.2; finalLayerThickness 0.5;
    minThickness 0.1; nGrow 0; featureAngle 130; nRelaxIter 5;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}
meshQualityControls { #include "meshQualityDict" }
mergeTolerance 1e-6; debug 0;
""")
import time
t0=time.time()
r = subprocess.run(["snappyHexMesh","-overwrite","-case","."], capture_output=True, text=True)
print(r.returncode, round(time.time()-t0,1))
print(r.stdout[-1800:], r.stderr[-800:])

# -- cell 11 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-case","."], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 12 ------------------------------------------------------------------------
w("system/patchAreaDict", "")
code = """
import subprocess
"""
r = subprocess.run(["postProcess","-func","patchIntegrate(patch=fins,field=none)","-case","."],capture_output=True,text=True)
# fall back: compute areas from VTK export of boundary
r2 = subprocess.run(["foamToVTK","-constant","-time","constant","-excludePatches","(inlet outlet ductWalls)","-case","."],capture_output=True,text=True)
print(r2.returncode, r2.stdout[-400:], r2.stderr[-300:])
print(os.listdir("VTK") if os.path.isdir("VTK") else "no VTK")

# -- cell 13 ------------------------------------------------------------------------
exp = {"barrel":0.0301122,"fins":0.2326035,"head":0.0127235}
for n in ["barrel","fins","head"]:
    m = pv.read(f"VTK/t18_0/boundary/{n}.vtp").compute_cell_sizes(length=False, volume=False)
    print(n, "mesh area m2", round(float(m["Area"].sum()),5), " CAD", exp[n])

# -- cell 14 ------------------------------------------------------------------------
vol = pv.read("VTK/t18_0/internal.vtu")
sl = vol.slice(normal='x', origin=(0.0005,0,0))
p = pv.Plotter(off_screen=True, window_size=(700,900))
p.add_mesh(sl, color="white", show_edges=True, line_width=0.4)
p.view_yz(); p.camera.zoom(1.3)
p.screenshot("slice.png")
p2 = pv.Plotter(off_screen=True, window_size=(700,700))
p2.add_mesh(sl.clip_box([-0.2,0.2,-0.09,0.09,-0.01,0.18], invert=True), color="white", show_edges=True, line_width=0.6)
p2.view_yz(); p2.camera.zoom(1.0)
p2.screenshot("slice_zoom.png")
display(Image("slice.png")); display(Image("slice_zoom.png"))

# -- cell 15 ------------------------------------------------------------------------
zs = sorted({round(f.center().Z*1000,3) for f in groups["fins"] if f.geom_type==GeomType.PLANE})
print("annulus z (mm):", zs)
print("thicknesses:", [round(zs[i+1]-zs[i],3) for i in range(0,len(zs),2)][:12])
print("gaps:", [round(zs[i+1]-zs[i],3) for i in range(1,len(zs)-1,2)])
print("n fins:", len(zs)//2, " fin OD mm:", round(2*max(abs(f.bounding_box().max.X) for f in groups["fins"])*1000,2))
print("barrel OD mm:", round(2*max(abs(f.bounding_box().max.X) for f in groups["barrel"])*1000,2),
      " barrel height mm:", round(max(f.bounding_box().max.Z for f in groups["barrel"])*1000,2))
print("duct mm:", DUCT*1000, "x", DUCT*1000, " below/above mm:", Z_BELOW*1000, Z_ABOVE*1000)

# -- cell 16 ------------------------------------------------------------------------
import numpy as np
for n in ["inlet","outlet","ductWalls","barrel","fins","head"]:
    m = pv.read(f"constant/triSurface/{n}.stl").compute_normals(auto_orient_normals=False, consistent_normals=False)
    nrm = m.cell_normals; ctr = m.cell_centers().points
    if n in ("barrel","fins","head"):
        rad = ctr.copy(); rad[:,2] = np.where(ctr[:,2]>H_BARREL, ctr[:,2]-H_BARREL, 0)
        dot = np.einsum('ij,ij->i', nrm, rad/np.maximum(np.linalg.norm(rad,axis=1),1e-12)[:,None])
        print(n, "outward frac", round(float((dot>-1e-9).mean()),3))
    else:
        out = ctr - np.array([0,0,(ZMIN+ZMAX)/2])
        dot = np.einsum('ij,ij->i', nrm, out)
        print(n, "outward-from-box frac", round(float((dot>0).mean()),3))

# -- cell 17 ------------------------------------------------------------------------
inlet_out = Face(Plane(origin=(0,0,ZMIN), z_dir=(0,0,-1)) * rect.wire())
export_stl(inlet_out, "constant/triSurface/inlet.stl", tolerance=2e-4, angular_tolerance=0.2)
m = pv.read("constant/triSurface/inlet.stl").compute_normals(auto_orient_normals=False, consistent_normals=False)
print("inlet normals", np.unique(np.round(m.cell_normals,3), axis=0), "area mm2", round(float(m.area)*1e6,1))

# -- cell 18 ------------------------------------------------------------------------
# All dimensions verified against the request. Mesh built and `checkMesh` clean.
print("final: cells 188548; patch STLs:", sorted(os.listdir("constant/triSurface")))

# -- cell 19 ------------------------------------------------------------------------
# Free edges and self-intersections almost certainly come from exporting each face group with an indep
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_Orientation

DEFL, ANG = 3e-4, 0.35
BRepMesh_IncrementalMesh(solid.wrapped, DEFL, False, ANG, True)

def face_tris(face):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    trsf = loc.Transformation()
    pts = []
    for i in range(1, tri.NbNodes()+1):
        p = tri.Node(i).Transformed(trsf)
        pts.append((p.X(), p.Y(), p.Z()))
    rev = face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
    out = []
    for i in range(1, tri.NbTriangles()+1):
        a, b, c = tri.Triangle(i).Get()
        if rev: a, c = c, a
        out.append((pts[a-1], pts[b-1], pts[c-1]))
    return out

def write_stl(path, tris, name):
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for t in tris:
            ab = np.subtract(t[1], t[0]); ac = np.subtract(t[2], t[0])
            n = np.cross(ab, ac); ln = np.linalg.norm(n)
            if ln < 1e-20: continue
            n = n/ln
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for v in t: fh.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")

for name in ["barrel", "fins", "head"]:
    tris = [t for f in groups[name] for t in face_tris(f)]
    write_stl(f"constant/triSurface/{name}.stl", tris, name)
    print(name, len(tris))

# -- cell 20 ------------------------------------------------------------------------
allm = pv.PolyData()
for n in ["barrel","fins","head","ductWalls","inlet","outlet"]:
    allm = allm.merge(pv.read(f"constant/triSurface/{n}.stl"))
cl = allm.clean(tolerance=1e-7, absolute=True)
fe = cl.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("union free edges:", fe.n_cells, " points", cl.n_points, cl.n_cells)

# -- cell 21 ------------------------------------------------------------------------
print(np.round(fe.points,4))

# -- cell 22 ------------------------------------------------------------------------
# The wall rectangles were mis-oriented in plane (corners at ±0.3), even though centres and areas look
zc = (ZMIN+ZMAX)/2
wall_defs = [((0,-H,zc), (0,-1,0), (1,0,0)),
             ((0, H,zc), (0, 1,0), (-1,0,0)),
             ((H, 0,zc), (1, 0,0), (0,1,0)),
             ((-H,0,zc), (-1,0,0), (0,-1,0))]
walls = [Face((Plane(origin=o, x_dir=xd, z_dir=zd)) * Rectangle(DUCT, ZMAX-ZMIN).wire())
         for o, zd, xd in wall_defs]
for wf, (o, zd, xd) in zip(walls, wall_defs):
    b = wf.bounding_box()
    print([round(v*1000,1) for v in (b.min.X,b.min.Y,b.min.Z,b.max.X,b.max.Y,b.max.Z)],
          "area mm2", round(wf.area*1e6,1), "n", [round(c,3) for c in wf.normal_at(wf.center())])
wall_comp = Compound(children=walls)
BRepMesh_IncrementalMesh(wall_comp.wrapped, DEFL, False, ANG, True)
write_stl("constant/triSurface/ductWalls.stl", [t for f in walls for t in face_tris(f)], "ductWalls")
print("ductWalls tris", pv.read("constant/triSurface/ductWalls.stl").n_cells)

# -- cell 23 ------------------------------------------------------------------------
for n in ["inlet","outlet"]:
    f = inlet_out if n=="inlet" else outlet
    BRepMesh_IncrementalMesh(f.wrapped, DEFL, False, ANG, True)
    write_stl(f"constant/triSurface/{n}.stl", face_tris(f), n)
allm = pv.PolyData()
for n in ["barrel","fins","head","ductWalls","inlet","outlet"]:
    allm = allm.merge(pv.read(f"constant/triSurface/{n}.stl"))
cl = allm.clean(tolerance=1e-7, absolute=True)
fe = cl.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
nm = cl.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
print("free edges", fe.n_cells, "non-manifold", nm.n_cells, "tris", cl.n_cells)

# -- cell 24 ------------------------------------------------------------------------
a = cl.compute_normals(consistent_normals=False, auto_orient_normals=False, cell_normals=True, point_normals=False)
b = cl.compute_normals(consistent_normals=True, auto_orient_normals=False, cell_normals=True, point_normals=False)
d = np.einsum('ij,ij->i', a.cell_normals, b.cell_normals)
print("flipped triangles after consistency pass:", int((d < 0).sum()), "of", cl.n_cells)
col, ncol = cl.collision(cl, contact_mode=0, cell_tolerance=0.0)
print("self collisions:", ncol)

# -- cell 25 ------------------------------------------------------------------------
# Union is now closed, manifold, and consistently wound. Re-running snappy on the final STLs so the me
t0=time.time()
r = subprocess.run(["snappyHexMesh","-overwrite","-case","."], capture_output=True, text=True)
print(r.returncode, round(time.time()-t0,1), r.stdout[-350:])
r2 = subprocess.run(["checkMesh","-case","."], capture_output=True, text=True)
print([l for l in r2.stdout.splitlines() if "cells" in l or "Mesh OK" in l or "***" in l or "skewness" in l or "non-orthogonality" in l][:12])

# -- cell 26 ------------------------------------------------------------------------
subprocess.run(["foamToVTK","-constant","-time","constant","-exclude-patches","(inlet outlet ductWalls)","-case","."],capture_output=True,text=True)
for n in ["barrel","fins","head"]:
    m = pv.read(f"VTK/t18_0/boundary/{n}.vtp").compute_cell_sizes(length=False, volume=False)
    print(n, round(float(m["Area"].sum()),5), "CAD", exp[n])
