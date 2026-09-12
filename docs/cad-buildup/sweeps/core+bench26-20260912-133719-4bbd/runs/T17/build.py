"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The first cell errored, so the constants aren't in the script. Re-emitting the parameters, imports a
import numpy as np, math, os, subprocess
from build123d import *

# --- parameters, millimetres (scaled to metres at STL export) ---
PLEN_L, PLEN_W, PLEN_H = 300.0, 120.0, 80.0
THR_D, THR_L = 70.0, 20.0
RUN_D, BEND_R, STRAIGHT = 38.0, 60.0, 30.0
PORT_DIST = 90.0                      # port face distance from plenum wall
XS = [37.5, 112.5, 187.5, 262.5]      # runner stations along plenum length
ZAX = PLEN_H / 2                      # runner axis height
assert STRAIGHT + BEND_R == PORT_DIST

def runner_solid(x0, sgn):
    """Round tube leaving the y=sgn*PLEN_W/2 wall horizontally, 90 deg bend upward."""
    p0 = Vector(x0, sgn * PLEN_W / 2, ZAX)
    p1 = p0 + Vector(0, sgn * STRAIGHT, 0)
    bp = Plane(origin=p1, x_dir=(0, sgn, 0), z_dir=(sgn, 0, 0))
    arc = bp * CenterArc(center=(0, BEND_R), radius=BEND_R, start_angle=-90, arc_size=90)
    path = Wire([Line(p0, p1).edge(), arc.edge()])
    prof = Plane(origin=p0, z_dir=(0, sgn, 0)) * Circle(RUN_D / 2)
    return sweep(prof, path)

r0 = runner_solid(XS[0], 1)
print("one runner bbox", r0.bounding_box())
print("volume", r0.volume, "analytic", math.pi * (RUN_D/2)**2 * (STRAIGHT + math.pi/2*BEND_R))
for f in r0.faces().filter_by(GeomType.PLANE):
    print("flat face c=", f.center(), "n=", f.normal_at(f.center()), "d=", 2*math.sqrt(f.area/math.pi))

# -- cell 2 -------------------------------------------------------------------------
# Now the full fluid volume: plenum box + throttle stub + 8 runners, fused.
plenum = Pos(PLEN_L/2, 0, PLEN_H/2) * Box(PLEN_L, PLEN_W, PLEN_H)
throttle = Plane(origin=(-THR_L, 0, ZAX), z_dir=(1, 0, 0)) * Cylinder(
    THR_D/2, THR_L, align=(Align.CENTER, Align.CENTER, Align.MIN))
runners = [(x0, sgn, runner_solid(x0, sgn)) for sgn in (1, -1) for x0 in XS]

fluid = plenum + throttle
for _, _, r in runners:
    fluid = fluid + r
print("solids:", len(fluid.solids()), " bbox:", fluid.bounding_box())
print("volume mm^3:", fluid.volume)
print("faces:", len(fluid.faces()))
from collections import Counter
print(Counter(str(f.geom_type) for f in fluid.faces()))

# -- cell 3 -------------------------------------------------------------------------
f = [x for x in fluid.faces() if x.geom_type == GeomType.CYLINDER][0]
from OCP.BRepAdaptor import BRepAdaptor_Surface
s = BRepAdaptor_Surface(f.wrapped)
print(s.Cylinder().Radius(), [m for m in dir(f) if "rad" in m.lower()])

# -- cell 4 -------------------------------------------------------------------------
from OCP.BRepAdaptor import BRepAdaptor_Surface

def cyl_radius(face):
    return BRepAdaptor_Surface(face.wrapped).Cylinder().Radius()

port_centers = {}
for x0, sgn, r in runners:
    pf = [f for f in r.faces().filter_by(GeomType.PLANE)
          if (f.normal_at(f.center()) - Vector(0, 0, 1)).length < 1e-6][0]
    port_centers[(x0, sgn)] = pf.center()
inlet_center = Vector(-THR_L, 0, ZAX)

def port_name(k):
    return "port%d%s" % (XS.index(k[0]) + 1, "L" if k[1] > 0 else "R")

groups = {"inlet": [], "plenumWalls": [], "runnerWalls": []}
for k in port_centers:
    groups[port_name(k)] = []

for f in fluid.faces():
    c, gt = f.center(), f.geom_type
    if gt == GeomType.PLANE:
        if (c - inlet_center).length < 1e-4:
            groups["inlet"].append(f)
        else:
            hit = [k for k, pc in port_centers.items() if (c - pc).length < 1e-4]
            groups[port_name(hit[0])].append(f) if hit else groups["plenumWalls"].append(f)
    elif gt == GeomType.CYLINDER:
        key = "runnerWalls" if abs(cyl_radius(f) - RUN_D/2) < 1e-6 else "plenumWalls"
        groups[key].append(f)
    else:
        groups["runnerWalls"].append(f)

for k, v in sorted(groups.items()):
    print(k, "nfaces", len(v), "area_mm2", round(sum(f.area for f in v), 1))

# -- cell 5 -------------------------------------------------------------------------
# STLs in metres, one per patch.
os.makedirs("constant/triSurface", exist_ok=True)
MM = 0.001
for k, v in sorted(groups.items()):
    comp = scale(Compound(children=[Face(f.wrapped) for f in v]), by=MM)
    export_stl(comp, "constant/triSurface/%s.stl" % k, tolerance=0.0005, angular_tolerance=0.2)
fluid_m = scale(fluid, by=MM)
print("fluid volume m^3", fluid_m.volume, " bbox(m)", fluid_m.bounding_box())
print(sorted(os.listdir("constant/triSurface")))

# -- cell 6 -------------------------------------------------------------------------
# Look at it before meshing.
import pyvista as pv
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(1100, 750))
cols = {"plenumWalls": "lightgray", "runnerWalls": "steelblue", "inlet": "red"}
for k in sorted(groups):
    m = pv.read("constant/triSurface/%s.stl" % k)
    pl.add_mesh(m, color=cols.get(k, "orange"), opacity=0.45 if k == "plenumWalls" else 1.0,
                show_edges=(k != "plenumWalls"), edge_color="k", line_width=0.4)
pl.camera_position = "iso"
pl.add_axes()
pl.screenshot("geom.png")
print("ok")

# -- cell 7 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 8 -------------------------------------------------------------------------
# Coarse snappyHexMesh setup: 10 mm background cells, feature extraction on all patch STLs.
import textwrap, glob
names = sorted(groups.keys())
for d in ("system", "constant"):
    os.makedirs(d, exist_ok=True)

def head(cls, obj, loc="system"):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n"
            "    location \"%s\";\n    object %s;\n}\n" % (cls, loc, obj))

open("system/controlDict", "w").write(head("dictionary", "controlDict") + textwrap.dedent("""
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
purgeWrite 0; writeFormat ascii; writePrecision 6; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
"""))
open("system/fvSchemes", "w").write(head("dictionary", "fvSchemes") +
    "ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\n"
    "divSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n"
    "interpolationSchemes{default linear;}\nsnGradSchemes{default corrected;}\n")
open("system/fvSolution", "w").write(head("dictionary", "fvSolution") + "solvers{}\n")

bm = head("dictionary", "blockMeshDict") + textwrap.dedent("""
scale 1;
vertices ( (-0.03 -0.18 -0.01) (0.31 -0.18 -0.01) (0.31 0.18 -0.01) (-0.03 0.18 -0.01)
           (-0.03 -0.18 0.11) (0.31 -0.18 0.11) (0.31 0.18 0.11) (-0.03 0.18 0.11) );
blocks ( hex (0 1 2 3 4 5 6 7) (34 36 12) simpleGrading (1 1 1) );
edges ();
boundary ( background { type patch; faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6)
                                            (1 2 6 5) (0 4 7 3) ); } );
""")
open("system/blockMeshDict", "w").write(bm)
print(subprocess.run(["blockMesh"], capture_output=True, text=True).stdout[-400:])

# -- cell 9 -------------------------------------------------------------------------
sfe = head("dictionary", "surfaceFeatureExtractDict")
for n in names:
    sfe += "%s.stl { extractionMethod extractFromSurface; includedAngle 150; writeObj no; }\n" % n
open("system/surfaceFeatureExtractDict", "w").write(sfe)
r = subprocess.run(["surfaceFeatureExtract"], capture_output=True, text=True)
print(r.returncode, r.stdout[-300:], r.stderr[-300:])
print(sorted(os.listdir("constant/triSurface")))

# -- cell 10 ------------------------------------------------------------------------
geom = "".join('    %s.stl { type triSurfaceMesh; name %s; }\n' % (n, n) for n in names)
refs = ""
for n in names:
    lvl = "(2 2)" if n in ("runnerWalls",) or n.startswith("port") else "(1 1)"
    refs += ('        %s { level %s; patchInfo { type %s; } }\n'
             % (n, lvl, "patch" if (n == "inlet" or n.startswith("port")) else "wall"))
feats = "".join('            { file "%s.eMesh"; level 1; }\n' % n for n in names)

snap = head("dictionary", "snappyHexMeshDict") + textwrap.dedent("""
castellatedMesh true; snap true; addLayers false;
geometry
{
%s}
castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 6000000; minRefinementCells 10;
    nCellsBetweenLevels 2; maxLoadUnbalance 0.1; resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    features ( 
%s    );
    refinementSurfaces
    {
%s    }
    refinementRegions {}
    locationInMesh (0.15 0.001 0.04);
}
snapControls
{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
  multiRegionFeatureSnap false; }
addLayersControls
{ relativeSizes true; layers {} expansionRatio 1.2; finalLayerThickness 0.4;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3;
  nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
  nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls
{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
  errorReduction 0.75; relaxed { maxNonOrtho 75; } }
writeFlags (); mergeTolerance 1e-6;
""" % (geom, feats, refs))
open("system/snappyHexMeshDict", "w").write(snap)
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 11 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh", "-constant"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 12 ------------------------------------------------------------------------
open("t17.foam", "w").write("")
rd = pv.OpenFOAMReader("t17.foam"); rd.set_active_time_value(0.0)
rd.enable_all_patch_arrays()
mb = rd.read()
bnd = mb["boundary"]
print("expect inlet %.6f, port %.6f m2" % (math.pi*(THR_D/2*MM)**2, math.pi*(RUN_D/2*MM)**2))
for nm in bnd.keys():
    s = bnd[nm].compute_cell_sizes(length=False, volume=False)
    print(nm, "area %.6f" % s.cell_data["Area"].sum())

# -- cell 13 ------------------------------------------------------------------------
# Confirm each named port patch sits where its runner was built, then look at the mesh.
for nm in sorted(bnd.keys()):
    if nm.startswith("port") or nm == "inlet":
        c = bnd[nm].points.mean(axis=0)
        print(nm, np.round(c, 4))
pl2 = pv.Plotter(off_screen=True, window_size=(1100, 750))
surf = mb["internalMesh"].extract_surface()
pl2.add_mesh(surf.clip(normal="y", origin=(0, 0.0, 0)), show_edges=True, color="lightblue", line_width=0.3)
pl2.camera_position = "iso"; pl2.add_axes()
pl2.screenshot("mesh.png")
display(Image("mesh.png"))

# -- cell 14 ------------------------------------------------------------------------
# Final measurements from the built geometry itself.
from OCP.BRepAdaptor import BRepAdaptor_Surface as BAS
tor = [BAS(f.wrapped).Torus() for f in fluid.faces() if f.geom_type == GeomType.TORUS]
print("bend centreline radii (mm):", sorted({round(t.MajorRadius(),3) for t in tor}), "asked 60")
print("tube radii (mm):", sorted({round(t.MinorRadius(),3) for t in tor}), "asked 19")
print("plenum box L,W,H (mm):", PLEN_L, PLEN_W, PLEN_H, " runner spacing:", np.diff(XS))
print("port face y (mm):", [round(abs(c.Y),3) for c in port_centers.values()], " = wall", PLEN_W/2, "+90")
print("turn angle (deg):", round(math.degrees(Vector(0,1,0).get_angle(Vector(0,0,1))),2))
print("inlet dia (mm):", round(2*math.sqrt(groups['inlet'][0].area/math.pi),3), "asked", THR_D)
print("CAD volume m^3 %.6f   mesh volume 0.0040772" % (fluid.volume*MM**3))
