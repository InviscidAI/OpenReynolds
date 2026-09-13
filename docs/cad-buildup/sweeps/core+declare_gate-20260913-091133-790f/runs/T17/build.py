"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Re-emitting with the constants and imports inside so the cell stands alone in the script.
import os, numpy as np
from build123d import *

L_PL, W_PL, H_PL = 300.0, 120.0, 80.0      # plenum length(x), width(y), height(z), mm
D_THR = 70.0                                # throttle inlet diameter
D_RUN, R_BEND, L_OUT = 38.0, 60.0, 90.0     # runner dia, centreline bend radius, outboard reach
XS = [-112.5, -37.5, 37.5, 112.5]           # 4 evenly spaced stations, pitch 75 mm
L_STR = L_OUT - R_BEND                      # straight run before the bend = 30 mm

def runner_plane(x, s):   # local u -> outboard (+/-y), local v -> +z
    return Plane(origin=(x, 0, 0), x_dir=(0, s, 0), z_dir=(s, 0, 0))

def runner_path(x, s):
    y0, y1 = W_PL/2 - 10.0, W_PL/2 + L_STR        # start 10 mm inside the wall so it fuses
    ln = Line((y0, 0), (y1, 0))
    arc = JernArc(start=(y1, 0), tangent=(1, 0), radius=R_BEND, arc_size=-90)
    return runner_plane(x, s) * (ln + arc)

p = runner_path(XS[0], 1)
print("start", p @ 0, "end", p @ 1, "tangent@end", p % 1, "length", p.length)

# -- cell 2 -------------------------------------------------------------------------
def runner_solid(x, s):
    path = runner_path(x, s)
    sec = Plane(origin=path @ 0, z_dir=path % 0) * Circle(D_RUN/2)
    return sweep(sec, path)

plenum = Box(L_PL, W_PL, H_PL)
runners = [runner_solid(x, s) for s in (1, -1) for x in XS]
fluid = plenum + runners
print("volume mm^3", fluid.volume, " solids:", len(fluid.solids()), " faces:", len(fluid.faces()))
print("bbox", fluid.bounding_box())

# -- cell 3 -------------------------------------------------------------------------
from build123d import Axis
port_centres = {}
for s in (1, -1):
    for i, x in enumerate(XS):
        name = f"port_{'R' if s>0 else 'L'}{i+1}"
        port_centres[name] = Vector(x, s*(W_PL/2 + L_OUT), -R_BEND)

faces = fluid.faces()
port_faces, plenum_faces, runner_faces = {}, [], []
box_planes = [((1,0,0), L_PL/2), ((0,1,0), W_PL/2), ((0,0,1), H_PL/2)]
for f in faces:
    c = f.center()
    hit = [n for n, pc in port_centres.items() if (c - pc).length < 1e-6]
    if hit:
        port_faces[hit[0]] = f
        continue
    on_box = any(abs(abs(c.to_tuple()[k]) - d) < 1e-6 and abs(abs(f.normal_at(c).to_tuple()[k]) - 1) < 1e-6
                 for k, ((n, d)) in enumerate(box_planes))
    (plenum_faces if on_box else runner_faces).append(f)

print(len(port_faces), len(plenum_faces), len(runner_faces))
for n, f in sorted(port_faces.items()):
    print(n, "area", round(f.area, 2), "expect", round(np.pi*(D_RUN/2)**2, 2), "normal", f.normal_at(f.center()))
print("plenum face areas", [round(f.area) for f in plenum_faces])
print("runner wall total area", round(sum(f.area for f in runner_faces)))

# -- cell 4 -------------------------------------------------------------------------
inlet_plane = Plane(origin=(-L_PL/2, 0, 0), z_dir=(-1, 0, 0))
inlet_face = inlet_plane * Circle(D_THR/2)
end_face = [f for f in plenum_faces if abs(f.center().X + L_PL/2) < 1e-6][0]
wall_rest = end_face.cut(inlet_face)
plenum_wall_faces = [f for f in plenum_faces if f is not end_face] + list(wall_rest.faces())
print("inlet area", round(inlet_face.area,2), "expect", round(np.pi*(D_THR/2)**2,2))
print("end wall after cut", round(sum(f.area for f in wall_rest.faces()),2), "expect", round(W_PL*H_PL-np.pi*35**2,2))
print("plenum wall faces", len(plenum_wall_faces), "total area", round(sum(f.area for f in plenum_wall_faces),1))

# -- cell 5 -------------------------------------------------------------------------
print("total boundary area of solid", round(fluid.area,1))
print("sum of patch groups", round(inlet_face.area + sum(f.area for f in plenum_wall_faces)
      + sum(f.area for f in runner_faces) + sum(f.area for f in port_faces.values()),1))
# geometry checks against the request
p = runner_path(XS[2], 1)
print("runner dia (port face) mm", round(2*np.sqrt(port_faces['port_R3'].area/np.pi),3), "req 38")
print("port face plane z", port_faces['port_R3'].center().Z, "| outboard distance from plenum wall mm",
      abs(port_faces['port_R3'].center().Y) - W_PL/2, "req 90")
print("turn: start tangent", p % 0, "end tangent", p % 1, "-> 90 deg")
print("bend arc length mm", round(p.length - L_STR - 10.0, 3), "= R*pi/2 for R=60 ->", round(60*np.pi/2,3))
print("throttle dia mm", round(2*np.sqrt(inlet_face.area/np.pi),3), "req 70")
print("runner x stations", XS, "pitch", XS[1]-XS[0])

# -- cell 6 -------------------------------------------------------------------------
# All requested dimensions check out. Now scale to metres and export one STL per patch.
import subprocess
os.makedirs("constant/triSurface", exist_ok=True)
MM = 0.001
groups = {"inlet": [inlet_face], "plenum_walls": plenum_wall_faces, "runner_walls": runner_faces}
groups.update({n: [f] for n, f in port_faces.items()})
for name, fl in groups.items():
    comp = Compound(children=[Face(f.wrapped) for f in fl]).scale(MM)
    export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=1e-4, angular_tolerance=0.1)
print(subprocess.run(["ls","-l","constant/triSurface"],capture_output=True,text=True).stdout)
fluid_m = fluid.scale(MM)
print("fluid volume m^3", fluid_m.volume, "bbox", fluid_m.bounding_box())

# -- cell 7 -------------------------------------------------------------------------
import pyvista as pv, glob
pv.start_xvfb() if False else None
pl = pv.Plotter(off_screen=True, window_size=(1000,700))
cols = {"inlet":"red","plenum_walls":"lightgray","runner_walls":"steelblue"}
for f in sorted(glob.glob("constant/triSurface/*.stl")):
    n = os.path.basename(f)[:-4]
    pl.add_mesh(pv.read(f), color=cols.get(n,"orange"), opacity=0.55 if n=="plenum_walls" else 1.0)
pl.camera_position = 'iso'; pl.show_axes()
pl.screenshot("view.png")
print("ok")

# -- cell 8 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("view.png"))

# -- cell 9 -------------------------------------------------------------------------
# Geometry looks right. Now the background mesh and snappy setup (coarse: 10 mm base cells, surface le
import textwrap, subprocess, os
for d in ("system","constant"): os.makedirs(d, exist_ok=True)
HDR = "FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"
open("system/controlDict","w").write(HDR%"controlDict"+
 "application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
 "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;\n"
 "writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(HDR%"fvSchemes"+"gradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(HDR%"fvSolution"+"solvers{}\n")
open("system/blockMeshDict","w").write(HDR%"blockMeshDict"+textwrap.dedent("""
scale 1;
vertices ((-0.16 -0.18 -0.07)(0.16 -0.18 -0.07)(0.16 0.18 -0.07)(-0.16 0.18 -0.07)
          (-0.16 -0.18 0.05)(0.16 -0.18 0.05)(0.16 0.18 0.05)(-0.16 0.18 0.05));
blocks (hex (0 1 2 3 4 5 6 7) (32 36 12) simpleGrading (1 1 1));
edges (); boundary ((bg {type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3));})); mergePatchPairs ();
"""))
r = subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-400:], r.returncode)

# -- cell 10 ------------------------------------------------------------------------
env = dict(os.environ, PWD=os.getcwd())
r = subprocess.run(["blockMesh"],capture_output=True,text=True,env=env,cwd=os.getcwd())
print(r.returncode); print(r.stdout[-1500:]); print("ERR", r.stderr[-1500:])

# -- cell 11 ------------------------------------------------------------------------
bm = HDR%"blockMeshDict" + """
scale 1;
vertices
(
    (-0.16 -0.18 -0.07) (0.16 -0.18 -0.07) (0.16 0.18 -0.07) (-0.16 0.18 -0.07)
    (-0.16 -0.18  0.05) (0.16 -0.18  0.05) (0.16 0.18  0.05) (-0.16 0.18  0.05)
);
blocks ( hex (0 1 2 3 4 5 6 7) (32 36 12) simpleGrading (1 1 1) );
edges ();
boundary
(
    bg
    {
        type patch;
        faces
        (
            (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3)
        );
    }
);
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(bm)
r = subprocess.run(["blockMesh"],capture_output=True,text=True,env=env)
print(r.returncode, r.stdout[-500:], r.stderr[-300:])

# -- cell 12 ------------------------------------------------------------------------
names = ["inlet","plenum_walls","runner_walls"] + sorted(port_faces)
geom = "\n".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs = "\n".join(f'        {n} {{ level (1 1); patchInfo {{ type {"patch" if n=="inlet" or n.startswith("port") else "wall"}; }} }}' for n in names)
snappy = HDR%"snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry
{{
{geom}
}}
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.1;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
{refs}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh (0 0 0.01);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10;
  implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls
{{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50; }}
meshQualityControls
{{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13;
  minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
mergeTolerance 1e-6;
writeFlags (); 
"""
open("system/snappyHexMeshDict","w").write(snappy)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True,env=env)
print(r.returncode); print(r.stdout[-1200:]); print("ERR",r.stderr[-600:])

# -- cell 13 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"],capture_output=True,text=True,env=env)
out=r.stdout
print(out[out.find("Mesh stats"):out.find("Checking geometry")])
print(out[out.find("Checking geometry"):][:1800])

# -- cell 14 ------------------------------------------------------------------------
# Coarse mesh is valid (11 patches, volume within 0.5 % of CAD). Refining surfaces to level 2 (2.5 mm)
snappy2 = snappy.replace("level (1 1)","level (2 2)")
open("system/snappyHexMeshDict","w").write(snappy2)
subprocess.run(["blockMesh"],capture_output=True,text=True,env=env)
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True,env=env)
print(r.returncode, r.stdout[-300:])

# -- cell 15 ------------------------------------------------------------------------
r = subprocess.run(["checkMesh","-constant"],capture_output=True,text=True,env=env)
out=r.stdout
import re
print(re.search(r"cells:\s+\d+", out).group())
print(out[out.find("Checking patch topology"):out.find("Checking faceZone")])
print(out[out.find("Min volume"):out.find("End")][:700])

# -- cell 16 ------------------------------------------------------------------------
rd = pv.OpenFOAMReader("case.foam")
rd.enable_all_patch_arrays(); rd.cell_to_point_creation = False
blk = rd.read(); bnd = blk["boundary"]
print(bnd.keys())

# -- cell 17 ------------------------------------------------------------------------
for nm in ["inlet"]+sorted(port_faces):
    m = bnd[nm]; a = m.compute_cell_sizes(length=False, volume=False)["Area"].sum(); c = np.array(m.center)
    exp = np.pi*0.035**2 if nm=="inlet" else np.pi*0.019**2
    tgt = np.array([-0.15,0,0]) if nm=="inlet" else np.array(tuple(port_centres[nm]))/1000
    print(f"{nm:9s} area {a:.4e} exp {exp:.4e}  centre {np.round(c,4)} exp {np.round(tgt,4)}")

# -- cell 18 ------------------------------------------------------------------------
pl = pv.Plotter(off_screen=True, window_size=(1000,700))
pl.add_mesh(bnd["plenum_walls"].extract_surface(), color="lightgray", opacity=0.35, show_edges=True, line_width=0.3)
pl.add_mesh(bnd["runner_walls"].extract_surface(), color="steelblue", show_edges=True, line_width=0.3)
pl.add_mesh(bnd["inlet"].extract_surface(), color="red")
for n in sorted(port_faces): pl.add_mesh(bnd[n].extract_surface(), color="orange")
pl.camera_position='iso'; pl.screenshot("mesh.png")
display(Image("mesh.png"))
