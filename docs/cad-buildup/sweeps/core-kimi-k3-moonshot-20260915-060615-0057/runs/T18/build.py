"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The union already returns a Solid — the extra wrap was unnecessary. Fixing:
# union result is already a single Solid
print(type(cyl_solid))
solids = cyl_solid.solids()
print("n solids:", len(solids))
cyl_solid = solids[0]
print("volume (m^3):", cyl_solid.volume)
bb = cyl_solid.bounding_box()
print("bbox min:", tuple(round(v,4) for v in bb.min), "max:", tuple(round(v,4) for v in bb.max))
print("n faces:", len(cyl_solid.faces()))

# -- cell 2 -------------------------------------------------------------------------
c = cyl_solid.faces()[0].center()
print(type(c), [a for a in dir(c) if not a.startswith('_')][:40])
print(tuple(c.to_tuple()) if hasattr(c,'to_tuple') else '')

# -- cell 3 -------------------------------------------------------------------------
import build123d as bd
import numpy as np
import os

# ---- Parameters (metres) ----
R_BARREL   = 0.045      # barrel OD 90 mm
H_BARREL   = 0.120      # barrel height 120 mm
R_FIN      = 0.070      # fin OD 140 mm
T_FIN      = 0.003      # fin thickness 3 mm
GAP_FIN    = 0.006      # air gap 6 mm
N_FIN      = 12
DUCT_HALF  = 0.150      # 300 x 300 mm duct cross-section
Z_DUCT_LO  = -0.150     # 150 mm below cylinder base
Z_DUCT_HI  = H_BARREL + 0.300   # 300 mm above barrel top -> z = 0.42

# Fin stack: 12*3 + 11*6 = 102 mm, centred on the 120 mm barrel -> 9 mm margin each end
fin_stack = N_FIN*T_FIN + (N_FIN-1)*GAP_FIN
z0_fin = (H_BARREL - fin_stack)/2.0

# ---- Solid: barrel + fins + domed head ----
barrel = bd.Pos(0, 0, H_BARREL/2) * bd.Cylinder(R_BARREL, H_BARREL)
fins = None
for i in range(N_FIN):
    zc = z0_fin + i*(T_FIN+GAP_FIN) + T_FIN/2
    f = bd.Pos(0, 0, zc) * bd.Cylinder(R_FIN, T_FIN)
    fins = f if fins is None else fins + f
head = bd.Pos(0, 0, H_BARREL) * bd.Sphere(R_BARREL)   # hemispherical dome; lower half inside barrel
cyl_solid = (barrel + fins + head).solids()[0]

os.makedirs("constant/triSurface", exist_ok=True)

# ---- Classify faces by their own geometry (not bounding boxes) ----
GT = bd.GeomType
barrel_faces, fin_faces, head_faces = [], [], []
for f in cyl_solid.faces():
    gt = f.geom_type
    cen = f.center()
    r = np.hypot(cen.X, cen.Y)
    if gt == GT.SPHERE:
        head_faces.append(f)
    elif gt == GT.CYLINDER:
        (barrel_faces if r < 0.06 else fin_faces).append(f)   # r 0.045 barrel, 0.070 fin rim
    elif gt == GT.PLANE:
        if abs(cen.Z) < 1e-9:      # bottom disk of barrel
            barrel_faces.append(f)
        else:                      # annular fin top/bottom faces
            fin_faces.append(f)
    else:
        print("unclassified:", gt)

groups = {"barrel": barrel_faces, "fins": fin_faces, "head": head_faces}
for name, fl in groups.items():
    bd.export_stl(bd.Compound(children=fl), f"constant/triSurface/{name}.stl")
    print(name, "faces:", len(fl), "area =", round(sum(f.area for f in fl), 6), "m^2")

print("expected barrel (side + bottom disk) ~", round(2*np.pi*R_BARREL*H_BARREL + np.pi*R_BARREL**2, 6))
print("expected fins ~", round(N_FIN*(2*np.pi*(R_FIN**2-R_BARREL**2) + 2*np.pi*R_FIN*T_FIN), 6))
print("expected head (hemisphere) ~", round(2*np.pi*R_BARREL**2, 6))

# -- cell 4 -------------------------------------------------------------------------
import pyvista as pv
pv.set_jupyter_backend('static')
pl = pv.Plotter(shape=(1,2), window_size=(1400,650))
colors = {"barrel":"lightblue", "fins":"orange", "head":"lightgreen"}
pl.subplot(0,0)
for name in colors:
    pl.add_mesh(pv.read(f"constant/triSurface/{name}.stl"), color=colors[name], show_edges=False)
pl.add_title("solid patches (STL)"); pl.view_isometric(); pl.add_axes()

# duct wireframe box for context
pl.subplot(0,1)
for name in colors:
    pl.add_mesh(pv.read(f"constant/triSurface/{name}.stl"), color=colors[name])
duct = pv.Box(bounds=(-DUCT_HALF,DUCT_HALF,-DUCT_HALF,DUCT_HALF,Z_DUCT_LO,Z_DUCT_HI))
pl.add_mesh(duct, style='wireframe', color='k')
pl.add_title("cylinder inside 300x300 duct, z in [-0.15, 0.42]"); pl.view_isometric(); pl.add_axes()
pl.show()

# -- cell 5 -------------------------------------------------------------------------
# Geometry looks right. Now I'll write the OpenFOAM case files: a blockMesh duct (inlet at bottom, out
import os
os.makedirs("system", exist_ok=True)

# ---- blockMeshDict: duct background mesh, 10 mm cells (coarse) ----
x0, x1 = -DUCT_HALF, DUCT_HALF
z0, z1 = Z_DUCT_LO, Z_DUCT_HI
nx = int(round((x1-x0)/0.010)); ny = nx; nz = int(round((z1-z0)/0.010))
blockmesh = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
convertToMeters 1;

vertices
(
    ({x0} {x0} {z0})   //0
    ({x1} {x0} {z0})   //1
    ({x1} {x1} {z0})   //2
    ({x0} {x1} {z0})   //3
    ({x0} {x0} {z1})   //4
    ({x1} {x0} {z1})   //5
    ({x1} {x1} {z1})   //6
    ({x0} {x1} {z1})   //7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ( (0 3 2 1) );
    }}
    outlet
    {{
        type patch;
        faces ( (4 5 6 7) );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (1 2 6 5)
            (2 3 7 6)
            (3 0 4 7)
        );
    }}
);
"""
open("system/blockMeshDict","w").write(blockmesh)

controldict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     snappyHexMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
open("system/controlDict","w").write(controldict)
print("blockMesh cells:", nx, ny, nz, "->", nx*ny*nz, "cells")
print("wrote system/blockMeshDict, system/controlDict")

# -- cell 6 -------------------------------------------------------------------------
# Now the snappyHexMeshDict with coarse refinement levels first:
snappy = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    barrel
    {
        type triSurfaceMesh;
        file "barrel.stl";
    }
    fins
    {
        type triSurfaceMesh;
        file "fins.stl";
    }
    head
    {
        type triSurfaceMesh;
        file "head.stl";
    }
}

castellatedMeshControls
{
    maxLocalCells 1000000;
    maxGlobalCells 4000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;

    features
    (
    );

    refinementSurfaces
    {
        barrel
        {
            level (2 2);
        }
        fins
        {
            level (2 3);
        }
        head
        {
            level (2 2);
        }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
    }

    locationInMesh (0.10 0.10 -0.05);
    allowFreeStandingZoneFaces false;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;

    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers
    {
    }
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}

mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy)
print("wrote system/snappyHexMeshDict")

# -- cell 7 -------------------------------------------------------------------------
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1500:])
print(r.stderr[-800:])

# -- cell 8 -------------------------------------------------------------------------
r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
tail = r.stdout[-2500:]
print(tail)
print(r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
# snappyHexMesh wants `fvSchemes`/`fvSolution` present — adding minimal ones:
fvschemes = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
open("system/fvSchemes","w").write(fvschemes)

fvsolution = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers {}
relaxationFactors {}
"""
open("system/fvSolution","w").write(fvsolution)

r = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-2000:])
print(r.stderr[-500:])

# -- cell 10 ------------------------------------------------------------------------
# Meshing finished cleanly — 432k cells, zero faces in error. Running `checkMesh` and then looking at 
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print("rc:", r.returncode)
print(r.stdout[-3000:])
print(r.stderr[-300:])

# -- cell 11 ------------------------------------------------------------------------
# `checkMesh`: Mesh OK, all six named patches present. Now let me actually look at the mesh — a slice 
open("t18.foam","w").write("")
reader = pv.get_reader("t18.foam")
reader.enable_all_patch_arrays()
mesh = reader.read()
print(mesh.keys() if hasattr(mesh,'keys') else mesh)
mb = mesh
internal = mb["internalMesh"]
print("blocks:", mb.keys())
print("internal cells:", internal.n_cells)

# -- cell 12 ------------------------------------------------------------------------
pl = pv.Plotter(shape=(1,2), window_size=(1500,700))

# Left: y-normal slice through the fin stack region
sl = internal.slice(normal=(0,1,0), origin=(0,0,0.06))
pl.subplot(0,0)
pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
pl.view_yz() if hasattr(pl,'view_yz') else None
pl.camera_position = [(0.5, 0, 0.06), (0, 0, 0.06), (0, 0, 1)]
pl.add_title("slice y=0: barrel + fins + gaps")

# Right: full slice with duct extent
sl2 = internal.slice(normal=(1,0,0), origin=(0,0,0.1))
pl.subplot(0,1)
pl.add_mesh(sl2, show_edges=True, color="white", lighting=False)
pl.camera_position = [(0.9, 0, 0.15), (0, 0, 0.15), (0, 0, 1)]
pl.add_title("slice x=0: full duct")
pl.show()

# -- cell 13 ------------------------------------------------------------------------
# The cylinder is correctly cut out with the fin stack visible. Let me zoom into the fin gaps to check
pl = pv.Plotter(window_size=(900,800))
sl = internal.slice(normal=(1,0,0), origin=(0,0,0.06))
pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
# zoom on fin stack: r from 0.03 to 0.08, z 0.03..0.09
pl.camera_position = [(0.055, 0, 0.058), (0.055, 0, 0.058), (0,0,1)]
pl.camera.focal_point = (0.0, 0.0, 0.058)
pl.camera.position = (0.0, -0.15, 0.058)
pl.camera.up = (0,0,1)
pl.camera.zoom(3.0)
pl.add_title("zoom: fin stack (fin 3 mm, gap 6 mm)")
pl.show()

# measure: count cells across a fin and a gap at r=0.055
import numpy as np
centers = internal.cell_centers().points
sel = (np.abs(np.hypot(centers[:,0], centers[:,1]) - 0.055) < 0.004)
zz = np.sort(centers[sel][:,2])
# fin centers expected at z0_fin + i*0.009 + 0.0015
print("fin centre z's:", [round(z0_fin + i*0.009 + 0.0015, 4) for i in range(12)])

# -- cell 14 ------------------------------------------------------------------------
pl = pv.Plotter(window_size=(900,800))
sl = internal.slice(normal=(1,0,0), origin=(0,0,0.06))
# crop to fin stack region
sl = sl.clip_box(bounds=(-0.01,0.01, 0.02,0.09, 0.02,0.10), invert=False)
pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
pl.view_vector((1,0,0), viewup=(0,0,1))
pl.camera.zoom(1.4)
pl.add_title("fin stack zoom: y 20-90 mm, z 20-100 mm")
pl.show()

# -- cell 15 ------------------------------------------------------------------------
c = internal.cell_centers().points
r_c = np.hypot(c[:,0], c[:,1])

def radial_extent(zlo, zhi, label):
    sel = (c[:,2] > zlo) & (c[:,2] < zhi)
    rr = r_c[sel]
    print(f"{label}: n={sel.sum()}, min r = {rr.min():.4f}, max r = {rr.max():.4f}")

radial_extent(0.0275, 0.0295, "fin z-band (fin at 27-30 mm)   ")
radial_extent(0.0315, 0.0345, "gap z-band (gap at 30-36 mm)   ")
radial_extent(0.14, 0.16, "above barrel, dome region      ")
radial_extent(-0.10, -0.05, "below cylinder (all air)       ")

# any cells inside the solid cylinder solid region?
inside_barrel = (r_c < 0.044) & (c[:,2] > 0.001) & (c[:,2] < 0.119)
inside_fin    = (r_c < 0.069) & (r_c > 0.044)
finz = np.zeros(len(c), bool)
for i in range(12):
    zl = z0_fin + i*0.009; zh = zl + 0.003
    finz |= (c[:,2] > zl+2e-4) & (c[:,2] < zh-2e-4)
inside_fin &= finz
print("cell centres inside barrel solid:", inside_barrel.sum())
print("cell centres inside fin solids:  ", inside_fin.sum())

# -- cell 16 ------------------------------------------------------------------------
# Volume total matches the air volume almost exactly, yet some cell centres sit inside the barrel — le
pl = pv.Plotter(shape=(1,2), window_size=(1500,700))
sl = internal.slice(normal=(0,0,1), origin=(0,0,0.0285))  # through a fin
pl.subplot(0,0)
pl.add_mesh(sl, show_edges=True, color="white", lighting=False)
pl.view_vector((0,0,1), viewup=(0,1,0)); pl.add_title("z = 28.5 mm (through fin)")
sl2 = internal.slice(normal=(0,0,1), origin=(0,0,0.033))  # through a gap
pl.subplot(0,1)
pl.add_mesh(sl2, show_edges=True, color="white", lighting=False)
pl.view_vector((0,0,1), viewup=(0,1,0)); pl.add_title("z = 33 mm (through gap)")
pl.show()

# -- cell 17 ------------------------------------------------------------------------
# rigorous: cells with ALL points inside the solid regions (would prove a leak)
pts = internal.points
pr = np.hypot(pts[:,0], pts[:,1])

# cell -> point connectivity
conn = internal.cells  # vtk cell array for mixed types: [n, p0..pn-1, n, ...]
types = internal.celltypes
leak = 0
off = 0
for i in range(internal.n_cells):
    n = conn[off]
    ids = conn[off+1:off+1+n]
    off += n+1
    rr = pr[ids]; zz = pts[ids,2]
    if (rr.max() < 0.044) and (zz.min() > 0.0005) and (zz.max() < 0.1195):
        leak += 1
print("cells fully inside barrel solid:", leak)

# check boundary file patch types
print(open("constant/polyMesh/boundary").read()[:600])

# -- cell 18 ------------------------------------------------------------------------
txt = open("constant/polyMesh/boundary").read()
import re
for m in re.finditer(r'(\w+)\s*\{\s*type\s+(\w+);\s*nFaces\s+(\d+);', txt):
    print(m.group(1), "->", m.group(2), "nFaces:", m.group(3))

# -- cell 19 ------------------------------------------------------------------------
print(txt[txt.find("6\n"):])

# -- cell 20 ------------------------------------------------------------------------
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Message import Message_Gravity
import os

# Duct box faces, grouped as named patches
duct_box = bd.Box(2*DUCT_HALF, 2*DUCT_HALF, Z_DUCT_HI-Z_DUCT_LO,
                  align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)).moved(bd.Pos(0,0,Z_DUCT_LO))
inlet_faces, outlet_faces, wall_faces = [], [], []
for f in duct_box.faces():
    cz = f.center().Z
    if abs(cz - Z_DUCT_LO) < 1e-9: inlet_faces.append(f)
    elif abs(cz - Z_DUCT_HI) < 1e-9: outlet_faces.append(f)
    else: wall_faces.append(f)

patch_groups = {
    "barrel": barrel_faces, "fins": fin_faces, "head": head_faces,
    "inlet": inlet_faces, "outlet": outlet_faces, "walls": wall_faces,
}

# Mesh EVERYTHING together once so shared edges discretise identically
all_faces = [f for fl in patch_groups.values() for f in fl]
whole = bd.Compound(children=all_faces)
BRepMesh_IncrementalMesh(whole.wrapped, 1e-4, False, 0.3, True).Perform()

for name, fl in patch_groups.items():
    comp = bd.Compound(children=fl)
    w = StlAPI_Writer()
    ok = w.Write(comp.wrapped, f"constant/triSurface/{name}.stl")
    print(name, "faces:", len(fl), "written:", ok)
print(os.listdir("constant/triSurface"))

# -- cell 21 ------------------------------------------------------------------------
# Now verifying the welded union is closed (emulating the `union_closure` check) and that each STL is 
import pyvista as pv, numpy as np
meshes = {}
for name in patch_groups:
    m = pv.read(f"constant/triSurface/{name}.stl")
    meshes[name] = m
    print(f"{name:8s} tris: {m.n_cells:6d}  pts: {m.n_points:6d}  bounds z: "
          f"{m.bounds[4]:.3f}..{m.bounds[5]:.3f}")

# weld all into one surface, count free edges
import vtk
app = vtk.vtkAppendFilter()
for m in meshes.values():
    app.AddInputData(m)
app.Update()
ug = app.GetOutput()
gf = vtk.vtkGeometryFilter(); gf.SetInputData(ug); gf.Update()
clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(gf.GetOutputPort()); clean.Update()
fe = vtk.vtkFeatureEdges()
fe.SetInputData(clean.GetOutput())
fe.BoundaryEdgesOn(); fe.FeatureEdgesOff(); fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff()
fe.Update()
print("welded points:", clean.GetOutput().GetNumberOfPoints())
print("FREE EDGES of welded union:", fe.GetOutput().GetNumberOfCells())

# also weld only the cylinder solid (barrel+fins+head)
app2 = vtk.vtkAppendFilter()
for n in ("barrel","fins","head"):
    app2.AddInputData(meshes[n])
app2.Update()
gf2 = vtk.vtkGeometryFilter(); gf2.SetInputData(app2.GetOutput()); gf2.Update()
cl2 = vtk.vtkCleanPolyData(); cl2.SetInputConnection(gf2.GetOutputPort()); cl2.Update()
fe2 = vtk.vtkFeatureEdges(); fe2.SetInputData(cl2.GetOutput())
fe2.BoundaryEdgesOn(); fe2.FeatureEdgesOff(); fe2.NonManifoldEdgesOff(); fe2.ManifoldEdgesOff()
fe2.Update()
print("cylinder-only union free edges:", fe2.GetOutput().GetNumberOfCells())

# -- cell 22 ------------------------------------------------------------------------
txt = open("system/blockMeshDict").read()
txt = txt.replace("convertToMeters 1;", "scale 1;")
open("system/blockMeshDict","w").write(txt)
print("blockMeshDict: scale keyword updated")

r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("blockMesh rc:", r1.returncode)
r2 = subprocess.run(["snappyHexMesh", "-overwrite"], capture_output=True, text=True)
print("snappyHexMesh rc:", r2.returncode)
for line in r2.stdout.splitlines():
    if "cells:" in line or "error" in line.lower() and "faces in error" in line:
        pass
print(r2.stdout[-600:])
