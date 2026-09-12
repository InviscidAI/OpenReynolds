"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, numpy as np
from build123d import *

MM = 1e-3
L, W, H = 60*MM, 40*MM, 8*MM      # plate
CW, CD = 4*MM, 5*MM               # channel width, depth
WEB = 3*MM                        # metal between passes
PITCH = CW + WEB                  # 7 mm
Y0 = (W - (3*CW + 2*WEB))/2       # first pass lower edge -> centred in plate
ZB, ZT = H - CD, H                # channel z range: 3..8 mm
XR, XL = 56*MM, 4*MM              # right/left extents of the U-turn legs

yc = [Y0 + CW/2 + i*PITCH for i in range(3)]   # pass centrelines

def rect(x0, x1, y0, y1):
    return Pos((x0+x1)/2, (y0+y1)/2, 0) * Rectangle(x1-x0, y1-y0)

foot = (rect(0*MM, XR, yc[0]-CW/2, yc[0]+CW/2)          # pass 1 (open at x=0)
        + rect(XR-CW, XR, yc[0]-CW/2, yc[1]+CW/2)        # right U-turn
        + rect(XL, XR, yc[1]-CW/2, yc[1]+CW/2)           # pass 2
        + rect(XL, XL+CW, yc[1]-CW/2, yc[2]+CW/2)        # left U-turn
        + rect(XL, L, yc[2]-CW/2, yc[2]+CW/2))           # pass 3 (open at x=L)

fluid = Pos(0, 0, ZB) * extrude(foot, CD)
print("pass centres y (mm):", [round(v/MM,3) for v in yc])
print("web gap (mm):", round((yc[1]-yc[0]-CW)/MM,3))
print("volume (mm^3):", round(fluid.volume/MM**3,1), " bbox:", fluid.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Inlet/outlet confirmed: 20 mm² each, at (0, 13, 5.5) and (60, 27, 5.5) mm. Redo the export with the 
import pyvista as pv
from build123d import export_stl, Compound
faces = fluid.faces()
inlet  = [f for f in faces if abs(f.center().X-0)<1e-9 and abs(f.normal_at(f.center()).X)>0.99]
outlet = [f for f in faces if abs(f.center().X-L)<1e-9 and abs(f.normal_at(f.center()).X)>0.99]
walls  = [f for f in faces if f not in inlet+outlet]
print("faces:", len(inlet), len(outlet), len(walls),
      "| inlet area mm2", round(inlet[0].area/MM**2,3), "outlet", round(outlet[0].area/MM**2,3),
      "| centres", [round(c/MM,2) for c in inlet[0].center()], [round(c/MM,2) for c in outlet[0].center()])
export_stl(fluid, "fluid_all.stl", tolerance=1e-5)
m = pv.read("fluid_all.stl")
p = pv.Plotter(off_screen=True, window_size=(900,650))
p.add_mesh(m, color="steelblue", show_edges=True)
p.add_mesh(pv.Box((0,L,0,W,0,H)), style="wireframe", color="grey")
p.camera_position = "iso"; p.screenshot("shape.png")

# -- cell 3 -------------------------------------------------------------------------
# Rendering fell back to a bad X connection (white image). Switch VTK to its OSMesa window class and r
import os
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
import pyvista as pv
m = pv.read("fluid_all.stl")
p = pv.Plotter(off_screen=True, window_size=(900,650))
p.add_mesh(m, color="steelblue", show_edges=True)
p.add_mesh(pv.Box((0,L,0,W,0,H)), style="wireframe", color="grey")
p.camera_position = "iso"
p.screenshot("shape.png")
print("written", os.path.getsize("shape.png"))

# -- cell 4 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("shape.png"))

# -- cell 5 -------------------------------------------------------------------------
# Shape is right: three passes, U-turns alternating ends, open ends at x=0 and x=60. Now the case skel
import os, textwrap
from build123d import Compound, export_stl
for d in ["system","constant/triSurface","constant/polyMesh"]:
    os.makedirs(d, exist_ok=True)
export_stl(Compound(children=inlet),  "constant/triSurface/inlet.stl",  tolerance=1e-5)
export_stl(Compound(children=outlet), "constant/triSurface/outlet.stl", tolerance=1e-5)
export_stl(Compound(children=walls),  "constant/triSurface/walls.stl",  tolerance=1e-5)

def foam(cls, obj, name):
    return ("FoamFile\n{ version 2.0; format ascii; class %s; object %s; }\n\n" % (cls, name)) + obj

open("system/controlDict","w").write(foam("dictionary","""application blockMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii;
writePrecision 8; runTimeModifiable true;
""","controlDict"))
open("system/fvSchemes","w").write(foam("dictionary","ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n","fvSchemes"))
open("system/fvSolution","w").write(foam("dictionary","solvers{}\n","fvSolution"))

NX, NY, NZ = 60, 40, 8   # 1 mm background cells, planes at integer mm
open("system/blockMeshDict","w").write(foam("dictionary",f"""scale 1;
vertices
(
 (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
 (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""","blockMeshDict"))
import subprocess
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 6 -------------------------------------------------------------------------
# Snappy dict: level 0 (base cells already match feature sizes), implicit feature snapping, no layers.
snappy = """
castellatedMesh true; snap true; addLayers false;

geometry
{
    inlet.stl  { type triSurfaceMesh; name inlet; }
    outlet.stl { type triSurfaceMesh; name outlet; }
    walls.stl  { type triSurfaceMesh; name walls; }
}

castellatedMeshControls
{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 0;
    nCellsBetweenLevels 1; maxLoadUnbalance 0.1; resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces
    {
        inlet  { level (0 0); patchInfo { type patch; } }
        outlet { level (0 0); patchInfo { type patch; } }
        walls  { level (0 0); patchInfo { type wall;  } }
    }
    refinementRegions {}
    locationInMesh (0.0305 0.0205 0.0055);
}

snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3;
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02;
    minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1;
    nSmoothScale 4; errorReduction 0.75;
}

writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(foam("dictionary",snappy,"snappyHexMeshDict"))
import subprocess
r = subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 7 -------------------------------------------------------------------------
# Exactly 3400 cells = 3400 mm³ at 1 mm³ each — the castellated mesh is the channel exactly. Check it 
import subprocess, re
r = subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[-1800:])

# -- cell 8 -------------------------------------------------------------------------
# `bg` picked up 680 faces = the 680 mm² open top of the milled channel (the cover face) — that's a wa
import subprocess
open("system/snappyHexMeshDict","w").write(foam("dictionary",snappy.replace("snap true;","snap false;"),"snappyHexMeshDict"))
open("system/createPatchDict","w").write(foam("dictionary","""pointSync false;
patches
(
    { name walls; patchInfo { type wall; } constructFrom patches; patches (walls bg); }
);
""","createPatchDict"))
for cmd in (["blockMesh"],["snappyHexMesh","-overwrite"],["createPatch","-overwrite"]):
    r = subprocess.run(cmd,capture_output=True,text=True); print(cmd[0], r.returncode)
r = subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Checking patch topology"):r.stdout.find("Checking faceZone")])
print(r.stdout[-900:])

# -- cell 9 -------------------------------------------------------------------------
# Coarse version is exact and orthogonal. Now the production refinement: 0.25 mm cells (16×20 cells ac
import subprocess
DX = 0.25*MM
NX, NY, NZ = int(round(L/DX)), int(round(W/DX)), int(round(H/DX))
open("system/blockMeshDict","w").write(foam("dictionary",f"""scale 1;
vertices
(
 (0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0)
 (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({NX} {NY} {NZ}) simpleGrading (1 1 1) );
edges ();
boundary ( bg {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3)); }} );
""","blockMeshDict"))
open("system/snappyHexMeshDict","w").write(foam("dictionary",
    snappy.replace("snap true;","snap false;").replace("(0.0305 0.0205 0.0055)","(0.0303 0.0203 0.0053)"),
    "snappyHexMeshDict"))
for cmd in (["blockMesh"],["snappyHexMesh","-overwrite"],["createPatch","-overwrite"]):
    r = subprocess.run(cmd,capture_output=True,text=True); print(cmd[0], r.returncode)
r = subprocess.run(["checkMesh","-constant"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Checking patch topology"):r.stdout.find("Checking faceZone")])
print(r.stdout[-800:])

# -- cell 10 ------------------------------------------------------------------------
import numpy as np, pyvista as pv
mb = pv.read("VTK/t4_0.vtm")
vol = mb[0].combine() if hasattr(mb[0],'combine') else mb[0]
print(vol.n_cells, vol.bounds)
surf = vol.extract_surface()
p = pv.Plotter(off_screen=True, window_size=(950,700))
p.add_mesh(surf, color="lightsteelblue", show_edges=True, line_width=0.3)
p.add_mesh(pv.Box((0,L,0,W,0,H)), style="wireframe", color="grey")
p.camera_position = [(0.09,-0.07,0.07),(0.03,0.02,0.005),(0,0,1)]
p.screenshot("mesh.png")
from IPython.display import Image, display; display(Image("mesh.png"))

# -- cell 11 ------------------------------------------------------------------------
# Mesh matches the serpentine. Final numeric audit taken from the mesh cell centres, not from the desi
import numpy as np
c = vol.cell_centers().points
sl = c[np.abs(c[:,0]-0.030) < 0.0002]           # cross-section at x = 30 mm
ys = np.sort(np.unique(np.round(sl[:,1]/MM,4)))
groups, cur = [], [ys[0]]
for a,b in zip(ys, ys[1:]):
    (cur.append(b) if b-a < 0.3 else (groups.append(cur), cur.__setitem__(slice(None),[b])))
groups.append(cur)
print("passes at x=30mm (y span mm, incl. half-cell):")
for g in groups: print("   %.3f .. %.3f  width %.3f" % (g[0]-0.125, g[-1]+0.125, g[-1]-g[0]+0.25))
print("webs (mm):", [round(groups[i+1][0]-0.125-(groups[i][-1]+0.125),3) for i in range(len(groups)-1)])
zs = np.sort(np.unique(np.round(sl[:,2]/MM,4)))
print("depth z: %.3f .. %.3f = %.3f mm (plate H=8)" % (zs[0]-0.125, zs[-1]+0.125, zs[-1]-zs[0]+0.25))
print("plate footprint used: x 0..%.1f, y %.1f..%.1f of 60x40 plate" % (c[:,0].max()/MM+0.125, c[:,1].min()/MM-0.125, c[:,1].max()/MM+0.125))
print("volume from mesh: %.1f mm^3 ; centreline length approx %.1f mm" % (vol.volume/MM**3, vol.volume/MM**3/20))

# -- cell 12 ------------------------------------------------------------------------
# Grouping loop was buggy (aliased list). Redo it cleanly.
import numpy as np
groups, cur = [], [ys[0]]
for a, b in zip(ys, ys[1:]):
    if b - a < 0.3:
        cur = cur + [b]
    else:
        groups.append(cur); cur = [b]
groups.append(cur)
spans = [(g[0]-0.125, g[-1]+0.125) for g in groups]
for lo, hi in spans: print("pass y %.3f .. %.3f mm, width %.3f" % (lo, hi, hi-lo))
print("webs (mm):", [round(spans[i+1][0]-spans[i][1],3) for i in range(len(spans)-1)])
print("n passes at x=30mm:", len(spans))
