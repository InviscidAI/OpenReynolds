"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the 8 mm cell width as the **clear opening across flats**, with 1 mm solid webs between a
import inspect, subprocess, sys
import build123d as bd
print('RegularPolygon:', inspect.signature(bd.RegularPolygon))
print('Face:', inspect.signature(bd.Face))
print('extrude:', inspect.signature(bd.extrude))
print('Box:', inspect.signature(bd.Box))

# -- cell 2 -------------------------------------------------------------------------
# The geometry cell failed before producing a fluid solid because this build123d version interprets th
import inspect, build123d as bd
print(inspect.signature(bd.Shape.fuse))

# -- cell 3 -------------------------------------------------------------------------
# The transformed extrusion returned an empty selection-like list, so I’ll avoid that operator and con
import inspect, build123d as bd
print(inspect.signature(bd.Wire.make_polygon))
print(inspect.signature(bd.Face))

# -- cell 4 -------------------------------------------------------------------------
# I’ll rebuild from scratch with explicit 3D wires. The corrected triangular lattice uses a 7.794 mm c
import math, numpy as np, build123d as bd
from build123d import Box, Face, Wire, extrude, Pos, Align
DUCT = 0.080
UPSTREAM = 0.040
CORE = 0.020
DOWNSTREAM = 0.080
CELL_AF = 0.008
WEB = 0.001
PITCH_AF = CELL_AF + WEB
R_OPEN = CELL_AF / math.sqrt(3.0)
R_PITCH = PITCH_AF / math.sqrt(3.0)
X0, X1, X2, X3 = 0.0, UPSTREAM, UPSTREAM + CORE, UPSTREAM + CORE + DOWNSTREAM
hex_vertices = [(0.0, R_OPEN*math.cos(math.radians(60*j)), R_OPEN*math.sin(math.radians(60*j))) for j in range(6)]
hex_face = Face(Wire.make_polygon(hex_vertices, close=True))
base_passage = extrude(hex_face, amount=CORE, dir=(1, 0, 0))
centres = []
for col in range(-8, 9):
    y = 1.5 * R_PITCH * col
    for row in range(-8, 9):
        z = PITCH_AF * (row + 0.5*(col % 2))
        verts = [(y + R_OPEN*math.cos(math.radians(60*j)), z + R_OPEN*math.sin(math.radians(60*j))) for j in range(6)]
        if max(abs(a) for p in verts for a in p) <= DUCT/2 + 1e-12:
            centres.append((y, z))
passages = [Pos(X1, y, z) * base_passage for y, z in centres]
upstream_air = Pos(X0, -DUCT/2, -DUCT/2) * Box(UPSTREAM, DUCT, DUCT, align=(Align.MIN, Align.MIN, Align.MIN))
downstream_air = Pos(X2, -DUCT/2, -DUCT/2) * Box(DOWNSTREAM, DUCT, DUCT, align=(Align.MIN, Align.MIN, Align.MIN))
fluid = upstream_air.fuse(*passages, downstream_air)
ys = [p[0] for p in centres]; zs = [p[1] for p in centres]
margin_y = DUCT/2 - (max(abs(y) for y in ys) + R_OPEN)
margin_z = DUCT/2 - max(max(abs(z + R_OPEN*math.sin(math.radians(60*j))) for j in range(6)) for z in zs)
print(f'cells={len(centres)}, connected solids={len(fluid.solids())}')
print(f'clear across flats={CELL_AF:.6f} m (requested 0.008000)')
print(f'normal centre pitch={PITCH_AF:.6f} m; web={PITCH_AF-CELL_AF:.6f} m (requested 0.001000)')
print(f'core streamwise extent={X2-X1:.6f} m (requested 0.020000)')
print(f'upstream={X1-X0:.6f} m, downstream={X3-X2:.6f} m')
print(f'edge margins: y={margin_y:.6f} m, z={margin_z:.6f} m')
print(f'fluid volume={fluid.volume:.9e} m^3; faces={len(fluid.faces())}')

# -- cell 5 -------------------------------------------------------------------------
# The first solid is connected and dimensionally correct, but the centered lattice places a few hex ve
candidates=[]
for oy in np.linspace(0, 1.5*R_PITCH, 41, endpoint=False):
    for oz in np.linspace(0, PITCH_AF, 41, endpoint=False):
        cc=[]; margins=[]
        for col in range(-9,10):
            y=1.5*R_PITCH*col+oy
            for row in range(-9,10):
                z=PITCH_AF*(row+0.5*(col%2))+oz
                vv=[(y+R_OPEN*math.cos(math.radians(60*j)),z+R_OPEN*math.sin(math.radians(60*j))) for j in range(6)]
                m=DUCT/2-max(abs(a) for p in vv for a in p)
                if m>=-1e-12: cc.append((y,z)); margins.append(m)
        candidates.append((len(cc),min(margins),oy,oz,cc))
best=max(candidates,key=lambda q:(q[0],q[1]))
print(f'best cells={best[0]}, minimum vertex-to-edge clearance={best[1]:.6f} m, offset y={best[2]:.6f}, z={best[3]:.6f}')

# -- cell 6 -------------------------------------------------------------------------
# The absolute maximum count (85) still requires point contact. I’ll select the largest count with at 
safe=[q for q in candidates if q[1]>=0.00025]
best_safe=max(safe,key=lambda q:(q[0],q[1]))
print(f'selected cells={best_safe[0]}, minimum clearance={best_safe[1]:.6f} m, offset y={best_safe[2]:.6f}, z={best_safe[3]:.6f}')

# -- cell 7 -------------------------------------------------------------------------
# A half-row vertical shift gives 72 complete passages and a 2.195 mm minimum frame margin, avoiding a
LATTICE_Y_OFFSET = best_safe[2]
LATTICE_Z_OFFSET = best_safe[3]
centres = best_safe[4]
passages = [Pos(X1, y, z) * base_passage for y, z in centres]
fluid = upstream_air.fuse(*passages, downstream_air)
actual_min_margin = min(DUCT/2-max(abs(a) for j in range(6) for a in (y+R_OPEN*math.cos(math.radians(60*j)),z+R_OPEN*math.sin(math.radians(60*j)))) for y,z in centres)
print(f'final passages={len(centres)}, connected solids={len(fluid.solids())}, faces={len(fluid.faces())}')
print(f'minimum core-frame margin={actual_min_margin:.6f} m')
print(f'fluid volume={fluid.volume:.9e} m^3')

# -- cell 8 -------------------------------------------------------------------------
# The final coarse CAD volume is one connected solid. I’ll now tessellate and render it with an obliqu
import pyvista as pv
from build123d import export_stl
export_stl(fluid, 'fluid_preview.stl', tolerance=2.5e-4, angular_tolerance=0.15)
pv.OFF_SCREEN=True
surf=pv.read('fluid_preview.stl')
cut=surf.clip(normal=(0,-1,1), origin=(0.070,0,0), invert=False)
pl=pv.Plotter(off_screen=True, window_size=(1000,700))
pl.add_mesh(cut, color='#79bce8', show_edges=True, edge_color='#24506c', line_width=0.5)
pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.25)
pl.add_text('Air volume cutaway: inlet left, outlet right', font_size=12)
pl.show(screenshot='coarse_geometry.png')
print(surf.n_cells, surf.bounds)

# -- cell 9 -------------------------------------------------------------------------
# The cutaway confirms the intended topology: open plenums connected only through the hexagonal passag
tol=1e-8
all_faces=list(fluid.faces())
inlet_faces=[]; outlet_faces=[]; duct_faces=[]; honeycomb_faces=[]
for f in all_faces:
    verts=[v.to_tuple() for v in f.vertices()]
    xs=[p[0] for p in verts]; ys_f=[p[1] for p in verts]; zs_f=[p[2] for p in verts]
    if max(abs(x-X0) for x in xs)<tol:
        inlet_faces.append(f)
    elif max(abs(x-X3) for x in xs)<tol:
        outlet_faces.append(f)
    elif (max(abs(y-DUCT/2) for y in ys_f)<tol or max(abs(y+DUCT/2) for y in ys_f)<tol or
          max(abs(z-DUCT/2) for z in zs_f)<tol or max(abs(z+DUCT/2) for z in zs_f)<tol):
        duct_faces.append(f)
    else:
        honeycomb_faces.append(f)
def area(fs): return sum(f.area for f in fs)
print('face counts:', {k:len(v) for k,v in [('inlet',inlet_faces),('outlet',outlet_faces),('duct_walls',duct_faces),('honeycomb_walls',honeycomb_faces)]})
print('areas m2:', {k:area(v) for k,v in [('inlet',inlet_faces),('outlet',outlet_faces),('duct_walls',duct_faces),('honeycomb_walls',honeycomb_faces)]})
print('assigned total=',sum(map(len,[inlet_faces,outlet_faces,duct_faces,honeycomb_faces])),'of',len(all_faces))

# -- cell 10 ------------------------------------------------------------------------
# All 444 faces are assigned exactly once, with inlet and outlet areas each matching 80×80 mm. I’ll ex
import sys
sys.path.insert(0,'.reference')
from cad_export import export_patches
export_patches(fluid, {'inlet':inlet_faces, 'outlet':outlet_faces, 'duct_walls':duct_faces, 'honeycomb_walls':honeycomb_faces}, tolerance=2.5e-4)

# -- cell 11 ------------------------------------------------------------------------
# The conformal surface is closed and clean, so I’ll immediately create a coarse snappyHexMesh setup: 
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/blockMeshDict').write_text(r'''FoamFile { format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ((-0.004 -0.044 -0.044) (0.144 -0.044 -0.044) (0.144 0.044 -0.044) (-0.004 0.044 -0.044)
          (-0.004 -0.044 0.044) (0.144 -0.044 0.044) (0.144 0.044 0.044) (-0.004 0.044 0.044));
blocks (hex (0 1 2 3 4 5 6 7) (37 22 22) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
''')
Path('system/snappyHexMeshDict').write_text(r'''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry {
 inlet.stl { type triSurfaceMesh; name inlet; }
 outlet.stl { type triSurfaceMesh; name outlet; }
 duct_walls.stl { type triSurfaceMesh; name duct_walls; }
 honeycomb_walls.stl { type triSurfaceMesh; name honeycomb_walls; }
}
castellatedMeshControls {
 maxLocalCells 200000; maxGlobalCells 400000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2;
 features ();
 refinementSurfaces {
  inlet { level (1 1); patchInfo { type patch; } }
  outlet { level (1 1); patchInfo { type patch; } }
  duct_walls { level (1 1); patchInfo { type wall; } }
  honeycomb_walls { level (1 1); patchInfo { type wall; } }
 }
 resolveFeatureAngle 25;
 refinementRegions {}
 locationInMesh (0.020 0 0);
 allowFreeStandingZoneFaces true;
}
snapControls {
 nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls { relativeSizes true; layers {}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #include "meshQualityDict" }
debug 0; mergeTolerance 1e-6;
''')
Path('system/meshQualityDict').write_text(r'''maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }''')
print('wrote blockMesh and snappyHexMesh dictionaries')

# -- cell 12 ------------------------------------------------------------------------
# blockMesh failed before snappy started. I’ll inspect its short log and correct only the reported dic
print(Path('log.blockMesh').read_text()[-4000:])

# -- cell 13 ------------------------------------------------------------------------
# The mesh dictionaries are valid so far; the case simply lacks the mandatory `controlDict`. I’ll add 
Path('system/controlDict').write_text(r'''FoamFile { format ascii; class dictionary; object controlDict; }
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0; writeFormat ascii; writePrecision 8; writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;
''')
with open('log.blockMesh','w') as f: r1=subprocess.run(['blockMesh'],stdout=f,stderr=subprocess.STDOUT)
print('blockMesh return',r1.returncode)
if r1.returncode==0:
    with open('log.snappyHexMesh','w') as f: r2=subprocess.run(['snappyHexMesh','-overwrite'],stdout=f,stderr=subprocess.STDOUT)
    print('snappy return',r2.returncode)
    print(Path('log.snappyHexMesh').read_text()[-1800:])
else: print(Path('log.blockMesh').read_text()[-1800:])

# -- cell 14 ------------------------------------------------------------------------
# This OpenFOAM build also insists on generic `fvSchemes` and `fvSolution` files even for snappy. I’ll
Path('system/fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
with open('log.snappyHexMesh','w') as f: r2=subprocess.run(['snappyHexMesh','-overwrite'],stdout=f,stderr=subprocess.STDOUT)
print('snappy return',r2.returncode)
print(Path('log.snappyHexMesh').read_text()[-2200:])
