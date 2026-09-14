"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os
print(bd.__version__ if hasattr(bd,'__version__') else 'ok', os.getcwd(), os.listdir('.'))

# -- cell 2 -------------------------------------------------------------------------
import build123d as bd
from build123d import Pos

# --- parameters (mm) ---  plate given: 60 x 40 x 8; channel 4 wide, 5 deep, 3 mm between passes
PL, PW, PT = 60.0, 40.0, 8.0
CW, CD, RIB = 4.0, 5.0, 3.0
PITCH = CW + RIB                      # 7 mm pass-to-pass centre distance
Y0 = (PW - (3*CW + 2*RIB))/2.0        # 11.0 -> three passes centred across the 40 mm width
ZTOP, ZBOT = PT, PT - CD              # channel milled from the top face, 5 mm deep

def blk(x0,x1,y0,y1,z0=ZBOT,z1=ZTOP):
    return Pos((x0+x1)/2,(y0+y1)/2,(z0+z1)/2) * bd.Box(x1-x0, y1-y0, z1-z0)

yA0,yA1 = Y0, Y0+CW              # pass 1: 11..15
yB0,yB1 = Y0+PITCH, Y0+PITCH+CW  # pass 2: 18..22
yC0,yC1 = Y0+2*PITCH, Y0+2*PITCH+CW  # pass 3: 25..29

pass1 = blk(0.0, PL-2.0, yA0, yA1)          # open at x=0 (inlet)
connA = blk(PL-6.0, PL-2.0, yA0, yB1)       # U-bend at the right-hand end
pass2 = blk(2.0, PL-2.0, yB0, yB1)
connB = blk(2.0, 6.0, yB0, yC1)             # U-bend at the left-hand end
pass3 = blk(2.0, PL, yC0, yC1)              # open at x=60 (outlet)

fluid_mm = pass1 + connA + pass2 + connB + pass3
print(fluid_mm.bounding_box(), fluid_mm.volume)

# -- cell 3 -------------------------------------------------------------------------
fluid = bd.scale(fluid_mm, 0.001)      # mm -> metres, once, here
print(fluid.bounding_box(), fluid.volume)
faces = fluid.faces()
inlet  = [f for f in faces if abs(f.center().X - 0.0)   < 1e-9]
outlet = [f for f in faces if abs(f.center().X - 0.060) < 1e-9]
print(len(faces), len(inlet), len(outlet), [f.area for f in inlet+outlet])

# -- cell 4 -------------------------------------------------------------------------
import os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)
walls = [f for f in faces if f not in inlet + outlet]
for name, fl in [("inlet", inlet), ("outlet", outlet), ("walls", walls)]:
    bd.export_stl(bd.Compound(children=[bd.Solid()]) if False else bd.Compound(fl),
                  f"constant/triSurface/{name}.stl", tolerance=1e-5, angular_tolerance=0.1)
    print(name, len(fl), sum(f.area for f in fl))
print(os.listdir("constant/triSurface"))

# -- cell 5 -------------------------------------------------------------------------
import os, textwrap
os.makedirs("system", exist_ok=True)
head = """FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"""
# background block: fluid bbox padded, 1 mm base cells (coarse first)
x0,x1,y0,y1,z0,z1 = -0.002,0.062, 0.009,0.031, 0.001,0.010
nx,ny,nz = 64,22,9
open("system/blockMeshDict","w").write(head%"blockMeshDict"+f"""
scale 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3) ); }} );
mergePatchPairs ();
""")
open("system/controlDict","w").write(head%"controlDict"+"""
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 10;
runTimeModifiable true;
""")
open("system/fvSchemes","w").write(head%"fvSchemes"+"gradSchemes{}\ndivSchemes{}\nlaplacianSchemes{}\nddtSchemes{default steadyState;}\n")
open("system/fvSolution","w").write(head%"fvSolution"+"solvers{}\n")
import subprocess
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-500:])

# -- cell 6 -------------------------------------------------------------------------
open("system/surfaceFeatureExtractDict","w").write(head%"surfaceFeatureExtractDict"+"""
inlet.stl  { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
outlet.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
walls.stl  { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
""")
r=subprocess.run(["surfaceFeatureExtract"],capture_output=True,text=True); print(r.stdout[-800:], r.returncode)

# -- cell 7 -------------------------------------------------------------------------
open("system/snappyHexMeshDict","w").write(head%"snappyHexMeshDict"+"""
castellatedMesh true; snap true; addLayers false;
geometry {
  inlet.stl  { type triSurfaceMesh; name inlet; }
  outlet.stl { type triSurfaceMesh; name outlet; }
  walls.stl  { type triSurfaceMesh; name walls; }
}
castellatedMeshControls {
  maxLocalCells 1000000; maxGlobalCells 5000000; minRefinementCells 0; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ( { file "inlet.eMesh"; level 0; } { file "outlet.eMesh"; level 0; } { file "walls.eMesh"; level 0; } );
  refinementSurfaces {
    inlet  { level (0 0); patchInfo { type patch; } }
    outlet { level (0 0); patchInfo { type patch; } }
    walls  { level (0 0); patchInfo { type wall;  } }
  }
  refinementRegions {}
  locationInMesh (0.030 0.013 0.0055);
}
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }
addLayersControls { relativeSizes true; layers{} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
  nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
  minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
mergeTolerance 1e-6; writeFlags ();
""")
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 8 -------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); print(r.stdout[-2500:])

# -- cell 9 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pv.OFF_SCREEN=True
m = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam","w").close()
mesh = pv.OpenFOAMReader("case.foam").read()
surf = mesh.combine().extract_surface()
p=pv.Plotter(off_screen=True,window_size=(1000,700))
p.add_mesh(surf,color="lightsteelblue",show_edges=True,line_width=0.5)
p.add_axes(); p.camera_position='xy'; p.show(screenshot="coarse.png")

# -- cell 10 ------------------------------------------------------------------------
# refine: 0.5 mm base cells (8 cells across the 4 mm width, 10 through the 5 mm depth)
nx,ny,nz = 128,44,18
open("system/blockMeshDict","w").write(head%"blockMeshDict"+f"""
scale 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3) ); }} );
mergePatchPairs ();
""")
print(subprocess.run(["blockMesh"],capture_output=True,text=True).returncode)
print(subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True).stdout[-400:])

# -- cell 11 ------------------------------------------------------------------------
r=subprocess.run(["checkMesh"],capture_output=True,text=True); s=r.stdout
print("\n".join([l for l in s.splitlines() if any(k in l for k in ["cells:","hexahedra","Total volume","non-orthogonality","Max skewness","inlet","outlet","walls","Mesh OK","***","aspect"])]))

# -- cell 12 ------------------------------------------------------------------------
mesh = pv.OpenFOAMReader("case.foam").read()
b = mesh.combine()
print("cells",b.n_cells,"bbox",np.round(np.array(b.bounds)*1000,3))
p=pv.Plotter(off_screen=True,window_size=(1000,600))
p.add_mesh(b.extract_surface(),color="lightsteelblue",show_edges=True,line_width=0.3)
p.camera_position=[(0.09,-0.06,0.07),(0.03,0.02,0.005),(0,0,1)]
p.show(screenshot="final.png")
