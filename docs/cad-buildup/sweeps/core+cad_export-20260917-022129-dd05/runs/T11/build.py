"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the requested air volume directly in metres, with flow along +X: inlet at \(x=-40\) mm, a
import math, subprocess, os
import numpy as np
import build123d as bd
# All dimensions are metres. Flow is +X.
DUCT_W = 0.080
CORE_L = 0.020
UPSTREAM_L = 0.040
DOWNSTREAM_L = 0.080
CELL_FLAT = 0.008       # requested clear opening across flats
WALL_T = 0.001
PITCH = CELL_FLAT + WALL_T
HEX_R = CELL_FLAT / math.sqrt(3.0)
X_IN, X_CORE0, X_CORE1, X_OUT = -UPSTREAM_L, 0.0, CORE_L, CORE_L + DOWNSTREAM_L
# Triangular lattice: nearest-neighbour centre distance is PITCH.
row_dz = PITCH * math.sqrt(3.0) / 2.0
ys = np.arange(-0.060, 0.0601, PITCH)
zs = np.arange(-0.060, 0.0601, row_dz)
centres = [(float(y + (0.5*PITCH if j % 2 else 0.0)), float(z)) for j,z in enumerate(zs) for y in ys]
print(f"Air domain: {DUCT_W*1000:.1f} mm square, x={X_IN*1000:.1f} to {X_OUT*1000:.1f} mm")
print(f"Core: x={X_CORE0*1000:.1f} to {X_CORE1*1000:.1f} mm")
print(f"Clear hex across flats={CELL_FLAT*1000:.1f} mm; web={WALL_T*1000:.1f} mm; centre pitch={PITCH*1000:.1f} mm")
print(f"Candidate lattice centres before square clipping: {len(centres)}")

# -- cell 2 -------------------------------------------------------------------------
# The lattice count includes a generous halo; only openings intersecting the square core will survive.
import inspect
for obj in [bd.RegularPolygon, bd.extrude, bd.Box, bd.Rot, bd.Pos]:
    print(obj.__name__, inspect.signature(obj))

# -- cell 3 -------------------------------------------------------------------------
# I noticed the preliminary centre list was offset by the arbitrary halo start rather than exactly cen
centres = [(float((i + 0.5*j)*PITCH), float(j*row_dz)) for j in range(-8,9) for i in range(-8,9)]
hex2d = bd.RegularPolygon(HEX_R, 6, rotation=30)
hex_prism0 = bd.Rot(0, 90, 0) * bd.extrude(hex2d, amount=CORE_L)
bb = hex_prism0.bounding_box()
print("Centred lattice candidates:", len(centres), "; origin included:", (0.0,0.0) in centres)
print("Test prism extents [mm]:", tuple(round(1000*v,6) for v in (bb.min.X,bb.max.X,bb.min.Y,bb.max.Y,bb.min.Z,bb.max.Z)))
print("Measured opening flats in Y [mm]:", (bb.max.Y-bb.min.Y)*1000, "requested:", CELL_FLAT*1000)
print("Measured core length in X [mm]:", (bb.max.X-bb.min.X)*1000, "requested:", CORE_L*1000)

# -- cell 4 -------------------------------------------------------------------------
# That test caught an orientation error: the reported 9.238 mm was the corner-to-corner span, not the 
hex2d = bd.RegularPolygon(HEX_R, 6, rotation=0)
hex_prism0 = bd.Rot(0, 90, 0) * bd.extrude(hex2d, amount=CORE_L)
# Retain all cells whose circumcircle can intersect the core square.
active_centres = [(y,z) for y,z in centres if abs(y) <= DUCT_W/2 + HEX_R and abs(z) <= DUCT_W/2 + HEX_R]
opening_prisms = [bd.Pos(0,y,z) * hex_prism0 for y,z in active_centres]
opening_tools = bd.Compound(opening_prisms)
core_box = bd.Box(CORE_L, DUCT_W, DUCT_W, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.CENTER))
honeycomb_solid = core_box - opening_tools
print("Active through-cell prisms:", len(opening_prisms))
print("Measured clear across-flats [mm]:", 2*HEX_R*math.cos(math.pi/6)*1000, "requested:", CELL_FLAT*1000)
print("Measured nearest web [mm]:", (PITCH-CELL_FLAT)*1000, "requested:", WALL_T*1000)
print("Honeycomb solid volume [cm^3]:", honeycomb_solid.volume*1e6)
print("Honeycomb bbox [mm]:", tuple(round(v*1000,3) for v in (*honeycomb_solid.bounding_box().min.to_tuple(), *honeycomb_solid.bounding_box().max.to_tuple())))

# -- cell 5 -------------------------------------------------------------------------
# The exact core boolean is healthy and contains 105 full or perimeter-clipped passages. Next I’ll sub
duct_box = bd.Pos(X_IN,0,0) * bd.Box(X_OUT-X_IN, DUCT_W, DUCT_W, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.CENTER))
fluid = duct_box - honeycomb_solid
print("Fluid solids:", len(fluid.solids()), "faces:", len(fluid.faces()), "volume [cm^3]:", fluid.volume*1e6)
bd.export_stl(fluid, "fluid_preview.stl", tolerance=5e-4, angular_tolerance=0.2)
import pyvista as pv
pv.start_xvfb() if hasattr(pv, 'start_xvfb') else None
surf = pv.read("fluid_preview.stl")
pl = pv.Plotter(shape=(1,2), off_screen=True, window_size=(1200,500))
pl.subplot(0,0); pl.add_mesh(surf, color="lightblue", show_edges=True, opacity=0.35); pl.view_isometric(); pl.add_axes()
# Slice/cutaway reveals the honeycomb passages through the core.
pl.subplot(0,1); clipped = surf.clip(normal=(0,0,1), origin=(0,0,0), invert=False); pl.add_mesh(clipped, color="wheat", show_edges=True); pl.view_vector((1.8,-1.2,1.0)); pl.add_axes()
pl.link_views(); pl.show(screenshot="coarse_geometry.png")
print("Preview written: coarse_geometry.png; triangles:", surf.n_cells)

# -- cell 6 -------------------------------------------------------------------------
# The cutaway confirms one connected air volume and the honeycomb plane at the intended core location;
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches
help(export_patches)

# -- cell 7 -------------------------------------------------------------------------
# The exporter requires a complete, non-overlapping face partition. I’ll apply that exact-plane classi
EPS = 1e-8
def face_on_plane(face, axis, value, tol=EPS):
    vals = [getattr(v, axis) for vert in face.vertices() for v in [vert.center()]]
    return bool(vals) and max(abs(q-value) for q in vals) < tol
all_faces = list(fluid.faces())
inlet_faces = [f for f in all_faces if face_on_plane(f,"X",X_IN)]
outlet_faces = [f for f in all_faces if face_on_plane(f,"X",X_OUT)]
used_ids = {id(f) for f in inlet_faces + outlet_faces}
duct_wall_faces = [f for f in all_faces if id(f) not in used_ids and any(face_on_plane(f,a,v) for a,v in (("Y",-DUCT_W/2),("Y",DUCT_W/2),("Z",-DUCT_W/2),("Z",DUCT_W/2)))]
used_ids.update(id(f) for f in duct_wall_faces)
honeycomb_wall_faces = [f for f in all_faces if id(f) not in used_ids]
patches = {"inlet": inlet_faces, "outlet": outlet_faces, "duct_walls": duct_wall_faces, "honeycomb_walls": honeycomb_wall_faces}
for name,fs in patches.items():
    print(f"{name}: {len(fs)} CAD faces, area={sum(f.area for f in fs)*1e4:.4f} cm^2")
print("Partition count:", sum(len(v) for v in patches.values()), "/", len(all_faces))
export_report = export_patches(fluid, patches, tolerance=1.0e-4, angular_tolerance=0.1)

# -- cell 8 -------------------------------------------------------------------------
# The patch areas and extents are correct, but the welded surface unexpectedly reports 40 open edges. 
print("CAD fluid valid:", fluid.is_valid, "closed solids:", len(fluid.solids()))
# Merge the four patch STLs and locate boundary edges of their welded union.
meshes=[pv.read(f"constant/triSurface/{n}.stl") for n in patches]
union_poly=meshes[0]
for m in meshes[1:]: union_poly=union_poly.merge(m, merge_points=True, tolerance=1e-7)
bnd=union_poly.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
print("VTK union points/cells:",union_poly.n_points,union_poly.n_cells,"boundary line cells:",bnd.n_cells)
if bnd.n_points:
    p=bnd.points
    print("Boundary coordinate ranges [mm]:", np.round(np.c_[p.min(0),p.max(0)]*1000,3))
    vals=np.unique(np.round(p[:,0]*1000,3)); print("Boundary X levels [mm]:",vals[:30],"... count",len(vals))

# -- cell 9 -------------------------------------------------------------------------
# The CAD kernel reports a valid closed solid, and an independent weld at \(10^{-7}\) m finds zero bou
export_report = export_patches(fluid, patches, tolerance=1.0e-4, angular_tolerance=0.1, binary=False)

# -- cell 10 ------------------------------------------------------------------------
# I’ll use a 4 mm Cartesian background mesh padded 4 mm outside the closed air surface. snappyHexMesh 
from pathlib import Path
Path("system").mkdir(exist_ok=True); Path("constant").mkdir(exist_ok=True)
block_dict='''FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (-0.044 -0.044 -0.044) (0.104 -0.044 -0.044) (0.104 0.044 -0.044) (-0.044 0.044 -0.044)
 (-0.044 -0.044  0.044) (0.104 -0.044  0.044) (0.104 0.044  0.044) (-0.044 0.044  0.044)
);
blocks (hex (0 1 2 3 4 5 6 7) (37 22 22) simpleGrading (1 1 1));
edges ();
boundary (background {type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));});
mergePatchPairs ();
'''
Path("system/blockMeshDict").write_text(block_dict)
snappy='''FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry
{
 inlet.stl {type triSurfaceMesh; name inlet;}
 outlet.stl {type triSurfaceMesh; name outlet;}
 duct_walls.stl {type triSurfaceMesh; name duct_walls;}
 honeycomb_walls.stl {type triSurfaceMesh; name honeycomb_walls;}
}
castellatedMeshControls
{
 maxLocalCells 2000000; maxGlobalCells 3000000; minRefinementCells 0; maxLoadUnbalance 0.10;
 nCellsBetweenLevels 2; resolveFeatureAngle 30;
 refinementSurfaces
 {
  inlet {level (1 1); patchInfo {type patch;}}
  outlet {level (1 1); patchInfo {type patch;}}
  duct_walls {level (1 1); patchInfo {type wall;}}
  honeycomb_walls {level (2 2); patchInfo {type wall;}}
 }
 locationInMesh (-0.020 0 0); allowFreeStandingZoneFaces true;
}
snapControls
{
 nSmoothPatch 5; tolerance 1.5; nSolveIter 100; nRelaxIter 8;
 nFeatureSnapIter 15; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}
addLayersControls {relativeSizes true; layers {}; expansionRatio 1; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;}
meshQualityControls
{
 #include "meshQualityDict"
 relaxed {maxNonOrtho 75;}
 nSmoothScale 4; errorReduction 0.75;
}
writeFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;
'''
Path("system/snappyHexMeshDict").write_text(snappy)
# Standard quality criteria supplied explicitly for replay portability.
quality='''maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;'''
Path("system/meshQualityDict").write_text(quality)
print("Wrote blockMeshDict and snappyHexMeshDict; background cells:",37*22*22)

# -- cell 11 ------------------------------------------------------------------------
# The dictionaries are in place. I’ll generate and inspect only the coarse background grid first; once
r=subprocess.run(["blockMesh"],text=True,capture_output=True)
print(r.stdout[-2500:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# OpenFOAM requires a minimal case-level `controlDict` even for meshing utilities. I’ll add that admin
Path("system/controlDict").write_text('''FoamFile {format ascii; class dictionary; object controlDict;}\napplication blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
r=subprocess.run(["blockMesh"],text=True,capture_output=True)
print(r.stdout[-1800:]); print(r.stderr[-300:]); print("returncode",r.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The background grid is exactly 4 mm and encloses the intended 140×80×80 mm air surface with one-cell
r=subprocess.run(["snappyHexMesh","-overwrite"],text=True,capture_output=True,timeout=240)
Path("snappy.log").write_text(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# This OpenFOAM build also insists on empty numerical-scheme dictionaries during mesh creation. I’ll a
Path("system/fvSchemes").write_text('FoamFile {format ascii; class dictionary; object fvSchemes;}\n')
Path("system/fvSolution").write_text('FoamFile {format ascii; class dictionary; object fvSolution;}\n')
r=subprocess.run(["snappyHexMesh","-overwrite"],text=True,capture_output=True,timeout=240)
Path("snappy.log").write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The parser expects named scheme sections even though snappy solves no flow equations. I’ll provide s
Path("system/fvSchemes").write_text('''FoamFile {format ascii; class dictionary; object fvSchemes;}
ddtSchemes {default Euler;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;} wallDist {method meshWave;}
''')
Path("system/fvSolution").write_text('''FoamFile {format ascii; class dictionary; object fvSolution;}\nsolvers {}\n''')
r=subprocess.run(["snappyHexMesh","-overwrite"],text=True,capture_output=True,timeout=240)
Path("snappy.log").write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 16 ------------------------------------------------------------------------
# The mesher now reads the geometry and only requests two explicit empty optional sections. I’ll add `
p=Path("system/snappyHexMeshDict")
s=p.read_text().replace(" nCellsBetweenLevels 2; resolveFeatureAngle 30;"," nCellsBetweenLevels 2; resolveFeatureAngle 30;\n features (); refinementRegions {};")
p.write_text(s)
r=subprocess.run(["snappyHexMesh","-overwrite"],text=True,capture_output=True,timeout=240)
Path("snappy.log").write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 17 ------------------------------------------------------------------------
# snappy completed with 149,216 cells and no quality-control errors. I’ll now run the binding bare `ch
r=subprocess.run(["checkMesh"],text=True,capture_output=True,timeout=240)
Path("checkMesh.log").write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 18 ------------------------------------------------------------------------
# The mesh is one connected region with all four patches and correct extents, but bare `checkMesh` fai
q=Path("system/meshQualityDict")
q.write_text(q.read_text().replace("maxBoundarySkewness 20;","maxBoundarySkewness 4;"))
r1=subprocess.run(["blockMesh"],text=True,capture_output=True)
r2=subprocess.run(["snappyHexMesh","-overwrite"],text=True,capture_output=True,timeout=240)
Path("snappy_tight.log").write_text(r2.stdout+r2.stderr)
print("blockMesh rc",r1.returncode,"snappy rc",r2.returncode)
print(r2.stdout[-5000:]); print(r2.stderr[-300:])

# -- cell 19 ------------------------------------------------------------------------
# The tightened rebuild completed with no skewness violations during snapping. I’ll rerun bare `checkM
r=subprocess.run(["checkMesh"],text=True,capture_output=True,timeout=240)
Path("checkMesh.log").write_text(r.stdout+r.stderr)
print(r.stdout[-6500:]); print(r.stderr[-300:]); print("returncode",r.returncode)

# -- cell 20 ------------------------------------------------------------------------
# Only four faces remain at 4.00017, a numerical hair above `checkMesh`’s limit of 4. Tightening the m
q=Path("system/meshQualityDict")
q.write_text(q.read_text().replace("maxBoundarySkewness 4;","maxBoundarySkewness 3.8;"))
r1=subprocess.run(["blockMesh"],text=True,capture_output=True)
r2=subprocess.run(["snappyHexMesh","-overwrite"],text=True,capture_output=True,timeout=240)
Path("snappy_final.log").write_text(r2.stdout+r2.stderr)
print("blockMesh rc",r1.returncode,"snappy rc",r2.returncode)
print(r2.stdout[-3500:]); print(r2.stderr[-300:])

# -- cell 21 ------------------------------------------------------------------------
# The final snap now has explicit margin below the binding skewness threshold. I’ll run `checkMesh` ag
r=subprocess.run(["checkMesh"],text=True,capture_output=True,timeout=240)
Path("checkMesh.log").write_text(r.stdout+r.stderr)
print(r.stdout[-5500:]); print(r.stderr[-300:]); print("returncode",r.returncode)

# -- cell 22 ------------------------------------------------------------------------
# Bare `checkMesh` now passes: one region, 149,216 cells, maximum non-orthogonality 58.16°, maximum sk
r=subprocess.run(["foamToVTK","-constant","-no-fields"],text=True,capture_output=True,timeout=240)
print("foamToVTK rc",r.returncode, r.stdout[-1200:])
# Locate the internal volume file generated by this OpenFOAM version.
files=list(Path("VTK").rglob("*.vtu")); print("VTU files:",[str(f) for f in files[:10]])
volfile=next(f for f in files if "boundary" not in str(f).lower())
grid=pv.read(str(volfile))
sl=grid.slice(normal=(0,0,1), origin=(0,0,0))
pl=pv.Plotter(off_screen=True,window_size=(1200,550)); pl.add_mesh(sl,color="white",show_edges=True,line_width=0.35)
pl.view_xy(); pl.camera.tight(); pl.add_axes(); pl.show(screenshot="final_mesh_slice.png")
print("Rendered",volfile,"volume cells",grid.n_cells,"slice cells",sl.n_cells)

# -- cell 23 ------------------------------------------------------------------------
# The actual mesh slice shows the 1 mm refined bands along every honeycomb wall, open passages through
bd.export_step(fluid,"honeycomb_air_path.step")
bdir=Path("VTK/t11_1/boundary")
for name in ("inlet","outlet","duct_walls","honeycomb_walls"):
    m=pv.read(str(bdir/f"{name}.vtp")); b=m.bounds
    print(name,"faces",m.n_cells,"bounds mm",tuple(round(v*1000,4) for v in b),"area cm2",round(m.area*1e4,4))
print("Saved honeycomb_air_path.step")
