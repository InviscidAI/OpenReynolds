"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll interpret “set into its floor” as the 10 mm-thick copper block occupying the lower 10 mm of the
import build123d as bd
# All dimensions are in metres.
L, W, H = 0.200, 0.030, 0.030
block_L, block_W, block_H = 0.040, 0.030, 0.010
block_x0 = (L - block_L) / 2

duct_volume = bd.Box(L, W, H)
copper = bd.Pos(block_x0, 0, 0) * bd.Box(block_L, block_W, block_H)
fluid = duct_volume - copper

print(f"Duct extent requested/measured: {L:.3f} / {duct_volume.bounding_box().size.X:.3f} m long, {W:.3f} / {duct_volume.bounding_box().size.Y:.3f} m wide, {H:.3f} / {duct_volume.bounding_box().size.Z:.3f} m high")
print(f"Copper requested/measured: {block_L:.3f} / {copper.bounding_box().size.X:.3f} m long, {block_W:.3f} / {copper.bounding_box().size.Y:.3f} m wide, {block_H:.3f} / {copper.bounding_box().size.Z:.3f} m thick")
print(f"Copper x-range: requested centered; measured {copper.bounding_box().min.X:.3f} to {copper.bounding_box().max.X:.3f} m")
print(f"Volumes: fluid={fluid.volume:.9e} m^3 (expected {L*W*H-block_L*block_W*block_H:.9e}), solid={copper.volume:.9e} m^3 (expected {block_L*block_W*block_H:.9e})")

# -- cell 2 -------------------------------------------------------------------------
# The corrected dimensions were right, but the diagnostic failed because build123d represents an empty
amin = (bd.Align.MIN, bd.Align.MIN, bd.Align.MIN)
duct_volume = bd.Box(L, W, H, align=amin)
copper = bd.Pos(block_x0, 0, 0) * bd.Box(block_L, block_W, block_H, align=amin)
fluid = duct_volume - copper
overlap = fluid.intersect(copper)
overlap_volume = 0.0 if overlap is None else overlap.volume
print(f"Duct bounds: x={duct_volume.bounding_box().min.X:.3f}..{duct_volume.bounding_box().max.X:.3f}, y={duct_volume.bounding_box().min.Y:.3f}..{duct_volume.bounding_box().max.Y:.3f}, z={duct_volume.bounding_box().min.Z:.3f}..{duct_volume.bounding_box().max.Z:.3f} m")
print(f"Copper x-range requested/measured: 0.080..0.120 / {copper.bounding_box().min.X:.3f}..{copper.bounding_box().max.X:.3f} m")
print(f"Fluid/solid overlap volume: {overlap_volume:.3e} m^3")

# -- cell 3 -------------------------------------------------------------------------
# Now I’ll export the two measured CAD volumes and render the fluid translucent so the full-width copp
from pathlib import Path
import pyvista as pv
geom_dir = Path("geometry"); geom_dir.mkdir(exist_ok=True)
bd.export_step(fluid, geom_dir / "fluid.step")
bd.export_step(copper, geom_dir / "copper.step")
bd.export_stl(fluid, geom_dir / "fluid_preview.stl", tolerance=2.5e-4)
bd.export_stl(copper, geom_dir / "copper_preview.stl", tolerance=2.5e-4)
p = pv.Plotter(off_screen=True, window_size=(1000, 600))
p.add_mesh(pv.read(geom_dir / "fluid_preview.stl"), color="lightskyblue", opacity=0.28, show_edges=True)
p.add_mesh(pv.read(geom_dir / "copper_preview.stl"), color="peru", opacity=1.0, show_edges=True)
p.add_axes(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot="geometry_preview.png")

# -- cell 4 -------------------------------------------------------------------------
# The preview matches the chosen shape. I’ll define the coarse conformal topology with 5 mm cells: 16/
case_system = Path("system"); case_system.mkdir(exist_ok=True)
xs, ys, zs = [0.0, 0.08, 0.12, 0.20], [0.0, 0.03], [0.0, 0.01, 0.03]
def vi(i,j,k): return i + 4*j + 8*k
vertices = [(x,y,z) for z in zs for y in ys for x in xs]
lines = ["FoamFile { format ascii; class dictionary; object blockMeshDict; }", "convertToMeters 1;", "vertices", "("]
lines += [f"    ({x:.6g} {y:.6g} {z:.6g})" for x,y,z in vertices]
lines += [");", "blocks", "("]
xcells, zcells = [16,8,16], [2,4]
for i in range(3):
    for k in range(2):
        vv = [vi(i,0,k),vi(i+1,0,k),vi(i+1,1,k),vi(i,1,k),vi(i,0,k+1),vi(i+1,0,k+1),vi(i+1,1,k+1),vi(i,1,k+1)]
        zone = "solid" if (i==1 and k==0) else "fluid"
        lines.append(f"    hex ({' '.join(map(str,vv))}) {zone} ({xcells[i]} 6 {zcells[k]}) simpleGrading (1 1 1)")
lines += [");", "edges ();", "boundary", "("]
inlet_faces = [[vi(0,0,k),vi(0,0,k+1),vi(0,1,k+1),vi(0,1,k)] for k in range(2)]
outlet_faces = [[vi(3,0,k),vi(3,1,k),vi(3,1,k+1),vi(3,0,k+1)] for k in range(2)]
wall_faces = []
for i in range(3):
    for k in range(2):
        wall_faces += [[vi(i,0,k),vi(i+1,0,k),vi(i+1,0,k+1),vi(i,0,k+1)], [vi(i,1,k),vi(i,1,k+1),vi(i+1,1,k+1),vi(i+1,1,k)]]
    wall_faces += [[vi(i,0,0),vi(i,1,0),vi(i+1,1,0),vi(i+1,0,0)], [vi(i,0,2),vi(i+1,0,2),vi(i+1,1,2),vi(i,1,2)]]
for name, ptype, faces in [("inlet","patch",inlet_faces),("outlet","patch",outlet_faces),("walls","wall",wall_faces)]:
    lines += [f"    {name}", "    {", f"        type {ptype};", "        faces", "        ("] + ["            ("+" ".join(map(str,f))+")" for f in faces] + ["        );", "    }"]
lines += [");", "mergePatchPairs ();" ]
(case_system / "blockMeshDict").write_text("\n".join(lines)+"\n")
print("Wrote system/blockMeshDict with", len(vertices), "vertices, 6 blocks, and", sum(xcells)*6*sum(zcells), "cells")

# -- cell 5 -------------------------------------------------------------------------
# `blockMesh` requires the case-level `controlDict` even though no solver will be run. I’ll add only t
control = """FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application blockMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
runTimeModifiable true;
"""
(case_system / "controlDict").write_text(control)
print("Wrote system/controlDict")

# -- cell 6 -------------------------------------------------------------------------
# I’ll re-emit the successful meshing step with its own `subprocess` import so the final build script 
import subprocess
bm = subprocess.run(["blockMesh"], text=True, capture_output=True)
print(bm.stdout[-3000:])
if bm.returncode != 0:
    print(bm.stderr)
    raise RuntimeError(f"blockMesh failed with code {bm.returncode}")

# -- cell 7 -------------------------------------------------------------------------
# This utility also initializes finite-volume infrastructure and therefore requests `fvSchemes`/`fvSol
(case_system / "fvSchemes").write_text("FoamFile { format ascii; class dictionary; object fvSchemes; }\n")
(case_system / "fvSolution").write_text("FoamFile { format ascii; class dictionary; object fvSolution; }\n")
print("Wrote utility-only empty fvSchemes and fvSolution dictionaries")

# -- cell 8 -------------------------------------------------------------------------
# OpenFOAM’s initialization expects the standard scheme subdictionaries even though the utility does n
schemes = """FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
"""
(case_system / "fvSchemes").write_text(schemes)
print("Added standard parser-complete fvSchemes entries")

# -- cell 9 -------------------------------------------------------------------------
# I’ll rerun the split with the parser-complete utility setup.
smr = subprocess.run(["splitMeshRegions", "-cellZones", "-overwrite"], text=True, capture_output=True)
print(smr.stdout[-5000:])
if smr.returncode != 0:
    print(smr.stderr)
    raise RuntimeError(f"splitMeshRegions failed with code {smr.returncode}")

# -- cell 10 ------------------------------------------------------------------------
# `splitMeshRegions` created both region meshes but did not write `constant/regionProperties`, which `
constant_dir = Path("constant")
region_props = """FoamFile
{
    format ascii;
    class dictionary;
    object regionProperties;
}
regions
(
    fluid (fluid)
    solid (solid)
);
"""
(constant_dir / "regionProperties").write_text(region_props)
for f in [constant_dir / "regionProperties", constant_dir / "fluid/polyMesh/boundary", constant_dir / "solid/polyMesh/boundary"]:
    print(f"\n--- {f} ---")
    print(f.read_text()[-2600:])

# -- cell 11 ------------------------------------------------------------------------
# The boundaries are correct: inlet/outlet only on the fluid, 72-face mapped-wall patches on each side
check_outputs = {}
for region in ("fluid", "solid"):
    cm = subprocess.run(["checkMesh", "-region", region], text=True, capture_output=True)
    check_outputs[region] = cm.stdout
    print(f"\n===== checkMesh {region} =====\n{cm.stdout[-3500:]}")
    if cm.returncode != 0 or "Mesh OK." not in cm.stdout:
        print(cm.stderr)
        raise RuntimeError(f"checkMesh failed for {region}")

# -- cell 12 ------------------------------------------------------------------------
# Both regions pass, with unit aspect ratio, zero non-orthogonality, and exact requested volumes. I’ll
Path("t8.foam").touch()
reader = pv.OpenFOAMReader("t8.foam")
reader.set_active_time_value(0.0)
mesh_data = reader.read()
print(mesh_data)
print("Top-level blocks:", mesh_data.keys())

# -- cell 13 ------------------------------------------------------------------------
# The reader sees `defaultRegion`, `fluid`, and `solid`; I’ll inspect the two split-region sub-blocks 
print("fluid:", mesh_data["fluid"])
print("fluid keys:", mesh_data["fluid"].keys())
print("solid:", mesh_data["solid"])
print("solid keys:", mesh_data["solid"].keys())

# -- cell 14 ------------------------------------------------------------------------
# I’ll render the actual internal meshes with cell edges: translucent blue fluid and opaque copper-col
fluid_grid = mesh_data["fluid"]["internalMesh"]
solid_grid = mesh_data["solid"]["internalMesh"]
mp = pv.Plotter(off_screen=True, window_size=(1100, 650))
mp.add_mesh(fluid_grid, color="lightskyblue", opacity=0.22, show_edges=True, edge_color="steelblue", line_width=1)
mp.add_mesh(solid_grid, color="peru", opacity=1.0, show_edges=True, edge_color="black", line_width=1)
mp.add_axes(); mp.view_isometric(); mp.camera.zoom(1.25)
mp.show(screenshot="coarse_multiregion_mesh.png")
print(f"Rendered actual region meshes: fluid={fluid_grid.n_cells} cells, solid={solid_grid.n_cells} cells")

# -- cell 15 ------------------------------------------------------------------------
# The interface blocks were found correctly; the diagnostic only lacked a local NumPy import. I’ll rer
import numpy as np
fb = mesh_data["fluid"]["boundary"]
sb = mesh_data["solid"]["boundary"]
fint = fb["fluid_to_solid"]
sint = sb["solid_to_fluid"]
fc = fint.cell_centers().points
sc = sint.cell_centers().points
fc_sorted = fc[np.lexsort((fc[:,2],fc[:,1],fc[:,0]))]
sc_sorted = sc[np.lexsort((sc[:,2],sc[:,1],sc[:,0]))]
print(f"Interface faces fluid/solid: {fint.n_cells}/{sint.n_cells}")
print(f"Interface area expected/measured fluid/solid: 0.001800 / {fint.area:.6f} / {sint.area:.6f} m^2")
print(f"Maximum paired face-centre mismatch: {np.max(np.linalg.norm(fc_sorted-sc_sorted, axis=1)):.3e} m")

# -- cell 16 ------------------------------------------------------------------------
# Conformality is exact: both mapped patches have 72 faces, identical 0.0018 m² area, and zero face-ce
import shutil
for stl in geom_dir.glob("*_preview.stl"):
    stl.unlink()
if Path("0").exists():
    shutil.rmtree("0")
print("Retained STEP geometry and constant region meshes; removed temporary STL previews and generated field files")
