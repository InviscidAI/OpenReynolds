"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid region as a 0.100 × 0.060 × 0.040 m rectangular volume and use a structured Car
import build123d as bd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# All dimensions are in metres.
L, W, H = 0.100, 0.060, 0.040
ZONE = 0.020
heater_min = (0.010, 0.020, 0.000)
cooler_min = (0.070, 0.020, 0.020)

cavity = bd.Box(L, W, H)
heater = bd.Pos(*heater_min) * bd.Box(ZONE, ZONE, ZONE)
cooler = bd.Pos(*cooler_min) * bd.Box(ZONE, ZONE, ZONE)

bd.export_step(cavity, "cavity.step")
bd.export_stl(cavity, "adiabaticWall.stl")

print(f"Cavity requested/measured: {(L,W,H)} m / {tuple(round(v,6) for v in cavity.bounding_box().size.to_tuple())} m")
print(f"Heater requested/measured cube edge: {ZONE} m / {heater.bounding_box().size.X:.6f} m")
print(f"Cooler requested/measured cube edge: {ZONE} m / {cooler.bounding_box().size.X:.6f} m")
print(f"Heater bounds: {heater.bounding_box().min.to_tuple()} to {heater.bounding_box().max.to_tuple()}")
print(f"Cooler bounds: {cooler.bounding_box().min.to_tuple()} to {cooler.bounding_box().max.to_tuple()}")
print(f"Volumes cavity/heater/cooler: {cavity.volume:.9g}, {heater.volume:.9g}, {cooler.volume:.9g} m^3")

fig = plt.figure(figsize=(8,5)); ax = fig.add_subplot(111, projection='3d')
def cuboid(ax, origin, size, color, alpha, label):
    x,y,z=origin; dx,dy,dz=size
    p=[(x,y,z),(x+dx,y,z),(x+dx,y+dy,z),(x,y+dy,z),(x,y,z+dz),(x+dx,y,z+dz),(x+dx,y+dy,z+dz),(x,y+dy,z+dz)]
    f=[[p[i] for i in q] for q in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]
    pc=Poly3DCollection(f, facecolors=color, edgecolors=color, alpha=alpha, linewidths=1.2, label=label); ax.add_collection3d(pc)
cuboid(ax,(0,0,0),(L,W,H),'steelblue',0.08,'cavity')
cuboid(ax,heater_min,(ZONE,)*3,'orangered',0.65,'heater')
cuboid(ax,cooler_min,(ZONE,)*3,'deepskyblue',0.65,'cooler')
ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',xlim=(0,L),ylim=(0,W),zlim=(0,H)); ax.set_box_aspect((L,W,H)); ax.view_init(24,-58); ax.legend(); plt.tight_layout(); plt.savefig('geometry_preview.png',dpi=160); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The numerical bounds exposed a build123d alignment issue: `Box` defaults to centered placement, so t
min_align = (bd.Align.MIN, bd.Align.MIN, bd.Align.MIN)
cavity = bd.Box(L, W, H, align=min_align)
heater = bd.Pos(*heater_min) * bd.Box(ZONE, ZONE, ZONE, align=min_align)
cooler = bd.Pos(*cooler_min) * bd.Box(ZONE, ZONE, ZONE, align=min_align)
bd.export_step(cavity, "cavity.step")
bd.export_stl(cavity, "adiabaticWall.stl")

for name, shape, expected_min, expected_max in [
    ("heater", heater, heater_min, tuple(v+ZONE for v in heater_min)),
    ("cooler", cooler, cooler_min, tuple(v+ZONE for v in cooler_min))]:
    actual_min, actual_max = tuple(shape.bounding_box().min), tuple(shape.bounding_box().max)
    print(f"{name} bounds requested: {expected_min} to {expected_max}")
    print(f"{name} bounds measured:  {actual_min} to {actual_max}")

fig = plt.figure(figsize=(8,5)); ax = fig.add_subplot(111, projection='3d')
cuboid(ax,(0,0,0),(L,W,H),'steelblue',0.08,'cavity')
cuboid(ax,heater_min,(ZONE,)*3,'orangered',0.65,'heater')
cuboid(ax,cooler_min,(ZONE,)*3,'deepskyblue',0.65,'cooler')
ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',xlim=(0,L),ylim=(0,W),zlim=(0,H)); ax.set_box_aspect((L,W,H)); ax.view_init(24,-58); ax.legend(); plt.tight_layout(); plt.savefig('geometry_preview.png',dpi=160); plt.show()

# -- cell 3 -------------------------------------------------------------------------
# `blockMesh` also requires a minimal `controlDict` in this distribution. I’ll first write a complete 
from pathlib import Path
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
foam_header = "FoamFile\n{\n    format ascii;\n    class dictionary;\n    object %s;\n}\n"
Path('system/controlDict').write_text(foam_header % 'controlDict' + "application blockMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
Path('system/blockMeshDict').write_text(foam_header % 'blockMeshDict' + r'''
convertToMeters 1;
vertices ((0 0 0) (0.1 0 0) (0.1 0.06 0) (0 0.06 0) (0 0 0.04) (0.1 0 0.04) (0.1 0.06 0.04) (0 0.06 0.04));
blocks (hex (0 1 2 3 4 5 6 7) (20 12 8) simpleGrading (1 1 1));
edges ();
boundary
(
 adiabaticWall
 {
  type wall;
  faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7));
 }
);
mergePatchPairs ();
''')
Path('system/topoSetDict').write_text(foam_header % 'topoSetDict' + r'''
actions
(
 { name heaterCells; type cellSet; action new; source boxToCell; sourceInfo { box (0.01 0.02 0) (0.03 0.04 0.02); } }
 { name heater; type cellZoneSet; action new; source setToCellZone; sourceInfo { set heaterCells; } }
 { name coolerCells; type cellSet; action new; source boxToCell; sourceInfo { box (0.07 0.02 0.02) (0.09 0.04 0.04); } }
 { name cooler; type cellZoneSet; action new; source setToCellZone; sourceInfo { set coolerCells; } }
);
''')
print('Wrote minimal OpenFOAM case dictionaries for a 20 x 12 x 8 structured mesh and two box-selected cell zones.')

# -- cell 4 -------------------------------------------------------------------------
# The case dictionaries are now complete. I’ll generate the coarse mesh and zones and inspect the util
import subprocess
r1=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r1.stdout[-1400:]); print(r1.stderr); r1.check_returncode()
r2=subprocess.run(['topoSet'],text=True,capture_output=True); print(r2.stdout[-2200:]); print(r2.stderr); r2.check_returncode()

# -- cell 5 -------------------------------------------------------------------------
# The render failed before constructing the view because this OSMesa build does not provide the option
import numpy as np, pyvista as pv
grid = pv.ImageData(dimensions=(21,13,9), spacing=(0.005,0.005,0.005), origin=(0,0,0))
centres = grid.cell_centers().points
heater_mask = ((centres[:,0]>=0.01)&(centres[:,0]<=0.03)&(centres[:,1]>=0.02)&(centres[:,1]<=0.04)&(centres[:,2]>=0)&(centres[:,2]<=0.02))
cooler_mask = ((centres[:,0]>=0.07)&(centres[:,0]<=0.09)&(centres[:,1]>=0.02)&(centres[:,1]<=0.04)&(centres[:,2]>=0.02)&(centres[:,2]<=0.04))
heater_grid=grid.extract_cells(np.flatnonzero(heater_mask)); cooler_grid=grid.extract_cells(np.flatnonzero(cooler_mask))
print('Rendered mesh cells:',grid.n_cells)
print('heater cells / union bounds:',heater_grid.n_cells,tuple(round(v,5) for v in heater_grid.bounds))
print('cooler cells / union bounds:',cooler_grid.n_cells,tuple(round(v,5) for v in cooler_grid.bounds))
p=pv.Plotter(off_screen=True,window_size=(1000,650))
p.add_mesh(grid.extract_surface(),style='wireframe',color='gray',opacity=0.32,line_width=1)
p.add_mesh(heater_grid,color='orangered',show_edges=True,edge_color='darkred',label='heater (64 cells)')
p.add_mesh(cooler_grid,color='deepskyblue',show_edges=True,edge_color='navy',label='cooler (64 cells)')
p.add_axes(); p.add_legend(); p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot='coarse_mesh_preview.png')

# -- cell 6 -------------------------------------------------------------------------
# The coarse view confirms the cavity topology and zone placement. For the delivered mesh I’ll uniform
from pathlib import Path
import subprocess
p=Path('system/blockMeshDict')
p.write_text(p.read_text().replace('(20 12 8)', '(40 24 16)'))
r1=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r1.stdout[-1200:]); print(r1.stderr); r1.check_returncode()
r2=subprocess.run(['topoSet'],text=True,capture_output=True); print(r2.stdout[-1300:]); print(r2.stderr); r2.check_returncode()

# -- cell 7 -------------------------------------------------------------------------
# The full diagnostic mode in OpenFOAM 2512 reads numerical-scheme dictionaries even though no solve i
from pathlib import Path
Path('system/fvSchemes').write_text(r'''FoamFile
{
 format ascii;
 class dictionary;
 object fvSchemes;
}
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
Path('system/fvSolution').write_text(r'''FoamFile
{
 format ascii;
 class dictionary;
 object fvSolution;
}
solvers {}
''')
print('Wrote minimal diagnostic dictionaries required by checkMesh full-check mode.')

# -- cell 8 -------------------------------------------------------------------------
# I’ll rerun the complete mesh diagnostics now.
import subprocess
r=subprocess.run(['checkMesh','-allTopology','-allGeometry'],text=True,capture_output=True)
print(r.stdout[-6000:]); print(r.stderr); r.check_returncode()
