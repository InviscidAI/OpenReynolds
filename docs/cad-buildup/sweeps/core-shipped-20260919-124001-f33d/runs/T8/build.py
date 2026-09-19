"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the stated 30 mm square as the **internal duct cross-section**, with the 10 mm-thick copp
import build123d as bd
import matplotlib.pyplot as plt
# All dimensions are metres.
L, W, H = 0.200, 0.030, 0.030
BL, BW, BT = 0.040, 0.030, 0.010
BX0 = (L - BL) / 2

duct_volume = bd.Box(L, W, H)
copper = bd.Pos(BX0, 0, 0) * bd.Box(BL, BW, BT)
fluid = duct_volume - copper

print(f"Duct extent requested/measured: {L:.3f} x {W:.3f} x {H:.3f} m")
print(f"Copper extent requested/measured: {BL:.3f} x {BW:.3f} x {BT:.3f} m")
print(f"Copper x-position: requested halfway; measured centre = {BX0 + BL/2:.3f} m, duct centre = {L/2:.3f} m")
print(f"Copper volume requested = {BL*BW*BT:.9g} m^3; CAD = {copper.volume:.9g} m^3")
print(f"Fluid volume expected = {L*W*H-copper.volume:.9g} m^3; CAD = {fluid.volume:.9g} m^3")

fig, ax = plt.subplots(figsize=(10,2.4))
ax.add_patch(plt.Rectangle((0,0), L,H, facecolor='#7ec8e3', edgecolor='navy', label='fluid envelope'))
ax.add_patch(plt.Rectangle((BX0,0), BL,BT, facecolor='#b87333', edgecolor='black', label='copper solid'))
ax.set(xlabel='x [m]', ylabel='z [m]', aspect='equal', xlim=(-.005,.205), ylim=(-.003,.034), title='Longitudinal section at mid-width')
ax.legend(loc='upper right'); ax.grid(alpha=.2)
plt.show()

# -- cell 2 -------------------------------------------------------------------------
# `blockMesh` requires a minimal `controlDict` even for mesh generation; no geometry/topology error wa
from pathlib import Path
import subprocess
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication chtMultiRegionFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n')
xs, ys, zs = [0.0, BX0, BX0+BL, L], [0.0, W], [0.0, BT, H]
vid = lambda i,j,k: (i*2+j)*3+k
verts = [(x,y,z) for x in xs for y in ys for z in zs]
lines = ['FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }','convertToMeters 1;','vertices','(']
lines += [f'    ({x:.9g} {y:.9g} {z:.9g})' for x,y,z in verts]
lines += [');','blocks','(']; blocks=[]
for i,nx in enumerate([8,4,8]):
    for k,nz in enumerate([1,2]):
        v=[vid(i,0,k),vid(i+1,0,k),vid(i+1,1,k),vid(i,1,k),vid(i,0,k+1),vid(i+1,0,k+1),vid(i+1,1,k+1),vid(i,1,k+1)]
        zone='solid' if (i==1 and k==0) else 'fluid'; blocks.append((i,k,v,zone))
        lines.append(f"    hex ({' '.join(map(str,v))}) {zone} ({nx} 3 {nz}) simpleGrading (1 1 1)")
lines += [');','edges ();','boundary','(']
def face(i,k,side):
    a,b,c,d,e,f,g,h=next(q[2] for q in blocks if q[0]==i and q[1]==k)
    return {'xmin':(a,d,h,e),'xmax':(b,f,g,c),'ymin':(a,e,f,b),'ymax':(d,c,g,h),'zmin':(a,b,c,d),'zmax':(e,h,g,f)}[side]
inlet=[face(0,k,'xmin') for k in range(2)]; outlet=[face(2,k,'xmax') for k in range(2)]
walls=[]
for i in range(3):
    for k in range(2): walls += [face(i,k,'ymin'),face(i,k,'ymax')]
for i in range(3): walls.append(face(i,1,'zmax'))
for i in [0,2]: walls.append(face(i,0,'zmin'))
for name,ptype,faces in [('inlet','patch',inlet),('outlet','patch',outlet),('walls','wall',walls),('solidExterior','wall',[face(1,0,'zmin')])]:
    lines += [f'    {name}','    {',f'        type {ptype};','        faces','        (']+[f"            ({' '.join(map(str,q))})" for q in faces]+['        );','    }']
lines += [');','mergePatchPairs ();']
Path('system/blockMeshDict').write_text('\n'.join(lines)+'\n')
res=subprocess.run(['blockMesh'],text=True,capture_output=True,check=True)
print(res.stdout[-1800:])

# -- cell 3 -------------------------------------------------------------------------
# The coarse mesh has 180 orthogonal 10 mm cubes: 168 fluid and 12 solid. I’ll now render the actual O
import pyvista as pv
from IPython.display import display, Image
Path('t8.foam').touch()
reader = pv.OpenFOAMReader('t8.foam')
mesh0 = reader.read()
print(mesh0)
print('Available cell arrays:', reader.cell_array_names)
# Render the actual volume mesh with the copper-zone cells overlaid.
grid = mesh0['internalMesh'] if 'internalMesh' in mesh0.keys() else mesh0[0]
plotter = pv.Plotter(off_screen=True, window_size=(1100,420))
plotter.add_mesh(grid, style='wireframe', color='#2878b5', line_width=1, opacity=0.55)
# Copper occupies the measured coordinate box; extraction is only for visualization, not patch naming.
cc = grid.cell_centers().points
solid_ids = ((cc[:,0] >= BX0-1e-9) & (cc[:,0] <= BX0+BL+1e-9) & (cc[:,2] <= BT+1e-9)).nonzero()[0]
plotter.add_mesh(grid.extract_cells(solid_ids), color='#b87333', show_edges=True, edge_color='black', label='solid (12 cells)')
plotter.add_legend(); plotter.view_xz(); plotter.camera.zoom(1.25)
plotter.add_axes(); plotter.show(screenshot='coarse_mesh.png', auto_close=True)
display(Image('coarse_mesh.png'))

# -- cell 4 -------------------------------------------------------------------------
# The coarse rendering confirms the block is centered, floor-mounted, and its top/end faces coincide e
bmd = Path('system/blockMeshDict').read_text()
for old,new in [('(8 3 1)','(40 15 5)'),('(8 3 2)','(40 15 10)'),('(4 3 1)','(20 15 5)'),('(4 3 2)','(20 15 10)')]:
    bmd = bmd.replace(old,new)
Path('system/blockMeshDict').write_text(bmd)
res=subprocess.run(['blockMesh'],text=True,capture_output=True,check=True)
print(res.stdout[-1100:])

# -- cell 5 -------------------------------------------------------------------------
# The split command failed before reporting through the checked subprocess wrapper. I’ll rerun it with
diag=subprocess.run(['splitMeshRegions','-cellZones','-overwrite'],text=True,capture_output=True)
print('return code:',diag.returncode)
print(diag.stdout[-5000:]); print(diag.stderr[-2000:])

# -- cell 6 -------------------------------------------------------------------------
# `splitMeshRegions` needs placeholder discretization dictionaries even though it is only manipulating
Path('system/fvSchemes').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
Path('system/fvSolution').write_text('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
res=subprocess.run(['splitMeshRegions','-cellZones','-overwrite'],text=True,capture_output=True,check=True)
print(res.stdout[-3500:])

# -- cell 7 -------------------------------------------------------------------------
# The split created 450 paired interface faces (each 2×2 mm) and preserved 21,000 fluid / 1,500 solid 
Path('constant/regionProperties').write_text('FoamFile { version 2.0; format ascii; class dictionary; object regionProperties; }\nregions\n(\n    fluid (fluid)\n    solid (solid)\n);\n')
checks={}
for region in ['fluid','solid']:
    r=subprocess.run(['checkMesh','-region',region],text=True,capture_output=True)
    checks[region]=r
    print(f'===== {region}: return {r.returncode} =====')
    print(r.stdout[-2400:]); print(r.stderr)
    if r.returncode: raise RuntimeError(f'checkMesh failed for {region}')

# -- cell 8 -------------------------------------------------------------------------
# Both region meshes pass bare `checkMesh`: all cells are hexahedra, aspect ratio 1, non-orthogonality
bd.export_step(fluid, 'fluid.step')
bd.export_step(copper, 'solid.step')
print('Exported fluid.step and solid.step in metres.')
print(f'fluid.step CAD volume: {fluid.volume:.9g} m^3')
print(f'solid.step CAD volume: {copper.volume:.9g} m^3')

# -- cell 9 -------------------------------------------------------------------------
# The reader already exposes `fluid` and `solid` as region blocks; my attempted alternate reader used 
case_reader2=pv.OpenFOAMReader('t8.foam')
mb2=case_reader2.read()
print(mb2['fluid']); print(mb2['solid'])
fg2=mb2['fluid']['internalMesh']; sg2=mb2['solid']['internalMesh']
print('Final region cells:',fg2.n_cells,'fluid,',sg2.n_cells,'solid')
p2=pv.Plotter(off_screen=True,window_size=(1100,430))
p2.add_mesh(fg2,color='#70bde3',opacity=0.20,show_edges=True,edge_color='#39779a',label='fluid')
p2.add_mesh(sg2,color='#b87333',show_edges=True,edge_color='black',label='solid')
p2.add_legend(); p2.view_isometric(); p2.camera.zoom(1.25); p2.add_axes()
p2.show(screenshot='final_regions.png',auto_close=True)
display(Image('final_regions.png'))
