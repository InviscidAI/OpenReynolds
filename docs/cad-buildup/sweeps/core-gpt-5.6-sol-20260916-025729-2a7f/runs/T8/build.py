"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the block as embedded below the duct floor, with its 40 × 30 mm top face flush with the 3
from pathlib import Path
import subprocess, textwrap, json, os, numpy as np
# All dimensions are metres. Requested geometry:
L, W, H = 0.200, 0.030, 0.030
block_L, block_W, block_T = 0.040, 0.030, 0.010
x0, x1 = (L-block_L)/2, (L+block_L)/2
cell = 0.005
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True); Path('0').mkdir(exist_ok=True)
# Vertex table for three fluid blocks and the central solid block.
verts=[]; vid={}
def V(x,y,z):
    key=(x,y,z)
    if key not in vid: vid[key]=len(verts); verts.append(key)
    return vid[key]
def hexb(xa,xb,za,zb,zone,nx,nz):
    ids=[V(xa,0,za),V(xb,0,za),V(xb,W,za),V(xa,W,za),V(xa,0,zb),V(xb,0,zb),V(xb,W,zb),V(xa,W,zb)]
    return f"    hex ({' '.join(map(str,ids))}) {zone} ({nx} 6 {nz}) simpleGrading (1 1 1)"
blocks=[hexb(0,x0,0,H,'fluid',16,6),hexb(x0,x1,0,H,'fluid',8,6),hexb(x1,L,0,H,'fluid',16,6),hexb(x0,x1,-block_T,0,'solid',8,2)]
# Named exterior faces are selected explicitly from the blocks at creation time.
def F(*pts): return '('+' '.join(str(V(*p)) for p in pts)+')'
inlet=[F((0,0,0),(0,0,H),(0,W,H),(0,W,0))]
outlet=[F((L,0,0),(L,W,0),(L,W,H),(L,0,H))]
fluidWalls=[]
for xa,xb in [(0,x0),(x0,x1),(x1,L)]:
    fluidWalls += [F((xa,0,0),(xb,0,0),(xb,0,H),(xa,0,H)),F((xa,W,0),(xa,W,H),(xb,W,H),(xb,W,0)),F((xa,0,H),(xb,0,H),(xb,W,H),(xa,W,H))]
for xa,xb in [(0,x0),(x1,L)]: fluidWalls += [F((xa,0,0),(xa,W,0),(xb,W,0),(xb,0,0))]
solidExternal=[F((x0,0,-block_T),(x1,0,-block_T),(x1,W,-block_T),(x0,W,-block_T)),F((x0,0,-block_T),(x0,0,0),(x1,0,0),(x1,0,-block_T)),F((x0,W,-block_T),(x1,W,-block_T),(x1,W,0),(x0,W,0)),F((x0,0,-block_T),(x0,W,-block_T),(x0,W,0),(x0,0,0)),F((x1,0,-block_T),(x1,0,0),(x1,W,0),(x1,W,-block_T))]
def patch(name, typ, faces): return f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n            "+'\n            '.join(faces)+"\n        );\n    }"
dct=f'''FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
{chr(10).join('    ('+' '.join(f'{q:.9g}' for q in p)+')' for p in verts)}
);
blocks
(
{chr(10).join(blocks)}
);
edges ();
boundary
(
{patch('inlet','patch',inlet)}
{patch('outlet','patch',outlet)}
{patch('ductWalls','wall',fluidWalls)}
{patch('solidExternal','wall',solidExternal)}
);
mergePatchPairs ();
'''
Path('system/blockMeshDict').write_text(dct)
print(f"Requested duct: {L*1000:.1f} x {W*1000:.1f} x {H*1000:.1f} mm")
print(f"Requested block: {block_L*1000:.1f} x {block_W*1000:.1f} x {block_T*1000:.1f} mm")
print(f"Block x extents: {x0*1000:.1f} to {x1*1000:.1f} mm; centre = {(x0+x1)*500:.1f} mm (requested halfway = {L*500:.1f} mm)")
print(f"Coarse nominal cell edge: {cell*1000:.1f} mm; expected cells fluid={40*6*6}, solid={8*6*2}, total={40*6*6+8*6*2}")

# -- cell 2 -------------------------------------------------------------------------
# `blockMesh` requires a minimal case `controlDict`; the geometry dictionary itself was read only afte
Path('system/controlDict').write_text('''FoamFile { format ascii; class dictionary; object controlDict; }
application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
''')
print('Wrote minimal system/controlDict for meshing utilities.')

# -- cell 3 -------------------------------------------------------------------------
# With the case metadata present, I’ll build the coarse mesh and review the resulting block and patch 
bm = subprocess.run(['blockMesh'], text=True, capture_output=True)
print('\n'.join((bm.stdout+bm.stderr).splitlines()[-40:]))
if bm.returncode: raise RuntimeError(f'blockMesh failed: {bm.returncode}')

# -- cell 4 -------------------------------------------------------------------------
# The image shows the intended flush interface and centered block. The reproducibility checker correct
import pyvista as pv
Path('t8.foam').touch()
reader=pv.OpenFOAMReader('t8.foam'); reader.set_active_time_value(0.0); mesh=reader.read()
def leaves(obj, prefix=''):
    out=[]
    if isinstance(obj,pv.MultiBlock):
        for i in range(obj.n_blocks):
            name=obj.get_block_name(i) or str(i); out += leaves(obj[i], prefix+'/'+name)
    elif obj is not None: out.append((prefix,obj))
    return out
lf=leaves(mesh)
print('OpenFOAM reader leaves:',[(n,g.n_cells) for n,g in lf])
pl=pv.Plotter(off_screen=True, window_size=(1100,500))
colors=['lightsteelblue','tomato','silver','gold']
for (name,g),c in zip(lf,colors*4):
    pl.add_mesh(g, color=c, opacity=0.20 if 'internalMesh' in name else 0.65, show_edges=True, edge_color='gray', line_width=0.5)
pl.camera_position=[(0.27,-0.18,0.14),(0.10,0.015,0.005),(0,0,1)]
pl.add_axes(); pl.show(screenshot='coarse_mesh.png', auto_close=True)
print('Wrote coarse_mesh.png from the actual polyMesh; bounds:',mesh.bounds)

# -- cell 5 -------------------------------------------------------------------------
# This OpenFOAM build loads `fvSchemes` even for region splitting. I’ll add minimal empty discretizati
Path('system/fvSchemes').write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
Path('system/fvSolution').write_text('''FoamFile { format ascii; class dictionary; object fvSolution; }
solvers {}
''')
print('Wrote meshing-only fvSchemes and fvSolution prerequisites.')

# -- cell 6 -------------------------------------------------------------------------
# I’ll rerun region splitting and inspect the generated interface naming and face counts.
sp=subprocess.run(['splitMeshRegions','-cellZones','-overwrite'],text=True,capture_output=True)
print('\n'.join((sp.stdout+sp.stderr).splitlines()[-65:]))
if sp.returncode: raise RuntimeError(f'splitMeshRegions failed: {sp.returncode}')

# -- cell 7 -------------------------------------------------------------------------
# The splitter found exactly 48 shared faces. At 5 × 5 mm each, their total area is 1,200 mm², equal t
for region in ['fluid','solid']:
    p=Path(f'constant/{region}/polyMesh/boundary')
    txt=p.read_text()
    print(f'--- {region} boundary interface excerpt ---')
    key=f'{region}_to_'+('solid' if region=='fluid' else 'fluid')
    i=txt.find(key); print(txt[i:i+420])

# -- cell 8 -------------------------------------------------------------------------
# The fluid’s numerical checks are perfect, but `-region fluid` alone cannot complete its mapped-patch
h=subprocess.run(['checkMesh','-help'],text=True,capture_output=True)
print(h.stdout+h.stderr)

# -- cell 9 -------------------------------------------------------------------------
# `checkMesh` supports `-allRegions` via `constant/regionProperties`. I’ll write that standard region 
Path('constant/regionProperties').write_text('''FoamFile { format ascii; class dictionary; object regionProperties; }
regions
(
    fluid (fluid)
    solid (solid)
);
''')
print(Path('constant/regionProperties').read_text())

# -- cell 10 ------------------------------------------------------------------------
# Now I’ll run the binding, simultaneous full check across both regions.
cp=subprocess.run(['checkMesh','-allRegions','-allTopology','-allGeometry'],text=True,capture_output=True)
check_all=cp.stdout+cp.stderr
print('\n'.join(check_all.splitlines()[-95:]))
if cp.returncode: raise RuntimeError(f'multi-region checkMesh failed: {cp.returncode}')

# -- cell 11 ------------------------------------------------------------------------
# Both regions pass full topology and geometry checks. The interface is exact: 48↔48 faces, zero coupl
import pyvista as pv
Path('t8.foam').touch()
r2=pv.OpenFOAMReader('t8.foam'); r2.set_active_time_value(0.0); m2=r2.read()
lf2=leaves(m2)
print([(n,g.n_cells) for n,g in lf2])

# -- cell 12 ------------------------------------------------------------------------
# The reader exposes the split-region patches exactly as expected. I’ll write five patch STLs from tho
surfdir=Path('constant/triSurface'); surfdir.mkdir(parents=True,exist_ok=True)
# Remove only generated STL outputs in this case subdirectory for reproducible reruns.
for f in surfdir.glob('*.stl'): f.unlink()
leafmap={n:g for n,g in lf2}
exports={
'inlet':'/fluid/boundary/inlet',
'outlet':'/fluid/boundary/outlet',
'ductWalls':'/fluid/boundary/ductWalls',
'solidExternal':'/solid/boundary/solidExternal',
'fluid_solid_interface':'/fluid/boundary/fluid_to_solid'}
for name,key in exports.items():
    s=leafmap[key].extract_surface().triangulate().clean()
    s.save(surfdir/f'{name}.stl',binary=False)
    print(name, 'quads=',leafmap[key].n_cells,'triangles=',s.n_cells,'bounds=',tuple(round(v,6) for v in s.bounds))

# -- cell 13 ------------------------------------------------------------------------
# The exported surface bounds match every requested location. As a final geometric inspection, I’ll re
fluid_grid=leafmap['/fluid/internalMesh']; solid_grid=leafmap['/solid/internalMesh']
fi=leafmap['/fluid/boundary/fluid_to_solid']; si=leafmap['/solid/boundary/solid_to_fluid']
def sorted_centres(poly):
    a=poly.cell_centers().points
    return a[np.lexsort((a[:,2],a[:,1],a[:,0]))]
interface_err=float(np.max(np.linalg.norm(sorted_centres(fi)-sorted_centres(si),axis=1)))
print(f'Fluid bounds [m]: {tuple(float(v) for v in fluid_grid.bounds)}; volume={fluid_grid.volume:.9g} m3 (expected {L*W*H:.9g})')
print(f'Solid bounds [m]: {tuple(float(v) for v in solid_grid.bounds)}; volume={solid_grid.volume:.9g} m3 (expected {block_L*block_W*block_T:.9g})')
print(f'Interface area={fi.area:.9g} m2 (expected {block_L*block_W:.9g}); max paired face-centre mismatch={interface_err:.3g} m')
pl2=pv.Plotter(off_screen=True,window_size=(1100,520))
pl2.add_mesh(fluid_grid.extract_surface(),color='lightskyblue',opacity=0.28,show_edges=True,edge_color='steelblue')
pl2.add_mesh(solid_grid.extract_surface(),color='orangered',opacity=1.0,show_edges=True,edge_color='darkred')
pl2.camera_position=[(0.27,-0.18,0.14),(0.10,0.015,0.005),(0,0,1)]
pl2.add_axes(); pl2.show(screenshot='final_regions.png',auto_close=True)
print('Wrote final_regions.png.')
