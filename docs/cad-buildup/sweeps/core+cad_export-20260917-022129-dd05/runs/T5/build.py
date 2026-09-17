"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll first inspect the supplied STEP’s declared units and import it with OCCT, then report its topol
from pathlib import Path
import build123d as bd
STEP_PATH = Path('/home/qiuzi/.openreynolds-buildup/work/T5-20260917-030245-0ec8/geometry/assembly.step')
header = STEP_PATH.read_text(errors='ignore')[:20000]
unit_lines = [ln.strip() for ln in header.splitlines() if 'SI_UNIT' in ln or 'CONVERSION_BASED_UNIT' in ln]
assembly = bd.import_step(STEP_PATH)
print('STEP unit declarations:')
print('\n'.join(unit_lines[:20]) or '(none found in first 20 kB)')
print(f'imported type: {type(assembly).__name__}')
print(f'solids: {len(assembly.solids())}, shells: {len(assembly.shells())}, faces: {len(assembly.faces())}')
print(f'OCCT valid: {assembly.is_valid}')
print(f'bbox min: {assembly.bounding_box().min}, max: {assembly.bounding_box().max}, size: {assembly.bounding_box().size}')
print(f'volume: {assembly.volume}')

# -- cell 2 -------------------------------------------------------------------------
# The import is a valid three-solid assembly about 54 × 46 × 31 in its native coordinates. I’ll now in
import re
step_text = STEP_PATH.read_text(errors='ignore')
unit_records = sorted(set(re.findall(r'#[0-9]+\s*=\s*(?:SI_UNIT|CONVERSION_BASED_UNIT|LENGTH_MEASURE_WITH_UNIT)\([^;]*;', step_text, flags=re.I)))
print('Unit-related STEP records:')
print('\n'.join(unit_records) or '(none)')
for i, solid in enumerate(assembly.solids(), 1):
    bb = solid.bounding_box()
    print(f'solid {i}: valid={solid.is_valid}, volume={solid.volume:.6g}, faces={len(solid.faces())}, bbox=({bb.min}) to ({bb.max}), size={bb.size}')
print('labels:', assembly.label, [s.label for s in assembly.solids()])

# -- cell 3 -------------------------------------------------------------------------
# No single-line unit record was found because STEP entities can split the unit definition across link
for token in ['MILLI', 'METRE', 'LENGTH_UNIT', 'GLOBAL_UNIT_ASSIGNED_CONTEXT']:
    hits = [ln.strip() for ln in step_text.splitlines() if token in ln.upper()]
    print(f'[{token}]', *(hits[:12] or ['(none)']), sep='\n  ')
import pyvista as pv
pv.OFF_SCREEN = True
pl = pv.Plotter(off_screen=True, window_size=(900,700))
colors = ['tomato','seagreen','royalblue']
for solid, color in zip(assembly.solids(), colors):
    vs, fs = solid.tessellate(0.15)
    pts = [[v.X, v.Y, v.Z] for v in vs]
    cells = []
    for f in fs: cells.extend([3, f[0], f[1], f[2]])
    pl.add_mesh(pv.PolyData(pts, cells), color=color, opacity=0.65, show_edges=True)
pl.add_axes(); pl.view_isometric(); pl.show(screenshot='assembly_overview.png')
print('wrote assembly_overview.png')

# -- cell 4 -------------------------------------------------------------------------
# The STEP context declares metres, while OCCT has converted the imported geometry to its millimetre w
pl = pv.Plotter(shape=(1,3), off_screen=True, window_size=(1500,520))
for j, (solid, color) in enumerate(zip(assembly.solids(), colors)):
    pl.subplot(0,j)
    vs, fs = solid.tessellate(0.08)
    pts = [[v.X,v.Y,v.Z] for v in vs]; cells=[]
    for f in fs: cells.extend([3,*f])
    mesh = pv.PolyData(pts,cells)
    pl.add_mesh(mesh, color=color, opacity=1, show_edges=True)
    pl.add_text(f'Solid {j+1}\nV={solid.volume:.1f} mm^3', font_size=11)
    pl.add_axes(); pl.view_isometric()
pl.show(screenshot='solids_separate.png')
print('wrote solids_separate.png')

# -- cell 5 -------------------------------------------------------------------------
# The first interface test hit an empty Boolean result and therefore did not become part of the reprod
solids = list(assembly.solids())
for i in range(3):
    for j in range(i+1,3):
        common_ij = solids[i] & solids[j]
        cv = 0.0 if common_ij is None else common_ij.volume
        cf = 0 if common_ij is None else len(common_ij.faces())
        print(f'{i+1}-{j+1}: common volume={cv:.9g} mm^3, common faces={cf}')
fluid_native = solids[0] + solids[1] + solids[2]
print(f'fused result: type={type(fluid_native).__name__}, solids={len(fluid_native.solids())}, valid={fluid_native.is_valid}, volume={fluid_native.volume:.9g}')
print(f'sum volume={sum(s.volume for s in solids):.9g}, fuse faces={len(fluid_native.faces())}')

# -- cell 6 -------------------------------------------------------------------------
# The three bodies do not overlap volumetrically, and the fuse still has two disconnected solids, so t
pl = pv.Plotter(shape=(1,2), off_screen=True, window_size=(1200,600))
allmeshes=[]
for solid, color in zip(solids, colors):
    vs, fs = solid.tessellate(0.06); pts=[[v.X,v.Y,v.Z] for v in vs]; cells=[]
    for f in fs: cells.extend([3,*f])
    allmeshes.append((pv.PolyData(pts,cells),color))
for k, normal in enumerate([(1,0,0),(0,1,0)]):
    pl.subplot(0,k)
    for mesh,color in allmeshes:
        clipped=mesh.clip(normal=normal, origin=(0,0,0), invert=False)
        pl.add_mesh(clipped,color=color,show_edges=True)
    pl.add_text('Cutaway at '+('x=0' if k==0 else 'y=0'),font_size=12)
    pl.add_axes()
    if k==0: pl.view_yz()
    else: pl.view_xz()
pl.show(screenshot='central_cutaways.png')
print('wrote central_cutaways.png')

# -- cell 7 -------------------------------------------------------------------------
# The cutaway confirms there is no hollow core inside the blue body: it is the central flow-space soli
product_lines=[ln.strip() for ln in step_text.splitlines() if 'PRODUCT(' in ln or 'MANIFOLD_SOLID_BREP' in ln]
print('\n'.join(product_lines[:30]))
for i in range(3):
    for j in range(i+1,3):
        d,p1,p2 = solids[i].distance_to_with_closest_points(solids[j])
        print(f'distance solid {i+1}-{j+1}: {d:.9g} mm at {p1} / {p2}')
for i,s in enumerate(solids,1):
    zfaces=sorted([(f.center().Z, f.area, type(f.geom_type).__name__ if hasattr(f,'geom_type') else '') for f in s.faces()])
    print(f'solid {i} lowest/highest face centers/areas:', zfaces[:3], zfaces[-3:])

# -- cell 8 -------------------------------------------------------------------------
# The CAD names the parts `BASE`, `EMITTER`, and `BODY`, and every imported item is already a closed s
bb=assembly.bounding_box(); margin=5.0
probe_box = bd.Pos(bb.min.X-margin,bb.min.Y-margin,bb.min.Z-margin) * bd.Box(bb.size.X+2*margin,bb.size.Y+2*margin,bb.size.Z+2*margin)
void_probe = probe_box - assembly
probe_solids=list(void_probe.solids())
print(f'void components after subtracting assembly from surrounding box: {len(probe_solids)}')
for i,s in enumerate(probe_solids,1):
    print(f'  component {i}: volume={s.volume:.6g} mm^3, bbox={s.bounding_box().size}')
print('Interpretation: one component means all non-material space is connected to the exterior; no sealed internal cavity exists.')

# -- cell 9 -------------------------------------------------------------------------
# The complement has only the exterior component, so there is no sealed cavity. I’ll check each closed
for i,s in enumerate(solids,1):
    V,E,F=len(s.vertices()),len(s.edges()),len(s.faces())
    chi=V-E+F
    genus=(2-chi)/2
    print(f'solid {i} ({["BASE","EMITTER","BODY"][i-1]}): V={V}, E={E}, F={F}, Euler={chi}, closed-shell genus={genus:g}')
