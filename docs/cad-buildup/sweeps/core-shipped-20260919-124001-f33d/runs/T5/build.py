"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# The import succeeded and found three solids, but validity is a property in this installed build123d 
from pathlib import Path
import re
import build123d as bd
STEP_PATH = Path('/home/qiuzi/.openreynolds-buildup/work/T5-20260919-134906-2bca/geometry/assembly.step')
step_text = STEP_PATH.read_text(errors='ignore')
unit_lines = sorted(set(re.findall(r'[^;]*(?:SI_UNIT|CONVERSION_BASED_UNIT)[^;]*;', step_text)))
print('STEP unit declarations:')
for line in unit_lines[:30]: print(line.strip())
assembly = bd.import_step(STEP_PATH)
solids = list(assembly.solids())
print('Imported solids:', len(solids))
print('OCCT valid:', assembly.is_valid)
print('Assembly type:', assembly.shape_type)
print('Bounding box:', assembly.bounding_box())
print('Volume:', assembly.volume)
for i, solid in enumerate(solids):
    print(f'solid {i}: valid={solid.is_valid}, volume={solid.volume:.9g}, bbox={solid.bounding_box()}')

# -- cell 2 -------------------------------------------------------------------------
# The file declares metres, while OCCT’s STEP reader has converted the coordinates into its millimetre
import pyvista as pv
preview_path = Path('assembly_preview.stl')
bd.export_stl(assembly, preview_path, tolerance=0.15, angular_tolerance=0.2)
mesh_preview = pv.read(preview_path)
plotter = pv.Plotter(off_screen=True, window_size=(900,700))
plotter.add_mesh(mesh_preview, color='lightgray', show_edges=True, edge_color='gray')
plotter.add_axes()
plotter.view_isometric()
plotter.show(screenshot='assembly_preview.png')
print('Preview:', mesh_preview.n_points, 'points,', mesh_preview.n_cells, 'triangles')
print('Dimensions in imported OCCT mm:', mesh_preview.bounds)

# -- cell 3 -------------------------------------------------------------------------
# The imported geometry is a compact lidar with a rectangular housing, top cylindrical scanner, and mo
from collections import Counter
for si, solid in enumerate(solids):
    faces = list(solid.faces())
    types = Counter(str(face.geom_type) for face in faces)
    print(f'solid {si}: {len(faces)} faces, types={dict(types)}')
    curved = [(i, str(face.geom_type), face.area) for i, face in enumerate(faces) if str(face.geom_type) != 'GeomType.PLANE']
    print(' curved faces (index,type,area mm2):', curved)

# -- cell 4 -------------------------------------------------------------------------
# The curved-face inventory shows several very small cylindrical and spline details on the scanner bod
from OCP.BRepAdaptor import BRepAdaptor_Surface
for si, solid in enumerate(solids):
    print('SOLID', si)
    for fi, face in enumerate(solid.faces()):
        typ = str(face.geom_type)
        if typ == 'GeomType.PLANE': continue
        ad = BRepAdaptor_Surface(face.wrapped)
        detail = ''
        try:
            if typ == 'GeomType.CYLINDER': detail = f'R={ad.Cylinder().Radius():.4g}'
            elif typ == 'GeomType.CONE': detail = f'R0={ad.Cone().RefRadius():.4g}, angle={ad.Cone().SemiAngle():.4g}'
            elif typ == 'GeomType.TORUS': detail = f'Rmajor={ad.Torus().MajorRadius():.4g}, Rminor={ad.Torus().MinorRadius():.4g}'
        except Exception as exc: detail = str(exc)
        bb = face.bounding_box()
        print(f' face {fi:2d} {typ:17s} area={face.area:8.3f} {detail} bbox={bb}')
