"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model only the wetted volume: a 60 mm bore, 42 mm long, divided axially into 20 mm flange-bore,
from build123d import Cylinder, Pos, export_stl
import pyvista as pv
# Requested dimensions, converted from mm to metres
BORE_D = 0.060
FLANGE_T = 0.020
GASKET_T = 0.002
TOTAL_L = 2*FLANGE_T + GASKET_T
R = BORE_D/2
# Keep axial seams by forming one solid from three touching axial segments
left_seg = Cylinder(R, FLANGE_T)
gasket_seg = Pos(0, 0, FLANGE_T) * Cylinder(R, GASKET_T)
right_seg = Pos(0, 0, FLANGE_T + GASKET_T) * Cylinder(R, FLANGE_T)
fluid = left_seg + gasket_seg + right_seg
print(f"Requested/measured bore diameter: {BORE_D:.6f} / {2*R:.6f} m")
print(f"Requested/measured face-to-face length: {TOTAL_L:.6f} / {fluid.bounding_box().size.Z:.6f} m")
print(f"Fluid volume: {fluid.volume:.9e} m^3; CAD faces: {len(fluid.faces())}")
export_stl(fluid, "fluid_preview.stl", tolerance=5e-4, angular_tolerance=0.2)
mesh_preview = pv.read("fluid_preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(700, 520))
plotter.add_mesh(mesh_preview, color="deepskyblue", show_edges=True)
plotter.camera_position = [(0.10, -0.10, 0.09), (0, 0, TOTAL_L/2), (0, 0, 1)]
plotter.show(screenshot="fluid_preview.png")
print("Preview written: fluid_preview.png")

# -- cell 2 -------------------------------------------------------------------------
# The preview shows the intended cylindrical water column and the axial patch seams. I’ll now inspect 
for i, face in enumerate(fluid.faces()):
    c = face.center()
    print(i, face.geom_type, f"area={face.area:.9e}", f"center=({c.X:.6f},{c.Y:.6f},{c.Z:.6f})")

# -- cell 3 -------------------------------------------------------------------------
# The face audit exposed a placement error: `Cylinder` is centred axially by default, so the first con
from build123d import Align
left_seg = Cylinder(R, FLANGE_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
gasket_seg = Pos(0, 0, FLANGE_T) * Cylinder(R, GASKET_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
right_seg = Pos(0, 0, FLANGE_T + GASKET_T) * Cylinder(R, FLANGE_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
fluid = left_seg + gasket_seg + right_seg
expected_volume = 3.141592653589793 * R**2 * TOTAL_L
print(f"Requested/measured axial extent: {TOTAL_L:.6f} / {fluid.bounding_box().size.Z:.6f} m")
print(f"Expected/measured volume: {expected_volume:.9e} / {fluid.volume:.9e} m^3")
print(f"Solids={len(fluid.solids())}, faces={len(fluid.faces())}")
for i, face in enumerate(fluid.faces()):
    c=face.center()
    print(i, face.geom_type, f"area={face.area:.9e}", f"z={c.Z:.6f}")

# -- cell 4 -------------------------------------------------------------------------
# A valid solid fusion correctly removes the two purely geometric seams, but those seams are required 
from build123d import Shell, GeomType
# Patch faces are named directly from the primitives that create them.
left_cyl = left_seg.faces().filter_by(GeomType.CYLINDER)[0]
left_planes = sorted(left_seg.faces().filter_by(GeomType.PLANE), key=lambda f: f.center().Z)
inlet_face = left_planes[0]
left_wall_face = left_cyl
gasket_wall_face = gasket_seg.faces().filter_by(GeomType.CYLINDER)[0]
right_wall_face = right_seg.faces().filter_by(GeomType.CYLINDER)[0]
right_planes = sorted(right_seg.faces().filter_by(GeomType.PLANE), key=lambda f: f.center().Z)
outlet_face = right_planes[-1]
fluid = Shell([inlet_face, left_wall_face, gasket_wall_face, right_wall_face, outlet_face])
print(f"Closed shell faces: {len(fluid.faces())}; extent Z={fluid.bounding_box().size.Z:.6f} m")
for name, face, expected in [
    ("inlet", inlet_face, 3.141592653589793*R**2),
    ("outlet", outlet_face, 3.141592653589793*R**2),
    ("flange1_bore_wall", left_wall_face, 2*3.141592653589793*R*FLANGE_T),
    ("gasket_inner_face", gasket_wall_face, 2*3.141592653589793*R*GASKET_T),
    ("flange2_bore_wall", right_wall_face, 2*3.141592653589793*R*FLANGE_T)]:
    print(f"{name}: expected/measured area = {expected:.9e} / {face.area:.9e} m^2")

# -- cell 5 -------------------------------------------------------------------------
# The five shell areas match the requested 60 mm bore and the 20–2–20 mm axial segmentation exactly. I
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches
help(export_patches)

# -- cell 6 -------------------------------------------------------------------------
# The exporter correctly rejected the source primitive face handles because sewing creates new face id
fluid_cyls = sorted(fluid.faces().filter_by(GeomType.CYLINDER), key=lambda f: f.center().Z)
fluid_planes = sorted(fluid.faces().filter_by(GeomType.PLANE), key=lambda f: f.center().Z)
left_wall_face, gasket_wall_face, right_wall_face = fluid_cyls
inlet_face, outlet_face = fluid_planes
print("Shell face z-centres:", [round(f.center().Z, 6) for f in [inlet_face,left_wall_face,gasket_wall_face,right_wall_face,outlet_face]])
patch_report = export_patches(
    fluid,
    {"inlet": inlet_face, "outlet": outlet_face,
     "flange1_bore_wall": left_wall_face,
     "gasket_inner_face": gasket_wall_face,
     "flange2_bore_wall": right_wall_face},
    tolerance=1.0e-4, angular_tolerance=0.08)
print(patch_report)

# -- cell 7 -------------------------------------------------------------------------
# I’ll re-emit that minimal metadata cell with its import included so it is reproducible from an empty
from pathlib import Path
Path("system").mkdir(exist_ok=True)
Path("system/controlDict").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application blockMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
""")
print("Wrote minimal system/controlDict")

# -- cell 8 -------------------------------------------------------------------------
# Now I’ll generate the replay-safe O-grid dictionary, leaving execution for the following diagnostic 
import math
RI=0.012; s=RI/math.sqrt(2); q=R/math.sqrt(2)
zplanes=[0.0,FLANGE_T,FLANGE_T+GASKET_T,TOTAL_L]
xy=[(s,s),(-s,s),(-s,-s),(s,-s),(q,q),(-q,q),(-q,-q),(q,-q)]
b2=[(2,3,0,1),(3,7,4,0),(0,4,5,1),(1,5,6,2),(2,6,7,3)]
L=["FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }","scale 1;","vertices ("]
L += [f"({x:.12g} {y:.12g} {z:.12g})" for z in zplanes for x,y in xy] + [");","blocks ("]
for k,nz in enumerate([8,1,8]):
    for j,b in enumerate(b2):
        lo=[8*k+i for i in b]; hi=[8*(k+1)+i for i in b]; n=(8,8,nz) if j==0 else (5,8,nz)
        L.append(f"hex ({' '.join(map(str,lo+hi))}) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)")
L += [");","edges ("]
for p,z in enumerate(zplanes):
    for a,b,x,y in [(7,4,R,0),(4,5,0,R),(5,6,-R,0),(6,7,0,-R)]: L.append(f"arc {8*p+a} {8*p+b} ({x:.12g} {y:.12g} {z:.12g})")
L += [");","boundary ("]
for name,p,rev in [("inlet",0,True),("outlet",3,False)]:
    L += [f"{name} {{ type patch; faces ("]
    for b in b2:
        ids=[8*p+i for i in b]; ids=ids[::-1] if rev else ids; L.append(f"({' '.join(map(str,ids))})")
    L += ["); }"]
for k,name in enumerate(["flange1_bore_wall","gasket_inner_face","flange2_bore_wall"]):
    L += [f"{name} {{ type wall; faces ("]
    for a,b in [(7,4),(4,5),(5,6),(6,7)]: L.append(f"({8*k+a} {8*k+b} {8*(k+1)+b} {8*(k+1)+a})")
    L += ["); }"]
L += [");","mergePatchPairs ();",""]
Path("system/blockMeshDict").write_text("\n".join(L))
print("Wrote 15-block cylindrical O-grid dictionary")

# -- cell 9 -------------------------------------------------------------------------
# I’ll run the coarse mesher and inspect its topology summary.
import subprocess
bm=subprocess.run(["blockMesh"],text=True,capture_output=True)
print(bm.stdout[-3500:]); print(bm.stderr)
bm.check_returncode()

# -- cell 10 ------------------------------------------------------------------------
# `foamToVTK` expects field data that this mesh-only case intentionally does not contain. I’ll instead
Path("case.foam").touch()
reader=pv.OpenFOAMReader("case.foam")
reader.set_active_time_value(0.0)
mb=reader.read()
print(mb.keys())
print(mb)

# -- cell 11 ------------------------------------------------------------------------
# The OpenFOAM mesh reader confirms the exact 60 mm × 42 mm bounds. I’ll now display a clipped mesh vi
ug=mb["internalMesh"]
clipped=ug.clip(normal=(0,1,0),origin=(0,0,0),invert=False,crinkle=True)
p= pv.Plotter(off_screen=True,window_size=(760,540))
p.add_mesh(clipped,color="lightblue",show_edges=True,edge_color="navy",line_width=1)
p.camera_position=[(0.10,-0.12,0.075),(0,0,0.021),(0,0,1)]
p.show(screenshot="coarse_mesh.png")
print(f"Rendered clipped mesh: {clipped.n_cells} of {ug.n_cells} cells; coarse_mesh.png")

# -- cell 12 ------------------------------------------------------------------------
# `checkMesh` reached the mesh but this OpenFOAM build requires the standard empty `fvSchemes`/`fvSolu
Path("system/fvSchemes").write_text("FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n")
Path("system/fvSolution").write_text("FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n")
print("Wrote mesh-check support dictionaries")

# -- cell 13 ------------------------------------------------------------------------
# This release validates the standard scheme subdictionaries during startup. I’ll provide minimal conv
Path("system/fvSchemes").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")
print("Completed minimal fvSchemes structure")

# -- cell 14 ------------------------------------------------------------------------
# I’ll run `checkMesh` again.
cm=subprocess.run(["checkMesh"],text=True,capture_output=True)
print(cm.stdout[-5000:]); print(cm.stderr)
cm.check_returncode()

# -- cell 15 ------------------------------------------------------------------------
# The binding check passes: one region, 3,808 all-hexahedral cells, maximum non-orthogonality 24.37°, 
reader=pv.OpenFOAMReader("case.foam"); mb=reader.read(); bdry=mb["boundary"]
print("Boundary blocks:",bdry.keys())
for key in bdry.keys():
    surf=bdry[key]
    print(key, "faces=",surf.n_cells,"area=",f"{surf.area:.9e}","bounds=",tuple(round(v,6) for v in surf.bounds))
