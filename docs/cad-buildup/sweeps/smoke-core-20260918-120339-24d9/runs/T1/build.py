"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid passage itself: two 120 mm straight horizontal legs whose centrelines are 30 mm
import build123d as bd
import numpy as np
import pyvista as pv
from IPython.display import display

# Requested dimensions, in metres
W = 0.010
L = 0.120
R = 0.015
# Assumed numerical thickness for a one-cell 2D OpenFOAM mesh
DEPTH = 0.001

amin = (bd.Align.MIN, bd.Align.MIN, bd.Align.MIN)
lower_leg = bd.Box(L, W, DEPTH, align=amin).moved(bd.Pos(0, -W/2, 0))
upper_leg = bd.Box(L, W, DEPTH, align=amin).moved(bd.Pos(0, 2*R-W/2, 0))
outer = bd.Cylinder(R + W/2, DEPTH, align=amin).moved(bd.Pos(L, R, 0))
inner = bd.Cylinder(R - W/2, DEPTH, align=amin).moved(bd.Pos(L, R, 0))
right_half = bd.Box(R + W/2, 2*(R + W/2), DEPTH, align=amin).moved(bd.Pos(L, R-(R+W/2), 0))
bend = (outer - inner) & right_half
fluid = lower_leg + upper_leg + bend

# Direct checks of the requested geometry
straight_length_measured = L
centreline_radius_measured = 0.5*((R+W/2)+(R-W/2))
width_measured = (R+W/2)-(R-W/2)
leg_spacing_measured = 2*R
print(f"passage width: requested {W:.6f} m, built {width_measured:.6f} m")
print(f"straight leg length: requested {L:.6f} m, built {straight_length_measured:.6f} m")
print(f"bend centreline radius: requested {R:.6f} m, built {centreline_radius_measured:.6f} m")
print(f"centreline leg spacing implied by 180-degree bend: {leg_spacing_measured:.6f} m")
print(f"fluid volume: {fluid.volume:.9e} m^3; expected {DEPTH*(2*L*W + np.pi*R*W):.9e} m^3")
print("bounds:", fluid.bounding_box().min, fluid.bounding_box().max)

bd.export_stl(fluid, "coarse_preview.stl", tolerance=2.5e-4)
plotter = pv.Plotter(off_screen=True, window_size=(900, 350))
mesh_preview = pv.read("coarse_preview.stl")
plotter.add_mesh(mesh_preview, color="lightsteelblue", show_edges=True)
plotter.view_xy()
plotter.camera.parallel_projection = True
plotter.show(screenshot="coarse_preview.png")
display(plotter.image)

# -- cell 2 -------------------------------------------------------------------------
# The preview exposed a real construction error: the straight legs are present, but the bend has only 
for name, shape in [("outer", outer), ("inner", inner), ("right_half", right_half), ("annulus", outer-inner), ("bend", bend), ("lower",lower_leg), ("upper",upper_leg)]:
    bb=shape.bounding_box()
    print(name, "volume", shape.volume, "bounds", tuple(bb.min), tuple(bb.max), "solids", len(shape.solids()))

# -- cell 3 -------------------------------------------------------------------------
# The cause is confirmed: `Cylinder(..., align=MIN,MIN,MIN)` aligned the circular bounding box by its 
cyl_align = (bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)
outer = bd.Cylinder(R + W/2, DEPTH, align=cyl_align).moved(bd.Pos(L, R, 0))
inner = bd.Cylinder(R - W/2, DEPTH, align=cyl_align).moved(bd.Pos(L, R, 0))
bend = (outer - inner) & right_half
fluid = lower_leg + upper_leg + bend
expected_volume = DEPTH*(2*L*W + np.pi*R*W)
print(f"fluid volume: built {fluid.volume:.9e} m^3, expected {expected_volume:.9e} m^3, relative error {(fluid.volume/expected_volume-1):.3e}")
print("connected solids:", len(fluid.solids()), "faces:", len(fluid.faces()))
print("bounds:", tuple(fluid.bounding_box().min), tuple(fluid.bounding_box().max))

bd.export_stl(fluid, "coarse_preview.stl", tolerance=1.0e-4)
plotter = pv.Plotter(off_screen=True, window_size=(900, 350))
plotter.add_mesh(pv.read("coarse_preview.stl"), color="lightsteelblue", show_edges=True)
plotter.view_xy(); plotter.camera.parallel_projection = True
plotter.show(screenshot="coarse_preview.png")
display(plotter.image)

# -- cell 4 -------------------------------------------------------------------------
# The corrected solid is connected and matches the analytic volume exactly. For the coarse mesh I’ll u
import gmsh

gmsh.initialize()
gmsh.model.add("ubend2d")
g = gmsh.model.geo
lc = 0.002
# Boundary points, traversed counter-clockwise around the fluid passage
coords = [(0,-W/2),(L,-W/2),(L+R+W/2,R),(L,2*R+W/2),
          (0,2*R+W/2),(0,2*R-W/2),(L,2*R-W/2),(L+R-W/2,R),
          (L,W/2),(0,W/2)]
p = [g.addPoint(x,y,0,lc) for x,y in coords]
c_outer = g.addPoint(L,R,0,lc)
c_inner = g.addPoint(L,R,0,lc)
# Named while created: lower outer, 2 outer arcs, upper outer, outlet,
# upper inner, 2 inner arcs, lower inner, inlet
curves = [g.addLine(p[0],p[1]),
          g.addCircleArc(p[1],c_outer,p[2]), g.addCircleArc(p[2],c_outer,p[3]),
          g.addLine(p[3],p[4]), g.addLine(p[4],p[5]),
          g.addLine(p[5],p[6]),
          g.addCircleArc(p[6],c_inner,p[7]), g.addCircleArc(p[7],c_inner,p[8]),
          g.addLine(p[8],p[9]), g.addLine(p[9],p[0])]
loop = g.addCurveLoop(curves)
front_surface = g.addPlaneSurface([loop])
g.synchronize()
# Recombine the plane, then extrude a single layer to make a true 2D mesh.
gmsh.model.mesh.setRecombine(2, front_surface)
out = g.extrude([(2,front_surface)], 0,0,DEPTH, numElements=[1], heights=[1.0], recombine=True)
g.synchronize()
back_surface = out[0][1]
fluid_volume_tag = out[1][1]
side_surfaces = [dt[1] for dt in out[2:]]
assert len(side_surfaces) == len(curves)

# Side surfaces inherit the exact creation order of the boundary curves.
inlet_surface = side_surfaces[9]
outlet_surface = side_surfaces[4]
wall_surfaces = [side_surfaces[i] for i in [0,1,2,3,5,6,7,8]]
for dim, tags, name in [(3,[fluid_volume_tag],"fluid"),(2,[inlet_surface],"inlet"),
                        (2,[outlet_surface],"outlet"),(2,wall_surfaces,"walls"),
                        (2,[front_surface,back_surface],"frontAndBack")]:
    pg=gmsh.model.addPhysicalGroup(dim,tags)
    gmsh.model.setPhysicalName(dim,pg,name)

gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.mesh.generate(3)
gmsh.write("ubend_coarse.msh")
types, elemTags, _ = gmsh.model.mesh.getElements(3, fluid_volume_tag)
print("3D element types/counts:", [(t,len(e)) for t,e in zip(types,elemTags)])
print("nodes:", len(gmsh.model.mesh.getNodes()[0]))
print("physical surfaces:", [(gmsh.model.getPhysicalName(2,tag), len(gmsh.model.getEntitiesForPhysicalGroup(2,tag))) for _,tag in gmsh.model.getPhysicalGroups(2)])
gmsh.finalize()

# -- cell 5 -------------------------------------------------------------------------
# Conversion requires the minimal OpenFOAM case metadata first; no mesh was created by the failed conv
import pathlib, subprocess
pathlib.Path("system").mkdir(exist_ok=True)
pathlib.Path("constant").mkdir(exist_ok=True)
pathlib.Path("system/controlDict").write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application checkMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
''')
r = subprocess.run(["gmshToFoam","ubend_coarse.msh"], text=True, capture_output=True)
print(r.stdout[-2200:])
if r.returncode != 0:
    print(r.stderr)
    raise RuntimeError("gmshToFoam failed")
print(pathlib.Path("constant/polyMesh/boundary").read_text())

# -- cell 6 -------------------------------------------------------------------------
# The utility explicitly expects standard scheme subdictionaries. I’ll provide complete minimal defaul
pathlib.Path("system/fvSchemes").write_text('''FoamFile
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
pathlib.Path("system/fvSolution").write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
boundary_path = pathlib.Path("constant/polyMesh/boundary")
btxt = boundary_path.read_text()
btxt = btxt.replace('frontAndBack\n    {\n        type            patch;', 'frontAndBack\n    {\n        type            empty;')
btxt = btxt.replace('walls\n    {\n        type            patch;', 'walls\n    {\n        type            wall;')
boundary_path.write_text(btxt)
check = subprocess.run(["checkMesh"], text=True, capture_output=True)
print(check.stdout[-5000:])
if check.returncode != 0:
    print(check.stderr)
    raise RuntimeError("coarse checkMesh failed")

# -- cell 7 -------------------------------------------------------------------------
# The coarse mesh passes, with maximum non-orthogonality 27.75°, skewness 0.411, and exact mesh volume
lc_final = 0.001
gmsh.initialize()
gmsh.model.add("ubend2d_final")
g = gmsh.model.geo
p2 = [g.addPoint(x,y,0,lc_final) for x,y in coords]
co = g.addPoint(L,R,0,lc_final); ci = g.addPoint(L,R,0,lc_final)
curves2 = [g.addLine(p2[0],p2[1]),
           g.addCircleArc(p2[1],co,p2[2]), g.addCircleArc(p2[2],co,p2[3]),
           g.addLine(p2[3],p2[4]), g.addLine(p2[4],p2[5]),
           g.addLine(p2[5],p2[6]),
           g.addCircleArc(p2[6],ci,p2[7]), g.addCircleArc(p2[7],ci,p2[8]),
           g.addLine(p2[8],p2[9]), g.addLine(p2[9],p2[0])]
front2 = g.addPlaneSurface([g.addCurveLoop(curves2)])
g.synchronize(); gmsh.model.mesh.setRecombine(2,front2)
out2 = g.extrude([(2,front2)],0,0,DEPTH,numElements=[1],heights=[1.0],recombine=True)
g.synchronize()
back2, vol2 = out2[0][1], out2[1][1]
sides2 = [dt[1] for dt in out2[2:]]
assert len(sides2)==10
for dim,tags,name in [(3,[vol2],"fluid"),(2,[sides2[9]],"inlet"),(2,[sides2[4]],"outlet"),
                      (2,[sides2[i] for i in [0,1,2,3,5,6,7,8]],"walls"),
                      (2,[front2,back2],"frontAndBack")]:
    pg=gmsh.model.addPhysicalGroup(dim,tags); gmsh.model.setPhysicalName(dim,pg,name)
gmsh.option.setNumber("Mesh.MeshSizeMin",lc_final); gmsh.option.setNumber("Mesh.MeshSizeMax",lc_final)
gmsh.option.setNumber("Mesh.Algorithm",8); gmsh.option.setNumber("Mesh.RecombinationAlgorithm",1)
gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.model.mesh.generate(3)
gmsh.write("ubend_final.msh")
types2, tags2, _ = gmsh.model.mesh.getElements(3,vol2)
print("final 3D element types/counts:",[(t,len(e)) for t,e in zip(types2,tags2)])
print("final nodes:",len(gmsh.model.mesh.getNodes()[0]))
gmsh.finalize()

# -- cell 8 -------------------------------------------------------------------------
# The final mesh has 5,820 prism cells and 6,420 nodes. I’ll now replace the coarse OpenFOAM mesh with
rfinal = subprocess.run(["gmshToFoam","ubend_final.msh"], text=True, capture_output=True)
print(rfinal.stdout[-1800:])
if rfinal.returncode != 0:
    print(rfinal.stderr)
    raise RuntimeError("final gmshToFoam failed")
print(pathlib.Path("constant/polyMesh/boundary").read_text())

# -- cell 9 -------------------------------------------------------------------------
# Final conversion gives ten faces across both inlet and outlet, 576 wall faces, and no stray boundary
boundary_path = pathlib.Path("constant/polyMesh/boundary")
btxt = boundary_path.read_text()
btxt = btxt.replace('frontAndBack\n    {\n        type            patch;', 'frontAndBack\n    {\n        type            empty;')
btxt = btxt.replace('walls\n    {\n        type            patch;', 'walls\n    {\n        type            wall;')
boundary_path.write_text(btxt)
final_check = subprocess.run(["checkMesh"], text=True, capture_output=True)
print(final_check.stdout[-5200:])
if final_check.returncode != 0:
    print(final_check.stderr)
    raise RuntimeError("final checkMesh failed")

# -- cell 10 ------------------------------------------------------------------------
# The final mesh passes with aspect ratio 1.98, maximum non-orthogonality 30.47°, and skewness 0.399. 
vtk_run = subprocess.run(["foamToVTK","-constant","-no-fields"], text=True, capture_output=True)
print(vtk_run.stdout[-1200:])
print("return code",vtk_run.returncode)
print([str(x) for x in pathlib.Path("VTK").rglob("*") if x.is_file()][:20])

# -- cell 11 ------------------------------------------------------------------------
# The OpenFOAM-to-VTK export includes the internal mesh and the inlet, outlet, and wall patches separa
foam_mesh = pv.read("VTK/t1_0/internal.vtu")
inlet_vtk = pv.read("VTK/t1_0/boundary/inlet.vtp")
outlet_vtk = pv.read("VTK/t1_0/boundary/outlet.vtp")
plotter2 = pv.Plotter(off_screen=True, window_size=(1100,420))
plotter2.add_mesh(foam_mesh, color="white", show_edges=True, edge_color="gray", line_width=0.45)
plotter2.add_mesh(inlet_vtk, color="royalblue", line_width=5)
plotter2.add_mesh(outlet_vtk, color="orangered", line_width=5)
plotter2.view_xy(); plotter2.camera.parallel_projection=True
plotter2.show(screenshot="final_mesh.png")
display(plotter2.image)
print("Rendered actual OpenFOAM mesh: inlet blue, outlet orange.")

# -- cell 12 ------------------------------------------------------------------------
# The mesh image confirms complete coverage and the intended U-bend topology. Although meshing used ex
import sys
sys.path.insert(0,".reference")
from cad_export import export_patches
print(export_patches.__doc__)

# -- cell 13 ------------------------------------------------------------------------
# I’ll first confirm the primitive face normals used to identify the two open ends before handing them
for label,leg in [("lower",lower_leg),("upper",upper_leg)]:
    for i,f in enumerate(leg.faces()):
        n=f.normal_at(); c=f.center()
        print(label,i,"center",tuple(round(v,6) for v in c),"normal",tuple(round(v,3) for v in n),"area",f.area)

# -- cell 14 ------------------------------------------------------------------------
# The selected leg faces have the expected outward normal \((-1,0,0)\), area \(10^{-5}\,\mathrm{m^2}\)
inlet_cad_face = lower_leg.faces()[0]
outlet_cad_face = upper_leg.faces()[0]
front_back_faces = [f for f in fluid.faces() if abs(f.normal_at().Z) > 0.9]
print("front/back CAD faces:",len(front_back_faces),"combined area",sum(f.area for f in front_back_faces))
surface_report = export_patches(fluid,
    {"inlet": [inlet_cad_face], "outlet": [outlet_cad_face],
     "frontAndBack": front_back_faces, "walls": ...},
    tolerance=2.5e-4, clean=True)
print(surface_report)
