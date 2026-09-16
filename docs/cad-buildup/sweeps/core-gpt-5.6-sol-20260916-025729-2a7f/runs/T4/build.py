"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water volume as a sealed rectangular groove: three 60 mm-direction passes on 7 mm pit
import build123d as bd
import numpy as np
import pyvista as pv
from IPython.display import display

# Given dimensions, in metres
PLATE_L, PLATE_W, PLATE_H = 0.060, 0.040, 0.008
CHANNEL_W, CHANNEL_D, METAL_WEB = 0.004, 0.005, 0.003
PITCH = CHANNEL_W + METAL_WEB
Z0 = PLATE_H - CHANNEL_D
R_CENTER = PITCH / 2
R_OUTER, R_INNER = R_CENTER + CHANNEL_W/2, R_CENTER - CHANNEL_W/2
X_BEND = PLATE_L/2 - R_OUTER
Y_PASS = (-PITCH, 0.0, PITCH)

# Straight runs; U-bends are clipped annuli and joined to these boxes.
run1 = bd.Pos((-PLATE_L/2 + X_BEND)/2, Y_PASS[0], Z0 + CHANNEL_D/2) * bd.Box(X_BEND + PLATE_L/2, CHANNEL_W, CHANNEL_D)
run2 = bd.Pos(0, Y_PASS[1], Z0 + CHANNEL_D/2) * bd.Box(2*X_BEND, CHANNEL_W, CHANNEL_D)
run3 = bd.Pos((X_BEND + PLATE_L/2)/2, Y_PASS[2], Z0 + CHANNEL_D/2) * bd.Box(PLATE_L/2 + X_BEND, CHANNEL_W, CHANNEL_D)
right_ring = (bd.Pos(X_BEND, -PITCH/2, Z0) * bd.Cylinder(R_OUTER, CHANNEL_D)) - (bd.Pos(X_BEND, -PITCH/2, Z0) * bd.Cylinder(R_INNER, CHANNEL_D))
right_clip = bd.Pos(X_BEND + R_OUTER/2, -PITCH/2, Z0 + CHANNEL_D/2) * bd.Box(R_OUTER, 2*R_OUTER, CHANNEL_D)
left_ring = (bd.Pos(-X_BEND, PITCH/2, Z0) * bd.Cylinder(R_OUTER, CHANNEL_D)) - (bd.Pos(-X_BEND, PITCH/2, Z0) * bd.Cylinder(R_INNER, CHANNEL_D))
left_clip = bd.Pos(-X_BEND - R_OUTER/2, PITCH/2, Z0 + CHANNEL_D/2) * bd.Box(R_OUTER, 2*R_OUTER, CHANNEL_D)
fluid = run1 + run2 + run3 + (right_ring & right_clip) + (left_ring & left_clip)

bb = fluid.bounding_box()
print(f"Channel width requested/measured from straight section: {CHANNEL_W*1e3:.3f} / {run2.bounding_box().size.Y*1e3:.3f} mm")
print(f"Channel depth requested/measured: {CHANNEL_D*1e3:.3f} / {bb.size.Z*1e3:.3f} mm")
print(f"Metal between passes requested/measured: {METAL_WEB*1e3:.3f} / {(PITCH-CHANNEL_W)*1e3:.3f} mm")
print(f"Pass count requested/built: 3 / {len(Y_PASS)}")
print(f"Fluid bounds (mm): x=[{bb.min.X*1e3:.1f},{bb.max.X*1e3:.1f}], y=[{bb.min.Y*1e3:.1f},{bb.max.Y*1e3:.1f}], z=[{bb.min.Z*1e3:.1f},{bb.max.Z*1e3:.1f}]")
print(f"Fluid volume: {fluid.volume*1e6:.3f} cm^3; solids: {len(fluid.solids())}")

# Coarse visual inspection of the exact CAD volume.
verts, faces = fluid.tessellate(2e-4)
pts = np.array([[v.X,v.Y,v.Z] for v in verts])
tri = np.hstack([np.full((len(faces),1),3), np.array(faces,dtype=int)])
poly = pv.PolyData(pts, tri)
pl = pv.Plotter(off_screen=True, window_size=(900,550))
pl.add_mesh(poly, color="#39a9dc", show_edges=True, opacity=0.85)
pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.3)
pl.show(screenshot="coarse_geometry.png")
display("coarse_geometry.png")

# -- cell 2 -------------------------------------------------------------------------
# The first inspection caught a construction error: the right return extended to 54.5 mm and the volum
for name, shape in [("run1",run1),("run2",run2),("run3",run3),("right_ring",right_ring),("right_clip",right_clip),("right_half",right_ring & right_clip),("left_ring",left_ring),("left_clip",left_clip),("left_half",left_ring & left_clip)]:
    b=shape.bounding_box()
    print(name, "x", b.min.X, b.max.X, "y", b.min.Y, b.max.Y, "solids",len(shape.solids()))

# -- cell 3 -------------------------------------------------------------------------
# The U-bends are correctly located; the error is the third straight’s centre coordinate. It must run 
run3 = bd.Pos((-X_BEND + PLATE_L/2)/2, Y_PASS[2], Z0 + CHANNEL_D/2) * bd.Box(PLATE_L/2 + X_BEND, CHANNEL_W, CHANNEL_D)
fluid = run1 + run2 + run3 + (right_ring & right_clip) + (left_ring & left_clip)
bb = fluid.bounding_box()
print(f"Fluid bounds (mm): x=[{bb.min.X*1e3:.1f},{bb.max.X*1e3:.1f}], y=[{bb.min.Y*1e3:.1f},{bb.max.Y*1e3:.1f}], z=[{bb.min.Z*1e3:.1f},{bb.max.Z*1e3:.1f}]")
print(f"Connected solids requested/measured: 1 / {len(fluid.solids())}")
print(f"Fluid volume: {fluid.volume*1e6:.3f} cm^3")
verts, faces = fluid.tessellate(1.5e-4)
pts = np.array([[v.X,v.Y,v.Z] for v in verts]); tri=np.hstack([np.full((len(faces),1),3),np.array(faces,dtype=int)])
poly=pv.PolyData(pts,tri)
pl=pv.Plotter(off_screen=True,window_size=(900,550)); pl.add_mesh(poly,color="#39a9dc",show_edges=True,opacity=0.9)
pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.35); pl.show(screenshot="coarse_geometry_corrected.png")
display("coarse_geometry_corrected.png")

# -- cell 4 -------------------------------------------------------------------------
# The corrected CAD is one connected 60 × 18 × 5 mm water volume with both edge openings in the intend
import gmsh, os
bd.export_step(fluid, "fluid_volume.step")
gmsh.initialize()
gmsh.model.add("cold_plate_water")
gmsh.model.occ.importShapes("fluid_volume.step")
gmsh.model.occ.synchronize()
vols=gmsh.model.getEntities(3); surfs=gmsh.model.getEntities(2)
print("Imported volumes:", vols, "surface count:", len(surfs))
for _,tag in surfs:
    com=gmsh.model.occ.getCenterOfMass(2,tag)
    area=gmsh.model.occ.getMass(2,tag)
    print(f"surface {tag:2d}: area={area*1e6:8.3f} mm2, COM=({com[0]*1e3:7.2f},{com[1]*1e3:7.2f},{com[2]*1e3:6.2f}) mm")
gmsh.finalize()

# -- cell 5 -------------------------------------------------------------------------
# The conversion itself succeeded, but the reproducibility guard correctly rejected the step because i
import pathlib, subprocess, textwrap, gmsh
pathlib.Path("system").mkdir(exist_ok=True); pathlib.Path("constant").mkdir(exist_ok=True)
pathlib.Path("system/controlDict").write_text(textwrap.dedent('''FoamFile
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
'''))
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.model.add("cold_plate_water")
gmsh.model.occ.importShapes("fluid_volume.step"); gmsh.model.occ.synchronize()
surf_tags=[t for d,t in gmsh.model.getEntities(2)]; inlet=[]; outlet=[]
for t in surf_tags:
    x,y,z=gmsh.model.occ.getCenterOfMass(2,t); a=gmsh.model.occ.getMass(2,t)
    if abs(x+PLATE_L/2)<1e-8 and abs(y+PITCH)<1e-8 and abs(a-CHANNEL_W*CHANNEL_D)<1e-8: inlet.append(t)
    if abs(x-PLATE_L/2)<1e-8 and abs(y-PITCH)<1e-8 and abs(a-CHANNEL_W*CHANNEL_D)<1e-8: outlet.append(t)
assert len(inlet)==len(outlet)==1
walls=[t for t in surf_tags if t not in inlet+outlet]
for tags,name in [(inlet,"inlet"),(outlet,"outlet"),(walls,"walls")]:
    p=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,p,name)
vols=[t for d,t in gmsh.model.getEntities(3)]; p=gmsh.model.addPhysicalGroup(3,vols); gmsh.model.setPhysicalName(3,p,"fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin",0.001); gmsh.option.setNumber("Mesh.MeshSizeMax",0.001)
gmsh.model.mesh.generate(3); nn=len(gmsh.model.mesh.getNodes()[0])
gmsh.option.setNumber("Mesh.MshFileVersion",2.2); gmsh.write("cold_plate_water.msh"); gmsh.finalize()
res=subprocess.run(["gmshToFoam","cold_plate_water.msh"],text=True,capture_output=True)
print(f"Coarse Gmsh nodes: {nn}; inlet/outlet/wall CAD faces: {len(inlet)}/{len(outlet)}/{len(walls)}")
print(res.stdout[-1200:]); print(res.stderr[-300:])
if res.returncode: raise RuntimeError(f"gmshToFoam failed {res.returncode}")

# -- cell 6 -------------------------------------------------------------------------
# I’ll now inspect the converted patch counts and run the binding OpenFOAM mesh-quality check. This wi
boundary_text=pathlib.Path("constant/polyMesh/boundary").read_text()
print(boundary_text)
chk=subprocess.run(["checkMesh","-allGeometry","-allTopology"],text=True,capture_output=True)
pathlib.Path("checkMesh.log").write_text(chk.stdout+chk.stderr)
print(chk.stdout[-5000:]); print(chk.stderr[-500:])
print("checkMesh return code:",chk.returncode)

# -- cell 7 -------------------------------------------------------------------------
# The conversion produced only the three intended patches—54 inlet faces, 54 outlet faces, and 8,208 w
pathlib.Path("system/fvSchemes").write_text(textwrap.dedent('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''))
pathlib.Path("system/fvSolution").write_text(textwrap.dedent('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
'''))
chk=subprocess.run(["checkMesh","-allGeometry","-allTopology"],text=True,capture_output=True)
pathlib.Path("checkMesh.log").write_text(chk.stdout+chk.stderr)
print(chk.stdout[-6500:]); print(chk.stderr[-500:])
if chk.returncode: raise RuntimeError(f"checkMesh failed {chk.returncode}")

# -- cell 8 -------------------------------------------------------------------------
# All topology, volume, skewness, and non-orthogonality checks pass. The optional exhaustive determina
chk_default=subprocess.run(["checkMesh"],text=True,capture_output=True)
print(chk_default.stdout[-5000:]); print(chk_default.stderr[-300:])
pathlib.Path("checkMesh_default.log").write_text(chk_default.stdout+chk_default.stderr)
print("return code:",chk_default.returncode,"mesh OK marker:","Mesh OK" in chk_default.stdout)

# -- cell 9 -------------------------------------------------------------------------
# The binding check reports `Mesh OK`: one region, 18,410 tetrahedra, closed boundary, maximum non-ort
vtkres=subprocess.run(["foamToVTK","-constant","-no-fields"],text=True,capture_output=True)
print(vtkres.stdout[-1200:]); print(vtkres.stderr[-300:])
if vtkres.returncode: raise RuntimeError("foamToVTK failed")
files=list(pathlib.Path("VTK").rglob("*.vtu")); print("VTU files:",[str(f) for f in files])
mesh_vtk=pv.read(str(files[0])); surf_vtk=mesh_vtk.extract_surface()
pl=pv.Plotter(off_screen=True,window_size=(1000,600)); pl.add_mesh(surf_vtk,color="#57b7dd",show_edges=True,edge_color="#253746",line_width=0.4)
pl.add_axes(); pl.view_isometric(); pl.camera.zoom(1.35); pl.show(screenshot="openfoam_mesh.png")
display("openfoam_mesh.png")

# -- cell 10 ------------------------------------------------------------------------
# The final-mesh rendering confirms the three-pass serpentine, both U-bends, and the two edge openings
patch_dir=pathlib.Path("geometry"); patch_dir.mkdir(exist_ok=True)
patch_meshes=[]
for name in ("inlet","outlet","walls"):
    p=pv.read(f"VTK/t4_0/boundary/{name}.vtp").triangulate()
    p.save(str(patch_dir/f"{name}.stl"))
    patch_meshes.append(p)
    print(name,"triangles",p.n_cells,"bounds",np.array(p.bounds).reshape(3,2))
union=patch_meshes[0].merge(patch_meshes[1:],merge_points=True,tolerance=1e-10).clean(tolerance=1e-10)
free_edges=union.extract_feature_edges(boundary_edges=True,feature_edges=False,manifold_edges=False,non_manifold_edges=False)
nonman=union.extract_feature_edges(boundary_edges=False,feature_edges=False,manifold_edges=False,non_manifold_edges=True)
print("Welded STL union: triangles",union.n_cells,"free edges",free_edges.n_cells,"non-manifold edges",nonman.n_cells)
