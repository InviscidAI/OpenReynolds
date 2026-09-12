"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, subprocess
from build123d import *
p = "/home/qiuzi/.openreynolds-buildup/work/T6-20260912-125146-ec8c/geometry/customer_part.step"
print(os.path.getsize(p))
print(subprocess.run(["head","-45",p],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
print(subprocess.run(["grep","-n","UNIT\\|CONICAL\\|CYLINDRICAL\\|TOROIDAL\\|PLANE\\|MANIFOLD",p],capture_output=True,text=True).stdout[-2500:])

# -- cell 3 -------------------------------------------------------------------------
print(subprocess.run(["sed","-n","495,515p",p],capture_output=True,text=True).stdout)

# -- cell 4 -------------------------------------------------------------------------
# The cell that defined `shp` errored, so it isn't in the script. Re-emitting import + inspection as o
import os, subprocess
from build123d import *

STEP_PATH = "/home/qiuzi/.openreynolds-buildup/work/T6-20260912-125146-ec8c/geometry/customer_part.step"
shp = import_step(STEP_PATH)
print("type:", type(shp), "| solids:", len(shp.solids()), "| faces:", len(shp.faces()))
print("valid:", shp.is_valid, "| bbox (file units):", shp.bounding_box())
print("volume:", shp.volume)
for f in shp.faces():
    print(f.geom_type, "center", f.center(), "area", round(f.area, 3))

# -- cell 5 -------------------------------------------------------------------------
cyl = [f for f in shp.faces() if f.geom_type==GeomType.CYLINDER][0]
print("cyl bbox:", cyl.bounding_box())
print("radius from surface area / (2*pi*L):", cyl.area/(2*3.141592653589793*200))
print("block volume - solid volume =", 200*100*100 - shp.volume)
print("pi r^2 L for r=20:", 3.141592653589793*400*200)

# -- cell 6 -------------------------------------------------------------------------
# Extracting the fluid volume as the negative of the part inside its own bounding block, then scaling 
bb = shp.bounding_box()
block = Box(bb.size.X, bb.size.Y, bb.size.Z).moved(Location(bb.center()))
fluid_mm = block - shp
print("fluid solids:", len(fluid_mm.solids()), "valid:", fluid_mm.is_valid)
print("fluid volume mm^3:", fluid_mm.volume, " bbox:", fluid_mm.bounding_box())

MM = 0.001
fluid = scale(fluid_mm, MM)
print("fluid volume m^3:", fluid.volume, " bbox m:", fluid.bounding_box())
for f in fluid.faces():
    print(f.geom_type, "area", round(f.area,6), "center", f.center())

# -- cell 7 -------------------------------------------------------------------------
import pyvista as pv, numpy as np
export_step(fluid, "fluid.step")
export_stl(fluid, "fluid_check.stl", tolerance=1e-4)
m = pv.read("fluid_check.stl")
pl = pv.Plotter(off_screen=True, window_size=(800,500))
pl.add_mesh(m, color="lightblue", show_edges=True)
pl.add_axes(); pl.camera_position='iso'
pl.screenshot("fluid_view.png")
print("step bytes", os.path.getsize("fluid.step"))

# -- cell 8 -------------------------------------------------------------------------
from IPython.display import Image, display
print(os.path.exists("fluid_view.png"), os.path.getsize("fluid_view.png") if os.path.exists("fluid_view.png") else 0)
display(Image("fluid_view.png"))

# -- cell 9 -------------------------------------------------------------------------
# Fluid is a straight circular duct Ø0.04 m × 0.2 m. Now mesh it coarsely in gmsh, naming patches from
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("fluid")
gmsh.model.occ.importShapes("fluid.step")
gmsh.model.occ.synchronize()
print("bbox from gmsh:", gmsh.model.getBoundingBox(-1,-1))
for dim,tag in gmsh.model.getEntities(2):
    print(tag, gmsh.model.getType(dim,tag), gmsh.model.occ.getCenterOfMass(dim,tag),
          gmsh.model.getNormal(tag,[0.5,0.5]))

# -- cell 10 ------------------------------------------------------------------------
import numpy as np
inlet=outlet=None; walls=[]
for dim,tag in gmsh.model.getEntities(2):
    t = gmsh.model.getType(dim,tag)
    if t=="Plane":
        n = np.array(gmsh.model.getNormal(tag,[0.5,0.5]))
        if n[0] < 0: inlet = tag
        else: outlet = tag
    else:
        walls.append(tag)
print("inlet",inlet,"outlet",outlet,"walls",walls)
gmsh.model.addPhysicalGroup(2,[inlet],name="inlet")
gmsh.model.addPhysicalGroup(2,[outlet],name="outlet")
gmsh.model.addPhysicalGroup(2,walls,name="wall")
gmsh.model.addPhysicalGroup(3,[v[1] for v in gmsh.model.getEntities(3)],name="internal")
gmsh.option.setNumber("Mesh.MeshSizeMin",0.006)
gmsh.option.setNumber("Mesh.MeshSizeMax",0.006)
gmsh.model.mesh.generate(3)
gmsh.write("fluid.msh")
print("nodes",len(gmsh.model.mesh.getNodes()[0]))
gmsh.finalize()

# -- cell 11 ------------------------------------------------------------------------
import gmsh, numpy as np, os
def build_mesh(lc, fname):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("fluid")
    gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
    inlet=outlet=None; walls=[]
    for dim,tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(dim,tag)=="Plane":
            n=np.array(gmsh.model.getNormal(tag,[0.5,0.5]))
            if n[0]<0: inlet=tag
            else: outlet=tag
        else: walls.append(tag)
    gmsh.model.addPhysicalGroup(2,[inlet],name="inlet")
    gmsh.model.addPhysicalGroup(2,[outlet],name="outlet")
    gmsh.model.addPhysicalGroup(2,walls,name="wall")
    gmsh.model.addPhysicalGroup(3,[v[1] for v in gmsh.model.getEntities(3)],name="internal")
    gmsh.option.setNumber("Mesh.MeshSizeMin",lc); gmsh.option.setNumber("Mesh.MeshSizeMax",lc)
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    n=len(gmsh.model.mesh.getNodes()[0]); gmsh.finalize(); return n

print("nodes:", build_mesh(0.006,"fluid.msh"))

# -- cell 12 ------------------------------------------------------------------------
# Now the OpenFOAM case skeleton and the conversion.
os.makedirs("system",exist_ok=True); os.makedirs("constant",exist_ok=True)
hdr = lambda cls,obj: f"""FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"""
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+
 "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} \n")
r = subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 13 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read())

# -- cell 14 ------------------------------------------------------------------------
b = open("constant/polyMesh/boundary").read().replace(
"""    wall
    {
        type            patch;
        physicalType    patch;""",
"""    wall
    {
        type            wall;
        physicalType    wall;""")
open("constant/polyMesh/boundary","w").write(b)
r = subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[-2500:])

# -- cell 15 ------------------------------------------------------------------------
# Coarse mesh is valid. Refining with curvature resolution so the bore circle is captured (coarse volu
import gmsh, numpy as np
def build_mesh2(lc, fname, curv=40):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("fluid")
    gmsh.model.occ.importShapes("fluid.step"); gmsh.model.occ.synchronize()
    inlet=outlet=None; walls=[]
    for dim,tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(dim,tag)=="Plane":
            n=np.array(gmsh.model.getNormal(tag,[0.5,0.5]))
            if n[0]<0: inlet=tag
            else: outlet=tag
        else: walls.append(tag)
    gmsh.model.addPhysicalGroup(2,[inlet],name="inlet")
    gmsh.model.addPhysicalGroup(2,[outlet],name="outlet")
    gmsh.model.addPhysicalGroup(2,walls,name="wall")
    gmsh.model.addPhysicalGroup(3,[v[1] for v in gmsh.model.getEntities(3)],name="internal")
    gmsh.option.setNumber("Mesh.MeshSizeMin",lc*0.5); gmsh.option.setNumber("Mesh.MeshSizeMax",lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",curv)
    gmsh.option.setNumber("Mesh.Optimize",1); gmsh.option.setNumber("Mesh.OptimizeNetgen",1)
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    n=len(gmsh.model.mesh.getNodes()[0]); gmsh.finalize(); return n
print("nodes:", build_mesh2(0.003,"fluid.msh"))

# -- cell 16 ------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
subprocess.run(["gmshToFoam","fluid.msh"],capture_output=True,text=True)
b = open("constant/polyMesh/boundary").read().replace("""    wall
    {
        type            patch;
        physicalType    patch;""","""    wall
    {
        type            wall;
        physicalType    wall;""")
open("constant/polyMesh/boundary","w").write(b)
r = subprocess.run(["checkMesh"],capture_output=True,text=True)
print(r.stdout[r.stdout.find("Mesh stats"):][:600])
print(r.stdout[r.stdout.find("Checking geometry"):][:1400])

# -- cell 17 ------------------------------------------------------------------------
# Mesh OK, volume within 0.27 % of exact. A look at it, with patch measurements.
open("case.foam","w").write("")
mesh = pv.OpenFOAMReader("case.foam"); mesh.set_active_time_value(0.0) if mesh.time_values else None
d = mesh.read()
internal = d["internalMesh"]
pl = pv.Plotter(off_screen=True, window_size=(900,500))
pl.add_mesh(internal.clip('y', origin=(0.1,0.05,0.05)), show_edges=True, color="lightsteelblue")
pl.add_mesh(d["boundary"]["inlet"], color="red")
pl.add_mesh(d["boundary"]["outlet"], color="green")
pl.camera_position='iso'; pl.add_axes()
pl.screenshot("mesh_view.png")
display(Image("mesh_view.png"))
import numpy as np
bp = d["boundary"]
for n in ["inlet","outlet","wall"]:
    print(n, "area", bp[n].compute_cell_sizes(length=False,volume=False)["Area"].sum())
print("exact inlet area pi*0.02^2 =", np.pi*0.02**2, " wall 2*pi*0.02*0.2 =", 2*np.pi*0.02*0.2)
