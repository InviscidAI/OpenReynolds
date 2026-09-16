"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the **air volume**, not the solid rotor: a 12 mm axial annulus from radius 95 to 160 mm, 
import build123d as bd
import numpy as np
from pathlib import Path
# All dimensions in metres (request converted once from mm).
RO, RI = 0.160, 0.095
FACE_T, GAP = 0.010, 0.012
N_VANES, VANE_W = 36, 0.008
PITCH_DEG = 360.0 / N_VANES
annular_gap = bd.Cylinder(RO, GAP) - bd.Cylinder(RI, GAP)
radial_vane = bd.Pos(RI, -VANE_W/2, 0) * bd.Box(RO-RI, VANE_W, GAP)
vane_solids = [bd.Rot(0, 0, i*PITCH_DEG) * radial_vane for i in range(N_VANES)]
vanes_union = vane_solids[0]
for v in vane_solids[1:]:
    vanes_union = vanes_union.fuse(v)
air = annular_gap - vanes_union
print(f"Requested/constructed radii: inner {RI:.6f} m, outer {RO:.6f} m")
print(f"Requested/constructed axial gap: {GAP:.6f} m")
print(f"Requested/constructed vane count: {N_VANES}; pitch {PITCH_DEG:.6f} deg; width {VANE_W:.6f} m")
print("Air solids/passages:", len(air.solids()), "volume [m^3]:", air.volume)
print("Bounds:", air.bounding_box().min, air.bounding_box().max)

# -- cell 2 -------------------------------------------------------------------------
# The first measurement exposed two construction issues before meshing: build123d centers cylinders ax
import inspect
print(inspect.signature(bd.Box))
print(inspect.signature(bd.Cylinder))
print(bd.Align.MIN, bd.Align.CENTER)

# -- cell 3 -------------------------------------------------------------------------
# I’ll use explicit `(MIN, CENTER, MIN)` alignment for each radial vane and `(CENTER, CENTER, MIN)` fo
annular_gap = bd.Cylinder(RO, GAP, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)) - bd.Cylinder(RI, GAP, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
radial_vane = bd.Pos(RI, 0, 0) * bd.Box(RO-RI, VANE_W, GAP, align=(bd.Align.MIN,bd.Align.CENTER,bd.Align.MIN))
vane_solids = [bd.Rot(0, 0, i*PITCH_DEG) * radial_vane for i in range(N_VANES)]
vanes_union = vane_solids[0]
for v in vane_solids[1:]:
    vanes_union = vanes_union.fuse(v)
air = annular_gap - vanes_union
bb = air.bounding_box()
print("Air solids/passages (target 36):", len(air.solids()))
print(f"Axial bounds (target 0 to {GAP} m): {bb.min.Z:.6f} to {bb.max.Z:.6f} m")
print(f"Each vane radial span target/measured: {RO-RI:.6f}/{radial_vane.bounding_box().max.X-radial_vane.bounding_box().min.X:.6f} m")
print(f"Vane tangential width target/measured: {VANE_W:.6f}/{radial_vane.bounding_box().max.Y-radial_vane.bounding_box().min.Y:.6f} m")
print("Fluid volume [m^3]:", air.volume)

# -- cell 4 -------------------------------------------------------------------------
# The corrected solid now has exactly 36 disconnected passages and the requested dimensions. I’ll tess
import pyvista as pv
Path("geometry").mkdir(exist_ok=True)
bd.export_stl(air, "geometry/air_volume_preview.stl", tolerance=2e-4, angular_tolerance=0.15)
pv.OFF_SCREEN = True
surf = pv.read("geometry/air_volume_preview.stl")
pl = pv.Plotter(off_screen=True, window_size=(900,700))
pl.add_mesh(surf, color="lightblue", show_edges=True, edge_color="navy", line_width=1)
pl.camera_position = [(0.28,-0.30,0.22),(0,0,0.006),(0,0,1)]
pl.add_axes(); pl.show(screenshot="geometry/coarse_geometry.png")
print("Preview facets:", surf.n_cells, "saved geometry/coarse_geometry.png")

# -- cell 5 -------------------------------------------------------------------------
# The preview shows the intended 36 open radial channels. I’ll now classify boundary faces by their ac
from collections import Counter
faces = list(air.faces())
print("Total faces:", len(faces))
print("Geometry types:", Counter(str(f.geom_type) for f in faces))
for j,f in enumerate(faces[:12]):
    c=f.center(); n=f.normal_at();
    print(j, str(f.geom_type), f"area={f.area:.8g}", f"c=({c.X:.4g},{c.Y:.4g},{c.Z:.4g})", f"n=({n.X:.3g},{n.Y:.3g},{n.Z:.3g})")

# -- cell 6 -------------------------------------------------------------------------
# The boolean has 288 faces: 72 horizontal friction faces, 72 cylindrical inlet/outlet strips, and 144
horizontal = [f for f in faces if f.geom_type == bd.GeomType.PLANE and abs(f.normal_at().Z) > 0.99]
frictionFace1 = [f for f in horizontal if f.normal_at().Z < 0]
frictionFace2 = [f for f in horizontal if f.normal_at().Z > 0]
cyls = [f for f in faces if f.geom_type == bd.GeomType.CYLINDER]
innerInlet = [f for f in cyls if (f.center().X**2+f.center().Y**2)**0.5 < (RI+RO)/2]
outerOutlet = [f for f in cyls if (f.center().X**2+f.center().Y**2)**0.5 > (RI+RO)/2]
used = {id(f) for f in frictionFace1+frictionFace2+innerInlet+outerOutlet}
vaneSurfaces = [f for f in faces if id(f) not in used]
patches = {"frictionFace1":frictionFace1,"frictionFace2":frictionFace2,"vaneSurfaces":vaneSurfaces,"innerInlet":innerInlet,"outerOutlet":outerOutlet}
for name,fs in patches.items(): print(name, "faces",len(fs),"area",sum(f.area for f in fs))
print("Coverage faces target/assigned:",len(faces),sum(map(len,patches.values())))

# -- cell 7 -------------------------------------------------------------------------
# All 288 boundary faces are assigned exactly once. I’ll export the requested five patch files—two fri
Path("constant/triSurface").mkdir(parents=True, exist_ok=True)
for name,fs in patches.items():
    shape = bd.Compound(children=fs)
    bd.export_stl(shape, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.10)
print("Exported:", sorted(p.name for p in Path("constant/triSurface").glob("*.stl")))
print("STL sizes:", {p.name:p.stat().st_size for p in Path("constant/triSurface").glob("*.stl")})

# -- cell 8 -------------------------------------------------------------------------
# The STL patches are in place. For meshing I’ll export the exact 36-solid air volume as STEP and impo
bd.export_step(air, "geometry/air_volume.step")
print("STEP exported bytes:", Path("geometry/air_volume.step").stat().st_size)

# -- cell 9 -------------------------------------------------------------------------
# I’ll import the STEP into Gmsh and inspect its entity types and counts before assigning physical gro
import gmsh
gmsh.initialize()
gmsh.model.add("vented_disc_air")
gmsh.model.occ.importShapes("geometry/air_volume.step", highestDimOnly=False)
gmsh.model.occ.synchronize()
ents3 = gmsh.model.getEntities(3); ents2 = gmsh.model.getEntities(2)
print("Gmsh volumes:",len(ents3),"surfaces:",len(ents2))
print("Surface types:",Counter(gmsh.model.getType(2,t) for _,t in ents2))

# -- cell 10 ------------------------------------------------------------------------
# The imported topology matches exactly. I’ll now assign Gmsh physical groups using surface type and a
gfaces={"frictionFace1":[],"frictionFace2":[],"vaneSurfaces":[],"innerInlet":[],"outerOutlet":[]}
for _,tag in ents2:
    typ=gmsh.model.getType(2,tag)
    xmin,ymin,zmin,xmax,ymax,zmax=gmsh.model.getBoundingBox(2,tag)
    if typ=="Cylinder":
        x,y,z=gmsh.model.occ.getCenterOfMass(2,tag); r=(x*x+y*y)**0.5
        gfaces["innerInlet" if r<(RI+RO)/2 else "outerOutlet"].append(tag)
    elif max(abs(zmin),abs(zmax)) < 1e-7:
        gfaces["frictionFace1"].append(tag)
    elif max(abs(zmin-GAP),abs(zmax-GAP)) < 1e-7:
        gfaces["frictionFace2"].append(tag)
    else:
        gfaces["vaneSurfaces"].append(tag)
for name,tags in gfaces.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
fluid_pg=gmsh.model.addPhysicalGroup(3,[t for _,t in ents3]); gmsh.model.setPhysicalName(3,fluid_pg,"fluid")
print({k:len(v) for k,v in gfaces.items()},"volumes",len(ents3))

# -- cell 11 ------------------------------------------------------------------------
# The first axial-plane test was too strict for STEP import tolerance: the lower friction faces were a
vals=[]
for _,tag in ents2:
 if gmsh.model.getType(2,tag)=="Plane":
  bbx=gmsh.model.getBoundingBox(2,tag); vals.append((bbx[2],bbx[5]))
print("Smallest distinct plane z bounds:", sorted(set((round(a,9),round(b,9)) for a,b in vals))[:8])

# -- cell 12 ------------------------------------------------------------------------
# The STEP face tolerance is ±0.1 µm. I’ll rebuild the physical groups using a 0.5 µm axial tolerance;
gmsh.model.removePhysicalGroups()
gfaces={"frictionFace1":[],"frictionFace2":[],"vaneSurfaces":[],"innerInlet":[],"outerOutlet":[]}
ZTOL=5e-7
for _,tag in ents2:
    typ=gmsh.model.getType(2,tag); bbx=gmsh.model.getBoundingBox(2,tag)
    if typ=="Cylinder":
        x,y,z=gmsh.model.occ.getCenterOfMass(2,tag); r=(x*x+y*y)**0.5
        gfaces["innerInlet" if r<(RI+RO)/2 else "outerOutlet"].append(tag)
    elif abs(bbx[2])<ZTOL and abs(bbx[5])<ZTOL:
        gfaces["frictionFace1"].append(tag)
    elif abs(bbx[2]-GAP)<ZTOL and abs(bbx[5]-GAP)<ZTOL:
        gfaces["frictionFace2"].append(tag)
    else: gfaces["vaneSurfaces"].append(tag)
for name,tags in gfaces.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
fluid_pg=gmsh.model.addPhysicalGroup(3,[t for _,t in ents3]); gmsh.model.setPhysicalName(3,fluid_pg,"fluid")
print({k:len(v) for k,v in gfaces.items()},"coverage",sum(map(len,gfaces.values())))

# -- cell 13 ------------------------------------------------------------------------
# I’ll re-emit the conversion cell with its own `subprocess` import so the final `build.py` is reprodu
import subprocess
Path("system").mkdir(exist_ok=True)
Path("system/controlDict").write_text('''FoamFile
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
r=subprocess.run(["gmshToFoam","ventedDiscAir.msh"],text=True,capture_output=True)
print(r.stdout[-1200:]); print(r.stderr[-300:]); print("returncode",r.returncode)
if r.returncode: raise RuntimeError("gmshToFoam failed")

# -- cell 14 ------------------------------------------------------------------------
# The coarse mesh now exists with 23,390 tetrahedra. I’ll run `checkMesh` immediately and inspect the 
r=subprocess.run(["checkMesh","-allTopology","-allGeometry"],text=True,capture_output=True)
Path("checkMesh.log").write_text(r.stdout+r.stderr)
print(r.stdout[-5000:]); print("returncode",r.returncode)
print("--- boundary ---")
print(Path("constant/polyMesh/boundary").read_text()[:5000])

# -- cell 15 ------------------------------------------------------------------------
# `checkMesh` exited before reporting topology, so I’ll inspect its captured stderr/log tail rather th
txt=Path("checkMesh.log").read_text()
print(txt[-6000:])

# -- cell 16 ------------------------------------------------------------------------
# The failure is only missing standard OpenFOAM dictionaries, not mesh topology. The converter’s regio
Path("system/fvSchemes").write_text('''FoamFile{format ascii; class dictionary; object fvSchemes;}
ddtSchemes{default steadyState;}
gradSchemes{default Gauss linear;}
divSchemes{default none;}
laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;}
snGradSchemes{default corrected;}
''')
Path("system/fvSolution").write_text('''FoamFile{format ascii; class dictionary; object fvSolution;}
solvers{}
''')
bp=Path("constant/polyMesh/boundary")
s=bp.read_text()
rename={"patch0":"outerOutlet","patch1":"vaneSurfaces","patch2":"frictionFace1","patch3":"frictionFace2","patch4":"innerInlet"}
for old,new in rename.items(): s=s.replace("    "+old+"\n","    "+new+"\n")
bp.write_text(s)
print([line.strip() for line in s.splitlines() if line.strip() in rename.values()])

# -- cell 17 ------------------------------------------------------------------------
# I’ll now run standard `checkMesh`, which is the binding validity check, and save the complete report
r=subprocess.run(["checkMesh"],text=True,capture_output=True)
Path("checkMesh.log").write_text(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print("returncode",r.returncode)

# -- cell 18 ------------------------------------------------------------------------
# The coarse mesh passes `checkMesh` (“Mesh OK”) and correctly reports 36 disconnected flow regions, o
r=subprocess.run(["foamToVTK","-constant","-no-fields"],text=True,capture_output=True)
print(r.stdout[-1500:],r.stderr[-500:],"returncode",r.returncode)
print([str(p) for p in Path("VTK").rglob("*.vtk")][:20])

# -- cell 19 ------------------------------------------------------------------------
# VTK export confirms all five named boundary patches are present. I’ll render half of the volume mesh
vol=pv.read("VTK/t22_0/internal.vtu")
half=vol.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
inlet=pv.read("VTK/t22_0/boundary/innerInlet.vtp")
outlet=pv.read("VTK/t22_0/boundary/outerOutlet.vtp")
pl=pv.Plotter(off_screen=True,window_size=(1000,750))
pl.add_mesh(half,color="lightsteelblue",show_edges=True,edge_color="navy",line_width=0.4)
pl.add_mesh(inlet,color="limegreen",label="innerInlet")
pl.add_mesh(outlet,color="tomato",label="outerOutlet")
pl.add_legend(); pl.add_axes()
pl.camera_position=[(0.34,-0.38,0.24),(0,0,0.006),(0,0,1)]
pl.show(screenshot="geometry/coarse_mesh.png")
print("Rendered clipped cells:",half.n_cells,"of",vol.n_cells,"to geometry/coarse_mesh.png")

# -- cell 20 ------------------------------------------------------------------------
# The rendered mesh covers all 36 passages and the inlet/outlet patches align correctly. One reproduci
if not gmsh.isInitialized():
    gmsh.initialize(); gmsh.model.add("vented_disc_air_replay")
    gmsh.model.occ.importShapes("geometry/air_volume.step", highestDimOnly=False); gmsh.model.occ.synchronize()
    ents3=gmsh.model.getEntities(3); ents2=gmsh.model.getEntities(2)
    gg={"frictionFace1":[],"frictionFace2":[],"vaneSurfaces":[],"innerInlet":[],"outerOutlet":[]}
    for _,tag in ents2:
        typ=gmsh.model.getType(2,tag); bbx=gmsh.model.getBoundingBox(2,tag)
        if typ=="Cylinder":
            x,y,z=gmsh.model.occ.getCenterOfMass(2,tag); r=(x*x+y*y)**0.5
            gg["innerInlet" if r<(RI+RO)/2 else "outerOutlet"].append(tag)
        elif abs(bbx[2])<5e-7 and abs(bbx[5])<5e-7: gg["frictionFace1"].append(tag)
        elif abs(bbx[2]-GAP)<5e-7 and abs(bbx[5]-GAP)<5e-7: gg["frictionFace2"].append(tag)
        else: gg["vaneSurfaces"].append(tag)
    for name,tags in gg.items():
        ptag=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,ptag,name)
    ptag=gmsh.model.addPhysicalGroup(3,[t for _,t in ents3]); gmsh.model.setPhysicalName(3,ptag,"fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin",0.004); gmsh.option.setNumber("Mesh.MeshSizeMax",0.005)
gmsh.option.setNumber("Mesh.Algorithm3D",1); gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.model.mesh.generate(3); gmsh.write("ventedDiscAir.msh")
print("Reproducible coarse MSH written; nodes:",len(gmsh.model.mesh.getNodes()[0]))
gmsh.finalize()

# -- cell 21 ------------------------------------------------------------------------
# The reproducible mesh-generation cell is now accepted, but it appears later than an already accepted
p=Path("build.py")
print("exists",p.exists(),"bytes",p.stat().st_size if p.exists() else None)
print(p.read_text()[:1000] if p.exists() else "")

# -- cell 22 ------------------------------------------------------------------------
# `build.py` is not materialized during the session, so there is no file to reorder manually; the actu
r=subprocess.run(["checkMesh","-allTopology","-allGeometry"],text=True,capture_output=True)
Path("checkMesh_all.log").write_text(r.stdout+r.stderr)
print(r.stdout[-7000:]); print(r.stderr[-500:]); print("returncode",r.returncode)

# -- cell 23 ------------------------------------------------------------------------
# The strict optional check reports two coarse-tet quality warnings (4,531 low-determinant cells and 4
pin=pv.read("VTK/t22_0/boundary/innerInlet.vtp"); pout=pv.read("VTK/t22_0/boundary/outerOutlet.vtp")
pf1=pv.read("VTK/t22_0/boundary/frictionFace1.vtp"); pf2=pv.read("VTK/t22_0/boundary/frictionFace2.vtp")
rin=np.sqrt(pin.points[:,0]**2+pin.points[:,1]**2); rout=np.sqrt(pout.points[:,0]**2+pout.points[:,1]**2)
print(f"Mesh inner radius target/range: {RI:.6f} / {rin.min():.6f}..{rin.max():.6f} m")
print(f"Mesh outer radius target/range: {RO:.6f} / {rout.min():.6f}..{rout.max():.6f} m")
print(f"Mesh friction z target/measured: 0/{pf1.points[:,2].min():.6f} m, {GAP:.6f}/{pf2.points[:,2].max():.6f} m")
print("Mesh cells:",vol.n_cells,"regions target/checked: 36/36")

# -- cell 24 ------------------------------------------------------------------------
# The union is closed, but 72 shared edges have inconsistent triangle winding. This is an STL tessella
patch_order=["frictionFace1","frictionFace2","vaneSurfaces","innerInlet","outerOutlet"]
parts=[]
for pid,name in enumerate(patch_order):
    m=pv.read(f"constant/triSurface/{name}.stl"); m.cell_data["patchId"]=np.full(m.n_cells,pid); parts.append(m)
joined=parts[0]
for m in parts[1:]: joined=joined.merge(m,merge_points=True,tolerance=1e-8)
joined=joined.clean(tolerance=1e-8)
fixed=joined.compute_normals(cell_normals=True,point_normals=False,consistent_normals=True,auto_orient_normals=True,non_manifold_traversal=True)
for pid,name in enumerate(patch_order):
    sub=fixed.threshold([pid-0.1,pid+0.1],scalars="patchId",preference="cell").extract_surface()
    sub.save(f"constant/triSurface/{name}.stl",binary=False)
    print(name,sub.n_cells)
print("Rewrote consistently wound patch STLs")

# -- cell 25 ------------------------------------------------------------------------
# PyVista’s normal filter made the split files worse because extracting each patch can reorient cells 
import importlib.util
print("trimesh available:",importlib.util.find_spec("trimesh") is not None)
print("meshio available:",importlib.util.find_spec("meshio") is not None)
