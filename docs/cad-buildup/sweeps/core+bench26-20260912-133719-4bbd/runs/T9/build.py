"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll send that same cell through the tool.
import os, subprocess, math
from build123d import *

# --- all dimensions in METRES (OpenFOAM reads metres) ---
R_RING_O, R_RING_I, H_RING = 0.030, 0.023, 0.010   # outer ring
R_SP_O,   R_SP_I,  H_SP    = 0.022, 0.015, 0.008   # inner spinner

ring    = Cylinder(R_RING_O, H_RING) - Cylinder(R_RING_I, H_RING)
spinner = Cylinder(R_SP_O,  H_SP)   - Cylinder(R_SP_I,  H_SP)
oil     = Cylinder(R_RING_I, H_RING) - spinner     # fluid = ring bore minus spinner

print("radial clearance (mm):", 1000*(R_RING_I - R_SP_O))
print("axial gap each end (mm):", 1000*(H_RING - H_SP)/2)
print("oil volume (mm^3):", oil.volume*1e9, " solids:", len(oil.solids()))
print("analytic  (mm^3):", 1e9*(math.pi*R_RING_I**2*H_RING - math.pi*(R_SP_O**2-R_SP_I**2)*H_SP))
print("oil bbox (mm):", [round(1000*v,3) for v in oil.bounding_box().min], [round(1000*v,3) for v in oil.bounding_box().max])
print("n faces:", len(oil.faces()))

# -- cell 2 -------------------------------------------------------------------------
# Now classify the 7 faces by their exact underlying surface (axis radius / plane height), not by boun
import numpy as np
def classify(f):
    g = f.geom_type
    c = f.center()
    if str(g).upper().endswith("CYLINDER") or "CYLIN" in str(g).upper():
        # radius from a sampled vertex
        r = max(math.hypot(v.X, v.Y) for v in f.vertices())
        return ("cyl", round(r,6))
    else:
        return ("plane", round(c.Z,6))

for i,f in enumerate(oil.faces()):
    print(i, f.geom_type, classify(f), "area_mm2=%.2f"%(f.area*1e6))

# -- cell 3 -------------------------------------------------------------------------
# Patch assignment follows from the surfaces themselves: r=23 cylinder is the ring's wetted bore, the 
os.makedirs("constant/triSurface", exist_ok=True)
F = oil.faces()
patches = {
    "ringWall":      [F[0]],              # cylinder r=23 mm, ring bore
    "spinnerWall":   [F[3], F[4], F[5], F[6]],  # r=22, z=+-4 annuli, r=15 bore
    "ringEndTop":    [F[1]],              # plane z=+5 mm
    "ringEndBottom": [F[2]],              # plane z=-5 mm
}
for name, faces in patches.items():
    shp = faces[0] if len(faces)==1 else Compound(children=[f for f in faces])
    export_stl(shp, f"constant/triSurface/{name}.stl", tolerance=2e-6, angular_tolerance=0.05)
    print(name, "area_mm2=%.2f"%(sum(f.area for f in faces)*1e6), os.path.getsize(f"constant/triSurface/{name}.stl"))
print(sorted(os.listdir("constant/triSurface")))

# -- cell 4 -------------------------------------------------------------------------
# Mesh with gmsh on the exact OCC solid (tets, patch names preserved), so the 1 mm gaps are meshed by 
export_step(oil, "oil.step")
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("oil")
gmsh.model.occ.importShapes("oil.step")
gmsh.model.occ.synchronize()
surfs = gmsh.model.getEntities(2)
info = []
for d,t in surfs:
    a = gmsh.model.occ.getMass(d,t); com = gmsh.model.occ.getCenterOfMass(d,t)
    info.append((t, a*1e6, [round(c*1000,3) for c in com]))
    print(t, "area_mm2=%.2f"%(a*1e6), info[-1][2])
print("volume mm^3", gmsh.model.occ.getMass(3,1)*1e9)

# -- cell 5 -------------------------------------------------------------------------
# Coarse pass: gmsh size field, 2 mm inside the bore, 0.8 mm in the gap regions; physical groups match
def tag_groups():
    want = {1:"ringWall", 2:"ringEndTop", 3:"ringEndBottom"}
    groups = {"ringWall":[], "ringEndTop":[], "ringEndBottom":[], "spinnerWall":[]}
    for d,t in gmsh.model.getEntities(2):
        a = gmsh.model.occ.getMass(d,t)*1e6; z = gmsh.model.occ.getCenterOfMass(d,t)[2]*1000
        if abs(a-1445.13) < 1: groups["ringWall"].append(t)
        elif abs(a-1661.90) < 1: groups["ringEndTop" if z>0 else "ringEndBottom"].append(t)
        else: groups["spinnerWall"].append(t)
    for n,(k,v) in enumerate(groups.items(), start=1):
        gmsh.model.addPhysicalGroup(2, v, n); gmsh.model.setPhysicalName(2, n, k)
    gmsh.model.addPhysicalGroup(3, [1], 100); gmsh.model.setPhysicalName(3, 100, "oil")
    print({k:v for k,v in groups.items()})

def mesh(h_bore, h_gap, fname):
    gmsh.model.mesh.clear()
    f = gmsh.model.mesh.field.add("Cylinder")
    gmsh.model.mesh.field.setNumber(f,"Radius",0.0155)
    gmsh.model.mesh.field.setNumber(f,"ZAxis",0.007)
    gmsh.model.mesh.field.setNumber(f,"XAxis",0.0); gmsh.model.mesh.field.setNumber(f,"YAxis",0.0)
    gmsh.model.mesh.field.setNumber(f,"VIn",h_bore); gmsh.model.mesh.field.setNumber(f,"VOut",h_gap)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0); gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",0)
    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    ts = gmsh.model.mesh.getElementsByType(4)[0]
    print(fname, "tets:", len(ts))

tag_groups()
mesh(0.002, 0.0008, "oil_coarse.msh")

# -- cell 6 -------------------------------------------------------------------------
# 38 k tets, all quality > 0.14. Convert to OpenFOAM (msh2 format for `gmshToFoam`) and look at it.
for d in ["system","constant"]: os.makedirs(d, exist_ok=True)
hdr = lambda cls,obj: f"FoamFile{{version 2.0; format ascii; class {cls}; object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
  "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
  "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; timeFormat general; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+"ddtSchemes{default Euler;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{} \n")

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("oil.msh")
p = subprocess.run(["gmshToFoam","oil.msh"], capture_output=True, text=True)
print(p.returncode, p.stdout[-1500:], p.stderr[-500:])

# -- cell 7 -------------------------------------------------------------------------
p = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(p.stdout[p.stdout.find("Mesh stats"):][:3000])

# -- cell 8 -------------------------------------------------------------------------
# Read the OpenFOAM mesh instead (pyvista can't read .msh here).
import pyvista as pv
pv.OFF_SCREEN = True
open("case.foam","w").close()
r = pv.OpenFOAMReader("case.foam"); r.enable_all_patch_arrays()
blocks = r.read()
names = list(blocks["boundary"].keys()) if "boundary" in blocks.keys() else blocks.keys()
print(blocks.keys(), names)

# -- cell 9 -------------------------------------------------------------------------
bnd = blocks["boundary"]
pl = pv.Plotter(shape=(1,2), window_size=(1500,700), off_screen=True)
cols = {"ringWall":"red","ringEndTop":"green","ringEndBottom":"blue","spinnerWall":"orange"}
pl.subplot(0,0)
for n,c in cols.items(): pl.add_mesh(bnd[n].extract_surface().clip("y"), color=c, show_edges=True, line_width=0.3, label=n)
pl.add_legend(); pl.camera_position="iso"
pl.subplot(0,1)
vol = blocks["internalMesh"].clip("y")
pl.add_mesh(vol, color="lightsteelblue", show_edges=True, line_width=0.3)
pl.camera_position = [(0.0,-0.09,0.03),(0,0,0),(0,0,1)]
pl.screenshot("look_coarse.png")
print("ok")

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look_coarse.png"))

# -- cell 11 ------------------------------------------------------------------------
# Shape is right. Now refine: 0.35 mm in the gap regions (≈3 cells across a 1 mm gap), 1.2 mm in the b
import time
t0=time.time()
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
mesh(0.0012, 0.00035, "oil.msh")
print("wall time", time.time()-t0)

# -- cell 12 ------------------------------------------------------------------------
# 383 k tets. Convert and check.
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
p = subprocess.run(["gmshToFoam","oil.msh"], capture_output=True, text=True)
print("gmshToFoam rc", p.returncode, p.stdout[p.stdout.find("Patch 0"):][:300])
c = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(c.stdout[c.stdout.find("Checking patch topology"):][:2600])

# -- cell 13 ------------------------------------------------------------------------
# Mesh OK. Now verify the gaps are actually resolved: count cells in the radial clearance and in an en
r2 = pv.OpenFOAMReader("case.foam"); r2.enable_all_patch_arrays()
b2 = r2.read(); im = b2["internalMesh"]
cc = im.cell_centers().points; vol = im.compute_cell_sizes(volume=True)["Volume"]
rr = np.hypot(cc[:,0],cc[:,1]); zz = cc[:,2]
sel_rad = (rr>0.022)&(rr<0.023)&(abs(zz)<0.004)
sel_ax  = (rr<0.022)&(abs(zz)>0.004)
for name, sel, gap in [("radial clearance",sel_rad,0.001),("end gaps",sel_ax,0.001)]:
    N=sel.sum(); V=vol[sel].sum()
    h=(8.5*V/N)**(1/3)
    print(f"{name}: cells={N}, V={V*1e9:.0f} mm^3, mean tet edge={h*1000:.3f} mm -> ~{gap/h:.1f} cells across the 1.0 mm gap")
print("total mesh volume mm^3 %.2f vs exact %.2f"%(vol.sum()*1e9, oil.volume*1e9))

# -- cell 14 ------------------------------------------------------------------------
# Two cells across a 1 mm gap is thin; tighten the gap size to 0.25 mm (bore 1.0 mm) for ≈3–4 cells ac
t0=time.time()
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
mesh(0.0010, 0.00025, "oil.msh")
print("wall time", time.time()-t0)

# -- cell 15 ------------------------------------------------------------------------
shutil.rmtree("constant/polyMesh", ignore_errors=True)
p = subprocess.run(["gmshToFoam","oil.msh"], capture_output=True, text=True); print("rc",p.returncode)
c = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(c.stdout[c.stdout.find("Checking patch topology"):][:2500])

# -- cell 16 ------------------------------------------------------------------------
# Mesh OK at 909 k cells. Re-measure gap resolution and take a zoomed look at the clearance and end ga
r3 = pv.OpenFOAMReader("case.foam"); r3.enable_all_patch_arrays(); b3 = r3.read()
im3 = b3["internalMesh"]; cc = im3.cell_centers().points
vol3 = im3.compute_cell_sizes(volume=True)["Volume"]
rr = np.hypot(cc[:,0],cc[:,1]); zz = cc[:,2]
for name, sel in [("radial clearance",(rr>0.022)&(rr<0.023)&(abs(zz)<0.004)),
                  ("end gaps",(rr<0.022)&(abs(zz)>0.004)),
                  ("bore",(rr<0.015)&(abs(zz)<0.004))]:
    N=sel.sum(); V=vol3[sel].sum(); h=(8.5*V/N)**(1/3)
    print(f"{name}: cells={N:7d} V={V*1e9:8.1f} mm^3  h={h*1000:.3f} mm  ({0.001/h:.1f} across 1 mm)")
sl = im3.slice(normal="y")
pl = pv.Plotter(window_size=(1400,700), off_screen=True)
pl.add_mesh(sl, color="lightsteelblue", show_edges=True, line_width=0.4)
pl.camera_position=[(0,-0.05,0),(0.0195,0,0.0),(0,0,1)]; pl.camera.parallel_projection=True; pl.camera.parallel_scale=0.006
pl.screenshot("gap_zoom.png"); display(Image("gap_zoom.png"))

# -- cell 17 ------------------------------------------------------------------------
# Final verification: patch areas in the mesh against analytic, and the STL files' extents (metres).
bnd3 = b3["boundary"]
exact = {"ringWall":2*math.pi*0.023*0.010, "ringEndTop":math.pi*0.023**2, "ringEndBottom":math.pi*0.023**2,
         "spinnerWall":2*math.pi*0.022*0.008+2*math.pi*(0.022**2-0.015**2)+2*math.pi*0.015*0.008}
for n,e in exact.items():
    a = bnd3[n].compute_cell_sizes(area=True)["Area"].sum()
    s = pv.read(f"constant/triSurface/{n}.stl")
    print(f"{n:14s} mesh {a*1e6:8.2f} mm^2  exact {e*1e6:8.2f}  STL {s.area*1e6:8.2f} mm^2  STL bbox(m) "
          f"{[round(v,4) for v in s.bounds]}")
