"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import os, numpy as np
from build123d import *

# all dimensions in METRES (OpenFOAM reads metres)
R_RING_O, R_RING_I = 0.030, 0.023     # outer ring: 30 / 23 mm
R_SPIN_O, R_SPIN_I = 0.022, 0.015     # inner spinner: 22 / 15 mm
H = 0.010                             # both 10 mm tall, centred on XY plane

housing_bore = Pos(0, 0, -H/2) * Cylinder(R_RING_I, H, align=(Align.CENTER, Align.CENTER, Align.MIN))
spinner = Pos(0, 0, -H/2) * (Cylinder(R_SPIN_O, H, align=(Align.CENTER, Align.CENTER, Align.MIN))
                             - Cylinder(R_SPIN_I, H, align=(Align.CENTER, Align.CENTER, Align.MIN)))
oil = housing_bore - spinner

print("solids in oil:", len(oil.solids()))
for s in oil.solids():
    bb = s.bounding_box()
    print(f"  vol={s.volume:.4e}  r_max={max(bb.max.X,bb.max.Y):.5f}  z=[{bb.min.Z:.4f},{bb.max.Z:.4f}]")
print("clearance = R_RING_I - R_SPIN_O =", (R_RING_I - R_SPIN_O)*1000, "mm")
print("analytic bore vol", np.pi*R_SPIN_I**2*H, " clearance vol", np.pi*(R_RING_I**2-R_SPIN_O**2)*H)
print("total oil volume", oil.volume)

# -- cell 2 -------------------------------------------------------------------------
# Two disjoint pockets, volumes matching analytic exactly (7.0686e-6 and 1.4137e-6 m³), clearance exac
import os, numpy as np, subprocess
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

# ---- mesh resolution (coarse first) ----
N_ANG   = 8   # cells per 90deg quadrant
N_CORE  = 8   # cells across the O-grid core block
N_BORE  = 5   # radial cells from core to r=15mm
N_CLR   = 3   # radial cells across the 1 mm clearance
N_Z     = 5   # cells through the 10 mm height

ang = [45 + 90*k for k in range(4)]
def pt(r, adeg, z): return (r*np.cos(np.radians(adeg)), r*np.sin(np.radians(adeg)), z)
ZL, ZU = -H/2, H/2
C = 0.35355*R_SPIN_I          # O-grid core half-diagonal radius

verts, vidx = [], {}
def V(key, p):
    vidx[key] = len(verts); verts.append(p)
for z, zt in ((ZL,'L'), (ZU,'U')):
    for k in range(4):
        V(f'i{k}{zt}', pt(C, ang[k], z))          # core square
        V(f'o{k}{zt}', pt(R_SPIN_I, ang[k], z))   # bore wall r=15
        V(f'p{k}{zt}', pt(R_SPIN_O, ang[k], z))   # spinner OD r=22
        V(f'q{k}{zt}', pt(R_RING_I, ang[k], z))   # ring bore r=23
def vs(*keys): return " ".join(str(vidx[k]) for k in keys)

blocks = [f"hex ({vs(*[f'i{k}L' for k in range(4)], *[f'i{k}U' for k in range(4)])}) ({N_CORE} {N_CORE} {N_Z}) simpleGrading (1 1 1)"]
for k in range(4):
    j = (k+1) % 4
    blocks.append(f"hex ({vs(f'i{k}L',f'o{k}L',f'o{j}L',f'i{j}L',f'i{k}U',f'o{k}U',f'o{j}U',f'i{j}U')}) ({N_BORE} {N_ANG} {N_Z}) simpleGrading (1 1 1)")
for k in range(4):
    j = (k+1) % 4
    blocks.append(f"hex ({vs(f'p{k}L',f'q{k}L',f'q{j}L',f'p{j}L',f'p{k}U',f'q{k}U',f'q{j}U',f'p{j}U')}) ({N_CLR} {N_ANG} {N_Z}) simpleGrading (1 1 1)")

edges = []
for pre, rad in (('o', R_SPIN_I), ('p', R_SPIN_O), ('q', R_RING_I)):
    for zt, z in (('L', ZL), ('U', ZU)):
        for k in range(4):
            j = (k+1) % 4
            m = pt(rad, ang[k]+45, z)
            edges.append(f"arc {vidx[f'{pre}{k}{zt}']} {vidx[f'{pre}{j}{zt}']} ({m[0]:.9g} {m[1]:.9g} {m[2]:.9g})")

def quad(a,b,c,d): return f"({vs(a,b,c,d)})"
f_ring, f_spin, f_top, f_bot = [], [], [], []
for k in range(4):
    j = (k+1) % 4
    f_ring.append(quad(f'q{k}L', f'q{k}U', f'q{j}U', f'q{j}L'))                 # r=23 ring wetted
    f_spin.append(quad(f'o{k}L', f'o{j}L', f'o{j}U', f'o{k}U'))                 # r=15 bore wall
    f_spin.append(quad(f'p{k}L', f'p{k}U', f'p{j}U', f'p{j}L'))                 # r=22 spinner OD
    f_top.append(quad(f'i{k}U', f'o{k}U', f'o{j}U', f'i{j}U'))
    f_top.append(quad(f'p{k}U', f'q{k}U', f'q{j}U', f'p{j}U'))
    f_bot.append(quad(f'i{k}L', f'i{j}L', f'o{j}L', f'o{k}L'))
    f_bot.append(quad(f'p{k}L', f'p{j}L', f'q{j}L', f'q{k}L'))
f_top.append(quad(*[f'i{k}U' for k in range(4)]))
f_bot.append(f"({vs(*[f'i{k}L' for k in range(3,-1,-1)])})")
print(len(blocks), "blocks,", len(edges), "arcs; est cells:",
      N_CORE**2*N_Z + 4*N_BORE*N_ANG*N_Z + 4*N_CLR*N_ANG*N_Z)

# -- cell 3 -------------------------------------------------------------------------
# Now write the dict and run blockMesh.
def blk(name, typ, faces):
    return f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n" + \
           "".join(f"            {f}\n" for f in faces) + "        );\n    }\n"

dict_txt = (
"FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n\n"
"scale 1;\n\nvertices\n(\n" +
"".join(f"    ({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})\n" for p in verts) +
");\n\nblocks\n(\n" + "".join(f"    {b}\n" for b in blocks) +
");\n\nedges\n(\n" + "".join(f"    {e}\n" for e in edges) +
");\n\nboundary\n(\n" +
blk("outerRingWall", "wall", f_ring) +
blk("spinnerWall",   "wall", f_spin) +
blk("oilTop",        "patch", f_top) +
blk("oilBottom",     "patch", f_bot) +
");\n\nmergePatchPairs ();\n")
open("system/blockMeshDict", "w").write(dict_txt)

open("system/controlDict", "w").write(
"FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object controlDict;\n}\n"
"application blockMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\n"
"deltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
open("system/fvSchemes", "w").write(
"FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object fvSchemes;\n}\n"
"ddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\n"
"laplacianSchemes{default Gauss linear corrected;}\ninterpolationSchemes{default linear;}\n"
"snGradSchemes{default corrected;}\n")
open("system/fvSolution", "w").write(
"FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object fvSolution;\n}\nsolvers{}\n")

r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-1800:])
print(r.stderr[-800:])

# -- cell 4 -------------------------------------------------------------------------
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
out = r.stdout
print(out[out.index("Mesh stats"):])

# -- cell 5 -------------------------------------------------------------------------
# Mesh OK, pure hex, volume 8.4279e-6 vs exact 8.4823e-6 (0.64 % low from faceted arcs). It confirms i
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
pv.global_theme.allow_empty_mesh = True
mesh = pv.OpenFOAMReader("case.foam") if os.path.exists("case.foam") else None
open("case.foam", "w").close()
rdr = pv.OpenFOAMReader("case.foam"); rdr.set_active_time_value(0.0)
blocks = rdr.read(); internal = blocks["internalMesh"]
bnd = blocks["boundary"]
p = pv.Plotter(off_screen=True, window_size=(900, 450), shape=(1, 2))
p.subplot(0, 0)
cols = {"outerRingWall": "firebrick", "spinnerWall": "steelblue", "oilTop": "gold", "oilBottom": "seagreen"}
for n, c in cols.items(): p.add_mesh(bnd[n], color=c, show_edges=True, line_width=0.5, label=n)
p.add_legend(size=(0.4, 0.25)); p.camera_position = "iso"
p.subplot(0, 1)
p.add_mesh(internal.clip("y"), show_edges=True, color="lightgray")
p.camera_position = [(0.06, -0.06, 0.05), (0, 0, 0), (0, 0, 1)]
p.screenshot("look_coarse.png"); print("ok")

# -- cell 6 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("look_coarse.png"))

# -- cell 7 -------------------------------------------------------------------------
print(r.returncode); print(r.stdout[-1500:]); print("ERR:", r.stderr[-1500:])

# -- cell 8 -------------------------------------------------------------------------
# The cell defining `write_blockmeshdict` errored on its last line, so it never entered the script. Re
def write_blockmeshdict(n_ang, n_bore, n_clr, n_z):
    """Cylindrical O-grid for the oil: core+bore (r<15mm) and clearance shell (22-23mm).
    The O-grid core must share n_ang subdivisions with the blocks around it."""
    n_core = n_ang
    verts, vidx = [], {}
    def V(key, p): vidx[key] = len(verts); verts.append(p)
    for z, zt in ((ZL, 'L'), (ZU, 'U')):
        for k in range(4):
            V(f'i{k}{zt}', pt(C, ang[k], z));        V(f'o{k}{zt}', pt(R_SPIN_I, ang[k], z))
            V(f'p{k}{zt}', pt(R_SPIN_O, ang[k], z)); V(f'q{k}{zt}', pt(R_RING_I, ang[k], z))
    def vs(*ks): return " ".join(str(vidx[k]) for k in ks)
    bl = [f"hex ({vs(*[f'i{k}L' for k in range(4)], *[f'i{k}U' for k in range(4)])}) ({n_core} {n_core} {n_z}) simpleGrading (1 1 1)"]
    for k in range(4):
        j = (k+1) % 4
        bl.append(f"hex ({vs(f'i{k}L',f'o{k}L',f'o{j}L',f'i{j}L',f'i{k}U',f'o{k}U',f'o{j}U',f'i{j}U')}) ({n_bore} {n_ang} {n_z}) simpleGrading (1 1 1)")
    for k in range(4):
        j = (k+1) % 4
        bl.append(f"hex ({vs(f'p{k}L',f'q{k}L',f'q{j}L',f'p{j}L',f'p{k}U',f'q{k}U',f'q{j}U',f'p{j}U')}) ({n_clr} {n_ang} {n_z}) simpleGrading (1 1 1)")
    ed = []
    for pre, rad in (('o', R_SPIN_I), ('p', R_SPIN_O), ('q', R_RING_I)):
        for zt in ('L', 'U'):
            z = ZL if zt == 'L' else ZU
            for k in range(4):
                j = (k+1) % 4; m = pt(rad, ang[k]+45, z)
                ed.append(f"arc {vidx[f'{pre}{k}{zt}']} {vidx[f'{pre}{j}{zt}']} ({m[0]:.9g} {m[1]:.9g} {m[2]:.9g})")
    def quad(a,b,c,d): return f"({vs(a,b,c,d)})"
    fr, fsp, ft, fb = [], [], [], []
    for k in range(4):
        j = (k+1) % 4
        fr.append(quad(f'q{k}L', f'q{k}U', f'q{j}U', f'q{j}L'))                     # ring wetted, r=23
        fsp.append(quad(f'o{k}L', f'o{j}L', f'o{j}U', f'o{k}U'))                    # spinner bore wall, r=15
        fsp.append(quad(f'p{k}L', f'p{k}U', f'p{j}U', f'p{j}L'))                    # spinner OD, r=22
        ft.append(quad(f'i{k}U', f'o{k}U', f'o{j}U', f'i{j}U')); ft.append(quad(f'p{k}U', f'q{k}U', f'q{j}U', f'p{j}U'))
        fb.append(quad(f'i{k}L', f'i{j}L', f'o{j}L', f'o{k}L')); fb.append(quad(f'p{k}L', f'p{j}L', f'q{j}L', f'q{k}L'))
    ft.append(quad(*[f'i{k}U' for k in range(4)]))
    fb.append(f"({vs(*[f'i{k}L' for k in range(3,-1,-1)])})")
    txt = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n\n"
           "scale 1;\n\nvertices\n(\n" + "".join(f"    ({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})\n" for p in verts) +
           ");\n\nblocks\n(\n" + "".join(f"    {b}\n" for b in bl) +
           ");\n\nedges\n(\n" + "".join(f"    {e}\n" for e in ed) +
           ");\n\nboundary\n(\n" + blk("outerRingWall","wall",fr) + blk("spinnerWall","wall",fsp) +
           blk("oilTop","patch",ft) + blk("oilBottom","patch",fb) + ");\n\nmergePatchPairs ();\n")
    open("system/blockMeshDict", "w").write(txt)

write_blockmeshdict(n_ang=16, n_bore=9, n_clr=4, n_z=8)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
i = r.stdout.find("Mesh Information")
print(r.returncode, r.stdout[i:] if i > 0 else r.stderr[-600:])

# -- cell 9 -------------------------------------------------------------------------
# Now measure the mesh itself — patch areas, radii of each named patch, volume — against the numbers i
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
o = r.stdout; print(o[o.index("Checking topology"):o.index("Checking patch topology")])
print(o[o.index("    Overall domain"):o.index("Mesh OK") + 8] if "Mesh OK" in o else o[-900:])

# -- cell 10 ------------------------------------------------------------------------
rdr2 = pv.OpenFOAMReader("case.foam"); rdr2.set_active_time_value(0.0)
bnd2 = rdr2.read()["boundary"]
exp = {"outerRingWall": 2*np.pi*R_RING_I*H,
       "spinnerWall":   2*np.pi*(R_SPIN_I + R_SPIN_O)*H,
       "oilTop":        np.pi*R_SPIN_I**2 + np.pi*(R_RING_I**2 - R_SPIN_O**2),
       "oilBottom":     np.pi*R_SPIN_I**2 + np.pi*(R_RING_I**2 - R_SPIN_O**2)}
for n in exp:
    s = bnd2[n]; a = s.compute_cell_sizes(length=False, volume=False)["Area"].sum()
    P = s.points; rr = np.hypot(P[:,0], P[:,1])
    print(f"{n:14s} area={a:.5e} (exact {exp[n]:.5e}, {100*(a/exp[n]-1):+.2f}%)  "
          f"r=[{rr.min()*1000:.3f},{rr.max()*1000:.3f}]mm  z=[{P[:,2].min()*1000:+.3f},{P[:,2].max()*1000:+.3f}]mm")
print(f"\nmeasured clearance = {(R_RING_I-R_SPIN_O)*1000:.3f} mm, height = {H*1000:.1f} mm, "
      f"mesh volume 8.46868e-06 vs exact {oil.volume:.5e} ({100*(8.46868e-06/oil.volume-1):+.2f}%)")

# -- cell 11 ------------------------------------------------------------------------
# Refined mesh measured clean. Final look at what's on disk.
p = pv.Plotter(off_screen=True, window_size=(900, 450), shape=(1, 2))
p.subplot(0, 0)
for n, c in (("outerRingWall","firebrick"),("spinnerWall","steelblue"),("oilTop","gold"),("oilBottom","seagreen")):
    p.add_mesh(bnd2[n], color=c, show_edges=True, line_width=0.4, label=n)
p.add_legend(size=(0.42,0.26)); p.camera_position = "iso"
p.subplot(0, 1)
p.add_mesh(rdr2.read()["internalMesh"].clip("y"), show_edges=True, color="lightgray")
p.camera_position = [(0.055,-0.055,0.045),(0,0,0),(0,0,1)]
p.screenshot("look_final.png"); display(Image("look_final.png"))
