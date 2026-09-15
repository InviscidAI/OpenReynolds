"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import subprocess, os, re
os.chdir("/home/qiuzi/.openreynolds-buildup/work/T5-20260915-061523-30aa/t5")
import build123d as bd
from build123d import import_step
path = "/home/qiuzi/.openreynolds-buildup/work/T5-20260915-061523-30aa/geometry/assembly.step"
raw = subprocess.run(["head","-c","12000",path], capture_output=True, text=True).stdout
print("units:", sorted(set(re.findall(r"SI_UNIT\([^)]*\)|CONVERSION_BASED_UNIT\([^)]*\)", raw))))
assy = import_step(path)
solids = assy.solids()
print("solids:", len(solids), "valid:", [bool(s.is_valid) for s in solids])
for i, s in enumerate(solids):
    bb = s.bounding_box()
    print(i, "size:", tuple(round(v,2) for v in bb.size), "min:", tuple(round(v,2) for v in bb.min), "vol:", round(s.volume,1), "faces:", len(s.faces()))

# -- cell 2 -------------------------------------------------------------------------
full = subprocess.run(["grep","-ao","SI_UNIT([^)]*)\\|MILLI\\|METRE",path], capture_output=True, text=True).stdout
print(set(full.split()))
# visualize the assembly
import matplotlib.pyplot as plt
from build123d import export_stl
export_stl(assy, "probe_assembly.stl", tolerance=0.05)
import pyvista as pv
pv.OFF_SCREEN = True
m = pv.read("probe_assembly.stl")
pl = pv.Plotter(off_screen=True)
pl.add_mesh(m, opacity=0.6, show_edges=False)
pl.camera_position = 'iso'
pl.screenshot("assy.png")
print(m.bounds)

# -- cell 3 -------------------------------------------------------------------------
# Units are declared as metres (plain `.METRE.`, no prefix). Let me look at the picture.
from IPython.display import Image
Image("assy.png")

# -- cell 4 -------------------------------------------------------------------------
out = subprocess.run(["grep","-aoE",".{80}(LENGTH_UNIT|SI_UNIT\\(\\$,\\.METRE\\.\\)).{40}", path], capture_output=True, text=True).stdout
print(out[:3000])

# -- cell 5 -------------------------------------------------------------------------
raw_all = open(path, errors="ignore").read()
import re
idxs = [m.start() for m in re.finditer(r"METRE", raw_all)]
print("n METRE:", len(idxs))
for i in idxs[:10]:
    print(repr(raw_all[max(0,i-160):i+80]))
    print("---")

# -- cell 6 -------------------------------------------------------------------------
# Section view through the middle (y-z plane cut) to see the interior
import numpy as np
import pyvista as pv
from build123d import export_stl, Plane
# clip each solid's mesh at x=0 plane to see inside
meshes = []
for i, s in enumerate(solids):
    export_stl(s, f"probe_s{i}.stl", tolerance=0.02)
    meshes.append(pv.read(f"probe_s{i}.stl"))
pl = pv.Plotter(off_screen=True, shape=(1,2))
for k, clip_normal in enumerate([(1,0,0),(0,-1,0)]):
    pl.subplot(0,k)
    for m in meshes:
        c = m.clip(normal=clip_normal, origin=(0.01,0,0) if k==0 else (0,0.01,0))
        pl.add_mesh(c, opacity=0.8, show_edges=True)
    pl.camera_position='iso'
pl.screenshot("sections.png")
Image("sections.png")

# -- cell 7 -------------------------------------------------------------------------
subprocess.run(["grep","-n","section",".reference/b123d_api.md"], capture_output=True, text=True).stdout[:800]

# -- cell 8 -------------------------------------------------------------------------
from build123d import section
# Decisive test: enclosing box minus fused hardware -> where is the void?
union = solids[0].fuse(solids[1], solids[2]); union = union.clean()
bb = union.bounding_box()
B = bd.Box(bb.size.X+2, bb.size.Y+2, bb.size.Z+2).locate(bd.Location(bb.center()))
neg = B - union
ns = neg.solids()
print("negative regions:", len(ns))
for s in ns:
    sb = s.bounding_box()
    touches = any(abs(a-b)<1e-6 for a,b in zip(sb.min, bb.min)) or any(abs(a-b)<1e-6 for a,b in zip(sb.max, bb.max))
    print("vol:", round(s.volume,1), "size:", tuple(round(v,2) for v in sb.size), "touches outer box:", touches)

# -- cell 9 -------------------------------------------------------------------------
# classify properly against B's boundary, and test identity of region2 vs solid 0
Bbb = B.bounding_box()
Bm = (Bbb.min.X, Bbb.min.Y, Bbb.min.Z); BM = (Bbb.max.X, Bbb.max.Y, Bbb.max.Z)
for j, s in enumerate(ns):
    sb = s.bounding_box()
    sm = (sb.min.X, sb.min.Y, sb.min.Z); sM = (sb.max.X, sb.max.Y, sb.max.Z)
    touch = any(abs(sm[k]-Bm[k])<1e-6 or abs(sM[k]-BM[k])<1e-6 for k in range(3))
    print(j, "vol", round(s.volume,1), "touches B wall:", touch)
inter = ns[1].intersect(solids[0])
print("region1 & solid0 common volume:", round(sum(x.volume for x in inter.solids()), 2))
print("solid0 volume:", round(solids[0].volume,2))
print("union solid volumes:", [round(s.volume,1) for s in union.solids()])

# -- cell 10 ------------------------------------------------------------------------
# The booleans are telling a confusing story; time to just look at each solid separately, side by side
pl = pv.Plotter(off_screen=True, shape=(1,3))
for k in range(3):
    pl.subplot(0,k)
    pl.add_mesh(pv.read(f"probe_s{k}.stl"), show_edges=True, opacity=1.0)
    pl.add_text(f"solid {k}", font_size=12)
    pl.camera_position='iso'
pl.screenshot("solids3.png")
Image("solids3.png")

# -- cell 11 ------------------------------------------------------------------------
s12 = solids[1].fuse(solids[2]); s12 = s12.clean()
export_stl(s12, "probe_s12.stl", tolerance=0.02)
export_stl(ns[1], "probe_neg1.stl", tolerance=0.02)
pl = pv.Plotter(off_screen=True, shape=(1,2))
pl.subplot(0,0)
pl.add_mesh(pv.read("probe_s12.stl").clip(normal=(1,0,0), origin=(0.05,0,0)), show_edges=True)
pl.add_text("s1+s2 clipped @ x=0")
pl.subplot(0,1)
pl.add_mesh(pv.read("probe_neg1.stl"), show_edges=True, color="orange")
pl.add_text("neg region 1 (the 'void')")
for k in range(2):
    pl.subplot(0,k); pl.camera_position='iso'
pl.screenshot("void.png")
Image("void.png")

# -- cell 12 ------------------------------------------------------------------------
n1bb = ns[1].bounding_box()
print("neg region1 bbox min:", tuple(round(v,2) for v in (n1bb.min.X,n1bb.min.Y,n1bb.min.Z)),
      "max:", tuple(round(v,2) for v in (n1bb.max.X,n1bb.max.Y,n1bb.max.Z)))
for j,s in enumerate(union.solids()):
    ub = s.bounding_box()
    print("union solid", j, "vol", round(s.volume,1),
          "min", tuple(round(v,2) for v in (ub.min.X,ub.min.Y,ub.min.Z)),
          "max", tuple(round(v,2) for v in (ub.max.X,ub.max.Y,ub.max.Z)))
# point membership tests
tests = {"s0 center": solids[0].center(), "s1 center": solids[1].center(),
         "s2 center": solids[2].center(), "region1 center": ns[1].center()}
for name, pt in tests.items():
    print(name, tuple(round(v,1) for v in (pt.X,pt.Y,pt.Z)),
          "| in union:", union.is_inside(pt), "| in neg:", neg.is_inside(pt),
          "| in s12:", s12.is_inside(pt))

# -- cell 13 ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14,7))
colors = ['tab:blue','tab:red','tab:green']
for ax, pl_plane, tag in [(axes[0], Plane.YZ, "x=0 (y-z cut)"), (axes[1], Plane.XZ, "y=0 (x-z cut)")]:
    ax.set_title(tag); ax.set_aspect('equal')
    for i, s in enumerate(solids):
        sec = section(s, section_by=pl_plane, height=0.0)
        for f in sec.faces():
            for w in [f.outer_wire()] + f.inner_wires():
                pts = [w.position_at(t) for t in np.linspace(0,1,400)]
                ax.plot([p.Y if pl_plane==Plane.YZ else p.X for p in pts],[p.Z for p in pts],
                        color=colors[i], lw=1.2, label=f"s{i}")
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(dict(zip(labels,handles)).values(), dict(zip(labels,handles)).keys())
plt.tight_layout(); plt.show()

# -- cell 14 ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(20,4.4))
for ax, z in zip(axes, [21.0, 23.0, 25.0, 27.0, 30.0]):
    sec = section(solids[1], section_by=Plane.XY, height=z)
    faces = sec.faces()
    ninner = sum(len(f.inner_wires()) for f in faces)
    ax.set_title(f"s1 @ z={z}: {len(faces)}f {ninner}i", fontsize=10)
    ax.set_aspect('equal')
    for f in faces:
        for w in [f.outer_wire()] + f.inner_wires():
            pts = [w.position_at(t) for t in np.linspace(0,1,400)]
            ax.plot([p.X for p in pts],[p.Y for p in pts], 'r-', lw=0.8)
plt.tight_layout(); plt.show()
