"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid region directly as a planar channel and later extrude it by a small, single-cel
import build123d as bd
import numpy as np
import matplotlib.pyplot as plt

# Requested dimensions, in metres
CHANNEL_W = 0.006
LENGTH = 0.120
N_LOOPS = 4

# Assumed Tesla-cell proportions, in metres
LOOP_SPAN = 0.024
LOOP_REACH = 0.015
LOOP_STARTS = [0.012, 0.039, 0.066, 0.093]
PATH_SAMPLES = 13

# A coarse planar footprint: main duct plus four smooth, alternating teardrop bypass ducts.
main_face = bd.Pos(LENGTH / 2, 0) * bd.Rectangle(LENGTH, CHANNEL_W)
fluid_face = main_face
loop_centrelines = []
for i, x0 in enumerate(LOOP_STARTS):
    side = 1 if i % 2 == 0 else -1
    t = np.linspace(0.0, 1.0, PATH_SAMPLES)
    # A pointed departure and broad downstream return form the teardrop centreline.
    x = x0 + LOOP_SPAN * t
    y = side * (CHANNEL_W * 0.22 + LOOP_REACH * np.sin(np.pi * t) * (0.72 + 0.28 * t))
    pts = [bd.Vector(float(xj), float(yj), 0) for xj, yj in zip(x, y)]
    loop_centrelines.append(np.column_stack((x, y)))
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        segment = bd.Pos(a.X, a.Y) * bd.Rot(0, 0, np.degrees(np.arctan2(d.Y, d.X))) * bd.Rectangle(d.length, CHANNEL_W, align=(bd.Align.MIN, bd.Align.CENTER))
        fluid_face = fluid_face + segment
    for p in pts:
        fluid_face = fluid_face + (bd.Pos(p.X, p.Y) * bd.Circle(CHANNEL_W / 2))

# Inspect numerical envelope and coarse topology.
bb = fluid_face.bounding_box()
print(f"Requested channel width: {CHANNEL_W*1e3:.3f} mm")
print(f"Built main-channel width: {(main_face.bounding_box().max.Y-main_face.bounding_box().min.Y)*1e3:.3f} mm")
print(f"Requested overall length: about {LENGTH*1e3:.1f} mm")
print(f"Built inlet-to-outlet length: {(bb.max.X-bb.min.X)*1e3:.3f} mm")
print(f"Bypass count requested/built: {N_LOOPS}/{len(loop_centrelines)}")
print(f"Assumed loop axial span: {LOOP_SPAN*1e3:.1f} mm; reach from centreline: {LOOP_REACH*1e3:.1f} mm")
print(f"Fluid footprint area: {fluid_face.area*1e6:.2f} mm^2; solids: {len(fluid_face.solids())}; faces: {len(fluid_face.faces())}")

# Plot the actual CAD boundary and the generating centrelines.
fig, ax = plt.subplots(figsize=(12, 4))
for edge in fluid_face.edges():
    us = np.linspace(0, 1, 30)
    q = np.array([[edge.position_at(float(u)).X, edge.position_at(float(u)).Y] for u in us])
    ax.plot(q[:,0]*1e3, q[:,1]*1e3, color='navy', lw=1.3)
for c in loop_centrelines:
    ax.plot(c[:,0]*1e3, c[:,1]*1e3, '--', color='tomato', lw=0.8)
ax.set_aspect('equal'); ax.grid(True, alpha=.25)
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('Coarse Tesla-valve fluid footprint (blue boundary; red bypass centrelines)')
plt.tight_layout()
plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The Boolean result is a build123d `Sketch` container rather than a bare `Face`; I’ll take its sole f
face0 = fluid_face.faces()[0]
outer = face0.outer_wire()
ordered_edges = list(outer.edges())
print(f"Outer boundary edges: {len(ordered_edges)}; closed: {outer.is_closed}")
terminal_edges = []
for j, e in enumerate(ordered_edges):
    p0, p1 = e.position_at(0), e.position_at(1)
    if abs(p1.X-p0.X) < 1e-10 and abs(e.length-CHANNEL_W) < 1e-8:
        if abs(p0.X) < 1e-10:
            terminal_edges.append(("inlet", j, e.length, p0, p1))
        elif abs(p0.X-LENGTH) < 1e-10:
            terminal_edges.append(("outlet", j, e.length, p0, p1))
for name, j, le, p0, p1 in terminal_edges:
    print(f"{name}: CAD edge {j}, length={le*1e3:.6f} mm, endpoints=({p0.X*1e3:.3f},{p0.Y*1e3:.3f}) to ({p1.X*1e3:.3f},{p1.Y*1e3:.3f}) mm")
print("Terminal patch identity check:", [x[0] for x in terminal_edges])

# -- cell 3 -------------------------------------------------------------------------
# The inlet and outlet are now tied to explicit CAD edges (0 and 52), each measured at exactly 6 mm. I
BOUNDARY_CHORD = 0.0008
boundary_xy = []
boundary_labels = []
current = None
for j, e in enumerate(ordered_edges):
    n = max(1, int(np.ceil(e.length / BOUNDARY_CHORD)))
    samples = np.array([[e.position_at(float(u)).X, e.position_at(float(u)).Y] for u in np.linspace(0, 1, n+1)])
    if current is not None and np.linalg.norm(samples[0]-current) > np.linalg.norm(samples[-1]-current):
        samples = samples[::-1]
    if current is not None and np.linalg.norm(samples[0]-current) > 1e-8:
        raise RuntimeError(f"CAD boundary discontinuity before edge {j}")
    label = "inlet" if j == 0 else ("outlet" if j == 52 else "walls")
    if not boundary_xy:
        boundary_xy.extend(samples[:-1])
    else:
        boundary_xy.extend(samples[1:-1])
    # Labels are rebuilt below from point-pair ownership, so retain edge polylines explicitly.
    boundary_labels.append((label, samples))
    current = samples[-1]

# Reassemble exact edge polylines in sequence and retain one label per segment.
poly = [boundary_labels[0][1][0]]
seg_labels = []
for label, samples in boundary_labels:
    if np.linalg.norm(samples[0]-poly[-1]) > np.linalg.norm(samples[-1]-poly[-1]): samples = samples[::-1]
    for q in samples[1:]:
        poly.append(q); seg_labels.append(label)
poly = np.asarray(poly)
closure_error = np.linalg.norm(poly[-1]-poly[0])
shoelace_area = 0.5*abs(np.dot(poly[:-1,0],poly[1:,1])-np.dot(poly[:-1,1],poly[1:,0]))
print(f"Boundary segments: {len(seg_labels)} (inlet {seg_labels.count('inlet')}, outlet {seg_labels.count('outlet')}, walls {seg_labels.count('walls')})")
print(f"Polyline closure error: {closure_error:.3e} m")
print(f"CAD area: {face0.area*1e6:.3f} mm^2; discretized area: {shoelace_area*1e6:.3f} mm^2; difference: {(shoelace_area-face0.area)/face0.area*100:.3f}%")

# -- cell 4 -------------------------------------------------------------------------
# The 19.8% area mismatch shows that `Wire.edges()` did not return traversal order even though the wir
import inspect
print("Wire ordering methods:", [n for n in dir(outer) if "order" in n.lower() or "edge" in n.lower()])
for n in [n for n in dir(outer) if "order" in n.lower()]:
    try: print(n, inspect.signature(getattr(outer, n)))
    except Exception: pass

# -- cell 5 -------------------------------------------------------------------------
# The installed API provides `Wire.order_edges()`. I’ll use that traversal and identify the two alread
ordered_edges = list(outer.order_edges())
boundary_labels = []
current = None
for e in ordered_edges:
    n = max(1, int(np.ceil(e.length / BOUNDARY_CHORD)))
    samples = np.array([[e.position_at(float(u)).X, e.position_at(float(u)).Y] for u in np.linspace(0, 1, n+1)])
    if current is not None and np.linalg.norm(samples[0]-current) > np.linalg.norm(samples[-1]-current): samples = samples[::-1]
    xmean = float(samples[:,0].mean())
    is_vertical_terminal = np.ptp(samples[:,0]) < 1e-10 and abs(e.length-CHANNEL_W) < 1e-8
    label = "inlet" if is_vertical_terminal and abs(xmean) < 1e-10 else ("outlet" if is_vertical_terminal and abs(xmean-LENGTH) < 1e-10 else "walls")
    boundary_labels.append((label, samples)); current = samples[-1]
poly = [boundary_labels[0][1][0]]; seg_labels = []
for label, samples in boundary_labels:
    if np.linalg.norm(samples[0]-poly[-1]) > np.linalg.norm(samples[-1]-poly[-1]): samples = samples[::-1]
    if np.linalg.norm(samples[0]-poly[-1]) > 1e-8: raise RuntimeError("ordered wire discontinuity")
    for q in samples[1:]: poly.append(q); seg_labels.append(label)
poly = np.asarray(poly)
closure_error = np.linalg.norm(poly[-1]-poly[0])
shoelace_area = 0.5*abs(np.dot(poly[:-1,0],poly[1:,1])-np.dot(poly[:-1,1],poly[1:,0]))
print(f"Ordered CAD edges: {len(ordered_edges)}; boundary segments: {len(seg_labels)}")
print(f"Terminal segment counts: inlet={seg_labels.count('inlet')}, outlet={seg_labels.count('outlet')}")
print(f"Closure error: {closure_error:.3e} m")
print(f"CAD area: {face0.area*1e6:.3f} mm^2; discretized area: {shoelace_area*1e6:.3f} mm^2; difference: {(shoelace_area-face0.area)/face0.area*100:.3f}%")

# -- cell 6 -------------------------------------------------------------------------
# The wire traversal is now correct; the remaining area difference is intentional topology that the fi
print("Face wire methods:", [n for n in dir(face0) if "wire" in n.lower()])
all_wires = list(face0.wires())
wire_polys = []; wire_seg_labels = []
for wi, wire in enumerate(all_wires):
    edges = list(wire.order_edges()); records=[]; current=None
    for e in edges:
        n=max(1,int(np.ceil(e.length/BOUNDARY_CHORD)))
        samples=np.array([[e.position_at(float(u)).X,e.position_at(float(u)).Y] for u in np.linspace(0,1,n+1)])
        if current is not None and np.linalg.norm(samples[0]-current)>np.linalg.norm(samples[-1]-current): samples=samples[::-1]
        xmean=float(samples[:,0].mean()); term=np.ptp(samples[:,0])<1e-10 and abs(e.length-CHANNEL_W)<1e-8
        label="inlet" if term and abs(xmean)<1e-10 else ("outlet" if term and abs(xmean-LENGTH)<1e-10 else "walls")
        records.append((label,samples)); current=samples[-1]
    pts=[records[0][1][0]]; labels=[]
    for label,samples in records:
        if np.linalg.norm(samples[0]-pts[-1])>np.linalg.norm(samples[-1]-pts[-1]): samples=samples[::-1]
        for q in samples[1:]: pts.append(q); labels.append(label)
    pts=np.asarray(pts)
    signed=0.5*(np.dot(pts[:-1,0],pts[1:,1])-np.dot(pts[:-1,1],pts[1:,0]))
    wire_polys.append(pts); wire_seg_labels.append(labels)
    print(f"wire {wi}: segments={len(labels)}, signed area={signed*1e6:.3f} mm^2, closure={np.linalg.norm(pts[-1]-pts[0]):.2e} m")
areas=sorted([abs(0.5*(np.dot(p[:-1,0],p[1:,1])-np.dot(p[:-1,1],p[1:,0]))) for p in wire_polys],reverse=True)
net_area=areas[0]-sum(areas[1:])
print(f"Wires found: {len(all_wires)} (1 outer + {len(all_wires)-1} islands)")
print(f"CAD area={face0.area*1e6:.3f} mm^2; polygon net area={net_area*1e6:.3f} mm^2; difference={(net_area-face0.area)/face0.area*100:.3f}%")

# -- cell 7 -------------------------------------------------------------------------
# The Gmsh construction itself reached extrusion, but a local typo referenced Python’s built-in `next`
import gmsh, os
THICKNESS = 0.001
MESH_SIZE = 0.0010
solid = bd.extrude(face0, THICKNESS)
bd.export_step(solid, "teslaValve.step")
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.model.add("teslaValve")
curve_labels={}; loop_tags=[]
for pts, labels in zip(wire_polys, wire_seg_labels):
    local_pts=[gmsh.model.geo.addPoint(float(x),float(y),0,MESH_SIZE) for x,y in pts[:-1]]
    curves=[]
    for k,label in enumerate(labels):
        c=gmsh.model.geo.addLine(local_pts[k],local_pts[(k+1)%len(local_pts)])
        curves.append(c); curve_labels[c]=label
    loop_tags.append(gmsh.model.geo.addCurveLoop(curves))
base_surface=gmsh.model.geo.addPlaneSurface(loop_tags)
ext=gmsh.model.geo.extrude([(2,base_surface)],0,0,THICKNESS,numElements=[1],recombine=False)
gmsh.model.geo.synchronize()
top_surface=ext[0][1]; volume=ext[1][1]
side_groups={"inlet":set(),"outlet":set(),"walls":set()}
for c,label in curve_labels.items():
    upward,_=gmsh.model.getAdjacencies(1,c)
    lateral=[int(s) for s in upward if int(s)!=base_surface]
    if len(lateral)!=1: raise RuntimeError(f"curve {c} has lateral surfaces {lateral}")
    side_groups[label].add(lateral[0])
for name in ("inlet","outlet","walls"):
    pg=gmsh.model.addPhysicalGroup(2,sorted(side_groups[name])); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(2,[base_surface,top_surface]); gmsh.model.setPhysicalName(2,pg,"frontAndBack")
pg=gmsh.model.addPhysicalGroup(3,[volume]); gmsh.model.setPhysicalName(3,pg,"fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin",MESH_SIZE); gmsh.option.setNumber("Mesh.MeshSizeMax",MESH_SIZE)
gmsh.option.setNumber("Mesh.Algorithm",6); gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.model.mesh.generate(3); gmsh.write("teslaValve.msh")
node_tags,_,_=gmsh.model.mesh.getNodes(); elem_types,elem_tags,_=gmsh.model.mesh.getElements(3,volume)
print(f"CAD solid volume: {solid.volume*1e9:.3f} mm^3 (area x depth: {face0.area*THICKNESS*1e9:.3f} mm^3)")
print(f"Gmsh nodes: {len(node_tags)}; volume elements: {sum(len(a) for a in elem_tags)}; element types: {elem_types}")
print("Patch surface counts:", {k:len(v) for k,v in side_groups.items()}, "; frontAndBack=2")
gmsh.finalize()

# -- cell 8 -------------------------------------------------------------------------
# The coarse Gmsh run produced 24,210 tetrahedra. Although valid 3D cells, tetrahedra are unsuitable f
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.model.add("teslaValve")
curve_labels={}; loop_tags=[]
for pts, labels in zip(wire_polys, wire_seg_labels):
    local_pts=[gmsh.model.geo.addPoint(float(x),float(y),0,MESH_SIZE) for x,y in pts[:-1]]; curves=[]
    for k,label in enumerate(labels):
        c=gmsh.model.geo.addLine(local_pts[k],local_pts[(k+1)%len(local_pts)])
        curves.append(c); curve_labels[c]=label
    loop_tags.append(gmsh.model.geo.addCurveLoop(curves))
base_surface=gmsh.model.geo.addPlaneSurface(loop_tags)
ext=gmsh.model.geo.extrude([(2,base_surface)],0,0,THICKNESS,numElements=[1],recombine=True)
gmsh.model.geo.synchronize(); top_surface=ext[0][1]; volume=ext[1][1]
side_groups={"inlet":set(),"outlet":set(),"walls":set()}
for c,label in curve_labels.items():
    upward,_=gmsh.model.getAdjacencies(1,c); lateral=[int(s) for s in upward if int(s)!=base_surface]
    if len(lateral)!=1: raise RuntimeError(f"curve {c}: {lateral}")
    side_groups[label].add(lateral[0])
for name in ("inlet","outlet","walls"):
    pg=gmsh.model.addPhysicalGroup(2,sorted(side_groups[name])); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(2,[base_surface,top_surface]); gmsh.model.setPhysicalName(2,pg,"frontAndBack")
pg=gmsh.model.addPhysicalGroup(3,[volume]); gmsh.model.setPhysicalName(3,pg,"fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin",MESH_SIZE); gmsh.option.setNumber("Mesh.MeshSizeMax",MESH_SIZE)
gmsh.option.setNumber("Mesh.Algorithm",6); gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.model.mesh.generate(3); gmsh.write("teslaValve.msh")
node_tags,_,_=gmsh.model.mesh.getNodes(); types,tags,_=gmsh.model.mesh.getElements(3,volume)
print(f"Recombined mesh nodes: {len(node_tags)}; volume cells: {sum(len(a) for a in tags)}; Gmsh element types: {types}")
print("Expected Gmsh type 6 = six-node triangular prism")
gmsh.finalize()

# -- cell 9 -------------------------------------------------------------------------
# The corrected mesh contains 8,070 six-node prisms in exactly one extruded layer. I’ll convert it to 
import subprocess, pathlib, re
pathlib.Path("system").mkdir(exist_ok=True)
pathlib.Path("constant").mkdir(exist_ok=True)
pathlib.Path("system/controlDict").write_text("FoamFile\n{ version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
r=subprocess.run(["gmshToFoam","teslaValve.msh"],text=True,capture_output=True)
print(r.stdout[-3000:]); print(r.stderr[-1000:])
if r.returncode: raise RuntimeError(f"gmshToFoam failed {r.returncode}")
bpath=pathlib.Path("constant/polyMesh/boundary")
txt=bpath.read_text()
# Change the explicitly named planar pair only; other patch types remain ordinary wall/inlet/outlet patches.
pat=r"(frontAndBack\s*\{[^}]*?type\s+)patch(\s*;)"
txt,n=re.subn(pat,r"\1empty\2",txt,flags=re.S)
if n!=1: raise RuntimeError(f"frontAndBack patch edit count {n}")
bpath.write_text(txt)
print("\nOpenFOAM boundary table:\n",txt)

# -- cell 10 ------------------------------------------------------------------------
# Conversion preserved all 8,070 prisms and the four intended patches; `frontAndBack` contains the pai
import pyvista as pv
pathlib.Path("t3.foam").touch()
reader=pv.OpenFOAMReader("t3.foam")
reader.set_active_time_value(0.0)
data=reader.read()
print(data)
# Collect internal mesh blocks and render all cell edges from the planar normal.
mesh_blocks=[]
for block in data:
    if block is not None:
        if isinstance(block,pv.MultiBlock): mesh_blocks.extend([b for b in block if b is not None and b.n_cells])
        elif block.n_cells: mesh_blocks.append(block)
mesh=mesh_blocks[0] if len(mesh_blocks)==1 else pv.merge(mesh_blocks)
print(f"Rendered dataset: {mesh.n_cells} cells, {mesh.n_points} points, bounds={mesh.bounds}")
pl=pv.Plotter(off_screen=True,window_size=(1400,500))
pl.add_mesh(mesh,show_edges=True,color="lightsteelblue",edge_color="navy",line_width=0.35)
pl.view_xy(); pl.camera.parallel_projection=True; pl.reset_camera()
pl.add_text("Coarse 1 mm Tesla-valve mesh: 8,070 prismatic cells",font_size=12)
pl.show(screenshot="coarseMesh.png")

# -- cell 11 ------------------------------------------------------------------------
# The top view confirms a connected straight passage with four alternating teardrop bypasses and no un
r=subprocess.run(["checkMesh","-allGeometry","-allTopology"],text=True,capture_output=True)
pathlib.Path("checkMesh.log").write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-7000:])
print("checkMesh return code:",r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# `checkMesh` did not reach the mesh because this OpenFOAM build requires placeholder `fvSchemes` and 
pathlib.Path("system/fvSchemes").write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
pathlib.Path("system/fvSolution").write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { }
''')
r=subprocess.run(["checkMesh","-allGeometry","-allTopology"],text=True,capture_output=True)
pathlib.Path("checkMesh.log").write_text(r.stdout+r.stderr)
print((r.stdout+r.stderr)[-7000:])
if r.returncode: raise RuntimeError(f"checkMesh failed with code {r.returncode}")

# -- cell 13 ------------------------------------------------------------------------
# The extended check found seven under-determined cells and 13 low-weight faces, traced to 10.6 µm sli
MIN_SEGMENT = 0.00020
clean_polys=[]; clean_labels=[]
for pts, labels in zip(wire_polys,wire_seg_labels):
    raw=pts[:-1]; n=len(raw); keep=[0]
    for i in range(1,n):
        if np.linalg.norm(raw[i]-raw[keep[-1]]) >= MIN_SEGMENT: keep.append(i)
    if len(keep)>2 and np.linalg.norm(raw[keep[-1]]-raw[keep[0]]) < MIN_SEGMENT: keep.pop()
    cp=raw[keep]
    cl=[]
    for a_idx,b_idx in zip(keep,keep[1:]+[keep[0]+n]):
        labs=[labels[k%n] for k in range(a_idx,b_idx)]
        nonwall=[x for x in labs if x!="walls"]
        cl.append(nonwall[0] if nonwall and all(x==nonwall[0] for x in nonwall) else "walls")
    clean_polys.append(np.vstack([cp,cp[0]])); clean_labels.append(cl)
areas_clean=[]
for p in clean_polys:
    areas_clean.append(abs(0.5*(np.dot(p[:-1,0],p[1:,1])-np.dot(p[:-1,1],p[1:,0]))))
net_clean=max(areas_clean)-(sum(areas_clean)-max(areas_clean))
all_lengths=[np.linalg.norm(p[i+1]-p[i]) for p in clean_polys for i in range(len(p)-1)]
print(f"Segments before/after: {sum(map(len,wire_seg_labels))}/{sum(map(len,clean_labels))}")
print(f"Minimum cleaned segment: {min(all_lengths)*1e3:.4f} mm")
print(f"CAD area={face0.area*1e6:.3f} mm^2; cleaned area={net_clean*1e6:.3f} mm^2; change={(net_clean-face0.area)/face0.area*100:.4f}%")
print(f"Clean inlet/outlet segment counts: {sum(x.count('inlet') for x in clean_labels)}/{sum(x.count('outlet') for x in clean_labels)}")

# -- cell 14 ------------------------------------------------------------------------
# Removing 16 sliver segments raises the minimum boundary edge to 0.272 mm while changing fluid area b
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.model.add("teslaValveClean")
curve_labels={}; loop_tags=[]
for pts, labels in zip(clean_polys,clean_labels):
    local_pts=[gmsh.model.geo.addPoint(float(x),float(y),0,MESH_SIZE) for x,y in pts[:-1]]; curves=[]
    for k,label in enumerate(labels):
        c=gmsh.model.geo.addLine(local_pts[k],local_pts[(k+1)%len(local_pts)])
        curves.append(c); curve_labels[c]=label
    loop_tags.append(gmsh.model.geo.addCurveLoop(curves))
base_surface=gmsh.model.geo.addPlaneSurface(loop_tags)
ext=gmsh.model.geo.extrude([(2,base_surface)],0,0,THICKNESS,numElements=[1],recombine=True)
gmsh.model.geo.synchronize(); top_surface=ext[0][1]; volume=ext[1][1]
side_groups={"inlet":set(),"outlet":set(),"walls":set()}
for c,label in curve_labels.items():
    upward,_=gmsh.model.getAdjacencies(1,c); lateral=[int(s) for s in upward if int(s)!=base_surface]
    if len(lateral)!=1: raise RuntimeError(f"curve {c}: {lateral}")
    side_groups[label].add(lateral[0])
for name in ("inlet","outlet","walls"):
    pg=gmsh.model.addPhysicalGroup(2,sorted(side_groups[name])); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(2,[base_surface,top_surface]); gmsh.model.setPhysicalName(2,pg,"frontAndBack")
pg=gmsh.model.addPhysicalGroup(3,[volume]); gmsh.model.setPhysicalName(3,pg,"fluid")
gmsh.option.setNumber("Mesh.MeshSizeMin",MESH_SIZE); gmsh.option.setNumber("Mesh.MeshSizeMax",MESH_SIZE)
gmsh.option.setNumber("Mesh.Algorithm",6); gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
gmsh.model.mesh.generate(3); gmsh.write("teslaValve.msh")
node_tags,_,_=gmsh.model.mesh.getNodes(); types,tags,_=gmsh.model.mesh.getElements(3,volume)
print(f"Clean mesh nodes={len(node_tags)}, cells={sum(len(a) for a in tags)}, element types={types}")
gmsh.finalize()

# -- cell 15 ------------------------------------------------------------------------
# The cleaned coarse mesh contains 7,518 prisms. I’ll replace the previous OpenFOAM mesh with this ver
import shutil
shutil.rmtree("constant/polyMesh",ignore_errors=True)
r=subprocess.run(["gmshToFoam","teslaValve.msh"],text=True,capture_output=True)
if r.returncode: print(r.stdout+r.stderr); raise RuntimeError("gmshToFoam failed")
bpath=pathlib.Path("constant/polyMesh/boundary"); txt=bpath.read_text()
txt,n=re.subn(r"(frontAndBack\s*\{[^}]*?type\s+)patch(\s*;)",r"\1empty\2",txt,flags=re.S)
if n!=1: raise RuntimeError(f"frontAndBack edit count {n}")
bpath.write_text(txt)
print("Conversion cell summary:")
for line in r.stdout.splitlines():
    if any(key in line for key in ("total:","prism:","Patch 0 gets","Patch 1 gets","Patch 2 gets","Patch 3 gets")): print(line)
print(re.sub(r"//.*","",txt)[-900:])

# -- cell 16 ------------------------------------------------------------------------
# The replacement mesh has the intended 7,518-prism single layer and patch counts. I’ll rerun the full
r=subprocess.run(["checkMesh","-allGeometry","-allTopology"],text=True,capture_output=True)
log=r.stdout+r.stderr; pathlib.Path("checkMesh.log").write_text(log)
print(log[-5200:])
if r.returncode or "Failed" in log: raise RuntimeError("checkMesh did not pass cleanly")

# -- cell 17 ------------------------------------------------------------------------
# The cleaned mesh now passes all extended OpenFOAM checks: maximum non-orthogonality 23.30°, maximum 
r=subprocess.run(["foamToSurface","-help"],text=True,capture_output=True)
print((r.stdout+r.stderr)[:5000])

# -- cell 18 ------------------------------------------------------------------------
# `foamToSurface` only offers a combined export. To preserve patch names, I’ll instead read the checke
tri_dir=pathlib.Path("constant/triSurface"); shutil.rmtree(tri_dir,ignore_errors=True); tri_dir.mkdir(parents=True)
# Parse OpenFOAM's ASCII point and face lists.
pt_lines=pathlib.Path("constant/polyMesh/points").read_text().splitlines()
pts=[]
for line in pt_lines:
    m=re.match(r"\s*\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)\s*$",line)
    if m: pts.append(tuple(map(float,m.groups())))
pts=np.asarray(pts)
face_lines=pathlib.Path("constant/polyMesh/faces").read_text().splitlines(); faces=[]
for line in face_lines:
    m=re.match(r"\s*\d+\(([^)]*)\)\s*$",line)
    if m: faces.append([int(x) for x in m.group(1).split()])
# Patch ranges are explicit in the boundary table.
btxt=pathlib.Path("constant/polyMesh/boundary").read_text(); patches={}
for name,body in re.findall(r"\n\s*(\w+)\s*\{([^}]*)\}",btxt,re.S):
    nf=re.search(r"nFaces\s+(\d+)",body); sf=re.search(r"startFace\s+(\d+)",body)
    if nf and sf: patches[name]=(int(sf.group(1)),int(nf.group(1)))
def write_stl(name,start,count):
    tris=[]
    for f in faces[start:start+count]:
        for k in range(1,len(f)-1): tris.append((f[0],f[k],f[k+1]))
    with open(tri_dir/f"{name}.stl","w") as out:
        out.write(f"solid {name}\n")
        for a,b,c in tris:
            pa,pb,pc=pts[[a,b,c]]; n=np.cross(pb-pa,pc-pa); nn=np.linalg.norm(n); n=n/nn if nn else n
            out.write(f" facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n")
            for p in (pa,pb,pc): out.write(f"   vertex {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n")
            out.write("  endloop\n endfacet\n")
        out.write(f"endsolid {name}\n")
    return len(tris)
counts={name:write_stl(name,*rng) for name,rng in patches.items()}
print(f"Parsed {len(pts)} points and {len(faces)} faces")
print("Exported patch STL triangle counts:",counts)

# -- cell 19 ------------------------------------------------------------------------
# The reader nests its blocks differently than expected, so the diagnostic tried to query a cell count
reader=pv.OpenFOAMReader("t3.foam"); reader.set_active_time_value(0.0); data=reader.read()
def walk_blocks(mb,prefix=""):
    found=[]
    for i in range(len(mb)):
        name=mb.get_block_name(i); b=mb[i]; path=f"{prefix}/{name}"
        if isinstance(b,pv.MultiBlock): found.extend(walk_blocks(b,path))
        elif b is not None:
            print(path,type(b).__name__,b.n_cells); found.append((name,b))
    return found
leaves=walk_blocks(data)
internal=next((b for name,b in leaves if name=="internalMesh"),None)
if internal is None: raise RuntimeError("internalMesh block not found")
pl=pv.Plotter(off_screen=True,window_size=(1500,540))
pl.add_mesh(internal,show_edges=True,color="aliceblue",edge_color="navy",line_width=0.4)
pl.view_xy(); pl.camera.parallel_projection=True; pl.reset_camera()
pl.add_text(f"Checked coarse Tesla-valve mesh: {internal.n_cells:,} one-layer prisms",font_size=12)
pl.show(screenshot="teslaValveMesh.png")
print(f"Rendered internalMesh: {internal.n_cells} cells, {internal.n_points} points")
