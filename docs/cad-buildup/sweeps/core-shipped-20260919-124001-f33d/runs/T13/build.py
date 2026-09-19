"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the leak path as a thin helical ribbon between two parallel 30° trapezoidal-thread flanks
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Requested dimensions, in metres
neck_od = 40e-3
pitch = 2e-3
thread_depth = 1e-3
thread_height = 12e-3
gap = 0.2e-3
starts = 1
# Assumption: standard metric trapezoidal thread, 30 degree included angle
included_angle_deg = 30.0
half_angle = np.deg2rad(included_angle_deg / 2)
turns = thread_height / pitch
r_crest = neck_od / 2
r_root = r_crest - thread_depth

def helix_xyz(u, s, t):
    """Helical clearance ribbon: u=turns, s=root-to-crest, t=neck-to-lid."""
    u, s, t = np.broadcast_arrays(u, s, t)
    rb = r_root + thread_depth * s
    r = rb - gap * np.sin(half_angle) * t
    th = 2 * np.pi * u
    z = pitch * u + np.tan(half_angle) * (rb - r_root) + gap * np.cos(half_angle) * t
    return r*np.cos(th), r*np.sin(th), z

# Coarse preview of the two opposing flank surfaces and boundary curves
u = np.linspace(0, turns, 145)
s = np.linspace(0, 1, 7)
U, S = np.meshgrid(u, s, indexing='ij')
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
for t, color, label in [(0, 'steelblue', 'neck flank'), (1, 'darkorange', 'lid flank')]:
    X, Y, Z = helix_xyz(U, S, t)
    ax.plot_surface(X*1e3, Y*1e3, Z*1e3, color=color, alpha=0.58, linewidth=0)
    ax.plot([], [], [], color=color, lw=6, label=label)
for sv in (0, 1):
    for tv in (0, 1):
        X, Y, Z = helix_xyz(u, sv, tv)
        ax.plot(X*1e3, Y*1e3, Z*1e3, color='k', lw=.55)
ax.set(xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]', title='Coarse helical gap preview (6 turns)')
ax.set_box_aspect((40, 40, 14))
ax.legend()
plt.tight_layout()
plt.show()

normal_gap = np.hypot(gap*np.sin(half_angle), gap*np.cos(half_angle))
print(f'turns: measured {turns:.6g}, requested {thread_height/pitch:.6g}')
print(f'pitch: measured {pitch*1e3:.6g} mm, requested 2 mm')
print(f'crest diameter: measured {2*r_crest*1e3:.6g} mm, requested 40 mm')
print(f'radial depth: measured {(r_crest-r_root)*1e3:.6g} mm, requested 1 mm')
print(f'normal flank gap: measured {normal_gap*1e3:.6g} mm, requested 0.2 mm')
print(f'axial helix span: measured {turns*pitch*1e3:.6g} mm, requested 12 mm')

# -- cell 2 -------------------------------------------------------------------------
# The preview shows the intended open-ended six-turn ribbon without turn-to-turn overlap. I’ll now dis
# Coarse structured discretisation; all patch STLs share these exact boundary vertices.
nu, ns, nt = 192, 5, 2  # 32 circumferential cells/turn, 5 radial, 2 across clearance
ug = np.linspace(0, turns, nu + 1)
sg = np.linspace(0, 1, ns + 1)
tg = np.linspace(0, 1, nt + 1)
UU, SS, TT = np.meshgrid(ug, sg, tg, indexing='ij')
X, Y, Z = helix_xyz(UU, SS, TT)
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
vid = np.arange(points.shape[0]).reshape((nu + 1, ns + 1, nt + 1))

# Each quad is ordered approximately outward; exact winding is corrected geometrically.
patch_quads = {
    'innerEnd': [[vid[0,j,k], vid[0,j,k+1], vid[0,j+1,k+1], vid[0,j+1,k]] for j in range(ns) for k in range(nt)],
    'outerEnd': [[vid[nu,j,k], vid[nu,j+1,k], vid[nu,j+1,k+1], vid[nu,j,k+1]] for j in range(ns) for k in range(nt)],
    'threadRoot': [[vid[i,0,k], vid[i+1,0,k], vid[i+1,0,k+1], vid[i,0,k+1]] for i in range(nu) for k in range(nt)],
    'threadCrest': [[vid[i,ns,k], vid[i,ns,k+1], vid[i+1,ns,k+1], vid[i+1,ns,k]] for i in range(nu) for k in range(nt)],
    'neckFlank': [[vid[i,j,0], vid[i,j+1,0], vid[i+1,j+1,0], vid[i+1,j,0]] for i in range(nu) for j in range(ns)],
    'lidFlank': [[vid[i,j,nt], vid[i+1,j,nt], vid[i+1,j+1,nt], vid[i,j+1,nt]] for i in range(nu) for j in range(ns)],
}

def outward_quad(q, interior):
    q = list(map(int, q)); p = points[q]
    n = np.cross(p[1]-p[0], p[2]-p[0])
    return q if np.dot(n, p.mean(axis=0)-interior) > 0 else q[::-1]

# Correct each boundary face against an adjacent interior point.
for name, qs in patch_quads.items():
    fixed = []
    for q in qs:
        fc = points[np.asarray(q)].mean(axis=0)
        # Global ribbon centre at the face's mean helical phase is a sufficient inward reference.
        theta = np.unwrap(np.arctan2(points[np.asarray(q),1], points[np.asarray(q),0]))
        um = (theta.mean()/(2*np.pi))
        if um < -0.25: um += turns
        rad = np.linalg.norm(fc[:2])
        sm = np.clip((rad-r_root)/thread_depth, 0, 1)
        interior = np.array(helix_xyz(um, sm, 0.5))
        fixed.append(outward_quad(q, interior))
    patch_quads[name] = fixed

tri_dir = Path('constant/triSurface'); tri_dir.mkdir(parents=True, exist_ok=True)
def write_ascii_stl(path, quads, solid_name):
    with open(path, 'w') as f:
        f.write(f'solid {solid_name}\n')
        for q in quads:
            for tri in ((q[0],q[1],q[2]), (q[0],q[2],q[3])):
                p = points[np.asarray(tri)]; n = np.cross(p[1]-p[0], p[2]-p[0]); n /= np.linalg.norm(n)
                f.write(f' facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n')
                for v in p: f.write(f'   vertex {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}\n')
                f.write('  endloop\n endfacet\n')
        f.write(f'endsolid {solid_name}\n')
for name, qs in patch_quads.items():
    write_ascii_stl(tri_dir/f'{name}.stl', qs, name)
    print(f'{name:12s}: {len(qs)*2:5d} triangles -> {tri_dir/name}.stl')
print(f'coarse volume grid: {nu*ns*nt} hexahedra; {len(points)} points')

# -- cell 3 -------------------------------------------------------------------------
# I’m correcting every surface winding against its actual adjacent cell (important because Cartesian c
# Rebuild boundary lists with exact adjacent-cell references and overwrite the STLs with robust outward winding.
def cell_vertices(i,j,k):
    return [vid[i,j,k],vid[i+1,j,k],vid[i+1,j+1,k],vid[i,j+1,k],
            vid[i,j,k+1],vid[i+1,j,k+1],vid[i+1,j+1,k+1],vid[i,j+1,k+1]]
def orient_from_cell(q, i, j, k):
    q = list(map(int,q)); p = points[q]; cc = points[cell_vertices(i,j,k)].mean(axis=0)
    n = np.cross(p[1]-p[0],p[2]-p[0])
    return q if np.dot(n,p.mean(axis=0)-cc)>0 else q[::-1]

patch_items = {n: [] for n in patch_quads}
for j in range(ns):
    for k in range(nt):
        patch_items['innerEnd'].append((orient_from_cell([vid[0,j,k],vid[0,j,k+1],vid[0,j+1,k+1],vid[0,j+1,k]],0,j,k),(0,j,k)))
        patch_items['outerEnd'].append((orient_from_cell([vid[nu,j,k],vid[nu,j+1,k],vid[nu,j+1,k+1],vid[nu,j,k+1]],nu-1,j,k),(nu-1,j,k)))
for i in range(nu):
    for k in range(nt):
        patch_items['threadRoot'].append((orient_from_cell([vid[i,0,k],vid[i+1,0,k],vid[i+1,0,k+1],vid[i,0,k+1]],i,0,k),(i,0,k)))
        patch_items['threadCrest'].append((orient_from_cell([vid[i,ns,k],vid[i,ns,k+1],vid[i+1,ns,k+1],vid[i+1,ns,k]],i,ns-1,k),(i,ns-1,k)))
for i in range(nu):
    for j in range(ns):
        patch_items['neckFlank'].append((orient_from_cell([vid[i,j,0],vid[i,j+1,0],vid[i+1,j+1,0],vid[i+1,j,0]],i,j,0),(i,j,0)))
        patch_items['lidFlank'].append((orient_from_cell([vid[i,j,nt],vid[i+1,j,nt],vid[i+1,j+1,nt],vid[i,j+1,nt]],i,j,nt-1),(i,j,nt-1)))
patch_quads = {n:[q for q,_ in items] for n,items in patch_items.items()}
for name, qs in patch_quads.items(): write_ascii_stl(tri_dir/f'{name}.stl', qs, name)

# Assemble owner-oriented cell faces, deduplicating internal faces.
face_map = {}
for i in range(nu):
  for j in range(ns):
    for k in range(nt):
      c = (i*ns+j)*nt+k
      v = cell_vertices(i,j,k)
      candidates = [[v[0],v[3],v[2],v[1]],[v[4],v[5],v[6],v[7]],
                    [v[0],v[1],v[5],v[4]],[v[1],v[2],v[6],v[5]],
                    [v[2],v[3],v[7],v[6]],[v[3],v[0],v[4],v[7]]]
      cc = points[v].mean(axis=0)
      for q in candidates:
        p=points[q]; n=np.cross(p[1]-p[0],p[2]-p[0])
        if np.dot(n,p.mean(axis=0)-cc)<0: q=q[::-1]
        key=tuple(sorted(q))
        if key in face_map: face_map[key]['neighbour']=c
        else: face_map[key]={'verts':q,'owner':c,'neighbour':None}

boundary_lookup = {tuple(sorted(q)):name for name,qs in patch_quads.items() for q in qs}
internal = [f for f in face_map.values() if f['neighbour'] is not None]
boundary_by_patch = {n:[] for n in patch_quads}
for key,f in face_map.items():
    if f['neighbour'] is None: boundary_by_patch[boundary_lookup[key]].append(f)
ordered = internal + [f for n in boundary_by_patch for f in boundary_by_patch[n]]
poly = Path('constant/polyMesh'); poly.mkdir(parents=True, exist_ok=True)

def foam_header(cls,obj):
    return f'''FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n'''
with open(poly/'points','w') as f:
    f.write(foam_header('vectorField','points')+f'{len(points)}\n(\n')
    for p in points: f.write(f'({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})\n')
    f.write(')\n')
with open(poly/'faces','w') as f:
    f.write(foam_header('faceList','faces')+f'{len(ordered)}\n(\n')
    for a in ordered: f.write(f"4({' '.join(map(str,a['verts']))})\n")
    f.write(')\n')
with open(poly/'owner','w') as f:
    f.write(foam_header('labelList','owner')+f'{len(ordered)}\n(\n')
    for a in ordered: f.write(f"{a['owner']}\n")
    f.write(')\n')
with open(poly/'neighbour','w') as f:
    f.write(foam_header('labelList','neighbour')+f'{len(internal)}\n(\n')
    for a in internal: f.write(f"{a['neighbour']}\n")
    f.write(')\n')
start=len(internal)
with open(poly/'boundary','w') as f:
    f.write(foam_header('polyBoundaryMesh','boundary')+f'{len(boundary_by_patch)}\n(\n')
    for name, fs in boundary_by_patch.items():
        f.write(f'{name}\n{{\n type patch;\n nFaces {len(fs)};\n startFace {start};\n}}\n'); start += len(fs)
    f.write(')\n')
print(f'wrote coarse OpenFOAM mesh: {nu*ns*nt} cells, {len(internal)} internal faces, {len(ordered)-len(internal)} boundary faces')

# -- cell 4 -------------------------------------------------------------------------
# The mesh itself was not read because this geometry-only case still needs a minimal `system/controlDi
system_dir = Path('system'); system_dir.mkdir(exist_ok=True)
(system_dir/'controlDict').write_text('''FoamFile
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
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
''')
print('wrote minimal system/controlDict for mesh utilities')

# -- cell 5 -------------------------------------------------------------------------
# This OpenFOAM build also requires empty finite-volume dictionaries even for `checkMesh`. I’ll add th
(system_dir/'fvSchemes').write_text('''FoamFile
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
''')
(system_dir/'fvSolution').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
print('wrote system/fvSchemes and system/fvSolution for mesh utilities')

# -- cell 6 -------------------------------------------------------------------------
# I’ll now run the coarse mesh check again.
import subprocess
cm = subprocess.run(['checkMesh'], text=True, capture_output=True)
print(cm.stdout)
if cm.returncode != 0:
    print(cm.stderr)
    raise RuntimeError(f'checkMesh returned {cm.returncode}')

# -- cell 7 -------------------------------------------------------------------------
# All geometry metrics are good, but `checkMesh` reports the internal face list is not in OpenFOAM’s p
internal = sorted(internal, key=lambda a: (a['owner'], a['neighbour']))
ordered = internal + [f for n in boundary_by_patch for f in boundary_by_patch[n]]
with open(poly/'faces','w') as f:
    f.write(foam_header('faceList','faces')+f'{len(ordered)}\n(\n')
    for a in ordered: f.write(f"4({' '.join(map(str,a['verts']))})\n")
    f.write(')\n')
with open(poly/'owner','w') as f:
    f.write(foam_header('labelList','owner')+f'{len(ordered)}\n(\n')
    for a in ordered: f.write(f"{a['owner']}\n")
    f.write(')\n')
with open(poly/'neighbour','w') as f:
    f.write(foam_header('labelList','neighbour')+f'{len(internal)}\n(\n')
    for a in internal: f.write(f"{a['neighbour']}\n")
    f.write(')\n')
print('sorted internal faces into owner/neighbour upper-triangular order')

# -- cell 8 -------------------------------------------------------------------------
# I’ll verify that the addressing correction clears the mesh check.
import subprocess
cm = subprocess.run(['checkMesh'], text=True, capture_output=True)
print(cm.stdout)
if 'Mesh OK.' not in cm.stdout:
    raise RuntimeError('bare checkMesh did not report Mesh OK')

# -- cell 9 -------------------------------------------------------------------------
# The coarse mesh passes. I’ll now render the actual boundary quads and mesh lines, highlighting the t
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
colors = {'neckFlank':'#4c78a8','lidFlank':'#f58518','innerEnd':'#54a24b','outerEnd':'#e45756',
          'threadRoot':'#777777','threadCrest':'#aaaaaa'}
# Draw filled ends and sparse grid lines on all longitudinal patches.
for name, qs in patch_quads.items():
    stride = 1 if 'End' in name else 8
    selected = qs if 'End' in name else qs[::stride]
    polys = [points[np.asarray(q)]*1e3 for q in selected]
    pc = Poly3DCollection(polys, facecolor=colors[name], edgecolor='k', linewidth=.2,
                          alpha=.9 if 'End' in name else .32)
    ax.add_collection3d(pc)
for sv in np.linspace(0,1,ns+1):
    for tv in np.linspace(0,1,nt+1):
        xx,yy,zz=helix_xyz(ug,sv,tv); ax.plot(xx*1e3,yy*1e3,zz*1e3,color='k',lw=.25,alpha=.65)
ax.set_xlim(-21,21); ax.set_ylim(-21,21); ax.set_zlim(0,13)
ax.set_box_aspect((42,42,16)); ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title='Actual coarse mesh boundary (1,920 hex cells)')
for n in ('neckFlank','lidFlank','innerEnd','outerEnd'): ax.plot([],[],[],color=colors[n],lw=6,label=n)
ax.legend(loc='upper right'); plt.tight_layout(); plt.show()

# -- cell 10 ------------------------------------------------------------------------
# The inspected mesh has the right connected helical topology and clean end patches. For the final mes
def build_helical_mesh(nu_f=384, ns_f=8, nt_f=3):
    ug_f=np.linspace(0,turns,nu_f+1); sg_f=np.linspace(0,1,ns_f+1); tg_f=np.linspace(0,1,nt_f+1)
    U,S,T=np.meshgrid(ug_f,sg_f,tg_f,indexing='ij'); x,y,z=helix_xyz(U,S,T)
    pts=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    ids=np.arange(len(pts)).reshape(nu_f+1,ns_f+1,nt_f+1)
    def cv(i,j,k): return [ids[i,j,k],ids[i+1,j,k],ids[i+1,j+1,k],ids[i,j+1,k],ids[i,j,k+1],ids[i+1,j,k+1],ids[i+1,j+1,k+1],ids[i,j+1,k+1]]
    def oq(q,i,j,k):
        q=list(map(int,q)); p=pts[q]; cc=pts[cv(i,j,k)].mean(0); n=np.cross(p[1]-p[0],p[2]-p[0])
        return q if np.dot(n,p.mean(0)-cc)>0 else q[::-1]
    pq={n:[] for n in ('innerEnd','outerEnd','threadRoot','threadCrest','neckFlank','lidFlank')}
    for j in range(ns_f):
      for k in range(nt_f):
        pq['innerEnd'].append(oq([ids[0,j,k],ids[0,j,k+1],ids[0,j+1,k+1],ids[0,j+1,k]],0,j,k))
        pq['outerEnd'].append(oq([ids[nu_f,j,k],ids[nu_f,j+1,k],ids[nu_f,j+1,k+1],ids[nu_f,j,k+1]],nu_f-1,j,k))
    for i in range(nu_f):
      for k in range(nt_f):
        pq['threadRoot'].append(oq([ids[i,0,k],ids[i+1,0,k],ids[i+1,0,k+1],ids[i,0,k+1]],i,0,k))
        pq['threadCrest'].append(oq([ids[i,ns_f,k],ids[i,ns_f,k+1],ids[i+1,ns_f,k+1],ids[i+1,ns_f,k]],i,ns_f-1,k))
      for j in range(ns_f):
        pq['neckFlank'].append(oq([ids[i,j,0],ids[i,j+1,0],ids[i+1,j+1,0],ids[i+1,j,0]],i,j,0))
        pq['lidFlank'].append(oq([ids[i,j,nt_f],ids[i+1,j,nt_f],ids[i+1,j+1,nt_f],ids[i,j+1,nt_f]],i,j,nt_f-1))
    # Export one consistently triangulated STL per patch.
    old_points=globals()['points']; globals()['points']=pts
    for name,qs in pq.items(): write_ascii_stl(tri_dir/f'{name}.stl',qs,name)
    globals()['points']=old_points
    # Build owner-oriented faces.
    fm={}
    for i in range(nu_f):
      for j in range(ns_f):
       for k in range(nt_f):
        c=(i*ns_f+j)*nt_f+k; v=cv(i,j,k); cc=pts[v].mean(0)
        candidates=[[v[0],v[3],v[2],v[1]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
                    [v[1],v[2],v[6],v[5]],[v[2],v[3],v[7],v[6]],[v[3],v[0],v[4],v[7]]]
        for q in candidates:
          p=pts[q]
          if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),p.mean(0)-cc)<0: q=q[::-1]
          key=tuple(sorted(q))
          if key in fm: fm[key]['neighbour']=c
          else: fm[key]={'verts':q,'owner':c,'neighbour':None}
    lookup={tuple(sorted(q)):name for name,qs in pq.items() for q in qs}
    ib=sorted([f for f in fm.values() if f['neighbour'] is not None],key=lambda a:(a['owner'],a['neighbour']))
    bp={n:[] for n in pq}
    for key,f in fm.items():
      if f['neighbour'] is None: bp[lookup[key]].append(f)
    allf=ib+[f for n in bp for f in bp[n]]
    with open(poly/'points','w') as f:
      f.write(foam_header('vectorField','points')+f'{len(pts)}\n(\n'); [f.write(f'({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})\n') for p in pts]; f.write(')\n')
    with open(poly/'faces','w') as f:
      f.write(foam_header('faceList','faces')+f'{len(allf)}\n(\n'); [f.write(f"4({' '.join(map(str,a['verts']))})\n") for a in allf]; f.write(')\n')
    with open(poly/'owner','w') as f:
      f.write(foam_header('labelList','owner')+f'{len(allf)}\n(\n'); [f.write(f"{a['owner']}\n") for a in allf]; f.write(')\n')
    with open(poly/'neighbour','w') as f:
      f.write(foam_header('labelList','neighbour')+f'{len(ib)}\n(\n'); [f.write(f"{a['neighbour']}\n") for a in ib]; f.write(')\n')
    start=len(ib)
    with open(poly/'boundary','w') as f:
      f.write(foam_header('polyBoundaryMesh','boundary')+f'{len(bp)}\n(\n')
      for name,fs in bp.items(): f.write(f'{name}\n{{\n type patch;\n nFaces {len(fs)};\n startFace {start};\n}}\n'); start+=len(fs)
      f.write(')\n')
    return pts,pq,nu_f*ns_f*nt_f

final_points, final_patch_quads, final_cells = build_helical_mesh()
print(f'final mesh written: {final_cells} hexahedra, {len(final_points)} points')
for name,qs in final_patch_quads.items(): print(f'{name:12s}: {2*len(qs):5d} STL triangles')

# -- cell 11 ------------------------------------------------------------------------
# The refined mesh and matching STLs are written. I’ll run the binding check on the final mesh and rep
import subprocess
cm_final = subprocess.run(['checkMesh'], text=True, capture_output=True)
print(cm_final.stdout)
if 'Mesh OK.' not in cm_final.stdout: raise RuntimeError('final bare checkMesh did not report Mesh OK')
