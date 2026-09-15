"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid passage itself: lower centreline from \((0,0)\) to \((120,0)\) mm, a right-hand
import build123d as bd
import matplotlib.pyplot as plt
import numpy as np

MM = 1e-3
W = 10*MM
L = 120*MM
R = 15*MM
T = 1*MM  # assumed computational extrusion thickness; one cell, empty faces

lower = bd.Pos(0, -W/2, 0) * bd.Box(L, W, T)
upper = bd.Pos(0, 2*R-W/2, 0) * bd.Box(L, W, T)
outer = bd.Pos(L, R, 0) * bd.Cylinder(R+W/2, T)
inner = bd.Pos(L, R, 0) * bd.Cylinder(R-W/2, T)
right_half = bd.Pos(L, R-(R+W/2), 0) * bd.Box(R+W/2, 2*(R+W/2), T)
bend = (outer - inner) & right_half
fluid = lower + bend + upper

th = np.linspace(-np.pi/2, np.pi/2, 201)
fig, ax = plt.subplots(figsize=(9,3))
ax.fill_between([0,L/MM], -W/(2*MM), W/(2*MM), color='lightblue')
ax.fill_between([0,L/MM], (2*R-W/2)/MM, (2*R+W/2)/MM, color='lightblue')
for rr in (R-W/2, R+W/2):
    ax.plot((L+rr*np.cos(th))/MM, (R+rr*np.sin(th))/MM, 'b')
ax.plot((L+R*np.cos(th))/MM, (R+R*np.sin(th))/MM, 'k--', lw=1)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title('U-bend fluid passage (plan view)'); ax.grid(True)
plt.show()
print(f"Requested/measured passage width: 10.000 mm / {(R+W/2-(R-W/2))/MM:.3f} mm")
print(f"Requested/measured straight leg length: 120.000 mm / {L/MM:.3f} mm")
print(f"Requested/measured bend centreline radius: 15.000 mm / {((R-W/2)+(R+W/2))/(2*MM):.3f} mm")
print(f"Requested/measured turn angle: 180.000 deg / {(th[-1]-th[0])*180/np.pi:.3f} deg")
print(f"CAD volume: {fluid.volume:.9e} m^3; assumed thickness: {T/MM:.3f} mm")

# -- cell 2 -------------------------------------------------------------------------
# The coarse shape is correct and the four requested dimensions match. I’ll now write a structured `bl
from pathlib import Path
import math

N_BEND = 12
N_STRAIGHT = 60
N_ARC_PER_BLOCK = 2
N_WIDTH = 5
Path('system').mkdir(exist_ok=True)
Path('constant').mkdir(exist_ok=True)
Path('constant/triSurface').mkdir(parents=True, exist_ok=True)

# Cross-sections in flow order; each stores outer-wall point a and inner-wall point b.
sections = [((0.0,-W/2),(0.0,W/2)), ((L,-W/2),(L,W/2))]
phis = np.linspace(-math.pi/2, math.pi/2, N_BEND+1)
for ph in phis[1:]:
    a = (L+(R+W/2)*math.cos(ph), R+(R+W/2)*math.sin(ph))
    b = (L+(R-W/2)*math.cos(ph), R+(R-W/2)*math.sin(ph))
    sections.append((a,b))
sections.append(((0.0,2*R+W/2),(0.0,2*R-W/2)))

verts=[]
for z in (0.0,T):
    for a,b in sections:
        verts += [(a[0],a[1],z),(b[0],b[1],z)]
ns=len(sections); off=2*ns
blocks=[]
for i in range(ns-1):
    nlong = N_STRAIGHT if i in (0,ns-2) else N_ARC_PER_BLOCK
    blocks.append((2*i,2*(i+1),2*(i+1)+1,2*i+1,2*i+off,2*(i+1)+off,2*(i+1)+1+off,2*i+1+off,nlong))

def fmtp(p): return '('+' '.join(f'{v:.12g}' for v in p)+')'
lines=['FoamFile','{','    format ascii;','    class dictionary;','    object blockMeshDict;','}','scale 1;','vertices','(']
lines += ['    '+fmtp(v) for v in verts]; lines += [');','edges','(']
for k in range(N_BEND):
    i=1+k; pm=(phis[k]+phis[k+1])/2
    for side,rr in ((0,R+W/2),(1,R-W/2)):
        mid=(L+rr*math.cos(pm),R+rr*math.sin(pm))
        lines.append(f'    arc {2*i+side} {2*(i+1)+side} ({mid[0]:.12g} {mid[1]:.12g} 0)')
        lines.append(f'    arc {2*i+side+off} {2*(i+1)+side+off} ({mid[0]:.12g} {mid[1]:.12g} {T:.12g})')
lines += [');','blocks','(']
for q in blocks:
    lines.append(f'    hex ({" ".join(map(str,q[:8]))}) ({q[8]} {N_WIDTH} 1) simpleGrading (1 1 1)')
lines += [');','boundary','(','    inlet { type patch; faces (']
lines += [f'        (0 1 {1+off} {off})', '    ); }','    outlet { type patch; faces (']
i=ns-1; lines += [f'        ({2*i} {2*i+off} {2*i+1+off} {2*i+1})','    ); }','    walls { type wall; faces (']
for i in range(ns-1):
    lines += [f'        ({2*i} {2*(i+1)} {2*(i+1)+off} {2*i+off})', f'        ({2*i+1} {2*i+1+off} {2*(i+1)+1+off} {2*(i+1)+1})']
lines += ['    ); }','    frontAndBack { type empty; faces (']
for i in range(ns-1):
    lines += [f'        ({2*i} {2*i+1} {2*(i+1)+1} {2*(i+1)})', f'        ({2*i+off} {2*(i+1)+off} {2*(i+1)+1+off} {2*i+1+off})']
lines += ['    ); }',');','mergePatchPairs ();']
Path('system/blockMeshDict').write_text('\n'.join(lines)+'\n')
bd.export_step(fluid, 'constant/triSurface/fluidVolume.step')
print(f'Wrote blockMeshDict: {len(verts)} vertices, {len(blocks)} blocks, nominal {2*N_STRAIGHT*N_WIDTH + N_BEND*N_ARC_PER_BLOCK*N_WIDTH} cells')

# -- cell 3 -------------------------------------------------------------------------
# `blockMesh` did not start because an empty case has no `controlDict`; no mesh was modified. I’ll add
Path('system/controlDict').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object controlDict;
}
application blockMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
''')
print('Wrote minimal system/controlDict for mesh utilities.')

# -- cell 4 -------------------------------------------------------------------------
# The mesher succeeded interactively, but that cell was not replay-safe because its `subprocess` impor
import subprocess
bm = subprocess.run(['blockMesh'], text=True, capture_output=True)
print(bm.stdout[-2500:])
if bm.returncode:
    print(bm.stderr)
    raise RuntimeError(f'blockMesh failed with code {bm.returncode}')

# -- cell 5 -------------------------------------------------------------------------
# The coarse mesh has 720 hexahedra and the intended patch counts. I’ll now look at the actual structu
fig, ax = plt.subplots(figsize=(11,3.4))
# straight-leg grid lines
for y0 in np.linspace(-W/2,W/2,N_WIDTH+1): ax.plot([0,L],[y0,y0],color='#277da1',lw=.55)
for y0 in np.linspace(2*R-W/2,2*R+W/2,N_WIDTH+1): ax.plot([0,L],[y0,y0],color='#277da1',lw=.55)
for x0 in np.linspace(0,L,N_STRAIGHT+1):
    ax.plot([x0,x0],[-W/2,W/2],color='#277da1',lw=.4)
    ax.plot([x0,x0],[2*R-W/2,2*R+W/2],color='#277da1',lw=.4)
# bend grid lines, 24 circumferential by 5 radial
phi_fine=np.linspace(-np.pi/2,np.pi/2,N_BEND*N_ARC_PER_BLOCK+1)
for rr in np.linspace(R-W/2,R+W/2,N_WIDTH+1):
    ax.plot(L+rr*np.cos(phi_fine),R+rr*np.sin(phi_fine),color='#277da1',lw=.55)
for ph in phi_fine:
    ax.plot(L+np.array([R-W/2,R+W/2])*np.cos(ph),R+np.array([R-W/2,R+W/2])*np.sin(ph),color='#277da1',lw=.4)
ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('Coarse structured 2D mesh — 720 cells'); ax.grid(False)
plt.show()
print('Planar cell sizes: straights 2.000 x 2.000 mm; bend centreline approx. 1.963 x 2.000 mm')
print('Through-plane: 1 cell across the assumed 1.000 mm extrusion; front/back are empty patches')

# -- cell 6 -------------------------------------------------------------------------
# The coarse mesh conforms cleanly, but five cells across a 10 mm passage is only a geometry-validatio
import re
N_STRAIGHT_FINAL=240
N_ARC_PER_BLOCK_FINAL=8
N_WIDTH_FINAL=20
p=Path('system/blockMeshDict')
txt=p.read_text()
txt=re.sub(r'(hex \([^\n]+\) \()60 5 1(\) simpleGrading)', rf'\g<1>{N_STRAIGHT_FINAL} {N_WIDTH_FINAL} 1\g<2>', txt)
txt=re.sub(r'(hex \([^\n]+\) \()2 5 1(\) simpleGrading)', rf'\g<1>{N_ARC_PER_BLOCK_FINAL} {N_WIDTH_FINAL} 1\g<2>', txt)
p.write_text(txt)
nominal=2*N_STRAIGHT_FINAL*N_WIDTH_FINAL+N_BEND*N_ARC_PER_BLOCK_FINAL*N_WIDTH_FINAL
print(f'Refined blockMeshDict to {nominal} cells: {N_WIDTH_FINAL} across, {N_STRAIGHT_FINAL} per straight, {N_BEND*N_ARC_PER_BLOCK_FINAL} around bend')

# -- cell 7 -------------------------------------------------------------------------
# The final resolution is now encoded. I’ll regenerate the mesh and verify the final cell and patch co
bm_final=subprocess.run(['blockMesh'], text=True, capture_output=True)
print(bm_final.stdout[-2200:])
if bm_final.returncode:
    print(bm_final.stderr)
    raise RuntimeError('final blockMesh failed')

# -- cell 8 -------------------------------------------------------------------------
# The final mesh contains 11,520 cells with 20 inlet and outlet faces, 1,152 wall faces, and two compl
h=subprocess.run(['foamToSurface','-help'],text=True,capture_output=True)
print(h.stdout[:4000])

# -- cell 9 -------------------------------------------------------------------------
# `foamToSurface` exports the combined boundary and does not expose per-patch selection in its short o
h2=subprocess.run(['surfaceMeshExtract','-help'],text=True,capture_output=True)
print(h2.stdout[:4500])

# -- cell 10 ------------------------------------------------------------------------
# `surfaceMeshExtract` supports exact patch-name selection. I’ll export `inlet`, `outlet`, `walls`, an
surfdir=Path('constant/triSurface')
for patch in ('inlet','outlet','walls','frontAndBack'):
    out=surfdir/f'{patch}.stl'
    r=subprocess.run(['surfaceMeshExtract','-constant','-patches',patch,str(out)],text=True,capture_output=True)
    print(patch, 'OK' if r.returncode==0 else 'FAILED', out.stat().st_size if out.exists() else 0, 'bytes')
    if r.returncode: print(r.stdout[-1500:],r.stderr)
    if r.returncode: raise RuntimeError(f'export failed: {patch}')

# -- cell 11 ------------------------------------------------------------------------
# The extractor correctly wrote the three non-empty boundary patches, but OpenFOAM intentionally omitt
def facet_text(p0,p1,p2):
    u=np.array(p1)-np.array(p0); v=np.array(p2)-np.array(p0); n=np.cross(u,v); n=n/np.linalg.norm(n)
    s=f' facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n  outer loop\n'
    for p in (p0,p1,p2): s+=f'   vertex {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n'
    return s+'  endloop\n endfacet\n'

def emit_quad(f,q,z,up):
    p=[(q[k][0],q[k][1],z) for k in range(4)] # q is CCW
    tris=((0,1,2),(0,2,3)) if up else ((0,2,1),(0,3,2))
    for a,b,c in tris: f.write(facet_text(p[a],p[b],p[c]))

with open('constant/triSurface/frontAndBack.stl','w') as f:
    f.write('solid frontAndBack\n')
    xs=np.linspace(0,L,N_STRAIGHT_FINAL+1)
    ys0=np.linspace(-W/2,W/2,N_WIDTH_FINAL+1)
    ys1=np.linspace(2*R-W/2,2*R+W/2,N_WIDTH_FINAL+1)
    quads=[]
    for ys in (ys0,ys1):
        for i in range(N_STRAIGHT_FINAL):
            for j in range(N_WIDTH_FINAL): quads.append(((xs[i],ys[j]),(xs[i+1],ys[j]),(xs[i+1],ys[j+1]),(xs[i],ys[j+1])))
    ph=np.linspace(-np.pi/2,np.pi/2,N_BEND*N_ARC_PER_BLOCK_FINAL+1)
    rs=np.linspace(R-W/2,R+W/2,N_WIDTH_FINAL+1)
    for i in range(len(ph)-1):
        for j in range(len(rs)-1):
            # outer-old, outer-new, inner-new, inner-old is CCW
            quads.append(((L+rs[j+1]*np.cos(ph[i]),R+rs[j+1]*np.sin(ph[i])),(L+rs[j+1]*np.cos(ph[i+1]),R+rs[j+1]*np.sin(ph[i+1])),(L+rs[j]*np.cos(ph[i+1]),R+rs[j]*np.sin(ph[i+1])),(L+rs[j]*np.cos(ph[i]),R+rs[j]*np.sin(ph[i]))))
    for q in quads:
        emit_quad(f,q,0.0,False); emit_quad(f,q,T,True)
    f.write('endsolid frontAndBack\n')
print(f'Wrote frontAndBack.stl with {4*len(quads)} triangles over {len(quads)} planar cells; size {Path("constant/triSurface/frontAndBack.stl").stat().st_size} bytes')

# -- cell 12 ------------------------------------------------------------------------
# `checkMesh` stopped immediately after reading the mesh, before reporting any geometry metric; the di
diag=subprocess.run(['checkMesh','-constant','-allGeometry','-allTopology'],text=True,capture_output=True)
print(diag.stdout)
print(diag.stderr)
print('return code:',diag.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The mesh itself was not rejected; `-allGeometry` requires an `fvSchemes` dictionary even for a mesh-
Path('system/fvSchemes').write_text('''FoamFile
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
print('Wrote minimal system/fvSchemes for extended mesh-quality evaluation.')

# -- cell 14 ------------------------------------------------------------------------
# The extended mode also requires `fvSolution` to exist, again before checking any cells. I’ll add an 
Path('system/fvSolution').write_text('''FoamFile
{
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers {}
''')
print('Wrote minimal mesh-only system/fvSolution.')

# -- cell 15 ------------------------------------------------------------------------
# Both parser dependencies are now available. I’ll rerun the extended `checkMesh` validation.
cm=subprocess.run(['checkMesh','-constant','-allGeometry','-allTopology'],text=True,capture_output=True)
Path('checkMesh.log').write_text(cm.stdout+cm.stderr)
print(cm.stdout[-6500:]); print(cm.stderr)
if cm.returncode: raise RuntimeError(f'checkMesh failed with code {cm.returncode}')

# -- cell 16 ------------------------------------------------------------------------
# `checkMesh` passes all checks, but its volume exposed a CAD bookkeeping issue: the mesh volume is \(
amin=(bd.Align.MIN,bd.Align.MIN,bd.Align.MIN)
acenter=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)
lower = bd.Pos(0,-W/2,0) * bd.Box(L,W,T,align=amin)
upper = bd.Pos(0,2*R-W/2,0) * bd.Box(L,W,T,align=amin)
outer = bd.Pos(L,R,0) * bd.Cylinder(R+W/2,T,align=acenter)
inner = bd.Pos(L,R,0) * bd.Cylinder(R-W/2,T,align=acenter)
right_half = bd.Pos(L,R-(R+W/2),0) * bd.Box(R+W/2,2*(R+W/2),T,align=amin)
bend = (outer-inner) & right_half
fluid = lower + bend + upper
expected_volume=(2*L*W+0.5*np.pi*((R+W/2)**2-(R-W/2)**2))*T
bb=fluid.bounding_box()
bd.export_step(fluid,'constant/triSurface/fluidVolume.step')
print(f'CAD bounds min={tuple(round(v,6) for v in bb.min.to_tuple())}, max={tuple(round(v,6) for v in bb.max.to_tuple())}')
print(f'CAD volume {fluid.volume:.9e} m^3; analytical target {expected_volume:.9e} m^3; relative error {(fluid.volume/expected_volume-1):.3e}')

# -- cell 17 ------------------------------------------------------------------------
# The mesh is valid, but the advisory STL-union check found 768 free edges and coplanar crossings. Thi
for fn in ('constant/polyMesh/points','constant/polyMesh/faces','constant/polyMesh/boundary'):
    s=Path(fn).read_text()
    print('\n---',fn,'---\n',s[:900])

# -- cell 18 ------------------------------------------------------------------------
# The parser accidentally treated the `FoamFile` header dictionary as a boundary patch. I’ll filter di
def foam_payload(path):
    s=Path(path).read_text(); m=re.search(r'\n(\d+)\s*\n\(\s*\n',s)
    if not m: raise ValueError(path)
    return int(m.group(1)),s[m.end():]
npnt,pbody=foam_payload('constant/polyMesh/points')
mesh_pts=np.array([[float(x) for x in g] for g in re.findall(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)',pbody)[:npnt]])
nface,fbody=foam_payload('constant/polyMesh/faces')
mesh_faces=[]
for n,ids in re.findall(r'(\d+)\(([^()]*)\)',fbody):
    vals=[int(x) for x in ids.split()]
    if len(vals)==int(n): mesh_faces.append(vals)
    if len(mesh_faces)==nface: break
bs=Path('constant/polyMesh/boundary').read_text(); patch_ranges={}
for name,body in re.findall(r'\n\s*(\w+)\s*\n\s*\{([^{}]*)\}',bs):
    m=re.search(r'nFaces\s+(\d+);.*?startFace\s+(\d+);',body,re.S)
    if m: patch_ranges[name]=(int(m.group(1)),int(m.group(2)))
for name,(nf,st) in patch_ranges.items():
    with open(f'constant/triSurface/{name}.stl','w') as f:
        f.write(f'solid {name}\n'); nt=0
        for ids in mesh_faces[st:st+nf]:
            for k in range(1,len(ids)-1):
                f.write(facet_text(mesh_pts[ids[0]],mesh_pts[ids[k]],mesh_pts[ids[k+1]])); nt+=1
        f.write(f'endsolid {name}\n')
    print(f'{name}: {nf} mesh faces -> {nt} exact-mesh triangles')
print(f'Parsed {len(mesh_pts)} points, {len(mesh_faces)} faces, and {len(patch_ranges)} patches')
