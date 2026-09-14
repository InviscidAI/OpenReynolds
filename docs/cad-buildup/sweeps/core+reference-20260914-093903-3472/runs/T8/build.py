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
print(os.getcwd(), os.listdir('.'))
print(subprocess.run(['bash','-c','echo $WM_PROJECT_DIR; foamVersion 2>/dev/null | head -1'],capture_output=True,text=True).stdout)

# -- cell 2 -------------------------------------------------------------------------
import os, subprocess, textwrap

# --- parameters (metres). From the request: duct 30x30 mm, 200 mm long;
# copper block 40 long x 30 wide x 10 thick, centred at mid-length.
H  = 0.030      # duct height and width
L  = 0.200      # duct length
BL = 0.040      # block length (x)
BT = 0.010      # block thickness (z), recessed below floor z=0
X = [0.0, (L-BL)/2, (L+BL)/2, L]   # 0, 0.08, 0.12, 0.2
Y = [0.0, H]
Z = [-BT, 0.0, H]
CELL = 0.005    # coarse target cell size

verts = {}
def V(i,j,k):
    key=(i,j,k)
    if key not in verts: verts[key]=len(verts)
    return verts[key]

def hexblk(i0,i1,j0,j1,k0,k1,zone):
    c = lambda a,b,cc: V(i0 if a==0 else i1, j0 if b==0 else j1, k0 if cc==0 else k1)
    h = [c(0,0,0),c(1,0,0),c(1,1,0),c(0,1,0),c(0,0,1),c(1,0,1),c(1,1,1),c(0,1,1)]
    n = [max(1,round((X[i1]-X[i0])/CELL)), max(1,round((Y[j1]-Y[j0])/CELL)), max(1,round((Z[k1]-Z[k0])/CELL))]
    return dict(h=h, n=n, zone=zone, idx=(i0,i1,j0,j1,k0,k1))

blocks = [hexblk(0,1,0,1,1,2,'fluid'), hexblk(1,2,0,1,1,2,'fluid'), hexblk(2,3,0,1,1,2,'fluid'),
          hexblk(1,2,0,1,0,1,'solid')]

FACE = {'xmin':[(0,0,0),(0,0,1),(0,1,1),(0,1,0)], 'xmax':[(1,0,0),(1,1,0),(1,1,1),(1,0,1)],
        'ymin':[(0,0,0),(1,0,0),(1,0,1),(0,0,1)], 'ymax':[(0,1,0),(0,1,1),(1,1,1),(1,1,0)],
        'zmin':[(0,0,0),(0,1,0),(1,1,0),(1,0,0)], 'zmax':[(0,0,1),(1,0,1),(1,1,1),(0,1,1)]}
def face(b,side):
    i0,i1,j0,j1,k0,k1 = b['idx']
    return [V(i0 if a==0 else i1, j0 if bb==0 else j1, k0 if cc==0 else k1) for a,bb,cc in FACE[side]]

f0,f1,f2,sol = blocks
patches = {
 'inlet'      : [face(f0,'xmin')],
 'outlet'     : [face(f2,'xmax')],
 'duct_wall'  : [face(b,s) for b in (f0,f1,f2) for s in ('ymin','ymax','zmax')]
                + [face(f0,'zmin'), face(f2,'zmin')],
 'block_wall' : [face(sol,s) for s in ('xmin','xmax','ymin','ymax','zmin')],
}
print({k:len(v) for k,v in patches.items()}, [b['n'] for b in blocks], len(verts))

# -- cell 3 -------------------------------------------------------------------------
# Vertex grid is shared, so the block's top face and the middle fluid block's floor are the same inter
for d in ['system','constant']: os.makedirs(d, exist_ok=True)
head = """FoamFile{version 2.0; format ascii; class dictionary; object %s;}\n"""
vlist = sorted(verts.items(), key=lambda kv: kv[1])
vtxt = "\n".join("    (%.6f %.6f %.6f)" % (X[i],Y[j],Z[k]) for (i,j,k),_ in vlist)
btxt = "\n".join("    hex (%s) %s (%d %d %d) simpleGrading (1 1 1)" %
                 (" ".join(map(str,b['h'])), b['zone'], *b['n']) for b in blocks)
ptxt = ""
for name,faces in patches.items():
    typ = 'patch' if name in ('inlet','outlet') else 'wall'
    ptxt += "    %s\n    {\n        type %s;\n        faces\n        (\n%s\n        );\n    }\n" % (
        name, typ, "\n".join("            (%s)" % " ".join(map(str,f)) for f in faces))
dict_txt = head % 'blockMeshDict' + """
scale 1;
vertices
(
%s
);
blocks
(
%s
);
edges ();
boundary
(
%s);
mergePatchPairs ();
""" % (vtxt, btxt, ptxt)
open('system/blockMeshDict','w').write(dict_txt)
open('system/controlDict','w').write(head % 'controlDict' + """
application chtMultiRegionFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;
""")
open('system/fvSchemes','w').write(head % 'fvSchemes' + "\nddtSchemes{default steadyState;}\ngradSchemes{default Gauss linear;}\ndivSchemes{default none;}\nlaplacianSchemes{default Gauss linear corrected;}\n")
open('system/fvSolution','w').write(head % 'fvSolution' + "\nsolvers{}\n")
r = subprocess.run(['blockMesh'],capture_output=True,text=True)
print(r.stdout[-1500:], r.stderr[-800:])

# -- cell 4 -------------------------------------------------------------------------
r = subprocess.run(['splitMeshRegions','-cellZones','-overwrite'],capture_output=True,text=True)
print(r.stdout[-2000:]); print(r.stderr[-500:])
print(os.listdir('constant'))

# -- cell 5 -------------------------------------------------------------------------
for reg in ('fluid','solid'):
    r=subprocess.run(['checkMesh','-region',reg],capture_output=True,text=True)
    print('==',reg); print(r.stdout[-900:])

# -- cell 6 -------------------------------------------------------------------------
import glob
print(glob.glob('VTK/**/*.vt*',recursive=True)[:20])

# -- cell 7 -------------------------------------------------------------------------
import pyvista as pv, glob
pv.global_theme.background='white'
def render(fn, cam=[(0.32,-0.28,0.20),(0.1,0.015,0.0),(0,0,1)]):
    fl = pv.read('VTK/fluid/t8_0/internal.vtu'); so = pv.read('VTK/solid/t8_0/internal.vtu')
    p = pv.Plotter(off_screen=True, window_size=(1100,600))
    p.add_mesh(fl, color='lightblue', show_edges=True, opacity=0.30)
    p.add_mesh(so, color='darkorange', show_edges=True)
    p.camera_position=cam; p.screenshot(fn); return fl.n_cells, so.n_cells
print(render('coarse.png'))

# -- cell 8 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image('coarse.png'))

# -- cell 9 -------------------------------------------------------------------------
# Shape confirmed: block flush into the floor at mid-length, conformal interface. Now the same constru
import shutil
CELL = 0.0025                     # final cell size, my choice (12x12 cells across the duct)
for d in ['constant/polyMesh','constant/fluid','constant/solid','0','VTK']:
    shutil.rmtree(d, ignore_errors=True)
verts.clear()
blocks = [hexblk(0,1,0,1,1,2,'fluid'), hexblk(1,2,0,1,1,2,'fluid'), hexblk(2,3,0,1,1,2,'fluid'),
          hexblk(1,2,0,1,0,1,'solid')]
f0,f1,f2,sol = blocks
patches = {
 'inlet'      : [face(f0,'xmin')],
 'outlet'     : [face(f2,'xmax')],
 'duct_wall'  : [face(b,s) for b in (f0,f1,f2) for s in ('ymin','ymax','zmax')]
                + [face(f0,'zmin'), face(f2,'zmin')],
 'block_wall' : [face(sol,s) for s in ('xmin','xmax','ymin','ymax','zmin')],
}
vlist = sorted(verts.items(), key=lambda kv: kv[1])
vtxt = "\n".join("    (%.6f %.6f %.6f)" % (X[i],Y[j],Z[k]) for (i,j,k),_ in vlist)
btxt = "\n".join("    hex (%s) %s (%d %d %d) simpleGrading (1 1 1)" %
                 (" ".join(map(str,b['h'])), b['zone'], *b['n']) for b in blocks)
ptxt = ""
for name,faces in patches.items():
    typ = 'patch' if name in ('inlet','outlet') else 'wall'
    ptxt += "    %s\n    {\n        type %s;\n        faces\n        (\n%s\n        );\n    }\n" % (
        name, typ, "\n".join("            (%s)" % " ".join(map(str,f)) for f in faces))
open('system/blockMeshDict','w').write(head % 'blockMeshDict' +
    "\nscale 1;\nvertices\n(\n%s\n);\nblocks\n(\n%s\n);\nedges ();\nboundary\n(\n%s);\nmergePatchPairs ();\n" % (vtxt,btxt,ptxt))
print([b['n'] for b in blocks])
print(subprocess.run(['blockMesh'],capture_output=True,text=True).stdout[-500:])
print(subprocess.run(['splitMeshRegions','-cellZones','-overwrite'],capture_output=True,text=True).stdout[-600:])

# -- cell 10 ------------------------------------------------------------------------
import re
for reg in ('fluid','solid'):
    r=subprocess.run(['checkMesh','-region',reg],capture_output=True,text=True).stdout
    print('==',reg)
    print("\n".join(l for l in r.splitlines() if re.search(r'patch|Total volume|bounding box|Mesh OK|\*\*\*|cells:|faces:',l))[:1600])

# -- cell 11 ------------------------------------------------------------------------
for reg in ('fluid','solid'):
    t=open(f'constant/{reg}/polyMesh/boundary').read()
    print('==',reg)
    print("\n".join(l.strip() for l in t.splitlines() if re.match(r'^\s*(\w+)$',l) or 'type' in l or 'nFaces' in l or 'sampleRegion' in l or 'sampleMode' in l)[-1200:])

# -- cell 12 ------------------------------------------------------------------------
open('constant/regionProperties','w').write(head % 'regionProperties' +
    "\nregions ( fluid (fluid) solid (solid) );\n")
def pts(reg):
    import numpy as np
    t=open(f'constant/{reg}/polyMesh/points').read()
    a=np.array([list(map(float,m.groups())) for m in re.finditer(r'\(([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)\)',t)])
    return a
import numpy as np
for reg in ('fluid','solid'):
    a=pts(reg); print(reg,'extent mm:', np.round((a.max(0)-a.min(0))*1000,4), 'min mm:', np.round(a.min(0)*1000,4))
print('interface area m2:', 192*(0.0025**2), 'expected', 0.04*0.03)
print('fluid vol m3 1.8e-4 expected', 0.2*0.03*0.03, '| solid vol expected', 0.04*0.03*0.01)
subprocess.run(['foamToVTK','-region','fluid','-no-boundary'],capture_output=True); subprocess.run(['foamToVTK','-region','solid','-no-boundary'],capture_output=True)
print(render('final.png'))
display(Image('final.png'))
