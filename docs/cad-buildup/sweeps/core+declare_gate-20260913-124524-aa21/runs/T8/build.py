"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll set up the parameters and confirm the block dimensions.
import os, subprocess, textwrap
os.makedirs("system", exist_ok=True); os.makedirs("constant", exist_ok=True)

# --- geometry parameters (metres) ---
H  = 0.030   # duct height  (30 mm square)
W  = 0.030   # duct width
L  = 0.200   # duct length
BL = 0.040   # copper block length
BT = 0.010   # copper block thickness (recessed into the floor, z<0)
x0, x3 = 0.0, L
x1 = 0.5*(L - BL)        # 0.080
x2 = 0.5*(L + BL)        # 0.120
print(x1, x2, "block length =", x2-x1, "width =", W, "thickness =", BT)

# -- cell 2 -------------------------------------------------------------------------
# Now the blockMeshDict: three fluid blocks in x (so the middle one's bottom face matches the solid to
xs=[x0,x1,x2,x3]; ys=[0.0,W]; zs=[-BT,0.0,H]
vid={}; verts=[]
for k,z in enumerate(zs):
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            vid[(i,j,k)]=len(verts); verts.append((x,y,z))
def hexv(i,k):  # block spanning xs[i]..xs[i+1], full y, zs[k]..zs[k+1]
    return [vid[(i,0,k)],vid[(i+1,0,k)],vid[(i+1,1,k)],vid[(i,1,k)],
            vid[(i,0,k+1)],vid[(i+1,0,k+1)],vid[(i+1,1,k+1)],vid[(i,1,k+1)]]
DX=0.005
nx=[int(round((xs[i+1]-xs[i])/DX)) for i in range(3)]; ny=int(round(W/DX))
nzF=int(round(H/DX)); nzS=int(round(BT/DX))
blocks=[]
for i in range(3):
    blocks.append(f"    hex ({' '.join(map(str,hexv(i,1)))}) fluid ({nx[i]} {ny} {nzF}) simpleGrading (1 1 1)")
blocks.append(f"    hex ({' '.join(map(str,hexv(1,0)))}) solid ({nx[1]} {ny} {nzS}) simpleGrading (1 1 1)")
print(nx,ny,nzF,nzS); print("\n".join(blocks))

# -- cell 3 -------------------------------------------------------------------------
# Boundary faces, with orderings chosen for outward normals. Fluid bottom is wall only outside the blo
def fx(i,k,side):  # constant-x face of block (i,k); side 'min'->-x, 'max'->+x
    ii = i if side=='min' else i+1
    a=[(ii,0,k),(ii,0,k+1),(ii,1,k+1),(ii,1,k)]
    return a if side=='min' else [(ii,0,k),(ii,1,k),(ii,1,k+1),(ii,0,k+1)]
def fy(i,k,j):     # constant-y face, j=0 -> -y, j=1 -> +y
    a=[(i,j,k),(i+1,j,k),(i+1,j,k+1),(i,j,k+1)]
    return a if j==0 else [(i,j,k),(i,j,k+1),(i+1,j,k+1),(i+1,j,k)]
def fz(i,k,side):  # constant-z face
    kk = k if side=='min' else k+1
    a=[(i,0,kk),(i+1,0,kk),(i+1,1,kk),(i,1,kk)]
    return a if side=='max' else [(i,0,kk),(i,1,kk),(i+1,1,kk),(i+1,0,kk)]
def s(f): return "        ("+" ".join(str(vid[v]) for v in f)+")"

inlet  = [fx(0,1,'min')]
outlet = [fx(2,1,'max')]
fluidWall = [fz(i,1,'max') for i in range(3)] + [fy(i,1,j) for i in range(3) for j in (0,1)] \
          + [fz(0,1,'min'), fz(2,1,'min')]          # floor either side of the block
solidWall = [fz(1,0,'min')] + [fy(1,0,j) for j in (0,1)] + [fx(1,0,'min'), fx(1,0,'max')]
print(len(inlet),len(outlet),len(fluidWall),len(solidWall))

# -- cell 4 -------------------------------------------------------------------------
def patch(name, typ, faces):
    return f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n" + "\n".join(s(f) for f in faces) + "\n        );\n    }}\n"
dic = """FoamFile{ version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;

vertices
(
""" + "\n".join(f"    ({x} {y} {z})" for x,y,z in verts) + """
);

blocks
(
""" + "\n".join(blocks) + """
);

edges ();

boundary
(
""" + patch("inlet","patch",inlet) + patch("outlet","patch",outlet) \
    + patch("fluidWall","wall",fluidWall) + patch("solidWall","wall",solidWall) + """);

mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(dic)
open("system/controlDict","w").write("""FoamFile{ version 2.0; format ascii; class dictionary; object controlDict; }
application chtMultiRegionFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
""")
for f,obj in [("fvSchemes","fvSchemes"),("fvSolution","fvSolution")]:
    open(f"system/{f}","w").write(f"FoamFile{{ version 2.0; format ascii; class dictionary; object {obj}; }}\n")
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-1500:], r.stderr[-800:])

# -- cell 5 -------------------------------------------------------------------------
print("\n".join(f"{i+1}: {l}" for i,l in enumerate(dic.split("\n")[40:62], start=40)))

# -- cell 6 -------------------------------------------------------------------------
# The trailing `}}` was in a plain string, not the f-string. Fix and rerun.
def patch(name, typ, faces):
    return (f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n"
            + "\n".join(s(f) for f in faces) + "\n        );\n    }\n")
dic = """FoamFile{ version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;

vertices
(
""" + "\n".join(f"    ({x} {y} {z})" for x,y,z in verts) + """
);

blocks
(
""" + "\n".join(blocks) + """
);

edges ();

boundary
(
""" + patch("inlet","patch",inlet) + patch("outlet","patch",outlet) \
    + patch("fluidWall","wall",fluidWall) + patch("solidWall","wall",solidWall) + """);

mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(dic)
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-1200:], r.stderr[-500:])

# -- cell 7 -------------------------------------------------------------------------
# Mesh is right: bbox (0 0 −0.01)→(0.2 0.03 0.03), cell zones present. Now split into regions with a c
r=subprocess.run(["splitMeshRegions","-cellZones","-overwrite"],capture_output=True,text=True)
print(r.stdout[-2000:], r.stderr[-500:])

# -- cell 8 -------------------------------------------------------------------------
open("system/fvSchemes","w").write("""FoamFile{ version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
open("system/fvSolution","w").write("""FoamFile{ version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {}
""")
r=subprocess.run(["splitMeshRegions","-cellZones","-overwrite"],capture_output=True,text=True)
print(r.stdout[-2500:], r.stderr[-500:])

# -- cell 9 -------------------------------------------------------------------------
# Interface has 48 faces (8×6 cells = 40 mm × 30 mm). Now checkMesh on both regions.
for reg in ("fluid","solid"):
    r=subprocess.run(["checkMesh","-region",reg],capture_output=True,text=True)
    tail=[l for l in r.stdout.split("\n") if any(k in l for k in ("Mesh OK","***","<<","cells:","faces:","boundingBox","Max aspect","Max skewness","non-orthogonality"))]
    print(reg, "\n ", "\n  ".join(tail), "\n")

# -- cell 10 ------------------------------------------------------------------------
# Both regions clean. Now measure the actual patch areas and extents from the meshes themselves, and d
import pyvista as pv, numpy as np
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
open("case.foam","w").write("")
rd=pv.OpenFOAMReader("case.foam")
print(rd.patch_array_names[:20])

# -- cell 11 ------------------------------------------------------------------------
rd.set_active_time_value(0.0)
rd.enable_all_patch_arrays(); rd.disable_patch_array('internalMesh')
m=rd.read()
def blk(path):
    o=m
    for p in path: o=o[p]
    return o
def rep(name, mesh):
    a=mesh.extract_surface().compute_cell_sizes(length=False, volume=False)["Area"].sum()
    b=np.array(mesh.bounds).reshape(3,2)
    print(f"{name:16s} area={a*1e6:9.1f} mm^2  x[{b[0,0]*1e3:6.1f},{b[0,1]*1e3:6.1f}] y[{b[1,0]*1e3:5.1f},{b[1,1]*1e3:5.1f}] z[{b[2,0]*1e3:6.1f},{b[2,1]*1e3:6.1f}] mm")
for nm in ['inlet','outlet','fluid_to_solid']: rep(nm, blk(['fluid','boundary',nm]))
for nm in ['solid_to_fluid','solidWall']: rep(nm, blk(['solid','boundary',nm]))
fv=blk(['fluid','internalMesh']).compute_cell_sizes(); sv=blk(['solid','internalMesh']).compute_cell_sizes()
print("fluid vol mm^3",fv['Volume'].sum()*1e9, " expected", (0.03*0.03*0.2)*1e9)
print("solid vol mm^3",sv['Volume'].sum()*1e9, " expected", (0.04*0.03*0.01)*1e9)

# -- cell 12 ------------------------------------------------------------------------
# Every number matches: inlet/outlet 30×30 mm, duct 200 mm, block 40×30×10 mm spanning x = 80–120 mm (
p=pv.Plotter(off_screen=True,window_size=(1100,600))
p.add_mesh(blk(['fluid','internalMesh']),color='lightblue',opacity=0.35,show_edges=True,line_width=1)
p.add_mesh(blk(['solid','internalMesh']),color='darkorange',show_edges=True)
p.add_mesh(blk(['fluid','boundary','inlet']),color='green')
p.add_mesh(blk(['fluid','boundary','outlet']),color='red')
p.camera_position=[(0.30,-0.30,0.22),(0.10,0.015,0.005),(0,0,1)]
p.add_axes(); p.screenshot("view.png"); print("ok")

# -- cell 13 ------------------------------------------------------------------------
import matplotlib.pyplot as plt, matplotlib.image as mpimg
img=mpimg.imread("view.png"); fig,ax=plt.subplots(figsize=(11,6)); ax.imshow(img); ax.axis('off'); plt.show()

# -- cell 14 ------------------------------------------------------------------------
# Shape is right: block flush with the floor, centred, conformal interface. Now refine to 2 mm cells b
DX=0.002
nx=[int(round((xs[i+1]-xs[i])/DX)) for i in range(3)]; ny=int(round(W/DX))
nzF=int(round(H/DX)); nzS=int(round(BT/DX))
assert all(abs((xs[i+1]-xs[i])/DX-nx[i])<1e-9 for i in range(3))
blocks=[f"    hex ({' '.join(map(str,hexv(i,1)))}) fluid ({nx[i]} {ny} {nzF}) simpleGrading (1 1 1)" for i in range(3)]
blocks.append(f"    hex ({' '.join(map(str,hexv(1,0)))}) solid ({nx[1]} {ny} {nzS}) simpleGrading (1 1 1)")
dic = dic.split("blocks\n(\n")[0] + "blocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges ();\n\nboundary\n(\n" \
    + patch("inlet","patch",inlet) + patch("outlet","patch",outlet) \
    + patch("fluidWall","wall",fluidWall) + patch("solidWall","wall",solidWall) + ");\n\nmergePatchPairs ();\n"
open("system/blockMeshDict","w").write(dic)
print(nx,ny,nzF,nzS)
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.returncode, r.stdout[-350:])
r=subprocess.run(["splitMeshRegions","-cellZones","-overwrite"],capture_output=True,text=True); print(r.returncode, r.stdout[-300:])

# -- cell 15 ------------------------------------------------------------------------
# Final verification on the refined mesh: checkMesh per region, plus re-measuring the patch areas and 
for reg in ("fluid","solid"):
    r=subprocess.run(["checkMesh","-region",reg],capture_output=True,text=True)
    print(reg, [l.strip() for l in r.stdout.split("\n") if any(k in l for k in ("Mesh OK","***","cells:","Max skewness","non-orthogonality Max"))])
rd2=pv.OpenFOAMReader("case.foam"); rd2.set_active_time_value(0.0)
rd2.enable_all_patch_arrays(); rd2.disable_patch_array('internalMesh'); m=rd2.read()
for reg,nm in [('fluid','fluid_to_solid'),('solid','solid_to_fluid')]: rep(nm, m[reg]['boundary'][nm])
print("fluid vol mm^3", m['fluid']['internalMesh'].compute_cell_sizes()['Volume'].sum()*1e9)
print("solid vol mm^3", m['solid']['internalMesh'].compute_cell_sizes()['Volume'].sum()*1e9)
