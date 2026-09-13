"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll send that as an actual cell.
import os, subprocess
for d in ["system","constant/triSurface","0"]:
    os.makedirs(d, exist_ok=True)

# --- parameters (metres) ---
LX, LY, LZ = 0.100, 0.060, 0.040        # cavity 100 x 60 x 40 mm
CUBE = 0.020                            # 20 mm cube zones
GAP  = 0.004                            # standoff from floor / ceiling
HEAT_ORIGIN = (0.010, (LY-CUBE)/2, GAP)                   # near floor, x-low end
COOL_ORIGIN = (LX-0.010-CUBE, (LY-CUBE)/2, LZ-GAP-CUBE)   # near ceiling, x-high end
CELL = 0.004                            # coarse: 4 mm cells
nx, ny, nz = int(round(LX/CELL)), int(round(LY/CELL)), int(round(LZ/CELL))
print(nx, ny, nz, nx*ny*nz, HEAT_ORIGIN, COOL_ORIGIN)

head = "FoamFile\n{ version 2.0; format ascii; class dictionary; object %s; }\n"
bm = head % "blockMeshDict" + f"""
scale 1;
vertices
(
 (0 0 0) ({LX} 0 0) ({LX} {LY} 0) (0 {LY} 0)
 (0 0 {LZ}) ({LX} 0 {LZ}) ({LX} {LY} {LZ}) (0 {LY} {LZ})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)
        );
    }}
);
mergePatchPairs ();
"""
open("system/blockMeshDict","w").write(bm)

ctrl = head % "controlDict" + """
application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
"""
open("system/controlDict","w").write(ctrl)
open("system/fvSchemes","w").write(head % "fvSchemes" + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(head % "fvSolution" + "solvers{}\n")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-500:])

# -- cell 2 -------------------------------------------------------------------------
# Cell zones via `topoSet`. I shift the end standoff to 12 mm so both cubes land exactly on cell bound
END_GAP = 0.012   # standoff from the end wall, multiple of the cell size
HEAT_ORIGIN = (END_GAP, (LY-CUBE)/2, GAP)                  # near floor, x-low end
COOL_ORIGIN = (LX-END_GAP-CUBE, (LY-CUBE)/2, LZ-GAP-CUBE)  # near ceiling, x-high end

def zone(name, o):
    mx, my, mz = o[0]+CUBE, o[1]+CUBE, o[2]+CUBE
    return f"""
    {{ name {name}Cells; type cellSet; action new; source boxToCell;
       box ({o[0]} {o[1]} {o[2]}) ({mx} {my} {mz}); }}
    {{ name {name};      type cellZoneSet; action new; source setToCellZone; set {name}Cells; }}
"""
ts = "FoamFile\n{ version 2.0; format ascii; class dictionary; object topoSetDict; }\nactions (" \
     + zone("heater", HEAT_ORIGIN) + zone("cooler", COOL_ORIGIN) + ");\n"
open("system/topoSetDict","w").write(ts)
print(HEAT_ORIGIN, COOL_ORIGIN)
r = subprocess.run(["topoSet"], capture_output=True, text=True)
print(r.stdout[-900:], r.stderr[-300:])

# -- cell 3 -------------------------------------------------------------------------
# 125 = 5³ cells each, exactly the 20 mm cube. Let me look at the zones in place.
import pyvista as pv, numpy as np
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
open("t7.foam","w").write("")
m = pv.OpenFOAMReader("t7.foam"); m.set_active_time_value(0)
mesh = m.read()["internalMesh"]
cz = {}
for nm, o in [("heater",HEAT_ORIGIN),("cooler",COOL_ORIGIN)]:
    c = mesh.cell_centers().points
    sel = np.all((c>=np.array(o)-1e-9)&(c<=np.array(o)+CUBE+1e-9),axis=1)
    cz[nm] = mesh.extract_cells(sel)
    b = cz[nm].bounds
    print(nm, "ncells", cz[nm].n_cells, "bounds mm", [round(v*1000,3) for v in b])
p = pv.Plotter(off_screen=True)
p.add_mesh(mesh, style="wireframe", color="lightgray", opacity=0.3)
p.add_mesh(cz["heater"], color="red"); p.add_mesh(cz["cooler"], color="blue")
p.show_bounds(); p.camera_position="iso"; p.screenshot("zones.png")

# -- cell 4 -------------------------------------------------------------------------
# Headless GL failed; I'll draw the geometry with matplotlib from the actual mesh data instead.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(11,4))
def box(ax,o,s,**kw):
    import itertools
    o=np.array(o); s=np.array(s)
    for i in range(3):
        for a,b in itertools.product([0,1],[0,1]):
            p=[o.copy(),o.copy()]; p[1][i]+=s[i]
            j,k=[d for d in range(3) if d!=i]
            for q in p: q[j]+=a*s[j]; q[k]+=b*s[k]
            ax.plot(*zip(*[q*1000 for q in p]),**kw)
ax=fig.add_subplot(121,projection="3d")
box(ax,(0,0,0),(LX,LY,LZ),color="k",lw=0.6)
box(ax,HEAT_ORIGIN,(CUBE,)*3,color="r",lw=1.5); box(ax,COOL_ORIGIN,(CUBE,)*3,color="b",lw=1.5)
ax.set_box_aspect((LX,LY,LZ)); ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("z mm")
ax2=fig.add_subplot(122)
c=mesh.cell_centers().points*1000
ax2.plot([0,100,100,0,0],[0,0,40,40,0],"k")
for nm,col in [("heater","r"),("cooler","b")]:
    pts=cz[nm].cell_centers().points*1000; ax2.scatter(pts[:,0],pts[:,2],c=col,s=6)
ax2.set_aspect("equal"); ax2.set_xlabel("x mm"); ax2.set_ylabel("z mm"); ax2.set_title("side view: heater(red)/cooler(blue)")
plt.tight_layout(); plt.show()

# -- cell 5 -------------------------------------------------------------------------
from IPython.display import display
display(fig)

# -- cell 6 -------------------------------------------------------------------------
# Shape is right. Refine to 2 mm cells (50×30×20 = 30 000) and rebuild zones, then checkMesh.
CELL_FINE = 0.002
nx, ny, nz = int(round(LX/CELL_FINE)), int(round(LY/CELL_FINE)), int(round(LZ/CELL_FINE))
bm_f = bm.replace(f"({25} {15} {10})", f"({nx} {ny} {nz})")
assert f"({nx} {ny} {nz})" in bm_f
open("system/blockMeshDict","w").write(bm_f)
for cmd in (["blockMesh"],["topoSet"],["checkMesh"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("====",cmd[0], r.returncode); print(r.stdout[-1200:])
