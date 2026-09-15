"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# A closed rectangular box — ideal for `blockMesh` with a single block, then two `topoSet` box selecti
import subprocess, os
os.chdir("/home/qiuzi/.openreynolds-buildup/work/T7-20260914-165512-f021/t7")
dict_txt = r"""
vertices
(
    (0    0    0)
    (0.1  0    0)
    (0.1  0.06 0)
    (0    0.06 0)
    (0    0    0.04)
    (0.1  0    0.04)
    (0.1  0.06 0.04)
    (0    0.06 0.04)
);
blocks ( hex (0 1 2 3 4 5 6 7) (25 15 10) simpleGrading (1 1 1) );
boundary ( allWalls { type wall; faces ((0 3 7 4)); } );
""".replace("allWalls { type wall; faces ((0 3 7 4)); }",
            "allWalls { type wall; faces ((0 3 7 4) (1 2 6 5) (0 1 5 4) (3 2 6 7) (4 5 6 7) (0 1 2 3)); }")
import pathlib; pathlib.Path("system").mkdir(exist_ok=True)
pathlib.Path("system/blockMeshDict").write_text("convertToMeters 1;\n" + dict_txt)
pathlib.Path("system/controlDict").write_text("// minimal")
print(subprocess.run(["blockMesh"], capture_output=True, text=True).returncode)

# -- cell 2 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-500:])

# -- cell 3 -------------------------------------------------------------------------
hdr = """FoamFile
{
    version 2.0; format ascii; class dictionary;
}
"""
ctrl = hdr + """application blockMesh; startTime 0; verbose 2;
"""
import pathlib
pathlib.Path("system/controlDict").write_text(ctrl)
bd = hdr + """
vertices
(
    (0 0 0) (0.1 0 0) (0.1 0.06 0) (0 0.06 0)
    (0 0 0.04) (0.1 0 0.04) (0.1 0.06 0.04) (0 0.06 0.04)
);
blocks ( hex (0 1 2 3 4 5 6 7) (25 15 10) simpleGrading (1 1 1) );
boundary ( allWalls { type wall; faces ( (0 3 7 4) (1 2 6 5) (0 1 5 4) (3 2 6 7) (4 5 6 7) (0 1 2 3) ); } );
"""
pathlib.Path("system/blockMeshDict").write_text(bd)
topo = hdr + """
actions
(
  { name heater; type cellSet; action new; source boxToCell;
    sourceInfo { box (0.01 0.02 0) (0.03 0.04 0.02); } }
  { name heater; type cellZoneSet; action new; source setToCellZone;
    sourceInfo { set heater; } }
  { name cooler; type cellSet; action new; source boxToCell;
    sourceInfo { box (0.07 0.02 0.02) (0.09 0.04 0.04); } }
  { name cooler; type cellZoneSet; action new; source setToCellZone;
    sourceInfo { set cooler; } }
);
"""
pathlib.Path("system/topoSetDict").write_text(topo)
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True); print("blockMesh", r1.returncode, r1.stdout[-300:])
r2 = subprocess.run(["topoSet"], capture_output=True, text=True); print("topoSet", r2.returncode, r2.stdout[-500:])

# -- cell 4 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-2500:])

# -- cell 5 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("rc", r.returncode); print(r.stderr[-2000:])

# -- cell 6 -------------------------------------------------------------------------
import re
def header(obj): return 'FoamFile\n{\n    version 2.0; format ascii; class dictionary; object "%s";\n}\n' % obj
bd_body = re.sub(r'F^|FoamFile.*\}', '', bd, flags=re.S)  # not reliable; just rebuild
bd_body = """
vertices
(
    (0 0 0) (0.1 0 0) (0.1 0.06 0) (0 0.06 0)
    (0 0 0.04) (0.1 0 0.04) (0.1 0.06 0.04) (0 0.06 0.04)
);
blocks ( hex (0 1 2 3 4 5 6 7) (25 15 10) simpleGrading (1 1 1) );
boundary ( allWalls { type wall; faces ( (0 3 7 4) (1 2 6 5) (0 1 5 4) (3 2 6 7) (4 5 6 7) (0 1 2 3) ); } );
"""
topo_body = """
actions
(
  { name heater; type cellSet; action new; source boxToCell;
    sourceInfo { box (0.01 0.02 0) (0.03 0.04 0.02); } }
  { name heater; type cellZoneSet; action new; source setToCellZone;
    sourceInfo { set heater; } }
  { name cooler; type cellSet; action new; source boxToCell;
    sourceInfo { box (0.07 0.02 0.02) (0.09 0.04 0.04); } }
  { name cooler; type cellZoneSet; action new; source setToCellZone;
    sourceInfo { set cooler; } }
);
"""
pathlib.Path("system/controlDict").write_text(header("controlDict") + "application blockMesh;\n")
pathlib.Path("system/blockMeshDict").write_text(header("blockMeshDict") + bd_body)
pathlib.Path("system/topoSetDict").write_text(header("topoSetDict") + topo_body)
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
r2 = subprocess.run(["topoSet"], capture_output=True, text=True)
print("blockMesh", r1.returncode, "topoSet", r2.returncode)
print(r2.stdout[-600:])

# -- cell 7 -------------------------------------------------------------------------
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stderr[-800:]); print("---"); print(r.stdout[-800:])

# -- cell 8 -------------------------------------------------------------------------
cd_txt = """application blockMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
runTimeModifiable true;
"""
pathlib.Path("system/controlDict").write_text(header("controlDict") + cd_txt)
r1 = subprocess.run(["blockMesh"], capture_output=True, text=True)
r2 = subprocess.run(["topoSet"], capture_output=True, text=True)
print("blockMesh", r1.returncode, "| topoSet", r2.returncode)
print(r1.stdout[-200:], r2.stdout[-600:])

# -- cell 9 -------------------------------------------------------------------------
# Mesh built: 3750 cells, single patch `allWalls`, zones heater (132 cells) and cooler (127 cells). Ch
cm = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(cm.returncode)
print("\n".join([l for l in (cm.stdout+cm.stderr).splitlines() if re.search(r"Mesh OK|FAIL|error|heater|cooler|allWalls|zones", l, re.I)][:20]))

# -- cell 10 ------------------------------------------------------------------------
print((cm.stdout+cm.stderr)[-1200:])

# -- cell 11 ------------------------------------------------------------------------
pathlib.Path("system/fvSchemes").write_text(header("fvSchemes"))
pathlib.Path("system/fvSolution").write_text(header("fvSolution"))
cm = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(cm.returncode)
print("\n".join([l for l in (cm.stdout+cm.stderr).splitlines() if re.search(r"Mesh OK|FAILED|error|heater|cooler|allWalls|zone", l, re.I)]))

# -- cell 12 ------------------------------------------------------------------------
print((cm.stdout+cm.stderr)[-1000:])

# -- cell 13 ------------------------------------------------------------------------
fvs = """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default Gauss linear; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
wallDist { method meshWave; }
"""
pathlib.Path("system/fvSchemes").write_text(header("fvSchemes") + fvs)
pathlib.Path("system/fvSolution").write_text(header("fvSolution") + "solvers { p_rgh { solver GAMG; tolerance 1e-7; relTol 0.01; smoother GaussSeidel; } }\nrelaxationFactors { fields { p_rgh 0.7; } }\n")
cm = subprocess.run(["checkMesh"], capture_output=True, text=True)
txt = cm.stdout + cm.stderr
print(cm.returncode)
print("\n".join([l for l in txt.splitlines() if re.search(r"Mesh OK|FAILED|heater|cooler|allWalls|zone", l, re.I)]))

# -- cell 14 ------------------------------------------------------------------------
topo_body = """
actions
(
  { name heater; type cellSet; action new; source boxToCell;
    sourceInfo { box (0.008 0.02 0) (0.032 0.04 0.02); } }
  { name heater; type cellZoneSet; action new; source setToCellZone;
    sourceInfo { set heater; } }
  { name cooler; type cellSet; action new; source boxToCell;
    sourceInfo { box (0.068 0.02 0.02) (0.092 0.04 0.04); } }
  { name cooler; type cellZoneSet; action new; source setToCellZone;
    sourceInfo { set cooler; } }
);
"""
pathlib.Path("system/topoSetDict").write_text(header("topoSetDict") + topo_body)
r = subprocess.run(["topoSet"], capture_output=True, text=True)
cm = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(cm.returncode)
print([l for l in cm.stdout.splitlines() if re.search(r"heater|cooler|Mesh OK", l)])

# -- cell 15 ------------------------------------------------------------------------
# Zone x-extent is 24 mm (6 cells) — needs 20 mm (5 cells). Fixing the box bounds.
topo_body = topo_body.replace("(0.008 0.02 0) (0.032 0.04 0.02)", "(0.008 0.02 0) (0.028 0.04 0.02)")
topo_body = topo_body.replace("(0.068 0.02 0.02) (0.092 0.04 0.04)", "(0.068 0.02 0.02) (0.088 0.04 0.04)")
pathlib.Path("system/topoSetDict").write_text(header("topoSetDict") + topo_body)
subprocess.run(["topoSet"], capture_output=True)
cm = subprocess.run(["checkMesh"], capture_output=True, text=True)
print([l for l in cm.stdout.splitlines() if re.search(r"heater|cooler|Mesh OK", l)])

# -- cell 16 ------------------------------------------------------------------------
import pyvista as pv, numpy as np
import pathlib
pathlib.Path("t7.foam").write_text("")
reader = pv.OpenFOAMReader("t7.foam")
m = reader.read()
blk = m[0]
c = blk.cell_centers().points
zone = np.zeros(len(c), dtype=int)
zone[(c[:, 0] > 0.008) & (c[:, 0] < 0.028) & (c[:, 1] > 0.02) & (c[:, 1] < 0.04) & (c[:, 2] < 0.02)] = 1
zone[(c[:, 0] > 0.068) & (c[:, 0] < 0.088) & (c[:, 1] > 0.02) & (c[:, 1] < 0.04) & (c[:, 2] > 0.02)] = 2
blk["zone"] = zone
p = pv.Plotter(shape=(1, 2), window_size=[900, 400])
sl = blk.slice(normal=[0, 1, 0], origin=(0.05, 0.03, 0.02))
p.subplot(0, 0)
p.add_mesh(sl, scalars="zone", cmap=["gray", "red", "blue"], clim=[0, 2], show_scalar_bar=False)
p.view_xz()
p.subplot(0, 1)
sl2 = blk.slice(normal=[1, 0, 0], origin=(0.05, 0.03, 0.02))
p.add_mesh(sl2, scalars="zone", cmap=["gray", "red", "blue"], clim=[0, 2], show_scalar_bar=False)
p.view_yz()
p.show(screenshot="plot.png")
from IPython.display import Image
Image("plot.png")

# -- cell 17 ------------------------------------------------------------------------
# Slices confirm heater at the floor/low-x end and cooler at ceiling/high-x end. Refining to 2 mm cubi
bd_body = """
vertices
(
    (0 0 0) (0.1 0 0) (0.1 0.06 0) (0 0.06 0)
    (0 0 0.04) (0.1 0 0.04) (0.1 0.06 0.04) (0 0.06 0.04)
);
blocks ( hex (0 1 2 3 4 5 6 7) (50 30 20) simpleGrading (1 1 1) );
boundary ( allWalls { type wall; faces ( (0 3 7 4) (1 2 6 5) (0 1 5 4) (3 2 6 7) (4 5 6 7) (0 1 2 3) ); } );
"""
pathlib.Path("system/blockMeshDict").write_text(header("blockMeshDict") + bd_body)
subprocess.run(["blockMesh"], capture_output=True)
subprocess.run(["topoSet"], capture_output=True)
cm = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(cm.returncode)
print([l for l in cm.stdout.splitlines() if re.search(r"heater|cooler|Mesh OK|cells", l)])
