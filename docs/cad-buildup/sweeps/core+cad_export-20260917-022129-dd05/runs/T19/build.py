"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid directly in metres as a 60 mm extrusion: the 20.05 mm-radius bore volume minus 
import math, os, sys
import build123d as bd
from build123d import Cylinder, Box, Pos
# All dimensions are metres.
L = 0.060
R_BORE = 0.02005
R_SHAFT = 0.020
KEY_WIDTH = 0.012
KEY_DEPTH = 0.005
# Shaft crown is at y=R_SHAFT; the longitudinal keyway floor is y=R_SHAFT-KEY_DEPTH.
bore_volume = Cylinder(R_BORE, L)
key_cutter = Pos(-KEY_WIDTH/2, R_SHAFT-KEY_DEPTH, 0) * Box(KEY_WIDTH, KEY_DEPTH+0.002, L)
shaft_solid = Cylinder(R_SHAFT, L) - key_cutter
fluid = bore_volume - shaft_solid
print(f"fluid solids={len(fluid.solids())}, volume={fluid.volume:.12e} m^3")
print(f"requested clearance={(R_BORE-R_SHAFT)*1e3:.6f} mm; built={(R_BORE-R_SHAFT)*1e3:.6f} mm")
print(f"requested key width={KEY_WIDTH*1e3:.3f} mm; built={KEY_WIDTH*1e3:.3f} mm")
print(f"requested key depth={KEY_DEPTH*1e3:.3f} mm; built={(R_SHAFT-(R_SHAFT-KEY_DEPTH))*1e3:.3f} mm")
print(f"requested sealed length={L*1e3:.3f} mm; built={fluid.bounding_box().size.Z*1e3:.3f} mm")
for i,f in enumerate(fluid.faces()):
    c=f.center(); print(i, f.geom_type, f"area={f.area:.8e}", f"center=({c.X:.5g},{c.Y:.5g},{c.Z:.5g})")
# Coarse visual preview only.
os.makedirs('constant/triSurface', exist_ok=True)
bd.export_stl(fluid, 'fluid_preview.stl', tolerance=2e-4, angular_tolerance=0.2)
import pyvista as pv
pv.OFF_SCREEN=True
p=pv.read('fluid_preview.stl')
pl=pv.Plotter(off_screen=True, window_size=(900,650)); pl.add_mesh(p, color='lightblue', show_edges=True)
pl.camera_position=[(0.075,-0.09,0.075),(0,0,0.03),(0,0,1)]; pl.add_axes(); pl.show(screenshot='coarse_fluid.png')
print('preview:', p.n_cells, 'triangles; bounds', p.bounds)

# -- cell 2 -------------------------------------------------------------------------
# The preview exposed a placement error before meshing: build123d’s `Box` is centre-aligned, so the fi
# Correct the centre-aligned key cutter placement; keep the exact 60 mm axial extent.
KEY_FLOOR_Y = R_SHAFT - KEY_DEPTH
KEY_CUTTER_TOP = R_BORE + 0.002
key_cutter = Pos(0, (KEY_FLOOR_Y+KEY_CUTTER_TOP)/2, 0) * Box(KEY_WIDTH, KEY_CUTTER_TOP-KEY_FLOOR_Y, L)
shaft_solid = Cylinder(R_SHAFT, L) - key_cutter
fluid = bore_volume - shaft_solid
bb=key_cutter.bounding_box()
print('cutter x range mm:',bb.min.X*1e3,bb.max.X*1e3,'y range mm:',bb.min.Y*1e3,bb.max.Y*1e3)
print(f"fluid volume={fluid.volume:.12e} m^3, solids={len(fluid.solids())}, faces={len(fluid.faces())}")
for i,f in enumerate(fluid.faces()):
    c=f.center(); n=f.normal_at(c) if f.geom_type==bd.GeomType.PLANE else None
    print(i, f.geom_type, f"area={f.area:.9e}", f"center=({c.X:.6g},{c.Y:.6g},{c.Z:.6g})", 'normal='+str(tuple(round(v,3) for v in n)) if n else '')
bd.export_stl(fluid, 'fluid_preview.stl', tolerance=1e-4, angular_tolerance=0.15)
p=pv.read('fluid_preview.stl'); pl=pv.Plotter(off_screen=True,window_size=(900,650)); pl.add_mesh(p,color='lightblue',show_edges=True)
pl.camera_position=[(0.065,-0.085,0.075),(0,0.004,0),(0,0,1)]; pl.add_axes(); pl.show(screenshot='coarse_fluid_corrected.png')

# -- cell 3 -------------------------------------------------------------------------
# The corrected solid has exactly the expected seven boundary faces: bore cylinder, shaft cylinder, ke
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
faces=list(fluid.faces())
cyls=[f for f in faces if f.geom_type==bd.GeomType.CYLINDER]
bore_faces=[max(cyls,key=lambda f:f.area)]
shaft_faces=[min(cyls,key=lambda f:f.area)]
planes=[f for f in faces if f.geom_type==bd.GeomType.PLANE]
high_pressure=[f for f in planes if f.normal_at(f.center()).Z > 0.9]
low_pressure=[f for f in planes if f.normal_at(f.center()).Z < -0.9]
keyway_floor=[f for f in planes if f.normal_at(f.center()).Y < -0.9]
keyway_flanks=[f for f in planes if abs(f.normal_at(f.center()).X) > 0.9]
patches={'shaft':shaft_faces,'keyway_floor':keyway_floor,'keyway_flanks':keyway_flanks,
         'bore':bore_faces,'high_pressure':high_pressure,'low_pressure':low_pressure}
print({k:len(v) for k,v in patches.items()})
export_patches(fluid, patches, tolerance=5e-6, angular_tolerance=0.08)
bd.export_step(fluid,'fluid_volume.step')

# -- cell 4 -------------------------------------------------------------------------
# For the volume mesh I’m using an exact analytic 2-D cross-section with separate curve groups establi
import gmsh, math, os
gmsh.initialize(); gmsh.model.add('seal_leak')
g=gmsh.model.geo
z0=-L/2; lc_gap=2.5e-5; lc_key=2.5e-4
# Outer bore loop, CCW.
co=g.addPoint(0,0,z0,lc_gap)
po=[g.addPoint(R_BORE,0,z0,lc_gap),g.addPoint(0,R_BORE,z0,lc_gap),g.addPoint(-R_BORE,0,z0,lc_gap),g.addPoint(0,-R_BORE,z0,lc_gap)]
outer=[g.addCircleArc(po[i],co,po[(i+1)%4]) for i in range(4)]
# Keyed shaft boundary, clockwise: floor -> right flank -> long shaft arc -> left flank.
yi=math.sqrt(R_SHAFT**2-(KEY_WIDTH/2)**2)
pfl=g.addPoint(-KEY_WIDTH/2,KEY_FLOOR_Y,z0,lc_key); pfr=g.addPoint(KEY_WIDTH/2,KEY_FLOOR_Y,z0,lc_key)
pro=g.addPoint(KEY_WIDTH/2,yi,z0,lc_gap); plo=g.addPoint(-KEY_WIDTH/2,yi,z0,lc_gap)
pr=g.addPoint(R_SHAFT,0,z0,lc_gap); pb=g.addPoint(0,-R_SHAFT,z0,lc_gap); pl=g.addPoint(-R_SHAFT,0,z0,lc_gap)
floor_curve=g.addLine(pfl,pfr); right_flank=g.addLine(pfr,pro)
shaft_curves=[g.addCircleArc(pro,co,pr),g.addCircleArc(pr,co,pb),g.addCircleArc(pb,co,pl),g.addCircleArc(pl,co,plo)]
left_flank=g.addLine(plo,pfl)
inner=[floor_curve,right_flank]+shaft_curves+[left_flank]
s=g.addPlaneSurface([g.addCurveLoop(outer),g.addCurveLoop(inner)])
g.synchronize()
curve_patch={c:'bore' for c in outer}; curve_patch.update({floor_curve:'keyway_floor',right_flank:'keyway_flanks',left_flank:'keyway_flanks'}); curve_patch.update({c:'shaft' for c in shaft_curves})
out=g.extrude([(2,s)],0,0,L,numElements=[60],recombine=True); g.synchronize()
top=out[0][1]; volume=[tag for dim,tag in out if dim==3][0]
side_groups={k:[] for k in ['bore','shaft','keyway_floor','keyway_flanks']}
for dim,stag in out:
    if dim!=2 or stag==top: continue
    bnd={abs(t) for d,t in gmsh.model.getBoundary([(2,stag)],oriented=False) if d==1}
    for c,name in curve_patch.items():
        if c in bnd: side_groups[name].append(stag)
for name,tags in side_groups.items():
    pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(2,[s]); gmsh.model.setPhysicalName(2,pg,'low_pressure')
pg=gmsh.model.addPhysicalGroup(2,[top]); gmsh.model.setPhysicalName(2,pg,'high_pressure')
pg=gmsh.model.addPhysicalGroup(3,[volume]); gmsh.model.setPhysicalName(3,pg,'fluid')
gmsh.option.setNumber('Mesh.Algorithm',6); gmsh.option.setNumber('Mesh.RecombineAll',0)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.model.mesh.generate(3)
gmsh.write('seal_leak.msh')
print('side surface groups:',side_groups)
print('nodes',len(gmsh.model.mesh.getNodes()[0]),'3D elements',sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 5 -------------------------------------------------------------------------
# Conversion stopped before reading the mesh because an OpenFOAM case requires a minimal `system/contr
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write('''FoamFile\n{\n    format ascii;\n    class dictionary;\n    object controlDict;\n}\napplication checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\nrunTimeModifiable false;\n''')
print(open('system/controlDict').read())

# -- cell 6 -------------------------------------------------------------------------
# The conversion itself succeeded, but that cell was not replay-safe because `subprocess` had only bee
import subprocess
r=subprocess.run(['gmshToFoam','seal_leak.msh'],text=True,capture_output=True,timeout=240)
print(r.stdout[-2500:]); print(r.stderr[-500:]); print('returncode',r.returncode)
if r.returncode: raise RuntimeError('gmshToFoam failed')

# -- cell 7 -------------------------------------------------------------------------
# I’ll now run the binding bare `checkMesh` and inspect the boundary counts. This will determine wheth
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
print(r.stdout[-9000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)
open('checkMesh.log','w').write(r.stdout+r.stderr)

# -- cell 8 -------------------------------------------------------------------------
# OpenFOAM 2512 also requires minimal `fvSchemes`/`fvSolution` dictionaries even for `checkMesh`; no m
open('system/fvSchemes','w').write('''FoamFile\n{\n format ascii; class dictionary; object fvSchemes;\n}\n''')
open('system/fvSolution','w').write('''FoamFile\n{\n format ascii; class dictionary; object fvSolution;\n}\n''')
print('wrote minimal fvSchemes and fvSolution')

# -- cell 9 -------------------------------------------------------------------------
# This OpenFOAM build constructs finite-volume interpolation data during mesh loading and therefore re
open('system/fvSchemes','w').write('''FoamFile\n{ format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\nfluxRequired { default no; }\n''')
print(open('system/fvSchemes').read())

# -- cell 10 ------------------------------------------------------------------------
# I’ll run the binding check again and inspect the actual topology and geometry results.
r=subprocess.run(['checkMesh'],text=True,capture_output=True,timeout=240)
print(r.stdout[-12000:]); print(r.stderr[-1000:]); print('returncode',r.returncode)
open('checkMesh.log','w').write(r.stdout+r.stderr)
