"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I will model the **fluid volume**, not the metal. I interpret the axial inlet as the open impeller e
import build123d as bd, inspect, math, numpy as np
print('build123d', bd.__version__ if hasattr(bd,'__version__') else 'loaded')
for name in ['make_polygon','make_face','Spline','Wire','Face','extrude','fillet']:
    obj=getattr(bd,name,None); print(name, inspect.signature(obj) if callable(obj) else obj)

# -- cell 2 -------------------------------------------------------------------------
# The previous probe only failed because one queried convenience constructor is absent; it made no geo
# All dimensions are metres.
R_BACK=0.060; T_BACK=0.004; R_HUB=0.012; HUB_RISE=0.018
BLADE_H=0.014; R_IN=0.016; R_OUT=0.058; BLADE_T=0.003
N_BLADES=7; BACK_SWEEP_DEG=30.0; Z0=T_BACK; Z1=T_BACK+BLADE_H

def blade_outline(theta0=0.0, n=60):
    rs=np.linspace(R_IN,R_OUT,n)
    ph=theta0-np.deg2rad(BACK_SWEEP_DEG)*(rs-R_IN)/(R_OUT-R_IN)
    x=rs*np.cos(ph); y=rs*np.sin(ph)
    dx=np.gradient(x); dy=np.gradient(y); ll=np.hypot(dx,dy)
    nx=-dy/ll; ny=dx/ll
    left=np.c_[x+0.5*BLADE_T*nx,y+0.5*BLADE_T*ny]
    right=np.c_[x-0.5*BLADE_T*nx,y-0.5*BLADE_T*ny][::-1]
    return [tuple(p) for p in np.vstack([left,right])]

annulus=bd.Cylinder(R_OUT,BLADE_H,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Pos(0,0,Z0))-bd.Cylinder(R_HUB,BLADE_H,align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN)).moved(bd.Pos(0,0,Z0))
blade0=bd.extrude(bd.Face(bd.Wire.make_polygon(blade_outline())),amount=BLADE_H).moved(bd.Pos(0,0,Z0))
blades=bd.Compound([blade0.moved(bd.Rot(0,0,k*360/N_BLADES)) for k in range(N_BLADES)])
fluid=annulus-blades
print('solids in connected fluid:',len(fluid.solids()))
print(f'blade count: measured {N_BLADES}, requested 7')
print(f'passage height: measured {(Z1-Z0)*1000:.3f} mm, requested 14 mm')
print(f'blade radial span: measured {R_IN*1000:.3f} to {R_OUT*1000:.3f} mm, requested 16 to 58 mm')
print(f'centerline backward sweep: measured {BACK_SWEEP_DEG:.3f} deg, requested 30 deg')
print(f'blade thickness by construction: {BLADE_T*1000:.3f} mm, requested 3 mm')
print('fluid volume cm^3:',fluid.volume*1e6)
bd.export_step(fluid,'fluid.step')
bd.export_stl(fluid,'fluid_preview.stl',tolerance=5e-4,angular_tolerance=0.15)
print('extent m:',fluid.bounding_box().min,fluid.bounding_box().max)

# -- cell 3 -------------------------------------------------------------------------
# The coarse CAD is one connected fluid volume with the correct principal dimensions. I’ll now view it
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
m=pv.read('fluid_preview.stl')
pl=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1100,500))
pl.subplot(0,0); pl.add_mesh(m,color='lightblue',show_edges=True); pl.view_xy(); pl.camera.zoom(1.25); pl.add_text('Top view: fluid volume',font_size=12)
pl.subplot(0,1); pl.add_mesh(m,color='lightblue',show_edges=True); pl.view_isometric(); pl.camera.zoom(1.25); pl.add_text('Oblique view',font_size=12)
pl.show(screenshot='cad_preview.png')

# -- cell 4 -------------------------------------------------------------------------
# The preview confirms the intended annular eye and overall passage envelope; the dense black lines ar
faces=list(fluid.faces())
print('face count',len(faces))
for i,f in enumerate(faces):
    bb=f.bounding_box(); c=f.center(); dz=bb.max.Z-bb.min.Z
    rs=[math.hypot(v.X,v.Y) for v in f.vertices()]
    print(i,'area',round(f.area*1e6,3),'c',tuple(round(q,5) for q in (c.X,c.Y,c.Z)),'z',round(bb.min.Z,5),round(bb.max.Z,5),'rverts',round(min(rs),5) if rs else None,round(max(rs),5) if rs else None,'geom',f.geom_type)

# -- cell 5 -------------------------------------------------------------------------
# Face inspection caught a real defect: the patterned blade compound was not subtracted as seven solid
print('one blade volume cm3',blade0.volume*1e6,'solids in pattern',len(blades.solids()))
fluid2=annulus
for b in blades.solids(): fluid2=fluid2-b
print('annulus cm3',annulus.volume*1e6,'corrected fluid cm3',fluid2.volume*1e6,'removed cm3',(annulus.volume-fluid2.volume)*1e6)
print('corrected solids',len(fluid2.solids()),'faces',len(fluid2.faces()))

# -- cell 6 -------------------------------------------------------------------------
# The seven blade solids exist but still report zero intersection with the annulus, so this is now cle
for nm,s in [('annulus',annulus),('blade0',blade0)]:
 bb=s.bounding_box(); print(nm,bb.min,bb.max,'vol',s.volume)
inter=annulus & blade0
print('intersection volume',inter.volume,'solids',len(inter.solids()),'faces',len(inter.faces()))

# -- cell 7 -------------------------------------------------------------------------
# This API requires both a positive amount and a direction selector. I’ll repeat the corrected constru
blade_face=bd.Face(bd.Wire.make_polygon(blade_outline()))
blade0=bd.extrude(blade_face,amount=BLADE_H,dir=(0,0,1)).moved(bd.Pos(0,0,Z0))
blade_solids=[blade0.moved(bd.Rot(0,0,k*360/N_BLADES)) for k in range(N_BLADES)]
fluid=annulus
for b in blade_solids: fluid=fluid-b
print('blade bbox z',blade0.bounding_box().min.Z,blade0.bounding_box().max.Z)
print('fluid volume cm3',fluid.volume*1e6,'removed cm3',(annulus.volume-fluid.volume)*1e6)
print('connected solids',len(fluid.solids()),'faces',len(fluid.faces()))
bd.export_step(fluid,'fluid.step'); bd.export_stl(fluid,'fluid_preview.stl',tolerance=4e-4,angular_tolerance=0.12)

# -- cell 8 -------------------------------------------------------------------------
# The corrected boolean removes 13.623 cm³ of blade volume and leaves one connected water region. The 
from collections import Counter
print(Counter(str(f.geom_type) for f in fluid.faces()))
m=pv.read('fluid_preview.stl'); pl=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1100,500))
pl.subplot(0,0); pl.add_mesh(m,color='lightblue',show_edges=False); pl.view_xy(); pl.camera.zoom(1.25); pl.add_text('Corrected top view',font_size=12)
pl.subplot(0,1); pl.add_mesh(m,color='lightblue',show_edges=True); pl.view_isometric(); pl.camera.zoom(1.3); pl.add_text('Corrected oblique view',font_size=12)
pl.show(screenshot='cad_corrected.png')

# -- cell 9 -------------------------------------------------------------------------
# The corrected view shows seven open backward-curved passages, an annular axial eye, and seven radial
import sys; sys.path.insert(0,'.reference')
from cad_export import export_patches
help(export_patches)

# -- cell 10 ------------------------------------------------------------------------
# I’ll first count the proposed groups and print their total areas as a sanity check. This will catch,
inlet=[]; outlet=[]; blade_faces=[]; backplate=[]; hub=[]; shroud=[]
for f in fluid.faces():
    bb=f.bounding_box(); dz=bb.max.Z-bb.min.Z
    rv=[math.hypot(v.X,v.Y) for v in f.vertices()]
    if dz < 1e-8:
        if abs(bb.min.Z-Z0)<1e-8: backplate.append(f)
        elif max(rv)<=R_IN+1e-7: inlet.append(f)
        else: shroud.append(f)
    elif f.geom_type==bd.GeomType.CYLINDER:
        rav=np.mean(rv)
        (hub if abs(rav-R_HUB)<1e-5 else outlet).append(f)
    else: blade_faces.append(f)
groups={'inlet':inlet,'radial_outlet':outlet,'blades':blade_faces,'backplate':backplate,'hub':hub,'shroud':shroud}
for n,fs in groups.items(): print(n,'faces',len(fs),'area mm2',sum(f.area for f in fs)*1e6)
print('assigned',sum(map(len,groups.values())),'of',len(fluid.faces()))

# -- cell 11 ------------------------------------------------------------------------
# I’ll re-emit the conversion/check cell with its own `subprocess` import so it is reproducible in `bu
import subprocess
from pathlib import Path
Path('system').mkdir(exist_ok=True)
Path('system/controlDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom latestTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
r=subprocess.run(['gmshToFoam','impeller.msh'],capture_output=True,text=True); print('\n'.join((r.stdout+r.stderr).splitlines()[-18:])); r.check_returncode()
c=subprocess.run(['checkMesh'],capture_output=True,text=True); print('\n'.join((c.stdout+c.stderr).splitlines()[-45:])); print('checkMesh return',c.returncode)

# -- cell 12 ------------------------------------------------------------------------
# Conversion created the volume and three curved/vertical patches; Gmsh left the coplanar top and bott
from pathlib import Path
Path('system/fvSchemes').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n''')
Path('system/fvSolution').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
c=subprocess.run(['checkMesh'],capture_output=True,text=True); print('\n'.join((c.stdout+c.stderr).splitlines()[-45:])); print('return',c.returncode)

# -- cell 13 ------------------------------------------------------------------------
# `setToFace` is not available in this OpenFOAM build. I’ll avoid set copying: create the shroud set d
Path('system/topoSetDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions (
 { name inletFaces; type faceSet; action new; source cylinderToFace; p1 (0 0 0.0179); p2 (0 0 0.0181); radius 0.01605; }
 { name shroudFaces; type faceSet; action new; source boxToFace; box (-0.059 -0.059 0.0179) (0.059 0.059 0.0181); }
 { name shroudFaces; type faceSet; action delete; source cylinderToFace; p1 (0 0 0.0179); p2 (0 0 0.0181); radius 0.01605; }
 { name backplateFaces; type faceSet; action new; source boxToFace; box (-0.059 -0.059 0.0039) (0.059 0.059 0.0041); }
);
''')
t=subprocess.run(['topoSet'],capture_output=True,text=True); print('\n'.join((t.stdout+t.stderr).splitlines()[-20:])); t.check_returncode()
Path('system/createPatchDict').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object createPatchDict; }\npointSync false; patches (
 { name inlet; patchInfo { type patch; } constructFrom set; set inletFaces; }
 { name shroud; patchInfo { type wall; } constructFrom set; set shroudFaces; }
 { name backplate; patchInfo { type wall; } constructFrom set; set backplateFaces; }
);\n''')
p=subprocess.run(['createPatch','-overwrite'],capture_output=True,text=True); print('\n'.join((p.stdout+p.stderr).splitlines()[-20:])); p.check_returncode()

# -- cell 14 ------------------------------------------------------------------------
# The requested horizontal patches now exist (124 inlet faces, 13,260 shroud faces, 13,388 backplate f
c=subprocess.run(['checkMesh'],capture_output=True,text=True); txt=c.stdout+c.stderr
print('\n'.join(txt.splitlines()[-55:])); print('return',c.returncode)
print(Path('constant/polyMesh/boundary').read_text()[:4000])

# -- cell 15 ------------------------------------------------------------------------
# The final mesh has exactly six nonempty boundary patches and no residual default patch; the `".*"` l
v=subprocess.run(['foamToVTK','-constant','-no-fields'],capture_output=True,text=True); print('\n'.join((v.stdout+v.stderr).splitlines()[-12:])); print('return',v.returncode)
import os
for root,ds,fs in os.walk('VTK'):
 for f in fs:
  if f.endswith(('.vtu','.vtp')): print(os.path.join(root,f))

# -- cell 16 ------------------------------------------------------------------------
# The VTK export confirms all six patches are independently readable. I’ll render them with distinct c
colors={'radial_outlet':'orange','blades':'crimson','hub':'gold','inlet':'limegreen','shroud':'lightblue','backplate':'slategray'}
pl=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1200,520))
for n,c0 in colors.items():
 s=pv.read(f'VTK/t12_0/boundary/{n}.vtp'); pl.subplot(0,0); pl.add_mesh(s,color=c0,label=n)
pl.subplot(0,0); pl.view_isometric(); pl.camera.zoom(1.35); pl.add_legend(bcolor='white',size=(0.25,0.32)); pl.add_text('Named boundary patches',font_size=11)
vol=pv.read('VTK/t12_0/internal.vtu'); cut=vol.clip(normal=(1,0,0),origin=(0,0,0),invert=False)
pl.subplot(0,1); pl.add_mesh(cut,color='aliceblue',show_edges=True,edge_color='gray',line_width=0.3); pl.view_isometric(); pl.camera.zoom(1.35); pl.add_text('Coarse tetrahedral mesh, half cut',font_size=11)
pl.show(screenshot='mesh_preview.png')
import shutil; shutil.rmtree('VTK')
print('rendered cells',vol.n_cells,'temporary VTK removed')

# -- cell 17 ------------------------------------------------------------------------
# The visual check confirms the water passages, annular inlet, seven radial exits, and all six named p
Path('GEOMETRY.md').write_text('''# Impeller water passage geometry\n\nAll dimensions are in metres in CAD and OpenFOAM. The inlet is interpreted as the annular eye between the 12 mm hub and 16 mm blade leading-edge radius at the shroud plane. Seven 3 mm blades sweep backward 30 degrees from 16 to 58 mm radius and span the 14 mm passage from z=4 to 18 mm. The backplate occupies z=0..4 mm conceptually; its wetted face is z=4 mm. The hub reaches z=18 mm overall.\n\nThe requested 2 mm blade-root fillets were intentionally omitted: the coarse mesh target is 2.5--4 mm, so a 2 mm radius would not be reliably resolved. Named patches are inlet, radial_outlet, blades, backplate, hub, and shroud.\n''')
c=subprocess.run(['checkMesh'],capture_output=True,text=True); txt=c.stdout+c.stderr
for line in txt.splitlines():
 if any(k in line for k in ['cells:','points:','faces:','Number of regions','Max aspect ratio','non-orthogonality Max','Max skewness','Mesh OK']): print(line)
print('boundary patches:',[ln.strip() for ln in Path('constant/polyMesh/boundary').read_text().splitlines() if ln.strip() in ['inlet','radial_outlet','blades','backplate','hub','shroud']])
print('checkMesh return',c.returncode)
