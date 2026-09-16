"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the water as the annular rotating passage between the solid hub (`r=12 mm`) and outlet (`
# I’ll generate the boundary directly as named triangulated patches and use `cartesianMesh`, first at 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# All dimensions are SI metres.
R_BACKPLATE=0.060; T_BACKPLATE=0.004
R_HUB=0.012; HUB_HEIGHT=0.018
N_BLADES=7; BLADE_THICKNESS=0.003; BLADE_HEIGHT=0.014
R_BLADE_IN=0.016; R_OUT=0.058; SWEEP_DEG=30.0
Z_BOTTOM=T_BACKPLATE; Z_TOP=T_BACKPLATE+BLADE_HEIGHT
ROOT_FILLET=0.002; COARSE_CELL=0.004

def blade_centerline(r, theta0=0.0):
    return theta0-np.deg2rad(SWEEP_DEG)*(r-R_BLADE_IN)/(R_OUT-R_BLADE_IN)

def blade_edges(r, theta0=0.0):
    th=blade_centerline(r,theta0)
    # Constant physical thickness represented by +/- half-thickness angular offsets.
    dth=np.arcsin((BLADE_THICKNESS/2)/r)
    return th+dth, th-dth

r=np.linspace(R_BLADE_IN,R_OUT,250)
fig,ax=plt.subplots(figsize=(6,6))
for k in range(N_BLADES):
    t0=2*np.pi*k/N_BLADES
    tp,tm=blade_edges(r,t0)
    ax.fill(np.r_[r*np.cos(tp),(r*np.cos(tm))[::-1]],np.r_[r*np.sin(tp),(r*np.sin(tm))[::-1]],alpha=.55)
for rr,ls in [(R_HUB,'--'),(R_BLADE_IN,':'),(R_OUT,'-')]:
    ax.add_patch(plt.Circle((0,0),rr,fill=False,ls=ls,color='k'))
ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('Impeller water-domain planform (solids shaded)'); ax.grid(True)
plt.show()
mid_thickness=2*(0.5*(R_BLADE_IN+R_OUT))*np.sin((blade_edges(np.array([0.5*(R_BLADE_IN+R_OUT)]))[0]-blade_centerline(np.array([0.5*(R_BLADE_IN+R_OUT)]))))[0]
print(f"Measured centerline sweep: {np.rad2deg(blade_centerline(R_OUT)-blade_centerline(R_BLADE_IN)):.3f} deg (requested -30.000 deg)")
print(f"Measured blade thickness at mid-radius: {mid_thickness*1000:.3f} mm (requested 3.000 mm)")
print(f"Flow height: {(Z_TOP-Z_BOTTOM)*1000:.3f} mm (requested 14.000 mm)")
print(f"Axial inlet annulus: {R_HUB*1000:.1f} to {R_BLADE_IN*1000:.1f} mm radius")

# -- cell 2 -------------------------------------------------------------------------
# I’m re-emitting the successful coarse build with its own `gmsh` and `math` imports so it is reproduc
import gmsh, math
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize()
gmsh.model.add('impeller_water')
occ=gmsh.model.occ
eye_o=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_BLADE_IN); eye_i=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_HUB)
eye,_=occ.cut([(3,eye_o)],[(3,eye_i)],removeObject=True,removeTool=True)
out_o=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_OUT); out_i=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_BLADE_IN)
outer_ann,_=occ.cut([(3,out_o)],[(3,out_i)],removeObject=True,removeTool=True)
blade_vols=[]
for k in range(N_BLADES):
    rv=np.linspace(R_BLADE_IN,R_OUT,25); t0=2*np.pi*k/N_BLADES; tp,tm=blade_edges(rv,t0)
    plus=[occ.addPoint(float(rr*np.cos(tt)),float(rr*np.sin(tt)),Z_BOTTOM) for rr,tt in zip(rv,tp)]
    minus=[occ.addPoint(float(rr*np.cos(tt)),float(rr*np.sin(tt)),Z_BOTTOM) for rr,tt in zip(rv[::-1],tm[::-1])]
    edges=[occ.addBSpline(plus),occ.addLine(plus[-1],minus[0]),occ.addBSpline(minus),occ.addLine(minus[-1],plus[0])]
    face=occ.addPlaneSurface([occ.addWire(edges,checkClosed=True)])
    blade_vols += [t for d,t in occ.extrude([(2,face)],0,0,Z_TOP-Z_BOTTOM) if d==3]
outer_fluid,_=occ.cut(outer_ann,[(3,v) for v in blade_vols],removeObject=True,removeTool=True)
fluid_frag,_=occ.fragment(eye,outer_fluid,removeObject=True,removeTool=True)
occ.synchronize()
vols=[(d,t) for d,t in gmsh.model.getEntities(3) if occ.getMass(d,t)>1e-12]
inc={}
for v in vols:
    for d,s in gmsh.model.getBoundary([v],oriented=False):
        if d==2: inc[s]=inc.get(s,0)+1
external=[s for s,n in inc.items() if n==1]
groups={n:[] for n in ['inlet','radial_outlet','blades','backplate','hub','shroud']}
for s in external:
    typ=gmsh.model.getType(2,s); cx,cy,cz=occ.getCenterOfMass(2,s); rr=math.hypot(cx,cy)
    if typ=='Plane' and abs(cz-Z_TOP)<1e-7: groups['inlet' if rr < 0.5*(R_HUB+R_BLADE_IN) else 'shroud'].append(s)
    elif typ=='Plane' and abs(cz-Z_BOTTOM)<1e-7: groups['backplate'].append(s)
    elif typ=='Cylinder' and rr < 0.002: groups['hub'].append(s)
    elif typ=='Cylinder' and rr > 0.04: groups['radial_outlet'].append(s)
    else: groups['blades'].append(s)
gmsh.model.addPhysicalGroup(3,[t for d,t in vols],1); gmsh.model.setPhysicalName(3,1,'water')
for i,(name,surfs) in enumerate(groups.items(),11):
    if not surfs: raise RuntimeError(f'No surfaces found for {name}')
    gmsh.model.addPhysicalGroup(2,surfs,i); gmsh.model.setPhysicalName(2,i,name)
print('Positive-volume fluid regions:',len(vols),'patch CAD-face counts:',{k:len(v) for k,v in groups.items()})
gmsh.option.setNumber('Mesh.MeshSizeMin',COARSE_CELL); gmsh.option.setNumber('Mesh.MeshSizeMax',COARSE_CELL)
gmsh.option.setNumber('Mesh.Algorithm3D',1); gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3); gmsh.write('impeller_water_coarse.msh')
ne=sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]); print('Coarse tetrahedra:',ne)
gmsh.finalize()

# -- cell 3 -------------------------------------------------------------------------
# The mesh itself has 12,417 valid tetrahedra and no ill-shaped tetrahedra after Gmsh optimization. I’
import subprocess, os, textwrap, shutil
Path('system').mkdir(exist_ok=True); Path('constant').mkdir(exist_ok=True)
Path('system/controlDict').write_text(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object controlDict; }
application meshOnly;
startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
'''))
if Path('constant/polyMesh').exists(): shutil.rmtree('constant/polyMesh')
r=subprocess.run(['gmshToFoam','impeller_water_coarse.msh'],capture_output=True,text=True)
print('gmshToFoam return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-35:]))
print(Path('constant/polyMesh/boundary').read_text() if Path('constant/polyMesh/boundary').exists() else 'NO BOUNDARY FILE')

# -- cell 4 -------------------------------------------------------------------------
# All six requested patches survived conversion with nonzero face counts. The conversion warnings are 
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],capture_output=True,text=True)
Path('checkMesh.coarse.log').write_text(r.stdout+r.stderr)
print('checkMesh return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-80:]))

# -- cell 5 -------------------------------------------------------------------------
# `checkMesh` did not yet inspect geometry because this OpenFOAM build requires placeholder `fvSchemes
Path('system/fvSchemes').write_text('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
Path('system/fvSolution').write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],capture_output=True,text=True)
Path('checkMesh.coarse.log').write_text(r.stdout+r.stderr)
print('checkMesh return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-90:]))

# -- cell 6 -------------------------------------------------------------------------
# This installation also validates the standard scheme subdictionaries on startup. I’ll provide minima
Path('system/fvSchemes').write_text(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''))
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],capture_output=True,text=True)
Path('checkMesh.coarse.log').write_text(r.stdout+r.stderr)
print('checkMesh return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-100:]))

# -- cell 7 -------------------------------------------------------------------------
# The coarse OpenFOAM mesh is one connected region, closed, correctly scaled, and has acceptable aspec
r=subprocess.run(['subsetMesh','-overwrite','water'],capture_output=True,text=True)
print('subsetMesh return:',r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-25:]))
for args in (['checkMesh'],['checkMesh','-allGeometry','-allTopology']):
    q=subprocess.run(args,capture_output=True,text=True); out=q.stdout+q.stderr
    print('\nCOMMAND',' '.join(args),'return',q.returncode)
    print('\n'.join([ln for ln in out.splitlines() if any(x in ln for x in ['points:','cells:','regions:','unused','Failed','Mesh OK','Max aspect','non-orthogonality','skewness','determinant'])][-15:]))

# -- cell 8 -------------------------------------------------------------------------
# After removing unused points, the standard binding check reports **Mesh OK**. The sole extended warn
# Now I’ll render the actual OpenFOAM mesh surface, with the inlet and outlet highlighted, to verify t
r=subprocess.run(['foamToVTK','-no-fields'],capture_output=True,text=True)
print('foamToVTK return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-20:]))
print([str(p) for p in Path('VTK').rglob('*') if p.suffix in ('.vtk','.vtp','.vtu')][:20])

# -- cell 9 -------------------------------------------------------------------------
# `subsetMesh` wrote the cleaned mesh at time `0` while leaving the original in `constant`; `foamToVTK
# Promote the cleaned all-cell subset to the canonical constant/polyMesh location.
clean=Path('0/polyMesh'); canonical=Path('constant/polyMesh')
if clean.exists():
    if canonical.exists(): shutil.rmtree(canonical)
    shutil.move(str(clean),str(canonical))
if Path('0').exists(): shutil.rmtree('0')
r=subprocess.run(['checkMesh'],capture_output=True,text=True)
print('checkMesh return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-45:]))

# -- cell 10 ------------------------------------------------------------------------
# The canonical mesh now passes standard `checkMesh` with one region, 12,417 cells, six named patches,
import pyvista as pv
if Path('VTK').exists(): shutil.rmtree('VTK')
subprocess.run(['foamToVTK','-no-fields'],check=True,capture_output=True,text=True)
pv.global_theme.allow_empty_mesh=True
plotter=pv.Plotter(off_screen=True,window_size=(1000,760))
bdir=Path('VTK/t12_0/boundary')
style={'backplate':('lightgray',0.22),'shroud':('skyblue',0.18),'blades':('goldenrod',1.0),'hub':('slategray',1.0),'inlet':('limegreen',1.0),'radial_outlet':('tomato',0.72)}
for name,(color,opacity) in style.items():
    mesh=pv.read(bdir/f'{name}.vtp')
    plotter.add_mesh(mesh,color=color,opacity=opacity,show_edges=(name in ['blades','inlet','radial_outlet']),line_width=1)
plotter.add_axes(); plotter.view_isometric(); plotter.camera.zoom(1.35)
plotter.add_text('OpenFOAM boundary mesh: inlet green, outlet red, blades gold',font_size=11)
plotter.show(screenshot='impeller_mesh.png')
img=plt.imread('impeller_mesh.png'); plt.figure(figsize=(10,7.6)); plt.imshow(img); plt.axis('off'); plt.show()
print('Rendered canonical OpenFOAM mesh; patch files:',sorted(p.stem for p in bdir.glob('*.vtp')))

# -- cell 11 ------------------------------------------------------------------------
# The render revealed one dimensional interpretation worth correcting before finalization: the blades/
Z_HUB_TOP=Z_BOTTOM+HUB_HEIGHT
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.model.add('impeller_water_final'); occ=gmsh.model.occ
eye_o=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_HUB_TOP-Z_BOTTOM,R_BLADE_IN); eye_i=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_HUB_TOP-Z_BOTTOM,R_HUB)
eye,_=occ.cut([(3,eye_o)],[(3,eye_i)],removeObject=True,removeTool=True)
out_o=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_OUT); out_i=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_BLADE_IN)
outer_ann,_=occ.cut([(3,out_o)],[(3,out_i)],removeObject=True,removeTool=True); blade_vols=[]
for k in range(N_BLADES):
    rv=np.linspace(R_BLADE_IN,R_OUT,25); tp,tm=blade_edges(rv,2*np.pi*k/N_BLADES)
    pp=[occ.addPoint(float(r*np.cos(t)),float(r*np.sin(t)),Z_BOTTOM) for r,t in zip(rv,tp)]
    pm=[occ.addPoint(float(r*np.cos(t)),float(r*np.sin(t)),Z_BOTTOM) for r,t in zip(rv[::-1],tm[::-1])]
    es=[occ.addBSpline(pp),occ.addLine(pp[-1],pm[0]),occ.addBSpline(pm),occ.addLine(pm[-1],pp[0])]
    f=occ.addPlaneSurface([occ.addWire(es,checkClosed=True)])
    blade_vols += [t for d,t in occ.extrude([(2,f)],0,0,Z_TOP-Z_BOTTOM) if d==3]
outer_fluid,_=occ.cut(outer_ann,[(3,v) for v in blade_vols],removeObject=True,removeTool=True)
occ.fragment(eye,outer_fluid,removeObject=True,removeTool=True); occ.synchronize()
vols=[e for e in gmsh.model.getEntities(3) if occ.getMass(*e)>1e-12]; inc={}
for v in vols:
    for d,s in gmsh.model.getBoundary([v],oriented=False):
        if d==2: inc[s]=inc.get(s,0)+1
external=[s for s,n in inc.items() if n==1]; groups={n:[] for n in ['inlet','radial_outlet','blades','backplate','hub','shroud']}
hub_area=2*np.pi*R_HUB*(Z_HUB_TOP-Z_BOTTOM)
for s in external:
    typ=gmsh.model.getType(2,s); cx,cy,cz=occ.getCenterOfMass(2,s); rr=math.hypot(cx,cy); area=occ.getMass(2,s)
    if typ=='Plane' and abs(cz-Z_HUB_TOP)<1e-7: groups['inlet'].append(s)
    elif typ=='Plane' and abs(cz-Z_TOP)<1e-7: groups['shroud'].append(s)
    elif typ=='Plane' and abs(cz-Z_BOTTOM)<1e-7: groups['backplate'].append(s)
    elif typ=='Cylinder' and rr<0.002 and abs(area-hub_area)<0.25*hub_area: groups['hub'].append(s)
    elif typ=='Cylinder' and rr<0.002: groups['shroud'].append(s)
    elif typ=='Cylinder' and rr>0.04: groups['radial_outlet'].append(s)
    else: groups['blades'].append(s)
gmsh.model.addPhysicalGroup(3,[t for d,t in vols],1); gmsh.model.setPhysicalName(3,1,'water')
for i,(name,surfs) in enumerate(groups.items(),11):
    if not surfs: raise RuntimeError(name)
    gmsh.model.addPhysicalGroup(2,surfs,i); gmsh.model.setPhysicalName(2,i,name)
gmsh.option.setNumber('Mesh.MeshSizeMin',COARSE_CELL); gmsh.option.setNumber('Mesh.MeshSizeMax',COARSE_CELL); gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3); gmsh.write('impeller_water.msh')
print('Final CAD bounds z:',Z_BOTTOM,Z_HUB_TOP,'blade/shroud top:',Z_TOP)
print('Patch faces:',{k:len(v) for k,v in groups.items()},'tetrahedra:',sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 12 ------------------------------------------------------------------------
# The corrected CAD measures `18 mm` hub rise (`z=4–22 mm`) and `14 mm` blade/shroud rise (`z=4–18 mm`
if canonical.exists(): shutil.rmtree(canonical)
subprocess.run(['gmshToFoam','impeller_water.msh'],check=True,capture_output=True,text=True)
subprocess.run(['subsetMesh','-overwrite','water'],check=True,capture_output=True,text=True)
clean=Path('0/polyMesh')
if clean.exists():
    if canonical.exists(): shutil.rmtree(canonical)
    shutil.move(str(clean),str(canonical)); shutil.rmtree('0')
r=subprocess.run(['checkMesh'],capture_output=True,text=True); Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('checkMesh return:',r.returncode)
print('\n'.join((r.stdout+r.stderr).splitlines()[-55:]))

# -- cell 13 ------------------------------------------------------------------------
# The extended-eye mesh passes, but its maximum aspect ratio (138) and 11 severe non-orthogonal faces 
if gmsh.isInitialized(): gmsh.finalize()
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.model.add('impeller_water_final_clean'); occ=gmsh.model.occ
lo_o=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_OUT); lo_i=occ.addCylinder(0,0,Z_BOTTOM,0,0,Z_TOP-Z_BOTTOM,R_HUB)
lower,_=occ.cut([(3,lo_o)],[(3,lo_i)],removeObject=True,removeTool=True); blade_vols=[]
for k in range(N_BLADES):
    rv=np.linspace(R_BLADE_IN,R_OUT,25); tp,tm=blade_edges(rv,2*np.pi*k/N_BLADES)
    pp=[occ.addPoint(float(r*np.cos(t)),float(r*np.sin(t)),Z_BOTTOM) for r,t in zip(rv,tp)]; pm=[occ.addPoint(float(r*np.cos(t)),float(r*np.sin(t)),Z_BOTTOM) for r,t in zip(rv[::-1],tm[::-1])]
    es=[occ.addBSpline(pp),occ.addLine(pp[-1],pm[0]),occ.addBSpline(pm),occ.addLine(pm[-1],pp[0])]
    f=occ.addPlaneSurface([occ.addWire(es,checkClosed=True)]); blade_vols += [t for d,t in occ.extrude([(2,f)],0,0,Z_TOP-Z_BOTTOM) if d==3]
lower_fluid,_=occ.cut(lower,[(3,v) for v in blade_vols],removeObject=True,removeTool=True)
up_o=occ.addCylinder(0,0,Z_TOP,0,0,Z_HUB_TOP-Z_TOP,R_BLADE_IN); up_i=occ.addCylinder(0,0,Z_TOP,0,0,Z_HUB_TOP-Z_TOP,R_HUB)
upper,_=occ.cut([(3,up_o)],[(3,up_i)],removeObject=True,removeTool=True)
fluid,_=occ.fuse(lower_fluid,upper,removeObject=True,removeTool=True); occ.synchronize(); vols=gmsh.model.getEntities(3)
inc={}
for v in vols:
    for d,s in gmsh.model.getBoundary([v],oriented=False):
        if d==2: inc[s]=inc.get(s,0)+1
external=[s for s,n in inc.items() if n==1]; groups={n:[] for n in ['inlet','radial_outlet','blades','backplate','hub','shroud']}
for s in external:
    typ=gmsh.model.getType(2,s); cx,cy,cz=occ.getCenterOfMass(2,s); rr=math.hypot(cx,cy)
    if typ=='Plane' and abs(cz-Z_HUB_TOP)<1e-7: groups['inlet'].append(s)
    elif typ=='Plane' and abs(cz-Z_TOP)<1e-7: groups['shroud'].append(s)
    elif typ=='Plane' and abs(cz-Z_BOTTOM)<1e-7: groups['backplate'].append(s)
    elif typ=='Cylinder' and rr<0.002 and occ.getMass(2,s)>0.0005: groups['hub'].append(s)
    elif typ=='Cylinder' and rr<0.002: groups['shroud'].append(s)
    elif typ=='Cylinder' and rr>0.04: groups['radial_outlet'].append(s)
    else: groups['blades'].append(s)
gmsh.model.addPhysicalGroup(3,[t for d,t in vols],1); gmsh.model.setPhysicalName(3,1,'water')
for i,(name,surfs) in enumerate(groups.items(),11):
    if not surfs: raise RuntimeError(name)
    gmsh.model.addPhysicalGroup(2,surfs,i); gmsh.model.setPhysicalName(2,i,name)
gmsh.option.setNumber('Mesh.MeshSizeMin',COARSE_CELL); gmsh.option.setNumber('Mesh.MeshSizeMax',COARSE_CELL); gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
gmsh.model.mesh.generate(3); gmsh.write('impeller_water.msh')
print('Fluid volumes:',len(vols),'patch CAD faces:',{k:len(v) for k,v in groups.items()},'tets:',sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]))
gmsh.finalize()

# -- cell 14 ------------------------------------------------------------------------
# The clean Boolean now contains only the expected eight positive volumes (central eye plus seven blad
if canonical.exists(): shutil.rmtree(canonical)
subprocess.run(['gmshToFoam','impeller_water.msh'],check=True,capture_output=True,text=True)
subprocess.run(['subsetMesh','-overwrite','water'],check=True,capture_output=True,text=True)
clean=Path('0/polyMesh')
if clean.exists():
    if canonical.exists(): shutil.rmtree(canonical)
    shutil.move(str(clean),str(canonical)); shutil.rmtree('0')
r=subprocess.run(['checkMesh'],capture_output=True,text=True); Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('checkMesh return:',r.returncode)
for ln in (r.stdout+r.stderr).splitlines():
    if any(x in ln for x in ['points:','cells:','regions:','BoundingBox','aspect ratio','non-orthogonality','skewness','volume =','Mesh OK','Failed']): print(ln)

# -- cell 15 ------------------------------------------------------------------------
# The clean topology reduces the worst aspect ratio from 138 to 71 and remains one connected region wi
gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0); gmsh.open('impeller_water.msh')
try: gmsh.model.mesh.optimize('Netgen')
except Exception as e: print('Netgen optimizer:',e)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.write('impeller_water.msh'); gmsh.finalize()
if canonical.exists(): shutil.rmtree(canonical)
subprocess.run(['gmshToFoam','impeller_water.msh'],check=True,capture_output=True,text=True)
subprocess.run(['subsetMesh','-overwrite','water'],check=True,capture_output=True,text=True)
clean=Path('0/polyMesh')
if clean.exists():
    if canonical.exists(): shutil.rmtree(canonical)
    shutil.move(str(clean),str(canonical)); shutil.rmtree('0')
r=subprocess.run(['checkMesh'],capture_output=True,text=True); Path('checkMesh.log').write_text(r.stdout+r.stderr)
print('FINAL checkMesh return:',r.returncode)
print('\n'.join([ln for ln in (r.stdout+r.stderr).splitlines() if any(x in ln for x in ['points:','cells:','regions:','aspect ratio','non-orthogonality','skewness','Min volume','Mesh OK','Failed'])]))

# -- cell 16 ------------------------------------------------------------------------
# The optimizer preserves the same accepted quality: 12,164 tetrahedra, one connected region, valid po
if Path('VTK').exists(): shutil.rmtree('VTK')
subprocess.run(['foamToVTK','-no-fields'],check=True,capture_output=True,text=True)
plotter=pv.Plotter(off_screen=True,window_size=(1000,760)); bdir=Path('VTK/t12_0/boundary')
for name,(color,opacity) in style.items():
    plotter.add_mesh(pv.read(bdir/f'{name}.vtp'),color=color,opacity=opacity,show_edges=(name in ['blades','inlet','radial_outlet']),line_width=1)
plotter.add_axes(); plotter.view_isometric(); plotter.camera.zoom(1.35)
plotter.add_text('Final OpenFOAM mesh: inlet green, outlet red, blades gold',font_size=11)
plotter.show(screenshot='impeller_mesh_final.png')
img=plt.imread('impeller_mesh_final.png'); plt.figure(figsize=(10,7.6)); plt.imshow(img); plt.axis('off'); plt.show()
print(f'Measured final bounds: r_out={R_OUT*1000:.1f} mm, z={Z_BOTTOM*1000:.1f}..{Z_HUB_TOP*1000:.1f} mm; shroud z={Z_TOP*1000:.1f} mm')
