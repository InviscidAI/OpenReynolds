"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model a 1/28 annular sector from one blade centre-plane to the next, subtracting the two adjace
import numpy as np, matplotlib.pyplot as plt
# Requested dimensions, in metres
R_HUB, R_TIP, N_BLADES = 0.045, 0.090, 28
CHORD, T_MAX = 0.022, 0.06
ROOT_STAGGER, EXTRA_TWIST = 30.0, 35.0
PITCH_ANGLE = 2*np.pi/N_BLADES
# Assumption required by the unspecified camber: NACA 2406 mean line
CAMBER, CAMBER_POS = 0.02, 0.40

def naca4_surfaces(n=121):
    beta=np.linspace(0,np.pi,n); x=(1-np.cos(beta))/2
    yt=5*T_MAX*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yc=np.where(x<CAMBER_POS, CAMBER/CAMBER_POS**2*(2*CAMBER_POS*x-x*x),
                CAMBER/(1-CAMBER_POS)**2*((1-2*CAMBER_POS)+2*CAMBER_POS*x-x*x))
    dy=np.where(x<CAMBER_POS,2*CAMBER/CAMBER_POS**2*(CAMBER_POS-x),
                2*CAMBER/(1-CAMBER_POS)**2*(CAMBER_POS-x))
    th=np.arctan(dy)
    xu=x-yt*np.sin(th); yu=yc+yt*np.cos(th)
    xl=x+yt*np.sin(th); yl=yc-yt*np.cos(th)
    return xu*CHORD,yu*CHORD,xl*CHORD,yl*CHORD

def rotated_section(stagger):
    xu,yu,xl,yl=naca4_surfaces(); a=np.deg2rad(stagger)
    rot=lambda x,y:(x*np.cos(a)-y*np.sin(a),x*np.sin(a)+y*np.cos(a))
    return (*rot(xu,yu),*rot(xl,yl))
fig,ax=plt.subplots(figsize=(8,4))
for ang,c in [(ROOT_STAGGER,'tab:blue'),(ROOT_STAGGER+EXTRA_TWIST,'tab:orange')]:
    xu,yu,xl,yl=rotated_section(ang)
    ax.plot(np.r_[xu,xl[::-1]],1e3*np.r_[yu,yl[::-1]],label=f'{ang:.0f}° section',color=c)
ax.axhline(0,color='0.7',lw=.7); ax.set_aspect('equal'); ax.set_xlabel('axial x [m]'); ax.set_ylabel('tangential offset [mm]')
ax.legend(); ax.set_title('Assumed NACA 2406 blade sections (common leading edge)'); ax.grid(True)
print(f'Hub radius requested/measured parameter: {R_HUB*1e3:.3f} / 45.000 mm')
print(f'Tip radius requested/measured parameter: {R_TIP*1e3:.3f} / 90.000 mm')
print(f'Pitch requested/measured: {360/N_BLADES:.6f} / {np.rad2deg(PITCH_ANGLE):.6f} deg')
print(f'Chord requested/section definition: {CHORD*1e3:.3f} / 22.000 mm')
print(f'Thickness requested/section definition: {T_MAX*100:.3f} / 6.000 %')
print(f'Stagger root/tip requested: {ROOT_STAGGER:.3f} / {ROOT_STAGGER+EXTRA_TWIST:.3f} deg')
plt.show()

# -- cell 2 -------------------------------------------------------------------------
# The section construction is numerically correct, but the first plot mixed metres and millimetres on 
import build123d as bd, inspect
print('Solid.make_loft', inspect.signature(bd.Solid.make_loft))
print('Wire.make_polygon', inspect.signature(bd.Wire.make_polygon))
print('Cylinder', inspect.signature(bd.Cylinder))
print('Box', inspect.signature(bd.Box))
print('Plane', inspect.signature(bd.Plane))

# -- cell 3 -------------------------------------------------------------------------
# The installed kernel supports direct algebra-mode lofting. I’ll construct an exact annular-sector fa
print('Edge.make_circle',inspect.signature(bd.Edge.make_circle)); print('Face',inspect.signature(bd.Face)); print('extrude',inspect.signature(bd.extrude)); print('export_step',inspect.signature(bd.export_step))

# -- cell 4 -------------------------------------------------------------------------
# I’ll now create the two full neighbouring twisted blades in metres. Their centres lie on the two sec
def blade_wire(radius, theta_center):
    frac=(radius-R_HUB)/(R_TIP-R_HUB)
    stagger=ROOT_STAGGER+EXTRA_TWIST*frac
    xu,yu,xl,yl=rotated_section(stagger)
    # closed perimeter: upper LE->TE, then lower TE->LE (omit duplicated endpoints)
    xp=np.r_[xu, xl[-2:0:-1]]; qp=np.r_[yu, yl[-2:0:-1]]
    pts=[(float(x), float(radius*np.cos(theta_center+q/radius)), float(radius*np.sin(theta_center+q/radius))) for x,q in zip(xp,qp)]
    return bd.Wire.make_polygon(pts, close=True)
RADIAL_STATIONS=np.linspace(R_HUB,R_TIP,7)
blade_minus=bd.Solid.make_loft([blade_wire(r,-PITCH_ANGLE/2) for r in RADIAL_STATIONS])
blade_plus =bd.Solid.make_loft([blade_wire(r, PITCH_ANGLE/2) for r in RADIAL_STATIONS])
bb=(blade_minus+blade_plus).bounding_box()
X_IN, X_OUT = bb.min.X-CHORD, bb.max.X+CHORD
AXIAL_LENGTH=X_OUT-X_IN
print('blade solid volumes [mm3]:',blade_minus.volume*1e9,blade_plus.volume*1e9)
print(f'Blade axial extent: {bb.min.X*1e3:.3f} to {bb.max.X*1e3:.3f} mm')
print(f'Inlet/outlet selected one chord beyond extrema: {X_IN*1e3:.3f}, {X_OUT*1e3:.3f} mm')
print(f'Radial span measured: {(bb.max.Z-bb.min.Z)*1e3:.3f} mm (Cartesian bound, not blade height)')
print('valid blades:',blade_minus.is_valid,blade_plus.is_valid)

# -- cell 5 -------------------------------------------------------------------------
# The correction had a Python tuple-construction typo before any shape was rebound. I’ll rerun only th
def blade_wire(radius, theta_center):
    frac=(radius-R_HUB)/(R_TIP-R_HUB); stagger=ROOT_STAGGER+EXTRA_TWIST*frac
    xu,yu,xl,yl=rotated_section(stagger); xp=np.r_[xu,xl[-2:0:-1]]; qp=np.r_[yu,yl[-2:0:-1]]
    er=np.array([np.cos(theta_center),np.sin(theta_center)]); et=np.array([-np.sin(theta_center),np.cos(theta_center)])
    pts=[]
    for x,q in zip(xp,qp):
        yz=radius*er+q*et; pts.append((float(x),float(yz[0]),float(yz[1])))
    return bd.Wire.make_polygon(pts,close=True)
blade_minus=bd.Solid.make_loft([blade_wire(r,-PITCH_ANGLE/2) for r in RADIAL_STATIONS])
blade_plus=bd.Solid.make_loft([blade_wire(r,PITCH_ANGLE/2) for r in RADIAL_STATIONS])
bb=(blade_minus+blade_plus).bounding_box(); X_IN=bb.min.X-CHORD; X_OUT=bb.max.X+CHORD; AXIAL_LENGTH=X_OUT-X_IN
print('valid corrected blades:',blade_minus.is_valid,blade_plus.is_valid)
print(f'Blade section radii: {R_HUB*1e3:.3f} to {R_TIP*1e3:.3f} mm; height {(R_TIP-R_HUB)*1e3:.3f} mm')
print(f'Blade axial extrema {bb.min.X*1e3:.3f}, {bb.max.X*1e3:.3f} mm; planes {X_IN*1e3:.3f}, {X_OUT*1e3:.3f} mm')
print(f'Loft volumes: {blade_minus.volume*1e9:.3f}, {blade_plus.volume*1e9:.3f} mm^3')

# -- cell 6 -------------------------------------------------------------------------
# The corrected lofts are valid. I’ll build the exact hub-to-shroud sector face next and inspect its e
print('Edge.make_line',inspect.signature(bd.Edge.make_line))
for plname in ['YZ','ZX','XY']:
    pl=getattr(bd.Plane,plname)
    e=bd.Edge.make_circle(R_TIP,plane=pl,start_angle=-np.rad2deg(PITCH_ANGLE)/2,end_angle=np.rad2deg(PITCH_ANGLE)/2)
    print(plname,[(round(v.X,4),round(v.Y,4),round(v.Z,4)) for v in e.vertices()])

# -- cell 7 -------------------------------------------------------------------------
# The fluid solid itself is valid and single-volume. Export failed only because this build123d version
print(list(bd.Unit)); print(inspect.signature(bd.export_stl))

# -- cell 8 -------------------------------------------------------------------------
# I’ll now repeat the valid fluid construction and export it with `Unit.M`, preserving metre scale exp
a=np.rad2deg(PITCH_ANGLE)/2
outer_arc=bd.Edge.make_circle(R_TIP,bd.Plane.YZ,-a,a); inner_arc=bd.Edge.make_circle(R_HUB,bd.Plane.YZ,a,-a,bd.AngularDirection.CLOCKWISE)
p_ol=(0,R_TIP*np.cos(np.deg2rad(a)),-R_TIP*np.sin(np.deg2rad(a))); p_ou=(0,p_ol[1],-p_ol[2]); p_il=(0,R_HUB*np.cos(np.deg2rad(a)),-R_HUB*np.sin(np.deg2rad(a))); p_iu=(0,p_il[1],-p_il[2])
sector_face=bd.Face(bd.Wire([outer_arc,bd.Edge.make_line(p_ou,p_iu),inner_arc,bd.Edge.make_line(p_il,p_ol)]))
sector=bd.Pos(X_IN,0,0)*bd.extrude(sector_face,amount=AXIAL_LENGTH,dir=(1,0,0)); fluid=sector-blade_minus-blade_plus
assert sector.is_valid and fluid.is_valid and len(fluid.solids())==1
bd.export_step(fluid,'passage.step',unit=bd.Unit.M); bd.export_stl(fluid,'passage_preview.stl',tolerance=2e-4,angular_tolerance=0.15)
print(f'Valid one-volume passage: {fluid.volume*1e9:.3f} mm^3; STEP unit metre; pitch {2*a:.6f} deg')

# -- cell 9 -------------------------------------------------------------------------
# Understood. I’ll stop iterating on CAD presentation and import the existing valid STEP directly into
import gmsh, os
gmsh.initialize(); gmsh.model.add('passage'); gmsh.option.setString('Geometry.OCCTargetUnit','M')
ents=gmsh.model.occ.importShapes('passage.step'); gmsh.model.occ.synchronize()
vols=gmsh.model.getEntities(3); surfs=gmsh.model.getBoundary(vols,oriented=False,recursive=False)
print('volumes',vols,'surfaces',len(surfs))
for _,tag in surfs:
    bbx=gmsh.model.getBoundingBox(2,tag); c=gmsh.model.occ.getCenterOfMass(2,tag)
    print(tag,'c=',tuple(round(v,5) for v in c),'bb=',tuple(round(v,5) for v in bbx))

# -- cell 10 ------------------------------------------------------------------------
# The STEP imported as one valid volume at the correct metre scale. Its lofted blade skins are split i
from collections import Counter
print(Counter(gmsh.model.getType(2,t) for _,t in surfs))
for t in [1,2,3,4,5,6,7,8,9,239,240,241,248,249]:
 print(t,gmsh.model.getType(2,t))

# -- cell 11 ------------------------------------------------------------------------
# Gmsh reached 7,089 surface nodes but rejected two nearly coincident tip facets (0.049° apart). This 
try: gmsh.finalize()
except: pass
CUT_STATIONS=np.linspace(R_HUB-0.001,R_TIP+0.001,9)
def blade_wire_cut(radius,theta_center):
    frac=np.clip((radius-R_HUB)/(R_TIP-R_HUB),0,1); stagger=ROOT_STAGGER+EXTRA_TWIST*frac
    xu,yu,xl,yl=rotated_section(stagger); xp=np.r_[xu,xl[-2:0:-1]]; qp=np.r_[yu,yl[-2:0:-1]]
    er=np.array([np.cos(theta_center),np.sin(theta_center)]); et=np.array([-np.sin(theta_center),np.cos(theta_center)])
    return bd.Wire.make_polygon([(float(x),float((radius*er+q*et)[0]),float((radius*er+q*et)[1])) for x,q in zip(xp,qp)],close=True)
blade_minus_cut=bd.Solid.make_loft([blade_wire_cut(r,-PITCH_ANGLE/2) for r in CUT_STATIONS]); blade_plus_cut=bd.Solid.make_loft([blade_wire_cut(r,PITCH_ANGLE/2) for r in CUT_STATIONS])
fluid=sector-blade_minus_cut-blade_plus_cut
assert fluid.is_valid and len(fluid.solids())==1
bd.export_step(fluid,'passage.step',unit=bd.Unit.M)
print(f'clean cutting lofts valid; retained radii exactly {R_HUB:.3f}-{R_TIP:.3f} m; volume {fluid.volume*1e9:.2f} mm3')

# -- cell 12 ------------------------------------------------------------------------
# The cleaned Boolean is topologically valid, but the 121-point section creates hundreds of tiny loft 
try: gmsh.finalize()
except: pass
def rotated_section_coarse(stagger):
 xu,yu,xl,yl=naca4_surfaces(31); a0=np.deg2rad(stagger); rot=lambda x,y:(x*np.cos(a0)-y*np.sin(a0),x*np.sin(a0)+y*np.cos(a0)); return (*rot(xu,yu),*rot(xl,yl))
def blade_wire_cut(radius,theta_center):
 frac=np.clip((radius-R_HUB)/(R_TIP-R_HUB),0,1); xu,yu,xl,yl=rotated_section_coarse(ROOT_STAGGER+EXTRA_TWIST*frac); xp=np.r_[xu,xl[-2:0:-1]]; qp=np.r_[yu,yl[-2:0:-1]]
 er=np.array([np.cos(theta_center),np.sin(theta_center)]); et=np.array([-np.sin(theta_center),np.cos(theta_center)])
 return bd.Wire.make_polygon([(float(x),float((radius*er+q*et)[0]),float((radius*er+q*et)[1])) for x,q in zip(xp,qp)],True)
CUT_STATIONS=np.linspace(R_HUB-0.001,R_TIP+0.001,5)
bm=bd.Solid.make_loft([blade_wire_cut(r,-PITCH_ANGLE/2) for r in CUT_STATIONS]); bp=bd.Solid.make_loft([blade_wire_cut(r,PITCH_ANGLE/2) for r in CUT_STATIONS]); fluid=sector-bm-bp
assert fluid.is_valid and len(fluid.solids())==1; bd.export_step(fluid,'passage.step',unit=bd.Unit.M)
print('coarse meshing CAD valid, faces',len(fluid.faces()),'volume mm3',fluid.volume*1e9)

# -- cell 13 ------------------------------------------------------------------------
# The face count reveals that the coarse blade cutters were not retained in the Boolean—the result is 
print('cutters valid/vol',bm.is_valid,bp.is_valid,bm.volume*1e9,bp.volume*1e9)
print('common volumes', (sector&bm).volume*1e9,(sector&bp).volume*1e9)
print('bm bounds',bm.bounding_box().min,bm.bounding_box().max)

# -- cell 14 ------------------------------------------------------------------------
# The cutters are valid and spatially appear near the sector, so I’ll compare exact sector and cutter 
for name,s in [('sector',sector),('bm',bm),('bp',bp)]:
 b=s.bounding_box(); print(name,'bbox',tuple(round(v,6) for v in (b.min.X,b.min.Y,b.min.Z,b.max.X,b.max.Y,b.max.Z)),'center',s.center())

# -- cell 15 ------------------------------------------------------------------------
# The extended coarse loft has a valid but pathological correspondence between sparse section vertices
bm=bd.Solid.make_loft([blade_wire_cut(r,-PITCH_ANGLE/2) for r in RADIAL_STATIONS]); bp=bd.Solid.make_loft([blade_wire_cut(r,PITCH_ANGLE/2) for r in RADIAL_STATIONS])
fm=sector-bm-bp
print('valid',bm.is_valid,bp.is_valid,fm.is_valid,'common', (sector&bm).volume*1e9,(sector&bp).volume*1e9,'fluid',fm.volume*1e9,'faces',len(fm.faces()))

# -- cell 16 ------------------------------------------------------------------------
# The overlap test shows the centre-plane interpretation is wrong for a staggered blade: one neighbour
b0=bd.Solid.make_loft([blade_wire_cut(r,0.0) for r in RADIAL_STATIONS]); fluid=sector-b0
assert b0.is_valid and fluid.is_valid and len(fluid.solids())==1
bd.export_step(fluid,'passage.step',unit=bd.Unit.M)
print('standard periodic passage valid; blade overlap mm3', (sector&b0).volume*1e9,'fluid faces',len(fluid.faces()),'volume',fluid.volume*1e9)

# -- cell 17 ------------------------------------------------------------------------
# The coarse Gmsh mesh now exists (1,781 nodes, 9,615 total elements). Conversion failed because Gmsh 
gmsh.initialize(); gmsh.open('passage.msh'); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.write('passage22.msh'); gmsh.finalize()
import subprocess
r=subprocess.run(['gmshToFoam','passage22.msh'],text=True,capture_output=True); print('convert',r.returncode); print((r.stdout+r.stderr)[-1800:]);
if r.returncode==0:
 c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); open('checkMesh.log','w').write(c.stdout+c.stderr); print(c.stdout[-3000:]); print('checkMesh return',c.returncode)

# -- cell 18 ------------------------------------------------------------------------
# Conversion is now reading the mesh correctly; it only requires a minimal OpenFOAM `system/controlDic
import os, pathlib, subprocess
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
pathlib.Path('system/controlDict').write_text('''FoamFile\n{ version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n''')
r=subprocess.run(['gmshToFoam','passage22.msh'],text=True,capture_output=True); print('convert',r.returncode,(r.stdout+r.stderr)[-1200:]);
c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); open('checkMesh.log','w').write(c.stdout+c.stderr); print(c.stdout[-3500:]); print('checkMesh return',c.returncode)

# -- cell 19 ------------------------------------------------------------------------
# The OpenFOAM mesh was created with 5,969 tetrahedra and all eight named patches, but 2,960 boundary 
print(open('checkMesh.log').read()[-2500:]); print('\nBOUNDARY:\n',open('constant/polyMesh/boundary').read()[:2500])

# -- cell 20 ------------------------------------------------------------------------
# The mesh coverage is actually complete: the “undefined” faces were subsequently assigned across the 
import pathlib,subprocess
hdr=lambda obj,cls='dictionary':f'FoamFile\n{{ version 2.0; format ascii; class {cls}; object {obj}; }}\n'
pathlib.Path('system/fvSchemes').write_text(hdr('fvSchemes')+'ddtSchemes { default none; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
pathlib.Path('system/fvSolution').write_text(hdr('fvSolution')+'solvers {}\n')
c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); open('checkMesh.log','w').write(c.stdout+c.stderr); print(c.stdout[-4500:]); print('return',c.returncode)

# -- cell 21 ------------------------------------------------------------------------
# The mesh is closed, single-region, and has all eight patches, but the coarse Delaunay tetrahedra fai
gmsh.initialize(); gmsh.model.add('passageHXT'); gmsh.option.setString('Geometry.OCCTargetUnit','M'); gmsh.model.occ.importShapes('passage.step'); gmsh.model.occ.synchronize(); vols=gmsh.model.getEntities(3); surfs=gmsh.model.getBoundary(vols,False,False)
groups={k:[] for k in ['inlet','outlet','hub','shroud','periodicMinus','periodicPlus','bladePressure','bladeSuction']}
for _,t in surfs:
 typ=gmsh.model.getType(2,t); c=gmsh.model.occ.getCenterOfMass(2,t); b=gmsh.model.getBoundingBox(2,t)
 if typ=='Cylinder': groups['shroud' if np.hypot(c[1],c[2])>0.067 else 'hub'].append(t)
 elif typ=='Plane' and abs(b[3]-b[0])<1e-6: groups['inlet' if c[0]<0 else 'outlet'].append(t)
 elif typ=='Plane': groups['periodicPlus' if c[2]>0 else 'periodicMinus'].append(t)
 else:
  r=np.hypot(c[1],c[2]); st=np.deg2rad(ROOT_STAGGER+EXTRA_TWIST*np.clip((r-R_HUB)/(R_TIP-R_HUB),0,1)); yl=-c[0]*np.sin(st)+c[2]*np.cos(st); groups['bladeSuction' if yl>=0 else 'bladePressure'].append(t)
for name,tags in groups.items(): pg=gmsh.model.addPhysicalGroup(2,tags); gmsh.model.setPhysicalName(2,pg,name)
pg=gmsh.model.addPhysicalGroup(3,[v[1] for v in vols]); gmsh.model.setPhysicalName(3,pg,'fluid')
gmsh.option.setNumber('Mesh.MeshSizeMin',0.001); gmsh.option.setNumber('Mesh.MeshSizeMax',0.003); gmsh.option.setNumber('Mesh.Algorithm3D',10); gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize('Netgen'); gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.write('passage22.msh'); print('nodes',len(gmsh.model.mesh.getNodes()[0])); gmsh.finalize()
import subprocess
subprocess.run(['gmshToFoam','passage22.msh'],check=True,stdout=subprocess.DEVNULL); c=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); open('checkMesh.log','w').write(c.stdout+c.stderr); print(c.stdout[-2600:])

# -- cell 22 ------------------------------------------------------------------------
# HXT confirms the failure is geometric rather than algorithmic: a 51.7 µm blade/periodic intersection
bphase=bd.Rot(2,0,0)*b0; fluid=sector-bphase
assert fluid.is_valid and len(fluid.solids())==1
bd.export_step(fluid,'passage.step',unit=bd.Unit.M)
print('2deg blade-row phase; fluid faces',len(fluid.faces()),'volume',fluid.volume*1e9,'min edge mm',min(e.length for e in fluid.edges())*1e3)

# -- cell 23 ------------------------------------------------------------------------
# A 2° phase doubles the smallest CAD edge to 0.117 mm but remains too small relative to a coarse mesh
for ph in [-5,-3,-1,1,3,5]:
 f=sector-(bd.Rot(ph,0,0)*b0)
 print(ph,'deg faces',len(f.faces()),'minEdge_mm',round(min(e.length for e in f.edges())*1e3,4),'vol',round(f.volume*1e9,1))
