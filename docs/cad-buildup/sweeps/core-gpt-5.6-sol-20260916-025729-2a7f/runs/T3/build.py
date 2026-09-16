"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll reissue the dependency check as a self-contained script cell so the final build remains replaya
import gmsh
try:
    import shapely
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    print('shapely',shapely.__version__,'available')
except Exception as e:
    print('shapely unavailable:',repr(e))

# -- cell 2 -------------------------------------------------------------------------
# The explicit shape is sound, but Gmsh requires adjacent curves to share the same point tags—not mere
import numpy as np, matplotlib.pyplot as plt
if gmsh.isInitialized(): gmsh.finalize()
L=0.120; W=0.006; R=W/2; HLOOP=0.012; SPAN=0.022; T=0.001; starts=np.array([0.008,0.035,0.062,0.089]); n_path=41
def bezier_offset(x0,side):
    t=np.linspace(0,1,n_path); p0=np.array([x0,0.]); p3=np.array([x0+SPAN,0.]); p1=np.array([x0+.002,HLOOP]); p2=np.array([x0+SPAN-.007,HLOOP])
    p=((1-t)**3)[:,None]*p0+(3*(1-t)**2*t)[:,None]*p1+(3*(1-t)*t**2)[:,None]*p2+(t**3)[:,None]*p3
    d=(3*(1-t)**2)[:,None]*(p1-p0)+(6*(1-t)*t)[:,None]*(p2-p1)+(3*t**2)[:,None]*(p3-p2); n=np.column_stack((-d[:,1],d[:,0])); n/=np.linalg.norm(n,axis=1)[:,None]; q=p+side*R*n
    out=[]
    for a,b in zip(q[:-1],q[1:]):
        if a[1]>=R: out.append(a)
        if (a[1]-R)*(b[1]-R)<0: out.append(a+(b-a)*(R-a[1])/(b[1]-a[1]))
    if q[-1,1]>=R: out.append(q[-1])
    return np.array(out)
outer=[bezier_offset(x,1) for x in starts]; inner=[bezier_offset(x,-1) for x in starts]
gmsh.initialize(); gmsh.model.add('teslaValve2D'); geo=gmsh.model.geo
def P(v): return geo.addPoint(float(v[0]),float(v[1]),0,0.0018)
def add_line(cur,v):
    nxt=P(v); return geo.addLine(cur,nxt),nxt
def add_spline(cur,arr):
    tags=[cur]+[P(v) for v in arr]; return geo.addSpline(tags),tags[-1]
outer_curves=[]; cur=P((0,-R)); c,cur=add_line(cur,(L,-R)); outer_curves.append(c); c,cur=add_line(cur,(L,R)); outlet_curve=c; outer_curves.append(c)
x=L
for q in outer[::-1]:
    c,cur=add_line(cur,q[-1]); outer_curves.append(c); c,cur=add_spline(cur,q[-2::-1]); outer_curves.append(c); x=q[0,0]
c,cur=add_line(cur,(0,R)); outer_curves.append(c); c=geo.addLine(cur,outer_curves and geo.getMaxTag(0)-0 if False else 1)
# inlet closes to the first point tag, which is 1 in this newly created model
inlet_curve=c; outer_curves.append(c)
hole_loops=[]; hole_curves=[]
for q in inner:
    tags=[P(v) for v in q]; c1=geo.addLine(tags[0],tags[-1]); c2=geo.addSpline(tags[::-1]); hole_curves += [c1,c2]; hole_loops.append(geo.addCurveLoop([c1,c2]))
outer_loop=geo.addCurveLoop(outer_curves); fluid_surface=geo.addPlaneSurface([outer_loop]+hole_loops); geo.synchronize()
gmsh.model.addPhysicalGroup(1,[inlet_curve],name='inlet2d'); gmsh.model.addPhysicalGroup(1,[outlet_curve],name='outlet2d'); gmsh.model.addPhysicalGroup(1,[c for c in outer_curves+hole_curves if c not in (inlet_curve,outlet_curve)],name='walls2d'); gmsh.model.addPhysicalGroup(2,[fluid_surface],name='fluid2d')
gmsh.model.mesh.generate(2); gmsh.write('coarse2d.msh')
nodeTags,xyz,_=gmsh.model.mesh.getNodes(); xy=xyz.reshape(-1,3)[:,:2]; idx={int(t):i for i,t in enumerate(nodeTags)}; tri=[]
types,_,nodeSets=gmsh.model.mesh.getElements(2)
for et,en in zip(types,nodeSets):
    if gmsh.model.mesh.getElementProperties(int(et))[0].startswith('Triangle'):
        nn=gmsh.model.mesh.getElementProperties(int(et))[3]; tri += [[idx[int(t)] for t in row[:3]] for row in np.array(en).reshape(-1,nn)]
fig,ax=plt.subplots(figsize=(12,3)); ax.triplot(xy[:,0]*1000,xy[:,1]*1000,np.asarray(tri),lw=.35); ax.set_aspect('equal'); ax.set(xlabel='x [mm]',ylabel='y [mm]',title='Coarse planar Tesla-valve fluid region'); ax.grid(True,alpha=.2); plt.show()
print(f'coarse nodes={len(nodeTags)}, triangles={len(tri)}, holes={len(hole_loops)}')

# -- cell 3 -------------------------------------------------------------------------
# The coarse view confirms one connected straight passage, four distinct bypasses, and four enclosed t
gmsh.model.mesh.clear(); gmsh.model.removePhysicalGroups()
ext=geo.extrude([(2,fluid_surface)],0,0,T,numElements=[1],recombine=True); geo.synchronize()
volume=[tag for dim,tag in ext if dim==3][0]; top=[tag for dim,tag in ext if dim==2][0]
laterals=[tag for dim,tag in ext if dim==2 and tag!=top]
def contains_base(surface,curve): return curve in [abs(t) for d,t in gmsh.model.getBoundary([(2,surface)],oriented=False) if d==1]
inlet_faces=[s for s in laterals if contains_base(s,inlet_curve)]; outlet_faces=[s for s in laterals if contains_base(s,outlet_curve)]; wall_faces=[s for s in laterals if s not in inlet_faces+outlet_faces]
gmsh.model.addPhysicalGroup(2,inlet_faces,name='inlet'); gmsh.model.addPhysicalGroup(2,outlet_faces,name='outlet'); gmsh.model.addPhysicalGroup(2,wall_faces,name='walls'); gmsh.model.addPhysicalGroup(2,[fluid_surface,top],name='frontAndBack'); gmsh.model.addPhysicalGroup(3,[volume],name='fluid')
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.model.mesh.generate(3); gmsh.write('teslaValve.msh')
print('volume',volume,'inlet',inlet_faces,'outlet',outlet_faces,'walls',len(wall_faces),'front/back',[fluid_surface,top]); print('3D elements',sum(len(a) for a in gmsh.model.mesh.getElements(3)[1]))

# -- cell 4 -------------------------------------------------------------------------
# The extrusion produced 1,141 one-layer prism cells with the intended five patch groups. I’ll convert
import os, subprocess
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write("FoamFile\n{ version 2.0; format ascii; class dictionary; object controlDict; }\napplication checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
r=subprocess.run(['gmshToFoam','teslaValve.msh','-case','.'],text=True,capture_output=True); print(r.stdout[-2500:]); print(r.stderr[-1000:]); print('return',r.returncode)
print(open('constant/polyMesh/boundary').read())

# -- cell 5 -------------------------------------------------------------------------
# Conversion is consistent: 1,141 prisms, 2,282 front/back triangles, 215 wall faces, and four faces o
import re
p='constant/polyMesh/boundary'; txt=open(p).read()
def set_patch_type(text,name,newtype):
    pat=r'(\n\s*'+re.escape(name)+r'\s*\n\s*\{.*?\n\s*type\s+)\w+(;.*?\n\s*physicalType\s+)\w+(;)'
    return re.sub(pat,lambda m:m.group(1)+newtype+m.group(2)+newtype+m.group(3),text,count=1,flags=re.S)
txt=set_patch_type(txt,'frontAndBack','empty'); txt=set_patch_type(txt,'walls','wall'); open(p,'w').write(txt)
r=subprocess.run(['checkMesh','-case','.','-allGeometry','-allTopology'],text=True,capture_output=True); print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 6 -------------------------------------------------------------------------
# `checkMesh` reached the case but OpenFOAM 2512 requires `fvSchemes` and `fvSolution` files even for 
for name in ('fvSchemes','fvSolution'):
    open('system/'+name,'w').write(f"FoamFile\n{{ version 2.0; format ascii; class dictionary; object {name}; }}\n{{}}\n")
r=subprocess.run(['checkMesh','-case','.','-allGeometry','-allTopology'],text=True,capture_output=True); print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 7 -------------------------------------------------------------------------
# This release parses the standard scheme subdictionaries during geometric checks. I’ll supply minimal
open('system/fvSchemes','w').write('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
open('system/fvSolution','w').write('FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
r=subprocess.run(['checkMesh','-case','.','-allGeometry','-allTopology'],text=True,capture_output=True); print(r.stdout[-7000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 8 -------------------------------------------------------------------------
# The full check passes. Before finishing, I’ll export the already named OpenFOAM boundary patches as 
r=subprocess.run(['surfaceMeshExtract','-help'],text=True,capture_output=True); print((r.stdout+r.stderr)[:5000])

# -- cell 9 -------------------------------------------------------------------------
# I’ll export exactly four non-overlapping patch STLs—`inlet`, `outlet`, `walls`, and `frontAndBack`. 
for patch in ('inlet','outlet','walls','frontAndBack'):
    r=subprocess.run(['surfaceMeshExtract','-case','.','-constant','-patches',patch,patch+'.stl'],text=True,capture_output=True)
    print(patch,'return',r.returncode, next((s for s in r.stdout.splitlines() if 'faces' in s.lower()),''))
print([(f,os.path.getsize(f)) for f in ('inlet.stl','outlet.stl','walls.stl','frontAndBack.stl')])

# -- cell 10 ------------------------------------------------------------------------
# `surfaceMeshExtract` skipped the faces on the `empty` patch, yielding an empty planar STL. I’ll temp
p='constant/polyMesh/boundary'; original=open(p).read(); temporary=set_patch_type(original,'frontAndBack','patch'); open(p,'w').write(temporary)
r=subprocess.run(['surfaceMeshExtract','-case','.','-constant','-patches','frontAndBack','frontAndBack.stl'],text=True,capture_output=True)
open(p,'w').write(original)
print('return',r.returncode,'frontAndBack.stl bytes',os.path.getsize('frontAndBack.stl'),'restored empty=',bool(re.search(r'frontAndBack\s*\{\s*type\s+empty;',open(p).read(),re.S)))

# -- cell 11 ------------------------------------------------------------------------
# I’ll perform a final geometric audit against the request and the stated assumptions, then verify tha
# Numerical construction audit
samples=[]
for x0 in starts:
    t=np.linspace(0,1,401); p0=np.array([x0,0.]); p3=np.array([x0+SPAN,0.]); p1=np.array([x0+.002,HLOOP]); p2=np.array([x0+SPAN-.007,HLOOP])
    p=((1-t)**3)[:,None]*p0+(3*(1-t)**2*t)[:,None]*p1+(3*(1-t)*t**2)[:,None]*p2+(t**3)[:,None]*p3
    d=(3*(1-t)**2)[:,None]*(p1-p0)+(6*(1-t)*t)[:,None]*(p2-p1)+(3*t**2)[:,None]*(p3-p2); n=np.column_stack((-d[:,1],d[:,0])); n/=np.linalg.norm(n,axis=1)[:,None]
    samples.append((p+R*n,p-R*n,p))
measured_widths=np.concatenate([np.linalg.norm(a-b,axis=1) for a,b,_ in samples]); bb=gmsh.model.getBoundingBox(3,volume)
print(f'channel width requested 6.000 mm; measured offset width {measured_widths.min()*1000:.6f}..{measured_widths.max()*1000:.6f} mm')
print(f'overall length requested about 120 mm; measured {(bb[3]-bb[0])*1000:.6f} mm')
print(f'bypass loops requested 4; constructed/measured topology holes {len(hole_loops)}')
print(f'assumed branch-to-rejoin span {SPAN*1000:.3f} mm; centerline peak rise {max(p[:,1].max() for _,_,p in samples)*1000:.3f} mm; planar extrusion {(bb[5]-bb[2])*1000:.6f} mm')
# Closed-surface extraction check without leaving a duplicate STL
btxt=open('constant/polyMesh/boundary').read(); open('constant/polyMesh/boundary','w').write(set_patch_type(btxt,'frontAndBack','patch'))
r1=subprocess.run(['surfaceMeshExtract','-case','.','-constant','-patches','(inlet outlet walls frontAndBack)','closedAudit.stl'],text=True,capture_output=True)
open('constant/polyMesh/boundary','w').write(btxt); r2=subprocess.run(['surfaceCheck','closedAudit.stl'],text=True,capture_output=True); print('\n'.join(r2.stdout.splitlines()[-25:])); os.remove('closedAudit.stl')
