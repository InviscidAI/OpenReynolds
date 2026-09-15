"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid directly in metres as the 12 mm annular gap, subtracting 36 constant-8 mm tange
import build123d as bd
import numpy as np, pyvista as pv
from pathlib import Path
# Requested dimensions, converted once from mm to m
RO, RI, GAP, VANE_W, N = 0.160, 0.095, 0.012, 0.008, 36
PITCH = 360.0/N
annulus = bd.Cylinder(RO, GAP) - bd.Cylinder(RI, GAP)
base_vane = bd.Box(RO-RI, VANE_W, GAP, align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.MIN)).moved(bd.Pos(RI,0,0))
vanes = bd.Compound([base_vane.moved(bd.Rot(0,0,k*PITCH)) for k in range(N)])
fluid = annulus - vanes
print(f"requested/measured OD: {2*RO:.6f} / {fluid.bounding_box().max.X-fluid.bounding_box().min.X:.6f} m")
print(f"requested ID={2*RI:.6f} m, gap={GAP:.6f} m, vane width={VANE_W:.6f} m, count={N}, pitch={PITCH:.6f} deg")
print(f"fluid solids (passages): {len(fluid.solids())}; volume={fluid.volume:.9e} m^3")
verts, tris = fluid.tessellate(2e-3)
pts=np.array([[v.X,v.Y,v.Z] for v in verts]); faces=np.hstack([np.full((len(tris),1),3),np.array(tris)]).ravel()
pl=pv.Plotter(off_screen=True, window_size=(800,650)); pl.add_mesh(pv.PolyData(pts,faces), color='lightskyblue', show_edges=True); pl.view_isometric(); pl.add_axes(); pl.show(screenshot='coarse_geometry.png')
print('wrote coarse_geometry.png')

# -- cell 2 -------------------------------------------------------------------------
# The image confirms the intended annular gap and radial vane pattern. The CAD kernel reports one comp
tri_dir=Path('constant/triSurface'); tri_dir.mkdir(parents=True,exist_ok=True)
ai, ao=np.arcsin(VANE_W/(2*RI)),np.arcsin(VANE_W/(2*RO)); pitch=2*np.pi/N
patch_tris={n:[] for n in ['frictionFace1','frictionFace2','vanes','innerInlet','outerOutlet']}
def quad(store,a,b,c,d): store += [(a,b,c),(a,c,d)]
def xyz(r,t,z): return (r*np.cos(t),r*np.sin(t),z)
NS=12
for k in range(N):
    t0=k*pitch
    # Shared grid around one passage: lower-vane side, outlet arc, upper-vane side, inlet arc
    inn=[xyz(RI,t0+ai+(pitch-2*ai)*j/NS,0) for j in range(NS+1)]
    out=[xyz(RO,t0+ao+(pitch-2*ao)*j/NS,0) for j in range(NS+1)]
    for j in range(NS):
        # bottom is outward -z; top is outward +z
        quad(patch_tris['frictionFace1'],inn[j],inn[j+1],out[j+1],out[j])
        quad(patch_tris['frictionFace2'],xyz(RI,np.arctan2(inn[j][1],inn[j][0]),GAP),xyz(RO,np.arctan2(out[j][1],out[j][0]),GAP),xyz(RO,np.arctan2(out[j+1][1],out[j+1][0]),GAP),xyz(RI,np.arctan2(inn[j+1][1],inn[j+1][0]),GAP))
    # lower and upper vane walls
    quad(patch_tris['vanes'],xyz(RI,t0+ai,0),xyz(RO,t0+ao,0),xyz(RO,t0+ao,GAP),xyz(RI,t0+ai,GAP))
    quad(patch_tris['vanes'],xyz(RI,t0+pitch-ai,0),xyz(RI,t0+pitch-ai,GAP),xyz(RO,t0+pitch-ao,GAP),xyz(RO,t0+pitch-ao,0))
    for j in range(NS):
        quad(patch_tris['outerOutlet'],out[j],xyz(RO,np.arctan2(out[j+1][1],out[j+1][0]),0),xyz(RO,np.arctan2(out[j+1][1],out[j+1][0]),GAP),xyz(RO,np.arctan2(out[j][1],out[j][0]),GAP))
        quad(patch_tris['innerInlet'],inn[j],xyz(RI,np.arctan2(inn[j][1],inn[j][0]),GAP),xyz(RI,np.arctan2(inn[j+1][1],inn[j+1][0]),GAP),inn[j+1])
def write_stl(path,triangles):
    with open(path,'w') as f:
        f.write('solid patch\n')
        for a,b,c in triangles:
            n=np.cross(np.subtract(b,a),np.subtract(c,a)); n=n/np.linalg.norm(n)
            f.write(f' facet normal {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}\n  outer loop\n')
            for p in (a,b,c): f.write(f'   vertex {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}\n')
            f.write('  endloop\n endfacet\n')
        f.write('endsolid patch\n')
for name,tt in patch_tris.items(): write_stl(tri_dir/f'{name}.stl',tt)
print('STLs exported before volume meshing:',{k:len(v) for k,v in patch_tris.items()})
print(f'passage angles at ID/OD: {np.degrees(pitch-2*ai):.6f} / {np.degrees(pitch-2*ao):.6f} deg')

# -- cell 3 -------------------------------------------------------------------------
# The exact passage opening is 5.174° at the 190 mm bore and 7.135° at the 320 mm rim; this variation 
import gmsh
gmsh.initialize(); gmsh.model.add('brakeDiscAir')
geo=gmsh.model.geo; center=geo.addPoint(0,0,0)
bottom=[]; top=[]; vane_s=[]; inlet_s=[]; outlet_s=[]; vols=[]
for k in range(N):
    t=k*pitch
    p1=geo.addPoint(*xyz(RI,t+ai,0)); p2=geo.addPoint(*xyz(RO,t+ao,0))
    p3=geo.addPoint(*xyz(RO,t+pitch-ao,0)); p4=geo.addPoint(*xyz(RI,t+pitch-ai,0))
    c1=geo.addLine(p1,p2); c2=geo.addCircleArc(p2,center,p3); c3=geo.addLine(p3,p4); c4=geo.addCircleArc(p4,center,p1)
    geo.mesh.setTransfiniteCurve(c1,14); geo.mesh.setTransfiniteCurve(c3,14)
    geo.mesh.setTransfiniteCurve(c2,5); geo.mesh.setTransfiniteCurve(c4,5)
    loop=geo.addCurveLoop([c1,c2,c3,c4]); s=geo.addPlaneSurface([loop]); geo.mesh.setTransfiniteSurface(s); geo.mesh.setRecombine(2,s)
    ex=geo.extrude([(2,s)],0,0,GAP,numElements=[3],recombine=True)
    bottom.append(s); top.append(ex[0][1]); vols.append(ex[1][1])
    vane_s += [ex[2][1],ex[4][1]]; outlet_s.append(ex[3][1]); inlet_s.append(ex[5][1])
geo.synchronize()
for dim,tags,name in [(2,bottom,'frictionFace1'),(2,top,'frictionFace2'),(2,vane_s,'vanes'),(2,inlet_s,'innerInlet'),(2,outlet_s,'outerOutlet'),(3,vols,'fluid')]:
    pg=gmsh.model.addPhysicalGroup(dim,tags); gmsh.model.setPhysicalName(dim,pg,name)
gmsh.option.setNumber('Mesh.MshFileVersion',2.2); gmsh.option.setNumber('Mesh.Binary',0)
gmsh.model.mesh.generate(3); gmsh.write('brakeDiscAir.msh')
types,etags,_=gmsh.model.mesh.getElements(3); print('3D element types/counts:',[(t,len(e)) for t,e in zip(types,etags)])
print('nodes:',len(gmsh.model.mesh.getNodes()[0]),'physical surfaces:',[(x[1],gmsh.model.getPhysicalName(*x)) for x in gmsh.model.getPhysicalGroups(2)])
gmsh.finalize()

# -- cell 4 -------------------------------------------------------------------------
# The coarse mesh contains exactly 5,616 hexahedra and all five requested physical surface groups. I’l
import subprocess
r=subprocess.run(['gmshToFoam','brakeDiscAir.msh'],text=True,capture_output=True)
Path('log.gmshToFoam').write_text(r.stdout+r.stderr); print('gmshToFoam rc',r.returncode); print('\n'.join((r.stdout+r.stderr).splitlines()[-18:]))
r2=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True)
Path('log.checkMesh').write_text(r2.stdout+r2.stderr); print('checkMesh rc',r2.returncode); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-35:]))

# -- cell 5 -------------------------------------------------------------------------
# Conversion stopped before touching the mesh because an empty OpenFOAM case lacks `system/controlDict
sysdir=Path('system'); sysdir.mkdir(exist_ok=True)
(sysdir/'controlDict').write_text('''FoamFile\n{\n version 2.0;\n format ascii;\n class dictionary;\n object controlDict;\n}\napplication checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n''')
r=subprocess.run(['gmshToFoam','brakeDiscAir.msh'],text=True,capture_output=True); Path('log.gmshToFoam').write_text(r.stdout+r.stderr)
print('gmshToFoam rc',r.returncode,'; tail:'); print('\n'.join((r.stdout+r.stderr).splitlines()[-12:]))
r2=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); Path('log.checkMesh').write_text(r2.stdout+r2.stderr)
print('checkMesh rc',r2.returncode,'; tail:'); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-38:]))

# -- cell 6 -------------------------------------------------------------------------
# Mesh conversion succeeded. This OpenFOAM release also requires empty `fvSchemes` and `fvSolution` di
for fn,obj in [('fvSchemes','fvSchemes'),('fvSolution','fvSolution')]:
    (sysdir/fn).write_text(f'FoamFile\n{{\n version 2.0;\n format ascii;\n class dictionary;\n object {obj};\n}}\n')
r2=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); Path('log.checkMesh').write_text(r2.stdout+r2.stderr)
print('checkMesh rc',r2.returncode); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-55:]))

# -- cell 7 -------------------------------------------------------------------------
# The checker reads mandatory scheme subdictionaries even though it does not solve. I’ll supply inert 
(sysdir/'fvSchemes').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n''')
(sysdir/'fvSolution').write_text('''FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n''')
r2=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True); Path('log.checkMesh').write_text(r2.stdout+r2.stderr)
print('checkMesh rc',r2.returncode); print('\n'.join((r2.stdout+r2.stderr).splitlines()[-65:]))

# -- cell 8 -------------------------------------------------------------------------
# `checkMesh` now passes all topology and geometry tests. It identifies the expected 36 disconnected f
rv=subprocess.run(['foamToVTK','-no-fields','-constant'],text=True,capture_output=True); Path('log.foamToVTK').write_text(rv.stdout+rv.stderr)
print('foamToVTK rc',rv.returncode)
files=list(Path('VTK').rglob('*.vtk'))+list(Path('VTK').rglob('*.vtu')); print('VTK files', [str(x) for x in files[:5]])
meshfile=next((x for x in files if 'boundary' not in str(x).lower()),files[0]); grid=pv.read(meshfile)
pl=pv.Plotter(off_screen=True,window_size=(900,700)); pl.add_mesh(grid.extract_surface(),color='wheat',show_edges=True,line_width=0.5); pl.view_isometric(); pl.add_axes(); pl.show(screenshot='openfoam_mesh.png')
btxt=Path('constant/polyMesh/boundary').read_text();
import re
print('cells',grid.n_cells,'bounds',tuple(round(x,6) for x in grid.bounds))
print('patch nFaces:',dict(re.findall(r'(frictionFace1|frictionFace2|vanes|innerInlet|outerOutlet)\s*\n\s*\{[^}]*?nFaces\s+(\d+);',btxt,re.S)))

# -- cell 9 -------------------------------------------------------------------------
# The converted mesh image shows all 36 open radial passages with the intended swept topology. One cor
FACE_T=0.010
fluid_exact=annulus
for k in range(N):
    vane_k=base_vane.moved(bd.Rot(0,0,k*PITCH))
    fluid_exact=fluid_exact-vane_k
lower_plate=(bd.Cylinder(RO,FACE_T)-bd.Cylinder(RI,FACE_T)).moved(bd.Pos(0,0,-FACE_T))
upper_plate=(bd.Cylinder(RO,FACE_T)-bd.Cylinder(RI,FACE_T)).moved(bd.Pos(0,0,GAP))
disc_material=bd.Compound([lower_plate,upper_plate,vanes])
expected_area=np.pi*(RO**2-RI**2)-N*VANE_W*(RO-RI)
print(f'friction plate thickness requested/measured: {FACE_T:.6f} / {lower_plate.bounding_box().size.Z:.6f} m')
print(f'inner-face separation requested/measured: {GAP:.6f} / {upper_plate.bounding_box().min.Z-lower_plate.bounding_box().max.Z:.6f} m')
print(f'vane radial span requested/measured: {RO-RI:.6f} / {base_vane.bounding_box().size.X:.6f} m; width {VANE_W:.6f} / {base_vane.bounding_box().size.Y:.6f} m')
print(f'fluid CAD volume={fluid_exact.volume:.9e}, analytical={expected_area*GAP:.9e}, mesh=4.00087e-4 m^3; CAD solids={len(fluid_exact.solids())}')

# -- cell 10 ------------------------------------------------------------------------
# The explicit plate dimensions are correct. The remaining CAD volume mismatch traces to composing rot
local_vane=bd.Box(RO-RI,VANE_W,GAP,align=(bd.Align.MIN,bd.Align.CENTER,bd.Align.MIN))
vane_shapes=[bd.Rot(0,0,k*PITCH)*(bd.Pos(RI,0,0)*local_vane) for k in range(N)]
fluid_cad=annulus
for vane_k in vane_shapes: fluid_cad=fluid_cad-vane_k
vanes_cad=bd.Compound(vane_shapes); disc_material=bd.Compound([lower_plate,upper_plate,vanes_cad])
# Exact area of clipped constant-width vane by numerical quadrature for comparison
ys=np.linspace(-VANE_W/2,VANE_W/2,10001)
vane_area_exact=np.trapezoid(np.sqrt(RO**2-ys**2)-np.sqrt(RI**2-ys**2),ys)
vol_exact=(np.pi*(RO**2-RI**2)-N*vane_area_exact)*GAP
print(f'vane count requested/built: {N}/{len(vane_shapes)}')
print(f'fluid volume CAD/exact/mesh: {fluid_cad.volume:.9e} / {vol_exact:.9e} / {4.00087e-4:.9e} m^3')
print('fluid CAD solids:',len(fluid_cad.solids()))

# -- cell 11 ------------------------------------------------------------------------
# The corrected list contains 36 vane objects, but the boolean still removes only 18 unique placements
for k in [0,1,17,18,19,35]:
    c=vane_shapes[k].center(); print(k,tuple(round(q,6) for q in (c.X,c.Y,c.Z)))
