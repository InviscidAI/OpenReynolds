"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid as the union of the cylindrical chamber and four horizontal rectangular duct pr
import math, os, subprocess
import build123d as bd
from build123d import Cylinder, Box, Pos, Rot, Align, export_step
# Requested dimensions, in metres
R = 0.060
H = 0.200
D_OUT = 0.040
PORT_W = 0.020
PORT_H = 0.030
PORT_Z0 = 0.040       # assumption: lower edge is 40 mm above floor
DUCT_L = 0.050
ANGLE_DEG = 15.0
OVERLAP = 0.035        # construction-only internal overlap, removed by union envelope
chamber = Cylinder(R, H)
ducts = []
duct_axes = []
for i in range(4):
    th = math.radians(90*i)
    r = (math.cos(th), math.sin(th))
    t = (-math.sin(th), math.cos(th))
    a = (math.cos(math.radians(ANGLE_DEG))*t[0] - math.sin(math.radians(ANGLE_DEG))*r[0],
         math.cos(math.radians(ANGLE_DEG))*t[1] - math.sin(math.radians(ANGLE_DEG))*r[1])
    pwall = (R*r[0], R*r[1])
    pout = (pwall[0] - DUCT_L*a[0], pwall[1] - DUCT_L*a[1])
    az = math.degrees(math.atan2(a[1], a[0]))
    local_duct = Box(DUCT_L + OVERLAP, PORT_W, PORT_H,
                     align=(Align.MIN, Align.CENTER, Align.MIN))
    duct = Pos(pout[0], pout[1], PORT_Z0) * Rot(0, 0, az) * local_duct
    ducts.append(duct)
    duct_axes.append((pout, pwall, a, az))
fluid = chamber
for duct in ducts:
    fluid = fluid + duct
os.makedirs('constant/triSurface', exist_ok=True)
export_step(fluid, 'fluid_coarse.step')
print('fluid solids:', len(fluid.solids()), 'volume [m3]:', fluid.volume)
print('requested diameter / built [m]:', 0.120, 2*R)
print('requested height / built [m]:', 0.200, H)
print('requested port WxH / built [m]:', (0.020,0.030), (PORT_W,PORT_H))
for i,(po,pw,a,az) in enumerate(duct_axes,1):
    dot_t = a[0]*(-math.sin(math.radians(90*(i-1)))) + a[1]*math.cos(math.radians(90*(i-1)))
    measured = math.degrees(math.acos(max(-1,min(1,dot_t))))
    length = math.dist(po,pw)
    print(f'inlet{i}: axis={a}, angle off tangent={measured:.6f} deg (requested 15), length={length:.6f} m (requested 0.05)')

# -- cell 2 -------------------------------------------------------------------------
# The union is a single connected solid and the requested axis angles and nominal duct lengths measure
from build123d import export_stl
import pyvista as pv
import matplotlib.pyplot as plt
export_stl(fluid, 'fluid_preview.stl', tolerance=0.0008, angular_tolerance=0.15)
mesh_preview = pv.read('fluid_preview.stl')
pl = pv.Plotter(shape=(1,2), off_screen=True, window_size=(1200,520))
pl.subplot(0,0); pl.add_mesh(mesh_preview, color='lightsteelblue', show_edges=True); pl.view_isometric(); pl.add_axes(); pl.camera.zoom(1.25)
pl.subplot(0,1); pl.add_mesh(mesh_preview, color='wheat', show_edges=True); pl.view_xy(); pl.add_axes(); pl.camera.zoom(1.15)
pl.screenshot('coarse_geometry.png')
print('preview:', mesh_preview.n_cells, 'triangles; bounds [m]:', mesh_preview.bounds)
from IPython.display import display, Image
display(Image('coarse_geometry.png'))

# -- cell 3 -------------------------------------------------------------------------
# The preview exposed a placement error: `Cylinder` defaults to vertical centring, so its floor was at
chamber = Cylinder(R, H, align=(Align.CENTER, Align.CENTER, Align.MIN))
fluid = chamber
for duct in ducts:
    fluid = fluid + duct
export_step(fluid, 'fluid_coarse.step')
export_stl(fluid, 'fluid_preview.stl', tolerance=0.0008, angular_tolerance=0.15)
mesh_preview = pv.read('fluid_preview.stl')
print('corrected bounds [m]:', tuple(round(x,6) for x in mesh_preview.bounds))
print('floor / top requested and built [m]:', (0.0,0.2), (mesh_preview.bounds[4],mesh_preview.bounds[5]))
print('port lower/upper elevations [m]:', PORT_Z0, PORT_Z0+PORT_H)

# -- cell 4 -------------------------------------------------------------------------
# With the corrected elevation, I’ll inspect the exact post-boolean boundary faces—their geometry type
faces = list(fluid.faces())
print('boundary face count:', len(faces))
for j,f in enumerate(faces):
    c=f.center(); n=f.normal_at(); gt=str(f.geom_type)
    print(f'{j:2d} {gt:18s} area={f.area:.8f} c=({c.X:.5f},{c.Y:.5f},{c.Z:.5f}) n=({n.X:.3f},{n.Y:.3f},{n.Z:.3f})')

# -- cell 5 -------------------------------------------------------------------------
# The boolean boundary is cleanly interpretable: one cylindrical vessel-wall face, floor and top disks
import inspect
print('Face:', inspect.signature(bd.Face))
print('Wire.make_circle:', inspect.signature(bd.Wire.make_circle))
print('Compound:', inspect.signature(bd.Compound))
print('export_stl:', inspect.signature(bd.export_stl))

# -- cell 6 -------------------------------------------------------------------------
# Before exporting, I’ll validate the exact face construction API and areas for the deliberately split
plane_top = bd.Plane(origin=(0,0,H), x_dir=(1,0,0), z_dir=(0,0,1))
w_outer = bd.Wire.make_circle(R, plane_top)
w_inner = bd.Wire.make_circle(D_OUT/2, plane_top)
outlet_face = bd.Face(w_inner)
top_face = bd.Face(w_outer, [w_inner])
print('outlet area built/expected:', outlet_face.area, math.pi*(D_OUT/2)**2)
print('top annulus area built/expected:', top_face.area, math.pi*(R**2-(D_OUT/2)**2))
print('sum / original top face:', outlet_face.area+top_face.area, faces[5].area)

# -- cell 7 -------------------------------------------------------------------------
# The split top areas sum exactly to the original chamber top area. I’ll now export the 12 requested p
# Classify exact post-boolean faces by source geometry and duct-axis provenance.
cylinder_faces = [f for f in faces if f.geom_type == bd.GeomType.CYLINDER]
floor_faces = [f for f in faces if f.geom_type == bd.GeomType.PLANE and abs(f.center().Z) < 1e-9]
original_top_faces = [f for f in faces if f.geom_type == bd.GeomType.PLANE and abs(f.center().Z-H) < 1e-9]
remaining = [f for f in faces if f not in cylinder_faces+floor_faces+original_top_faces]
inlet_faces = [[] for _ in range(4)]
duct_wall_faces = [[] for _ in range(4)]
for f in remaining:
    c=f.center(); n=f.normal_at()
    inlet_match = None
    for i,(po,pw,a,az) in enumerate(duct_axes):
        if math.hypot(c.X-po[0], c.Y-po[1]) < 1e-6 and abs(c.Z-(PORT_Z0+PORT_H/2))<1e-6 and (n.X*(-a[0])+n.Y*(-a[1]))>0.999:
            inlet_match=i
            break
    if inlet_match is not None:
        inlet_faces[inlet_match].append(f)
    else:
        # nearest duct centreline identifies provenance among four non-overlapping prisms
        distances=[]
        for po,pw,a,az in duct_axes:
            vx,vy=c.X-po[0],c.Y-po[1]
            q=vx*a[0]+vy*a[1]
            px,py=po[0]+q*a[0],po[1]+q*a[1]
            distances.append(math.hypot(c.X-px,c.Y-py))
        duct_wall_faces[distances.index(min(distances))].append(f)
patches = {'outlet': outlet_face, 'cylinderWall': bd.Compound(cylinder_faces),
           'floor': bd.Compound(floor_faces), 'top': top_face}
for i in range(4):
    patches[f'inlet{i+1}'] = bd.Compound(inlet_faces[i])
    patches[f'ductWall{i+1}'] = bd.Compound(duct_wall_faces[i])
for name,shape in patches.items():
    bd.export_stl(shape, f'constant/triSurface/{name}.stl', tolerance=0.00035, angular_tolerance=0.08, ascii_format=True)
print('patch face counts/areas:')
for name,shape in patches.items(): print(f'{name:12s}',len(shape.faces()),f'{shape.area:.9f}')
print('STLs:', sorted(os.listdir('constant/triSurface')))

# -- cell 8 -------------------------------------------------------------------------
# All requested patches are present with the expected counts and areas; each inlet is exactly \(20\tim
os.makedirs('system',exist_ok=True); os.makedirs('constant',exist_ok=True)
def write(path,text):
    with open(path,'w') as f: f.write(text)
foam_header='''FoamFile\n{\n    format ascii;\n    class dictionary;\n    object %s;\n}\n'''
write('system/controlDict', foam_header%'controlDict' + '''application snappyHexMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\nrunTimeModifiable true;\n''')
write('system/blockMeshDict', foam_header%'blockMeshDict' + '''convertToMeters 1;\nvertices\n(\n (-0.09 -0.09 -0.005) (0.09 -0.09 -0.005) (0.09 0.09 -0.005) (-0.09 0.09 -0.005)\n (-0.09 -0.09 0.205) (0.09 -0.09 0.205) (0.09 0.09 0.205) (-0.09 0.09 0.205)\n);\nblocks (hex (0 1 2 3 4 5 6 7) (36 36 42) simpleGrading (1 1 1));\nedges ();\nboundary\n(\n background { type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); }\n);\nmergePatchPairs ();\n''')
patch_names=['inlet1','inlet2','inlet3','inlet4','outlet','cylinderWall','floor','top','ductWall1','ductWall2','ductWall3','ductWall4']
geom='\n'.join([f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in patch_names])
refs='\n'.join([f'        {n} {{ level (1 1); patchInfo {{ type patch; }} }}' for n in patch_names])
write('system/snappyHexMeshDict', foam_header%'snappyHexMeshDict' + f'''castellatedMesh true;\nsnap true;\naddLayers false;\ngeometry\n{{\n{geom}\n}}\ncastellatedMeshControls\n{{\n maxLocalCells 1000000;\n maxGlobalCells 2000000;\n minRefinementCells 0;\n maxLoadUnbalance 0.10;\n nCellsBetweenLevels 2;\n features ();\n refinementSurfaces\n {{\n{refs}\n }}\n resolveFeatureAngle 30;\n refinementRegions {{}}\n locationInMesh (0 0 0.10);\n allowFreeStandingZoneFaces true;\n}}\nsnapControls\n{{\n nSmoothPatch 5;\n tolerance 2.0;\n nSolveIter 50;\n nRelaxIter 8;\n nFeatureSnapIter 10;\n implicitFeatureSnap true;\n explicitFeatureSnap false;\n multiRegionFeatureSnap false;\n}}\naddLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}\nmeshQualityControls\n{{\n maxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-16; minTetQuality 1e-20; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed {{ maxNonOrtho 75; }}\n}}\nmergeTolerance 1e-6;\n''')
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-2000:]); print(r.stderr[-500:]); print('return',r.returncode)

# -- cell 9 -------------------------------------------------------------------------
# The background mesh is 54,432 uniform 5 mm cells and fully encloses the geometry. I’ll now run the c
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-5000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 10 ------------------------------------------------------------------------
# SnappyHexMesh stopped before modifying the mesh because OpenFOAM 2512 requires placeholder discretis
write('system/fvSchemes', foam_header%'fvSchemes' + '''ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n''')
write('system/fvSolution', foam_header%'fvSolution' + '''solvers {}\n''')
print('added minimal meshing support dictionaries')

# -- cell 11 ------------------------------------------------------------------------
# The required support dictionaries now exist. I’ll rerun the same coarse snapping pass.
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# The coarse body-fitted mesh completed with 52,136 cells, all surface cells at 2.5 or 5 mm refinement
r=subprocess.run(['checkMesh','-allGeometry','-allTopology'],text=True,capture_output=True,timeout=240)
open('log.checkMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-8000:]); print(r.stderr[-1000:]); print('return',r.returncode)

# -- cell 13 ------------------------------------------------------------------------
# The exhaustive geometry mode reports 1,452 cells as concave under face-plane decomposition, although
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
print(r.stdout[-5000:]); print(r.stderr[-500:]); print('return',r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The binding standard check passes: one connected region, all 12 patches, positive cell volumes, max 
r=subprocess.run(['foamToVTK','-constant','-no-internal'],text=True,capture_output=True)
print('foamToVTK return',r.returncode, r.stdout[-700:])
import glob
files=glob.glob('VTK/**/*.vtp',recursive=True)+glob.glob('VTK/**/*.vtk',recursive=True)
print('surface files',len(files),files[:5])

# -- cell 15 ------------------------------------------------------------------------
# The converter confirms that every requested boundary exists as a distinct mesh patch. I’ll render th
colors={'inlet1':'red','inlet2':'orange','inlet3':'gold','inlet4':'magenta','outlet':'limegreen','cylinderWall':'lightsteelblue','floor':'tan','top':'skyblue','ductWall1':'salmon','ductWall2':'plum','ductWall3':'khaki','ductWall4':'aquamarine'}
pl=pv.Plotter(shape=(1,2),off_screen=True,window_size=(1300,550))
for k in range(2):
    pl.subplot(0,k)
    for n in patch_names:
        m=pv.read(f'VTK/t24_0/boundary/{n}.vtp')
        pl.add_mesh(m,color=colors[n],show_edges=True,line_width=0.35,label=n if k==0 else None)
    pl.add_axes()
pl.subplot(0,0); pl.view_isometric(); pl.camera.zoom(1.25); pl.add_legend(bcolor='white',size=(0.18,0.42))
pl.subplot(0,1); pl.view_xy(); pl.camera.zoom(1.2)
pl.screenshot('final_mesh.png')
display(Image('final_mesh.png'))

# -- cell 16 ------------------------------------------------------------------------
# I’ll re-emit the shared tessellation setup in a self-contained cell so it is reproducible in `build.
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
all_faces_compound = bd.Compound([shape for shape in patches.values()])
shared_mesher = BRepMesh_IncrementalMesh(all_faces_compound.wrapped, 0.00035, False, 0.08, True)
shared_mesher.Perform()
test_face = list(patches['inlet1'].faces())[0]
test_loc = TopLoc_Location()
test_tri = BRep_Tool.Triangulation_s(test_face.wrapped, test_loc)
print('shared inlet triangulation:',test_tri.NbNodes(),'nodes',test_tri.NbTriangles(),'triangles')

# -- cell 17 ------------------------------------------------------------------------
# The shared CAD assembly now has one persistent triangulation. I’ll overwrite each patch STL using tr
def write_shared_ascii_stl(path, name, shape):
    ntri=0
    with open(path,'w') as out:
        out.write(f'solid {name}\n')
        for face in shape.faces():
            loc=TopLoc_Location()
            tr=BRep_Tool.Triangulation_s(face.wrapped,loc)
            xf=loc.Transformation()
            rev=(face.wrapped.Orientation()==TopAbs_REVERSED)
            for k in range(1,tr.NbTriangles()+1):
                ids=list(tr.Triangle(k).Get())
                if rev: ids[1],ids[2]=ids[2],ids[1]
                pts=[]
                for idx in ids:
                    p=tr.Node(idx).Transformed(xf); pts.append((p.X(),p.Y(),p.Z()))
                ux,uy,uz=(pts[1][q]-pts[0][q] for q in range(3)); vx,vy,vz=(pts[2][q]-pts[0][q] for q in range(3))
                nx,ny,nz=uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
                mag=math.sqrt(nx*nx+ny*ny+nz*nz)
                if mag: nx,ny,nz=nx/mag,ny/mag,nz/mag
                out.write(f' facet normal {nx:.16g} {ny:.16g} {nz:.16g}\n  outer loop\n')
                for p in pts: out.write(f'   vertex {p[0]:.16g} {p[1]:.16g} {p[2]:.16g}\n')
                out.write('  endloop\n endfacet\n'); ntri+=1
        out.write(f'endsolid {name}\n')
    return ntri
for name,shape in patches.items():
    nt=write_shared_ascii_stl(f'constant/triSurface/{name}.stl',name,shape)
    print(name,nt)

# -- cell 18 ------------------------------------------------------------------------
# The shared export greatly reduces the triangle count while preserving the exact CAD tolerance. Befor
combined='constant/triSurface/_unionCheck.stl'
with open(combined,'w') as out:
    for n in patch_names: out.write(open(f'constant/triSurface/{n}.stl').read())
r=subprocess.run(['surfaceCheck',combined],text=True,capture_output=True)
print(r.stdout[-5000:]); print(r.stderr[-500:]); print('return',r.returncode)
os.remove(combined)

# -- cell 19 ------------------------------------------------------------------------
# The first shared attempt still assembled already-detached faces, so OpenCASCADE did not preserve the
from OCP.BRepFeat import BRepFeat_SplitShape
print(BRepFeat_SplitShape.__doc__)
print([x for x in dir(BRepFeat_SplitShape) if x in ('Add','Build','Shape','IsDone') or 'Add' in x])

# -- cell 20 ------------------------------------------------------------------------
# The installed kernel provides `BRepFeat_SplitShape.Add`, which can add a wire to a basis face. I’ll 
help(BRepFeat_SplitShape.Add)

# -- cell 21 ------------------------------------------------------------------------
# The split itself completed, but the generic build123d cast helper does not wrap this returned topolo
splitter = BRepFeat_SplitShape(fluid.wrapped)
splitter.Add(w_inner.wrapped, original_top_faces[0].wrapped)
splitter.Build()
fluid_split = bd.Solid(splitter.Shape())
print('split done:',splitter.IsDone(),'solids:',len(fluid_split.solids()),'faces:',len(fluid_split.faces()),'volume delta:',fluid_split.volume-fluid.volume)
for f in fluid_split.faces():
    if abs(f.center().Z-H)<1e-9: print('top split area:',f.area)

# -- cell 22 ------------------------------------------------------------------------
# The solid is now topologically correct: one solid, 28 faces, unchanged volume to machine precision, 
from OCP.BRepTools import BRepTools
split_faces=list(fluid_split.faces())
split_cyl=[f for f in split_faces if f.geom_type==bd.GeomType.CYLINDER]
split_floor=[f for f in split_faces if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z)<1e-9]
split_topall=[f for f in split_faces if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H)<1e-9]
split_outlet=[f for f in split_topall if abs(f.area-math.pi*(D_OUT/2)**2)<1e-9]
split_top=[f for f in split_topall if f not in split_outlet]
split_remaining=[f for f in split_faces if f not in split_cyl+split_floor+split_topall]
split_inlets=[[] for _ in range(4)]; split_walls=[[] for _ in range(4)]
for f in split_remaining:
    c=f.center(); n=f.normal_at(); found=None
    for i,(po,pw,a,az) in enumerate(duct_axes):
        if math.hypot(c.X-po[0],c.Y-po[1])<1e-6 and (n.X*(-a[0])+n.Y*(-a[1]))>0.999: found=i; break
    if found is not None: split_inlets[found].append(f)
    else:
        ds=[]
        for po,pw,a,az in duct_axes:
            q=(c.X-po[0])*a[0]+(c.Y-po[1])*a[1]
            ds.append(math.hypot(c.X-(po[0]+q*a[0]),c.Y-(po[1]+q*a[1])))
        split_walls[ds.index(min(ds))].append(f)
patches={'outlet':bd.Compound(split_outlet),'cylinderWall':bd.Compound(split_cyl),'floor':bd.Compound(split_floor),'top':bd.Compound(split_top)}
for i in range(4): patches[f'inlet{i+1}']=bd.Compound(split_inlets[i]); patches[f'ductWall{i+1}']=bd.Compound(split_walls[i])
BRepTools.Clean_s(fluid_split.wrapped)
solid_mesher=BRepMesh_IncrementalMesh(fluid_split.wrapped,0.00035,False,0.08,True); solid_mesher.Perform()
export_step(fluid_split,'fluid.step')
print('patch faces:',{n:len(s.faces()) for n,s in patches.items()})

# -- cell 23 ------------------------------------------------------------------------
# The final groups again contain exactly the intended faces. I’ll overwrite the 12 STLs from the solid
for name,shape in patches.items(): write_shared_ascii_stl(f'constant/triSurface/{name}.stl',name,shape)
combined='constant/triSurface/_unionCheck.stl'
with open(combined,'w') as out:
    for n in patch_names: out.write(open(f'constant/triSurface/{n}.stl').read())
r=subprocess.run(['surfaceCheck',combined],text=True,capture_output=True)
print(r.stdout[-3500:]); print('return',r.returncode)
for p in glob.glob('constant/triSurface/_unionCheck*'): os.remove(p)

# -- cell 24 ------------------------------------------------------------------------
# The welded STL union now passes: **closed**, one connected part, one consistent normal zone, and no 
r=subprocess.run(['blockMesh'],text=True,capture_output=True)
print(r.stdout[-1800:]); print('return',r.returncode)

# -- cell 25 ------------------------------------------------------------------------
# The clean background mesh is restored. I’ll run the final body-fitting pass against the watertight, 
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True,timeout=240)
open('log.snappyHexMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-3500:]); print(r.stderr[-500:]); print('return',r.returncode)

# -- cell 26 ------------------------------------------------------------------------
# The corrected STL set produces the same intended 52,136-cell mesh and snappy reports no quality erro
r=subprocess.run(['checkMesh'],text=True,capture_output=True)
open('log.checkMesh','w').write(r.stdout+r.stderr)
print(r.stdout[-4500:]); print('return',r.returncode)
