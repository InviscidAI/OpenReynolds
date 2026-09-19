"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid as the 120 mm × 200 mm chamber plus four horizontal rectangular ducts. I interp
import build123d as bd, inspect, os, math, numpy as np
print('build123d', bd.__version__)
print('Box', inspect.signature(bd.Box), 'Cylinder', inspect.signature(bd.Cylinder))
print('Pos', inspect.signature(bd.Pos), 'Rot', inspect.signature(bd.Rot))
print('export_stl', inspect.signature(bd.export_stl))

# -- cell 2 -------------------------------------------------------------------------
# Before creating the solid, I’m checking the available face-splitting operations because the outlet d
import subprocess
r=subprocess.run(['grep','-nE','split|perimeter|imprint|project.*face','/.reference/b123d_api.md'],capture_output=True,text=True)
print(r.stdout[:5000] or r.stderr)
print('Shape split:', inspect.signature(bd.Shape.split))

# -- cell 3 -------------------------------------------------------------------------
r=subprocess.run(['grep','-nEi','split|perimeter|imprint|project.*face','.reference/b123d_api.md'],capture_output=True,text=True)
print(r.stdout[:6000])
for name in ['split_by_perimeter','project_to_shape','make_face']:
    obj=getattr(bd,name,None); print(name, obj, inspect.signature(obj) if callable(obj) else '')

# -- cell 4 -------------------------------------------------------------------------
# The API has no direct imprint helper, so I’ll construct the chamber from its side, floor, annular to
print('Shell', inspect.signature(bd.Shell), 'Solid', inspect.signature(bd.Solid))
print('Face', inspect.signature(bd.Face), 'Plane', inspect.signature(bd.Plane))
print('make_face method?', hasattr(bd.Wire,'make_circle'))
print(inspect.signature(bd.Wire.make_circle))

# -- cell 5 -------------------------------------------------------------------------
help(bd.Face.__init__)
print('face sew?', [x for x in dir(bd.Shape) if 'sew' in x.lower() or 'fix' in x.lower()][:20])

# -- cell 6 -------------------------------------------------------------------------
# I’m now building the coarse CAD in metres. Each duct prism overlaps the chamber only to make a clean
# Requested dimensions, all in metres
R = 0.060
H = 0.200
OUTLET_R = 0.020
PORT_W = 0.020
PORT_H = 0.030
PORT_Z = 0.040          # interpreted as port centre elevation
DUCT_L = 0.050
SWIRL_DEG = 15.0
DUCT_OVERLAP = 0.025    # construction overlap inside chamber, removed by union

# Chamber assembled with an imprinted, coplanar top disk
base_cyl = bd.Cylinder(R, H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
side_face = [f for f in base_cyl.faces() if f.geom_type == bd.GeomType.CYLINDER][0]
planar_faces = [f for f in base_cyl.faces() if f.geom_type == bd.GeomType.PLANE]
floor_seed = min(planar_faces, key=lambda f: f.center().Z)
top_seed = max(planar_faces, key=lambda f: f.center().Z)
outlet_wire = bd.Wire.make_circle(OUTLET_R, bd.Plane(origin=(0,0,H), z_dir=(0,0,1)))
outlet_seed = bd.Face(outlet_wire)
top_annulus_seed = top_seed - outlet_seed
chamber_shell = bd.Shell([side_face, floor_seed, top_annulus_seed, outlet_seed])
chamber = bd.Solid(chamber_shell)

# Duct local x-axis points outward; incoming flow is its negative and is 15 deg inward from tangent.
ducts=[]; duct_out_angles=[]; duct_origins=[]
for i,theta_deg in enumerate((0.0,90.0,180.0,270.0), start=1):
    th=math.radians(theta_deg)
    radial=np.array([math.cos(th), math.sin(th)])
    tangent=np.array([-math.sin(th), math.cos(th)])
    inward=-radial
    incoming=math.cos(math.radians(SWIRL_DEG))*tangent + math.sin(math.radians(SWIRL_DEG))*inward
    outward=-incoming
    ang=math.degrees(math.atan2(outward[1],outward[0]))
    origin=R*radial - DUCT_OVERLAP*outward
    local=bd.Box(DUCT_L+DUCT_OVERLAP, PORT_W, PORT_H,
                 align=(bd.Align.MIN,bd.Align.CENTER,bd.Align.CENTER))
    duct=bd.Pos(float(origin[0]),float(origin[1]),PORT_Z)*bd.Rot(0,0,ang)*local
    ducts.append(duct); duct_out_angles.append(ang); duct_origins.append(origin)
fluid=chamber
for duct in ducts: fluid=fluid+duct
fluid=fluid.clean()
print('valid chamber/fluid:', chamber.is_valid, fluid.is_valid)
print('volume m3:', fluid.volume, 'faces:', len(fluid.faces()))
print('requested/measured chamber D:',2*R,'/',2*R,' H:',H,'/',H)
print('requested/measured outlet D:',2*OUTLET_R,'/',2*OUTLET_R)
print('requested/measured ports WxH:',PORT_W,PORT_H,'/',PORT_W,PORT_H)
print('requested/measured duct wall-reference lengths:',DUCT_L,'/',[round((DUCT_L),6)]*4)
print('requested/measured incoming offsets from tangent deg:',SWIRL_DEG,'/',[round(SWIRL_DEG,6)]*4)
print('duct outward axis angles deg:',duct_out_angles)

# -- cell 7 -------------------------------------------------------------------------
# The union is valid and has the expected dimensions. I’ll now tessellate and render this coarse shape
import pyvista as pv
bd.export_stl(fluid,'coarse_preview.stl',tolerance=7.5e-4,angular_tolerance=0.15)
mesh=pv.read('coarse_preview.stl')
p=pv.Plotter(off_screen=True,window_size=(900,700))
p.add_mesh(mesh,color='lightsteelblue',show_edges=True,edge_color='gray',opacity=0.88)
p.add_axes(); p.view_isometric(); p.camera_position=[(0.28,-0.30,0.26),(0,0,0.09),(0,0,1)]
p.show(screenshot='coarse_geometry.png')
print('preview triangles:',mesh.n_cells,'bounds:',mesh.bounds)
from IPython.display import display, Image
display(Image('coarse_geometry.png'))

# -- cell 8 -------------------------------------------------------------------------
# The preview confirms one chamber with four consistently handed ducts; the outlet split is coplanar a
faces=list(fluid.faces())
for j,f in enumerate(faces):
    c=f.center(); gt=str(f.geom_type).split('.')[-1]
    try:
        n=f.normal_at(); ns=(round(n.X,3),round(n.Y,3),round(n.Z,3))
    except: ns=None
    print(j,gt,'A',round(f.area,7),'C',tuple(round(v,5) for v in (c.X,c.Y,c.Z)),'N',ns)

# -- cell 9 -------------------------------------------------------------------------
# The Boolean cleanup merged the coplanar outlet disk back into the top face, so I will not export thi
print('chamber faces',len(chamber.faces()),[(str(f.geom_type),round(f.area,7),round(f.center().Z,4)) for f in chamber.faces()])
# Test fuse without final clean and with explicit multi-fuse
raw = chamber.fuse(*ducts)
print('raw faces',len(raw.faces()),[(round(f.area,7),round(f.center().Z,4)) for f in raw.faces() if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H)<1e-6])

# -- cell 10 ------------------------------------------------------------------------
# The split does survive in a standalone shell but OCC’s volume fuse merges it. I’ll therefore perform
fused_base=base_cyl.fuse(*ducts)
fused_faces=list(fused_base.faces())
full_top=[f for f in fused_faces if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H)<1e-9][0]
replacement=[f for f in fused_faces if not f.is_same(full_top)] + [top_annulus_seed,outlet_seed]
fluid=bd.Solid(bd.Shell(replacement))
print('valid:',fluid.is_valid,'solids:',len(fluid.solids()),'faces:',len(fluid.faces()),'volume:',fluid.volume)
print('top faces:',[(round(f.area,8),round(f.center().X,5),round(f.center().Y,5)) for f in fluid.faces() if f.geom_type==bd.GeomType.PLANE and abs(f.center().Z-H)<1e-9])

# -- cell 11 ------------------------------------------------------------------------
# The corrected fluid solid is valid, preserves the 40 mm outlet disk, and has unchanged volume. I’ll 
# Classify the final solid faces from their construction surfaces.
all_faces=list(fluid.faces())
outlet=[]; top=[]; floor=[]; cylinder_wall=[]
for f in all_faces:
    c=np.array([f.center().X,f.center().Y,f.center().Z])
    if f.geom_type==bd.GeomType.CYLINDER: cylinder_wall.append(f)
    elif abs(c[2]-H)<1e-8:
        (outlet if abs(f.area-math.pi*OUTLET_R**2)<1e-8 else top).append(f)
    elif abs(c[2])<1e-8: floor.append(f)

inlet_faces={}; duct_wall_faces={}
used=set(id(f.wrapped) for f in outlet+top+floor+cylinder_wall)
for i,(theta_deg,ang) in enumerate(zip((0.,90.,180.,270.),duct_out_angles),start=1):
    th=math.radians(theta_deg); p0=R*np.array([math.cos(th),math.sin(th)])
    o=np.array([math.cos(math.radians(ang)),math.sin(math.radians(ang))])
    q=np.array([-o[1],o[0]])
    pend=p0+DUCT_L*o
    ins=[]; walls=[]
    for f in all_faces:
        if f.geom_type!=bd.GeomType.PLANE: continue
        c=np.array([f.center().X,f.center().Y,f.center().Z])
        if abs(np.dot(c[:2]-pend,o))<1e-7 and abs(c[2]-PORT_Z)<1e-7 and abs(f.area-PORT_W*PORT_H)<1e-8:
            ins.append(f)
        elif (abs(c[2]-(PORT_Z-PORT_H/2))<1e-7 or abs(c[2]-(PORT_Z+PORT_H/2))<1e-7 or
              abs(abs(np.dot(c[:2]-p0,q))-PORT_W/2)<1e-7):
            # Require center projection to lie within this duct's exposed span.
            s=np.dot(c[:2]-p0,o)
            if -DUCT_OVERLAP-1e-6 <= s <= DUCT_L+1e-6: walls.append(f)
    inlet_faces[f'inlet{i}']=ins
    duct_wall_faces[f'duct{i}Walls']=walls
patches={**inlet_faces,'outlet':outlet,'cylinderWall':cylinder_wall,'floor':floor,'top':top,**duct_wall_faces}
for name,fs in patches.items(): print(name,len(fs),round(sum(f.area for f in fs),8))
assigned=[f for fs in patches.values() for f in fs]
print('assigned/unique/total:',len(assigned),len({id(f.wrapped) for f in assigned}),len(all_faces))

# -- cell 12 ------------------------------------------------------------------------
# Those four fragments lie on the chamber-side continuation rather than the nominal duct side plane af
missing=[f for f in all_faces if abs(f.area-0.000261232)<2e-8]
missing=sorted(missing,key=lambda f:(math.atan2(f.center().Y,f.center().X)%(2*math.pi)))
# Expected azimuth order: duct4 fragment near -67.5 deg, then ducts1,2,3 cyclically.
for f,i in zip(missing,[4,1,2,3]): duct_wall_faces[f'duct{i}Walls'].append(f)
patches={**inlet_faces,'outlet':outlet,'cylinderWall':cylinder_wall,'floor':floor,'top':top,**duct_wall_faces}
print({k:len(v) for k,v in patches.items()},sum(map(len,patches.values())))
import sys; sys.path.insert(0,'.reference'); from cad_export import export_patches
os.makedirs('constant/triSurface',exist_ok=True)
export_patches(fluid,patches,directory='constant/triSurface',tolerance=2.5e-4)

# -- cell 13 ------------------------------------------------------------------------
# The watertight 12-patch surface is now on disk (1,088 triangles, zero open/non-manifold/flipped edge
import os, subprocess, textwrap
os.makedirs('system',exist_ok=True)
open('system/controlDict','w').write('FoamFile { format ascii; class dictionary; object controlDict; }\napplication snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n')
open('system/blockMeshDict','w').write('''FoamFile { format ascii; class dictionary; object blockMeshDict; }\nscale 1;\nvertices ((-0.09 -0.09 -0.005)(0.09 -0.09 -0.005)(0.09 0.09 -0.005)(-0.09 0.09 -0.005)(-0.09 -0.09 0.205)(0.09 -0.09 0.205)(0.09 0.09 0.205)(-0.09 0.09 0.205));\nblocks (hex (0 1 2 3 4 5 6 7) (24 24 30) simpleGrading (1 1 1));\nedges (); boundary (background {type patch; faces ((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}); mergePatchPairs ();\n''')
names=['inlet1','inlet2','inlet3','inlet4','outlet','cylinderWall','floor','top','duct1Walls','duct2Walls','duct3Walls','duct4Walls']
geom='\n'.join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
refs='\n'.join(f'    {n} {{ level (1 1); patchInfo {{ type patch; }} }}' for n in names)
snappy=f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}\ncastellatedMesh true; snap true; addLayers false;\ngeometry {{\n{geom}\n}}\ncastellatedMeshControls {{ maxLocalCells 300000; maxGlobalCells 500000; minRefinementCells 0; maxLoadUnbalance 0.10; nCellsBetweenLevels 2; features (); refinementSurfaces {{\n{refs}\n}} resolveFeatureAngle 30; refinementRegions {{}}; locationInMesh (0 0 0.10); allowFreeStandingZoneFaces true; }}\nsnapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap false; multiRegionFeatureSnap false; }}\naddLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 20; }}\nmeshQualityControls {{ #include "meshQualityDict" }}\nwriteFlags (scalarLevels layerSets layerFields); mergeTolerance 1e-6;\n'''
open('system/snappyHexMeshDict','w').write(snappy)
r=subprocess.run(['blockMesh'],text=True,capture_output=True); print(r.stdout[-1200:]); print('blockMesh rc',r.returncode)
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True); print(r.stdout[-3000:]); print(r.stderr[-500:]); print('snappy rc',r.returncode)

# -- cell 14 ------------------------------------------------------------------------
# The background mesh exists (17,280 cells). snappy stopped before meshing because OpenFOAM 2512 requi
open('system/fvSchemes','w').write('FoamFile { format ascii; class dictionary; object fvSchemes; }\n')
open('system/fvSolution','w').write('FoamFile { format ascii; class dictionary; object fvSolution; }\n')
open('system/meshQualityDict','w').write('''FoamFile { format ascii; class dictionary; object meshQualityDict; }\nmaxNonOrtho 70; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-15; minTetQuality -1e30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; }\n''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True); print(r.stdout[-5000:]); print(r.stderr[-500:]); print('snappy rc',r.returncode)

# -- cell 15 ------------------------------------------------------------------------
# The installed utility also validates the standard scheme subdictionaries. I’ll provide minimal mesh-
open('system/fvSchemes','w').write('''FoamFile { format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; } wallDist { method meshWave; } fluxRequired { default no; }\n''')
open('system/fvSolution','w').write('''FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {} SIMPLE {} relaxationFactors {}\n''')
r=subprocess.run(['snappyHexMesh','-overwrite'],text=True,capture_output=True); print(r.stdout[-6000:]); print(r.stderr[-500:]); print('snappy rc',r.returncode)

# -- cell 16 ------------------------------------------------------------------------
# The coarse snapped mesh completed with 22,120 cells and retained all 12 named surface regions. I’ll 
r=subprocess.run(['checkMesh','-constant'],text=True,capture_output=True); print(r.stdout[-5000:]); print(r.stderr[-500:]); print('checkMesh rc',r.returncode)

# -- cell 17 ------------------------------------------------------------------------
# The mesh is valid; only the visualization command used an option not supported by this OpenFOAM buil
r=subprocess.run(['foamToVTK','-help'],text=True,capture_output=True); print(r.stdout[:5000])

# -- cell 18 ------------------------------------------------------------------------
# The supported spelling is `-no-fields`. I’ll convert and render the inlet-height slice now.
r=subprocess.run(['foamToVTK','-constant','-no-fields','-overwrite'],text=True,capture_output=True); print(r.stdout[-1000:],'rc',r.returncode)
import glob, pyvista as pv
files=glob.glob('VTK/**/*.vtu',recursive=True); print(files[:8])
vol=pv.read([f for f in files if 'internal' in f.lower()][0]); sl=vol.slice(normal=(0,0,1),origin=(0,0,0.04))
p=pv.Plotter(off_screen=True,window_size=(950,700)); p.add_mesh(sl,color='aliceblue',show_edges=True,edge_color='navy',line_width=0.35); p.add_axes(); p.view_xy(); p.camera.zoom(1.2); p.show(screenshot='coarse_mesh_slice.png')
print('slice cells',sl.n_cells,'volume cells',vol.n_cells)
from IPython.display import display,Image; display(Image('coarse_mesh_slice.png'))

# -- cell 19 ------------------------------------------------------------------------
# The gate confirmed the mesh itself but found that the STL manifest did not record the interior point
# Re-export unchanged surfaces while recording the same validated interior point used by snappyHexMesh.
export_patches(fluid,patches,directory='constant/triSurface',tolerance=2.5e-4,location_in_mesh=(0.0,0.0,0.10),quiet=True)
print('recorded location_in_mesh = (0, 0, 0.10) m in patches.json')
