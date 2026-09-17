"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the fluid as the union of the 120 mm × 200 mm chamber and four horizontal rectangular duc
import math, sys
from pathlib import Path
import build123d as bd
import pyvista as pv

# All dimensions are metres.
R, H = 0.060, 0.200
D_OUT = 0.040
PORT_W, PORT_H = 0.020, 0.030
PORT_Z = 0.040                 # assumed port centre height
DUCT_EXT, DUCT_INTERNAL = 0.050, 0.035
SWIRL_ANGLE = math.radians(15.0)

chamber = bd.Cylinder(R, H)
ducts = []
duct_axes = []
for i in range(4):
    theta = i * math.pi / 2
    er = (math.cos(theta), math.sin(theta))
    et = (-math.sin(theta), math.cos(theta))
    d = (math.cos(SWIRL_ANGLE)*et[0] - math.sin(SWIRL_ANGLE)*er[0],
         math.cos(SWIRL_ANGLE)*et[1] - math.sin(SWIRL_ANGLE)*er[1])
    pwall = (R*er[0], R*er[1])
    pin = (pwall[0] - DUCT_EXT*d[0], pwall[1] - DUCT_EXT*d[1])
    phi = math.degrees(math.atan2(d[1], d[0]))
    blank = bd.Box(DUCT_EXT + DUCT_INTERNAL, PORT_W, PORT_H,
                   align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.CENTER))
    duct = bd.Pos(pin[0], pin[1], PORT_Z) * bd.Rot(0, 0, phi) * blank
    ducts.append(duct)
    duct_axes.append((theta, d, pwall, pin, phi))

fluid = chamber
for duct in ducts:
    fluid = fluid + duct

bd.export_stl(fluid, "coarse-fluid-preview.stl", tolerance=8e-4, angular_tolerance=0.25)
mesh = pv.read("coarse-fluid-preview.stl")
plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
plotter.add_mesh(mesh, color="lightsteelblue", show_edges=True)
plotter.view_isometric(); plotter.camera.zoom(1.15)
plotter.show(screenshot="coarse-fluid-preview.png")
print(f"fluid solids={len(fluid.solids())}, volume={fluid.volume:.9g} m^3")
print(f"extent={tuple(round(x,6) for x in fluid.bounding_box().min.to_tuple())} to {tuple(round(x,6) for x in fluid.bounding_box().max.to_tuple())} m")
for i, (_, d, pwall, pin, phi) in enumerate(duct_axes, 1):
    measured = math.degrees(math.acos(abs(d[0]*(-math.sin((i-1)*math.pi/2)) + d[1]*math.cos((i-1)*math.pi/2))))
    print(f"inlet{i}: external axis length={math.dist(pin,pwall)*1000:.3f} mm (asked 50), off tangent={measured:.3f} deg (asked 15), axis azimuth={phi:.3f} deg")
print("preview: coarse-fluid-preview.png")

# -- cell 2 -------------------------------------------------------------------------
# The preview exposes one construction error: `Cylinder` defaulted to centred Z, giving −100 to +100 m
import subprocess, inspect
text = subprocess.run(["grep", "-n", "split_by_perimeter\|split.*perimeter\|make_splitter", ".reference/b123d_api.md"], capture_output=True, text=True).stdout
print(text if text else "No perimeter split entry in reading list")
for name in ["split_by_perimeter", "split"]:
    obj = getattr(bd.Shape, name, None)
    if obj:
        print(name, inspect.signature(obj))
        print((inspect.getdoc(obj) or "")[:900])

# -- cell 3 -------------------------------------------------------------------------
# The reconstructed solid was created; the check failed only because this installed build123d exposes 
# Rebuild with the vessel floor at z=0 and split the flush top boundary.
chamber = bd.Cylinder(R, H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
ducts = []
for _, d, pwall, pin, phi in duct_axes:
    blank = bd.Box(DUCT_EXT + DUCT_INTERNAL, PORT_W, PORT_H,
                   align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.CENTER))
    ducts.append(bd.Pos(pin[0], pin[1], PORT_Z) * bd.Rot(0, 0, phi) * blank)
fluid_unsplit = chamber
for duct in ducts:
    fluid_unsplit = fluid_unsplit + duct

top_whole = [f for f in fluid_unsplit.faces()
             if f.geom_type == bd.GeomType.PLANE and abs(f.center().Z-H) < 1e-8
             and abs(f.normal_at().Z) > 0.999][0]
outlet_perimeter = bd.Wire.make_circle(D_OUT/2, plane=bd.Plane.XY.offset(H))
outlet_face, top_face = top_whole.split_by_perimeter(outlet_perimeter, keep=bd.Keep.BOTH)
other_faces = [f for f in fluid_unsplit.faces() if not f.is_same(top_whole)]
fluid = bd.Solid(bd.Shell(other_faces + [outlet_face, top_face]))
print(f"valid={fluid.is_valid}, solids={len(fluid.solids())}, shells={len(fluid.shells())}, faces={len(fluid.faces())}")
print(f"z extent={fluid.bounding_box().min.Z:.6f} to {fluid.bounding_box().max.Z:.6f} m (asked 0 to 0.200)")
print(f"volume before/after split={fluid_unsplit.volume:.12g} / {fluid.volume:.12g} m^3")
print(f"outlet diameter from area={2*math.sqrt(outlet_face.area/math.pi)*1000:.3f} mm (asked 40)")
print(f"top annulus area={top_face.area:.9g} m^2; expected={math.pi*(R*R-(D_OUT/2)**2):.9g}")

# -- cell 4 -------------------------------------------------------------------------
# The corrected fluid is one valid closed solid, and splitting preserved its volume exactly. Next I’ll
for j, f in enumerate(fluid.faces()):
    c, n = f.center(), f.normal_at()
    print(f"{j:02d} {str(f.geom_type):9s} A={f.area:.8f} C=({c.X:+.4f},{c.Y:+.4f},{c.Z:+.4f}) N=({n.X:+.3f},{n.Y:+.3f},{n.Z:+.3f})")

# -- cell 5 -------------------------------------------------------------------------
# The face inspection revealed an important coarse-geometry issue: extending each near-tangential duct
import sys
sys.path.insert(0, ".reference")
from cad_export import export_patches

DUCT_INTERNAL = R * math.sin(SWIRL_ANGLE)
chamber = bd.Cylinder(R, H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
ducts = []
for _, d, pwall, pin, phi in duct_axes:
    blank = bd.Box(DUCT_EXT + DUCT_INTERNAL, PORT_W, PORT_H,
                   align=(bd.Align.MIN, bd.Align.CENTER, bd.Align.CENTER))
    ducts.append(bd.Pos(pin[0], pin[1], PORT_Z) * bd.Rot(0, 0, phi) * blank)
fluid_unsplit = chamber
for duct in ducts:
    fluid_unsplit = fluid_unsplit + duct

top_whole = [f for f in fluid_unsplit.faces() if f.geom_type == bd.GeomType.PLANE
             and abs(f.center().Z-H) < 1e-8 and f.normal_at().Z > 0.999][0]
outlet_perimeter = bd.Wire.make_circle(D_OUT/2, plane=bd.Plane.XY.offset(H))
outlet_face, top_face = top_whole.split_by_perimeter(outlet_perimeter, keep=bd.Keep.BOTH)
fluid = bd.Solid(bd.Shell([f for f in fluid_unsplit.faces() if not f.is_same(top_whole)] + [outlet_face, top_face]))

faces = list(fluid.faces())
cylinder_wall = [f for f in faces if f.geom_type == bd.GeomType.CYLINDER]
floor_faces = [f for f in faces if f.geom_type == bd.GeomType.PLANE and f.normal_at().Z < -0.999 and abs(f.center().Z) < 1e-8]
top_faces = [f for f in faces if f.is_same(top_face)]
outlet_faces = [f for f in faces if f.is_same(outlet_face)]
reserved = cylinder_wall + floor_faces + top_faces + outlet_faces
inlet_faces = [[] for _ in range(4)]
duct_wall_faces = [[] for _ in range(4)]
for f in faces:
    if any(f.is_same(g) for g in reserved):
        continue
    c = f.center(); n = f.normal_at()
    candidates = []
    for i, (_, d, _, pin, _) in enumerate(duct_axes):
        qdir = (-d[1], d[0])
        rx, ry = c.X-pin[0], c.Y-pin[1]
        s, q, zz = rx*d[0]+ry*d[1], rx*qdir[0]+ry*qdir[1], c.Z-PORT_Z
        plane_error = min(abs(s), abs(s-(DUCT_EXT+DUCT_INTERNAL)),
                          abs(abs(q)-PORT_W/2), abs(abs(zz)-PORT_H/2))
        candidates.append((plane_error, i, s, n.X*d[0]+n.Y*d[1]))
    err, i, s, nd = min(candidates)
    if err > 1e-6:
        raise RuntimeError(f"Unclassified face at {tuple(f.center())}, local plane error {err}")
    if abs(s) < 1e-6 and nd < -0.999:
        inlet_faces[i].append(f)
    else:
        duct_wall_faces[i].append(f)

patches = {f"inlet{i+1}": inlet_faces[i] for i in range(4)}
patches.update({"outlet": outlet_faces, "cylinderWall": cylinder_wall,
                "floor": floor_faces, "top": top_faces})
patches.update({f"duct{i+1}Walls": duct_wall_faces[i] for i in range(4)})
Path("constant/triSurface").mkdir(parents=True, exist_ok=True)
export_patches(fluid, patches, tolerance=2.5e-4)
print(f"valid={fluid.is_valid}, faces classified={sum(len(v) for v in patches.values())}/{len(faces)}")
for name, fs in patches.items():
    print(f"{name:14s}: faces={len(fs):2d}, area={sum(f.area for f in fs):.9g} m^2")

# -- cell 6 -------------------------------------------------------------------------
# The union surface is closed and consistently wound, but the first classifier exposed a tagging defec
# Diagnose local ownership of non-reserved planar faces before correcting patch tags.
for j, f in enumerate(faces):
    if any(f.is_same(g) for g in reserved) or any(f.is_same(g) for group in inlet_faces for g in group):
        continue
    c=f.center(); n=f.normal_at(); rows=[]
    for i, (_, d, _, pin, _) in enumerate(duct_axes):
        qdir=(-d[1],d[0]); rx,ry=c.X-pin[0],c.Y-pin[1]
        s=rx*d[0]+ry*d[1]; q=rx*qdir[0]+ry*qdir[1]
        inside=(-1e-6 <= s <= DUCT_EXT+DUCT_INTERNAL+1e-6 and abs(q)<=PORT_W/2+1e-6)
        align=max(abs(n.X*d[0]+n.Y*d[1]),abs(n.X*qdir[0]+n.Y*qdir[1]),abs(n.Z))
        if inside: rows.append((i+1,round(s,5),round(q,5),round(align,3)))
    print(j, tuple(round(x,4) for x in c), "candidates", rows)

# -- cell 7 -------------------------------------------------------------------------
# The diagnostic confirms every duct-wall face has exactly one valid local-blank owner, so the correct
inlet_faces = [[] for _ in range(4)]
duct_wall_faces = [[] for _ in range(4)]
for f in faces:
    if any(f.is_same(g) for g in reserved):
        continue
    c, n = f.center(), f.normal_at()
    owners=[]
    for i, (_, d, _, pin, _) in enumerate(duct_axes):
        qdir=(-d[1],d[0]); rx,ry=c.X-pin[0],c.Y-pin[1]
        s=rx*d[0]+ry*d[1]; q=rx*qdir[0]+ry*qdir[1]
        if -1e-6 <= s <= DUCT_EXT+DUCT_INTERNAL+1e-6 and abs(q) <= PORT_W/2+1e-6:
            owners.append((i,s,n.X*d[0]+n.Y*d[1]))
    if len(owners) != 1:
        raise RuntimeError(f"Face ownership is not unique: {owners}")
    i,s,nd=owners[0]
    (inlet_faces[i] if abs(s)<1e-6 and nd < -0.999 else duct_wall_faces[i]).append(f)
patches = {f"inlet{i+1}": inlet_faces[i] for i in range(4)}
patches.update({"outlet": outlet_faces, "cylinderWall": cylinder_wall,
                "floor": floor_faces, "top": top_faces})
patches.update({f"duct{i+1}Walls": duct_wall_faces[i] for i in range(4)})
export_patches(fluid, patches, tolerance=2.5e-4)
print("corrected patch symmetry:")
for i in range(4):
    fs=duct_wall_faces[i]
    print(f"duct{i+1}Walls faces={len(fs)}, area={sum(f.area for f in fs):.9g} m^2")

# -- cell 8 -------------------------------------------------------------------------
# The corrected STL union has zero open, flipped, and non-manifold edges; all four duct-wall patches a
import subprocess, textwrap
Path("system").mkdir(exist_ok=True)
Path("constant").mkdir(exist_ok=True)
Path("system/controlDict").write_text(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object controlDict; }
application snappyHexMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
'''))
Path("system/blockMeshDict").write_text(textwrap.dedent('''
FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (-0.09 -0.09 -0.005) (0.09 -0.09 -0.005) (0.09 0.09 -0.005) (-0.09 0.09 -0.005)
 (-0.09 -0.09 0.205) (0.09 -0.09 0.205) (0.09 0.09 0.205) (-0.09 0.09 0.205)
);
blocks (hex (0 1 2 3 4 5 6 7) (30 30 35) simpleGrading (1 1 1));
edges ();
boundary (background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
'''))
result=subprocess.run(["blockMesh"], text=True, capture_output=True)
print(result.stdout[-2500:]); print(result.stderr[-1000:])
if result.returncode: raise RuntimeError("blockMesh failed")

# -- cell 9 -------------------------------------------------------------------------
# The remaining failure is only an installation-path mismatch for the optional quality template. I’ll 
import os
print("WM_PROJECT_DIR=",os.environ.get("WM_PROJECT_DIR"))
r=subprocess.run(["find",os.environ.get("WM_PROJECT_DIR","/usr"),"-iname","*meshQuality*","-type","f"],capture_output=True,text=True)
print(r.stdout[:5000])

# -- cell 10 ------------------------------------------------------------------------
# The installed template is `caseDicts/mesh/generation/meshQualityDict.cfg` (with a `.cfg` suffix). I’
# Keep the accepted dictionary setup reproducible, now with the installed quality-template filename.
Path("system/fvSchemes").write_text('''FoamFile { format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; }
snGradSchemes { default corrected; } wallDist { method meshWave; }
''')
Path("system/fvSolution").write_text('FoamFile { format ascii; class dictionary; object fvSolution; }\nsolvers {}\n')
surface_names=[f"inlet{i}" for i in range(1,5)]+["outlet","cylinderWall","floor","top"]+[f"duct{i}Walls" for i in range(1,5)]
geom="\n".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in surface_names)
refs="\n".join(f'        {n} {{ level (1 1); patchInfo {{ type patch; }} }}' for n in surface_names)
Path("system/snappyHexMeshDict").write_text(f'''FoamFile {{ format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry {{ {geom} }}
castellatedMeshControls {{ maxLocalCells 500000; maxGlobalCells 1000000; minRefinementCells 0; nCellsBetweenLevels 2; features (); refinementSurfaces {{ {refs} }} resolveFeatureAngle 30; refinementRegions {{}} locationInMesh (0 0 0.1); allowFreeStandingZoneFaces true; }}
snapControls {{ nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 20; }}
meshQualityControls {{ #includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg" }}
mergeTolerance 1e-6;
''')
result=subprocess.run(["snappyHexMesh","-overwrite"], text=True, capture_output=True, timeout=230)
print(result.stdout[-7000:]); print(result.stderr[-1500:])
if result.returncode: raise RuntimeError("snappyHexMesh failed")

# -- cell 11 ------------------------------------------------------------------------
# The coarse snapped mesh completed with 33,108 cells and no final quality-control errors. I’ll run th
cm=subprocess.run(["checkMesh"],text=True,capture_output=True)
print(cm.stdout[-6500:]); print(cm.stderr[-1000:])
if cm.returncode: raise RuntimeError("checkMesh failed")
print("\nBoundary patch summary:")
btxt=Path("constant/polyMesh/boundary").read_text()
for n in surface_names:
    k=btxt.find("\n    "+n+"\n")
    print(n, "present" if k>=0 else "MISSING")

# -- cell 12 ------------------------------------------------------------------------
# `checkMesh` passes: one connected region, 12 boundary patches, max non-orthogonality 58.8°, max skew
vtk=subprocess.run(["foamToVTK","-constant"],text=True,capture_output=True,timeout=120)
print(vtk.stdout[-1800:]); print(vtk.stderr[-500:])
files=list(Path("VTK").rglob("*.vtk"))+list(Path("VTK").rglob("*.vtu"))
print("VTK files",[str(x) for x in files[:8]])
vol=pv.read(str(next(x for x in files if "boundary" not in str(x).lower())))
clip=vol.clip(normal=(1,-1,0), origin=(0,0,0.1), invert=False)
p=pv.Plotter(off_screen=True,window_size=(950,720))
p.add_mesh(clip,style="wireframe",color="navy",line_width=0.6,opacity=0.55)
p.add_mesh(clip.extract_surface(),color="lightsteelblue",opacity=0.35,show_edges=False)
p.view_isometric(); p.camera.zoom(1.25)
p.show(screenshot="final-mesh-clipped.png")
print(f"rendered cells={vol.n_cells}, clipped cells={clip.n_cells} -> final-mesh-clipped.png")

# -- cell 13 ------------------------------------------------------------------------
# The clipped volume render shows the chamber interior and a connected tangential duct with refinement
# Direct measurements from the authored faces and solids.
cb=chamber.bounding_box()
print(f"vessel internal diameter measured={cb.size.X*1000:.3f} mm (asked 120)")
print(f"vessel height measured={cb.size.Z*1000:.3f} mm (asked 200)")
print(f"floor/top measured z={cb.min.Z*1000:.3f}/{cb.max.Z*1000:.3f} mm")
for i,fs in enumerate(inlet_faces):
    f=fs[0]; verts=list(f.vertices()); _,d,pwall,pin,_=duct_axes[i]; qdir=(-d[1],d[0])
    qs=[v.X*qdir[0]+v.Y*qdir[1] for v in verts]; zs=[v.Z for v in verts]
    tangent=(-math.sin(i*math.pi/2),math.cos(i*math.pi/2))
    angle=math.degrees(math.acos(max(-1,min(1,d[0]*tangent[0]+d[1]*tangent[1]))))
    print(f"inlet{i+1}: {((max(qs)-min(qs))*1000):.3f} x {((max(zs)-min(zs))*1000):.3f} mm (asked 20 x 30), center z={f.center().Z*1000:.3f} mm, axis={math.dist(pin,pwall)*1000:.3f} mm, angle={angle:.3f} deg")
print(f"outlet diameter measured={2*math.sqrt(outlet_face.area/math.pi)*1000:.3f} mm (asked 40)")
mesh_vol=0.00238145
print(f"CAD/mesh volume={fluid.volume:.9g}/{mesh_vol:.9g} m^3; coarse mesh volume difference={(mesh_vol/fluid.volume-1)*100:.3f}%")
print("export_patches signature:",inspect.signature(export_patches))

# -- cell 14 ------------------------------------------------------------------------
# All requested dimensions measure exactly, and the coarse mesh volume is within 0.095% of the CAD flu
export_patches(fluid, patches, tolerance=2.5e-4, location_in_mesh=(0.0,0.0,0.1))
print("Recorded location_in_mesh=(0,0,0.1) m in the surface manifest; STL geometry unchanged.")
