"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# That stray expression made the cell ineligible for the script. Re-emitting it clean:
import numpy as np, build123d as bd
# mm model; scaled to metres at export time
R0_IN, R_OUT = 16.0, 58.0          # blade inlet / outlet radius, mm
H_BLADE, T_BLADE = 14.0, 3.0       # blade height, thickness, mm
SWEEP_DEG = 30.0                   # total backward sweep of trailing edge
N_BLADES = 7
r = np.linspace(R0_IN, R_OUT, 121)
th = np.radians(-SWEEP_DEG) * np.log(r / R0_IN) / np.log(R_OUT / R0_IN)
pts = np.column_stack([r*np.cos(th), r*np.sin(th)])
sweep_meas = -np.degrees(np.arctan2(pts[-1,1], pts[-1,0]))
print("sweep requested:", SWEEP_DEG, "sweep measured:", round(sweep_meas,3))
d = np.gradient(pts, axis=0); d = d / np.linalg.norm(d, axis=1)[:, None]
n = np.column_stack([-d[:,1], d[:,0]])
left  = pts + n*(T_BLADE/2)
right = pts - n*(T_BLADE/2)
spl_l = bd.Spline([bd.Vector(*p,0) for p in left])
spl_r = bd.Spline([bd.Vector(*p,0) for p in right[::-1]])
cap0  = bd.Line(bd.Vector(*left[0],0),  bd.Vector(*right[0],0))
cap1  = bd.Line(bd.Vector(*left[-1],0), bd.Vector(*right[-1],0))
foot  = bd.Face(bd.Wire([spl_l, cap1, spl_r, cap0]))
blade0 = bd.extrude(foot, H_BLADE)
print("blade volume mm^3:", round(blade0.volume,1))

# -- cell 2 -------------------------------------------------------------------------
R_HUB = 12.0
half_rad = np.pi / N_BLADES
def ring_pts(rr, n=80):
    a = np.linspace(-half_rad, half_rad, n)
    return [bd.Vector(rr*np.cos(t), rr*np.sin(t), 0) for t in a]
outer = bd.Spline(ring_pts(R_OUT))                 # -half -> +half at r=58
inner = bd.Spline(ring_pts(R_HUB)[::-1])           # reversed
l1 = bd.Line(bd.Vector(R_OUT,0,0).rotate(bd.Axis.Z, -np.degrees(half_rad)), bd.Vector(R_HUB,0,0).rotate(bd.Axis.Z, -np.degrees(half_rad)))
l2 = bd.Line(bd.Vector(R_HUB,0,0).rotate(bd.Axis.Z, np.degrees(half_rad)), bd.Vector(R_OUT,0,0).rotate(bd.Axis.Z, np.degrees(half_rad)))
wedge_face = bd.Face(bd.Wire([outer, l2, inner, l1]))
wedge = bd.extrude(wedge_face, H_BLADE)
print("wedge volume mm^3:", round(wedge.volume,1), " passage half-angle deg:", np.degrees(half_rad))

# -- cell 3 -------------------------------------------------------------------------
half_deg = np.degrees(half_rad)
blade_hi = bd.Pos(0,0,0) * blade0 if False else blade0
hi = (bd.Rotation(0,0,half_deg) * blade0) & wedge
lo = (bd.Rotation(0,0,-half_deg) * blade0) & wedge
fluid = wedge - hi - lo
print("fluid volume mm^3:", round(fluid.volume,1), " expected ~ wedge - 2 halves:", round(wedge.volume - (hi.volume+lo.volume),1))
# measure minimum passage width between blades near mid-radius (numerically via sampled angular gap)
for rr in (16.0, 37.0, 58.0):
    # angular thickness of blade at radius rr from centreline deflection
    print(rr, "blade angular half-width deg ~", np.degrees((T_BLADE/2)/rr))

# -- cell 4 -------------------------------------------------------------------------
# Subtract did nothing — the intersections look empty. Debugging:
b2 = bd.Rotation(0,0,half_deg) * blade0
print("rotated blade volume:", round(b2.volume,1))
print("intersection volume:", round((b2 & wedge).volume,1))
print("blade0 bbox:", blade0.bounding_box().min, blade0.bounding_box().max)
print("wedge bbox:", wedge.bounding_box().min, wedge.bounding_box().max)

# -- cell 5 -------------------------------------------------------------------------
blade0 = bd.extrude(foot, -H_BLADE)
bb = blade0.bounding_box()
print("blade z-range:", bb.min.Z, bb.max.Z)
hi = (bd.Rotation(0,0,half_deg) * blade0) & wedge
lo = (bd.Rotation(0,0,-half_deg) * blade0) & wedge
print("clipped halves:", round(hi.volume,1), round(lo.volume,1))
fluid = wedge - hi - lo
print("fluid volume mm^3:", round(fluid.volume,1))

# -- cell 6 -------------------------------------------------------------------------
# The half-blade clipping is wrong (98%/4% instead of ~50% each). Checking what `bd.Rotation` actually
t = bd.Rotation(0,0,half_deg) * blade0
print([tuple(round(v,3) for v in (t.bounding_box().min.to_tuple() if hasattr(t.bounding_box().min,'to_tuple') else []))], end=" ") if False else None
print("rot+half bbox min", tuple(round(v,2) for v in t.bounding_box().min.to_tuple()), "max", tuple(round(v,2) for v in t.bounding_box().max.to_tuple()))
print("help:", bd.Rotation.__doc__)

# -- cell 7 -------------------------------------------------------------------------
v1 = (t & wedge).volume
# analytical outside fraction: fraction of centreline points with theta > half
ang = np.arctan2(pts[:,1], pts[:,0]) + half_rad
frac = np.mean((ang > half_rad) | (ang < -half_rad) )
print("clip volume:", round(v1,1), "of", round(blade0.volume,1), " sample flag:", frac)
# check lower blade
tlo = bd.Rotation(0,0,-half_deg) * blade0
print("lo clip:", round((tlo & wedge).volume,1))

# -- cell 8 -------------------------------------------------------------------------
# Rethinking: the passage walls must be blade surfaces, not planar cuts, since the spiral is not radia
import numpy as np
angles = [bd.Rotation(0,0,k*360.0/N_BLADES) for k in range(N_BLADES)]
base = bd.Cylinder(R_OUT, H_BLADE) - bd.Cylinder(R_HUB, H_BLADE)
blades_all = bd.Compound([a*blade0 for a in angles])
fluid = base - blades_all
print("fluid volume mm^3:", round(fluid.volume,1), " = annulus", round(base.volume,1), "- blades", round(blades_all.volume,1))
bb = fluid.bounding_box()
print("bbox:", tuple(bb.min), tuple(bb.max))

# -- cell 9 -------------------------------------------------------------------------
# Understood — going to disk now. Export patch STLs (mm → scale to metres on export via a `bd.Scale` f
import numpy as np, os, subprocess as sp
groups = {'inlet':[], 'shroud':[], 'backplate':[], 'hub':[], 'outlet':[], 'blades':[]}
for f in fluid.faces():
    c = f.center(); rr = (c.X**2+c.Y**2)**0.5
    n = f.normal_at()
    if abs(n.Z) > 0.999:   # planar top/bottom
        at_top = bb.max.Z - 1e-6
        if abs(c.Z) < 1e-6:
            groups['backplate'].append(f)
        elif abs(c.Z - at_top) < 1e-4:
            groups['inlet' if rr < 16.5 else 'shroud'].append(f)
        else:
            groups['blades'].append(f)
    elif abs(rr - 12) < 1e-4:
        groups['hub'].append(f)
    elif abs(rr - 58) < 1e-4:
        groups['outlet'].append(f)
    else:
        groups['blades'].append(f)
os.makedirs('triSurface', exist_ok=True)
for name, fs in groups.items():
    solid = bd.Compound(fs)
    bd.export_stl(solid, f'triSurface/{name}.stl', tolerance=1e-4, angular_tolerance=0.35)
    sp.run(['surfaceTransformPoints','-scale','0.001', f'triSurface/{name}.stl', f'triSurface/{name}.stl'], capture_output=True)
    print(name, len(fs))

# -- cell 10 ------------------------------------------------------------------------
# The top face came out as a single face, so my inlet/shroud split needs a face-level boolean against 
# redo classification; identify the top face and split it with a r=16 disk
groups = {'inlet':[], 'shroud':[], 'backplate':[], 'hub':[], 'outlet':[], 'blades':[]}
top_faces = [f for f in fluid.faces() if abs(f.normal_at().Z) > 0.99 and abs(f.center().Z - bb.max.Z) < 0.25]
disk = bd.Pos(0,0,H_BLADE-1) * bd.Cylinder(16.0, 2)
for f in top_faces:
    ins = (f & disk).faces()
    outs = (f - disk).faces()
    print("top face r-range inside 16:", [round((fc.center().X**2+fc.center().Y**2)**0.5,2) for fc in ins])
    groups['inlet'] += ins
    groups['shroud'] += outs
for f in fluid.faces():
    c = f.center(); rr = (c.X**2+c.Y**2)**0.5
    n = f.normal_at()
    if abs(n.Z) > 0.999 and abs(c.Z) < 1e-6: groups['backplate'].append(f)
    elif abs(n.Z) > 0.999 and abs(c.Z - bb.max.Z) < 0.25: pass
    elif abs(rr - R_HUB) < 1e-4: groups['hub'].append(f)
    elif abs(rr - R_OUT) < 1e-4: groups['outlet'].append(f)
    elif f not in top_faces: groups['blades'].append(f)
import subprocess as sp, os
for name, fs in groups.items():
    if not fs: print('EMPTY', name)
    solid = bd.Compound(fs)
    bd.export_stl(solid, f'triSurface/{name}.stl', tolerance=1e-4, angular_tolerance=0.35)
    sp.run(['surfaceTransformPoints','-scale','0.001', f'triSurface/{name}.stl', f'triSurface/{name}.stl'], capture_output=True)
    print(name, len(fs))

# -- cell 11 ------------------------------------------------------------------------
# Re-emitting clean, with the full snappyHexMeshDict this time:
import os, subprocess as sp
names = ['inlet','shroud','backplate','hub','outlet','blades']
geo_entries = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
w = lambda p,s: open(p,'w').write(s)
w('system/blockMeshDict', """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ( (-0.07 -0.07 -0.005) (0.07 -0.07 -0.005) (0.07 0.07 -0.005) (-0.07 0.07 -0.005)
           (-0.07 -0.07  0.017) (0.07 -0.07  0.017) (0.07 0.07  0.017) (-0.07 0.07  0.017) );
blocks ( hex (0 1 2 3 4 5 6 7) (30 30 10) simpleGrading (1 1 1) );
edges ( ); faces ( ); boundary ( );
""")
w('system/snappyHexMeshDict', """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true; snap true; addLayers false;
geometry { GEOENTRIES }REPLACE_GE
castellatedMeshControls {
  maxLocalCells 4000000; maxGlobalCells 5000000; minRefinementCells 10; maxLoadUnbalance 0.2;
  nCellsBetweenLevels 2; allowFreeStandingZoneFaces true;
  locationInMesh (0.030 0.002 0.007);   // inside fluid between two blade passages, r~30mm
  refinementSurfaces { SURFENTRIES }REPLACE_SU
  resolveFeatureAngle 30; planarTolerance 0.001;
}
snapControls {
  nSmoothPatch 3; tolerance 2.0; nSolveTolerance 20; nRelaxIter 5;
  nFeatureSnapIter 15; implicitFeatureSnapshot false; explicitFeatureSnapshot true;
  multiRegionFeatureSnapshot false;
}
addLayersControls { layers ( ); relativeSizes false; }
meshQualityControls {
  maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02;
  minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -0.999;
  nSmoothScale 4; errorReduction 0.75;
}
mergeTolerance 1e-06;
""")
s = open('system/snappyHexMeshDict').read()
s = s.replace('GEOENTRIES', geo_entries).replace('SURFENTRIES', "\n".join(
  f'  {n} {{ level (1 1); }}' for n in names))
open('system/snappyHexMeshDict','w').write(s.replace('}REPLACE_GE','}').replace('}REPLACE_SU','}'))
r1 = sp.run(['blockMesh'], capture_output=True, text=True)
print('blockMesh', r1.returncode, r1.stdout[-150:], r1.stderr[-200:])

# -- cell 12 ------------------------------------------------------------------------
print(open('system/controlDict').read())
r = sp.run(['blockMesh'], capture_output=True, text=True)
print(r.stderr)
