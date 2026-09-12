"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import numpy as np, os
m, alpha = 3.0, np.radians(20.0)
Np, Nw   = 20, 30
BACKLASH = 0.3          # mm, total along the pitch line
F        = 18.0         # face width, mm
gam_p, gam_w = np.arctan2(Np, Nw), np.arctan2(Nw, Np)
A = 0.5*m*np.hypot(Np, Nw)      # outer cone distance

def profile(N, gam, nf=14, nt=4, nr=5):
    """closed back-cone tooth profile of one bevel gear, returns (n,3) local pts"""
    r  = 0.5*m*N; Rv = r/np.cos(gam); Nv = 2*Rv/m
    rb = Rv*np.cos(alpha); ra = Rv + m; rf = Rv - 1.25*m
    half = (0.5*np.pi*m/2 - 0.25*BACKLASH)/Rv          # half tooth angle at pitch
    inv  = lambda rr: np.tan(np.arccos(rb/rr)) - np.arccos(rb/rr)
    ph   = lambda rr: half + inv(Rv) - inv(rr)
    rs   = np.linspace(max(rb, rf), ra, nf)
    phs  = ph(rs)
    if rf < rb:                                         # radial extension below base circle
        rs  = np.concatenate([np.linspace(rf, rb, nr, endpoint=False), rs])
        phs = np.concatenate([np.full(nr, ph(rb)), phs])
    pts = []
    for k in range(N):
        c = 2*np.pi*k/Nv
        pts += [(rr, c-p) for rr, p in zip(rs, phs)]                       # left flank up
        pts += [(ra, c+t) for t in np.linspace(-phs[-1], phs[-1], nt)[1:-1]]  # tip land
        pts += [(rr, c+p) for rr, p in zip(rs[::-1], phs[::-1])]           # right flank down
        pts += [(rf, c+phs[0]+t) for t in np.linspace(0, 2*np.pi/Nv-2*phs[0], nr)[1:-1]]
    rho = np.array([p[0] for p in pts]); th = np.array([p[1] for p in pts])
    fi  = th/np.cos(gam); R = rho*np.cos(gam); z = A*np.cos(gam) - (rho-Rv)*np.sin(gam)
    return np.c_[R*np.cos(fi), R*np.sin(fi), z], dict(r=r, Rv=Rv, rb=rb, half=half)

prof_p, ip = profile(Np, gam_p)
prof_w, iw = profile(Nw, gam_w)
print(f"cone angles  pinion {np.degrees(gam_p):.3f}  wheel {np.degrees(gam_w):.3f}  sum {np.degrees(gam_p+gam_w):.3f}")
print(f"pitch radii  {ip['r']:.3f} / {iw['r']:.3f} mm   outer cone dist {A:.3f} mm")
print(f"circular thickness at pitch: {2*ip['half']*ip['Rv']:.4f} + {2*iw['half']*iw['Rv']:.4f}"
      f" = {2*ip['half']*ip['Rv']+2*iw['half']*iw['Rv']:.4f} mm   vs pitch {np.pi*m:.4f}"
      f"  -> backlash {np.pi*m-2*ip['half']*ip['Rv']-2*iw['half']*iw['Rv']:.4f} mm")
print("profile pts", prof_p.shape, prof_w.shape)

# -- cell 2 -------------------------------------------------------------------------
# Shell assembly: outer profile loop, the same loop scaled toward the pitch apex (straight bevel, flan
def shell_from_loop(loop, s, zc):
    n = len(loop); V = np.vstack([loop, s*loop, [[0,0,zc]], [[0,0,s*zc]]])
    co, ci = 2*n, 2*n+1; T = []
    for i in range(n):
        j = (i+1) % n
        T += [[i, j, n+j], [i, n+j, n+i]]        # lateral
        T += [[co, i, j], [ci, n+j, n+i]]        # caps
    T = np.array(T)
    p0, p1, p2 = V[T[:,0]], V[T[:,1]], V[T[:,2]]
    vol = np.einsum('ij,ij->i', p0, np.cross(p1-p0, p2-p0)).sum()/6.0
    if vol < 0: T = T[:, ::-1]; vol = -vol
    return V, T, vol

def cylinder(r, z0, z1, nseg=48):
    a = np.linspace(0, 2*np.pi, nseg, endpoint=False)
    c = np.c_[r*np.cos(a), r*np.sin(a)]
    V = np.vstack([np.c_[c, np.full(nseg, z0)], np.c_[c, np.full(nseg, z1)],
                   [[0,0,z0]], [[0,0,z1]]]); T = []
    for i in range(nseg):
        j = (i+1) % nseg
        T += [[i, j, nseg+j], [i, nseg+j, nseg+i], [2*nseg, j, i], [2*nseg+1, nseg+i, nseg+j]]
    T = np.array(T)
    p0, p1, p2 = V[T[:,0]], V[T[:,1]], V[T[:,2]]
    vol = np.einsum('ij,ij->i', p0, np.cross(p1-p0, p2-p0)).sum()/6.0
    if vol < 0: T = T[:, ::-1]; vol = -vol
    return V, T, vol

s_in = (A - F)/A
Vp, Tp, volp = shell_from_loop(prof_p, s_in, A*np.cos(gam_p))
Vw, Tw, volw = shell_from_loop(prof_w, s_in, A*np.cos(gam_w))
print(f"pinion shell {len(Tp)} tris  vol {volp/1000:.2f} cm^3  bbox {Vp.min(0).round(2)} {Vp.max(0).round(2)}")
print(f"wheel  shell {len(Tw)} tris  vol {volw/1000:.2f} cm^3  bbox {Vw.min(0).round(2)} {Vw.max(0).round(2)}")
def watertight(T):
    e = {}
    for t in T:
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            e[(min(a,b),max(a,b))] = e.get((min(a,b),max(a,b)),0)+1
    return set(e.values())
print("edge multiplicities:", watertight(Tp), watertight(Tw))

# -- cell 3 -------------------------------------------------------------------------
# Now place them in the global frame: pinion axis = +x, wheel axis = +z, common pitch apex at the orig
def rot_z(V, a):
    c, s = np.cos(a), np.sin(a)
    return V @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]).T   # rotate points by +a about z

def to_x_axis(V):        # local z -> global +x
    return np.c_[V[:,2], V[:,1], -V[:,0]]

Vp_g = to_x_axis(rot_z(Vp, np.pi))                 # pinion tooth centred on the contact line
Vw_g = rot_z(Vw, np.pi/Nw)                         # wheel space centred on the contact line
P_pitch = np.array([A*np.sin(gam_p)*np.cos(0)*0 + A*np.cos(gam_p), 0.0, A*np.sin(gam_p)])
print("outer pitch point (global):", P_pitch.round(3))
print("pinion bbox", Vp_g.min(0).round(2), Vp_g.max(0).round(2))
print("wheel  bbox", Vw_g.min(0).round(2), Vw_g.max(0).round(2))

def densify(V, T, ctr, rad, n=6):
    keep = T[(np.linalg.norm(V[T].mean(1) - ctr, axis=1) < rad)]
    bc = np.array([[i/n, j/n, 1-i/n-j/n] for i in range(n+1) for j in range(n+1-i)])
    p = np.einsum('kb,tbd->ktd', bc, V[keep]).reshape(-1, 3)
    return p
ap = densify(Vp_g, Tp, P_pitch, 22.0); aw = densify(Vw_g, Tw, P_pitch, 22.0)
d = np.linalg.norm(ap[:, None, :] - aw[None, :, :], axis=2)
print(f"sampled pts {len(ap)}/{len(aw)}   min pinion-wheel surface gap = {d.min():.4f} mm")

# -- cell 4 -------------------------------------------------------------------------
# No interference (min gap +0.095 mm) — the pair is separate. Adding the shafts and the cavity box, th
import pyvista as pv
pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
BOX_LO = np.array([-70.0, -70.0, -50.0]); BOX_HI = np.array([90.0, 70.0, 70.0])
RSH_P, RSH_W = 12.0, 15.0
Vsp, Tsp, _ = cylinder(RSH_P, 35.0, 95.0); Vsp = to_x_axis(Vsp*np.array([1,1,1]))
Vsp = np.c_[Vsp[:,0], Vsp[:,1], Vsp[:,2]]      # cylinder built about local z -> x axis
Vsw, Tsw, _ = cylinder(RSH_W, 25.0, 75.0)
print("cavity size", (BOX_HI-BOX_LO), " shaft-p bbox", Vsp.min(0).round(1), Vsp.max(0).round(1),
      " shaft-w bbox", Vsw.min(0).round(1), Vsw.max(0).round(1))

def pvmesh(V, T): return pv.PolyData(V, np.c_[np.full(len(T), 3), T].ravel())
pl = pv.Plotter(off_screen=True, window_size=(1100, 850))
pl.add_mesh(pvmesh(Vp_g, Tp), color="orange", show_edges=True, line_width=1)
pl.add_mesh(pvmesh(Vw_g, Tw), color="steelblue", show_edges=True, line_width=1)
pl.add_mesh(pvmesh(Vsp, Tsp), color="gray"); pl.add_mesh(pvmesh(Vsw, Tsw), color="gray")
pl.add_mesh(pv.Box(bounds=(BOX_LO[0],BOX_HI[0],BOX_LO[1],BOX_HI[1],BOX_LO[2],BOX_HI[2])),
            style="wireframe", color="k")
pl.camera_position = [(300, -320, 260), (10, 0, 10), (0, 0, 1)]
pl.screenshot("assembly.png"); print("ok")

# -- cell 5 -------------------------------------------------------------------------
from IPython.display import Image, display
print(os.path.getsize("assembly.png"))
display(Image("assembly.png"))

# -- cell 6 -------------------------------------------------------------------------
# A proper bevel pair. Close-up on the mesh zone to confirm the teeth interleave:
pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
pl.add_mesh(pvmesh(Vp_g, Tp), color="orange", show_edges=True, line_width=1)
pl.add_mesh(pvmesh(Vw_g, Tw), color="steelblue", show_edges=True, line_width=1)
pl.camera_position = [(120, -95, 105), (42, 0, 27), (0, 0, 1)]
pl.camera.zoom(2.2)
pl.screenshot("mesh_zone.png"); display(Image("mesh_zone.png"))

# -- cell 7 -------------------------------------------------------------------------
# Teeth interleave cleanly. Now write one STL per patch, in **metres** (the build is in mm, so divide 
os.makedirs("constant/triSurface", exist_ok=True)
def write_stl(name, V, T):
    Vm = V/1000.0; p0, p1, p2 = Vm[T[:,0]], Vm[T[:,1]], Vm[T[:,2]]
    n = np.cross(p1-p0, p2-p0); L = np.linalg.norm(n, axis=1); n = n/np.where(L>0, L, 1)[:, None]
    with open(f"constant/triSurface/{name}.stl", "w") as f:
        f.write(f"solid {name}\n")
        for k in range(len(T)):
            f.write(f" facet normal {n[k,0]:.6e} {n[k,1]:.6e} {n[k,2]:.6e}\n  outer loop\n")
            for p in (p0[k], p1[k], p2[k]): f.write(f"   vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")
    return len(T)

bx = np.array([[BOX_LO[0],BOX_LO[1],BOX_LO[2]],[BOX_HI[0],BOX_LO[1],BOX_LO[2]],[BOX_HI[0],BOX_HI[1],BOX_LO[2]],[BOX_LO[0],BOX_HI[1],BOX_LO[2]],
               [BOX_LO[0],BOX_LO[1],BOX_HI[2]],[BOX_HI[0],BOX_LO[1],BOX_HI[2]],[BOX_HI[0],BOX_HI[1],BOX_HI[2]],[BOX_LO[0],BOX_HI[1],BOX_HI[2]]])
bt = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
for nm, (V, T) in dict(pinion=(Vp_g,Tp), wheel=(Vw_g,Tw), bore_pinion=(Vsp,Tsp),
                       bore_wheel=(Vsw,Tsw), walls=(bx,bt)).items():
    print(nm, write_stl(nm, V, T), "tris")

for nm, V, gam in (("pinion", Vp, gam_p), ("wheel", Vw, gam_w)):   # teeth = radial maxima of the loop
    loop = prof_p if nm == "pinion" else prof_w
    rad = np.hypot(loop[:,0], loop[:,1]); mx = rad.max()
    cnt = np.sum((rad > mx-1e-9) & (np.roll(rad, 1) < mx-1e-9))
    print(f"{nm}: tip-land runs counted = {cnt}")

# -- cell 8 -------------------------------------------------------------------------
# Case files: a 5 mm background box (31k cells) and snappy cutting the four solids out of it, patches 
for d in ("system", "constant"): os.makedirs(d, exist_ok=True)
head = "FoamFile{ version 2.0; format ascii; class dictionary; object %s; }\n"
LO, HI = BOX_LO/1000.0, BOX_HI/1000.0
NB = np.round((HI-LO)/0.005).astype(int)
open("system/blockMeshDict","w").write(head % "blockMeshDict" + f"""
scale 1;
vertices ( ({LO[0]} {LO[1]} {LO[2]}) ({HI[0]} {LO[1]} {LO[2]}) ({HI[0]} {HI[1]} {LO[2]}) ({LO[0]} {HI[1]} {LO[2]})
           ({LO[0]} {LO[1]} {HI[2]}) ({HI[0]} {LO[1]} {HI[2]}) ({HI[0]} {HI[1]} {HI[2]}) ({LO[0]} {HI[1]} {HI[2]}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({NB[0]} {NB[1]} {NB[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( walls {{ type wall; faces ( (0 3 2 1) (4 5 6 7) (0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3) ); }} );
mergePatchPairs ();
""")
open("system/controlDict","w").write(head % "controlDict" + """
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 8;
runTimeModifiable true;
""")
open("system/fvSchemes","w").write(head % "fvSchemes" + "gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(head % "fvSolution" + "solvers{}\n")
import subprocess
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-600:], r.stderr[-400:])

# -- cell 9 -------------------------------------------------------------------------
# snappyHexMeshDict — each STL is its own named patch; no layers for now; coarse level 2 on the gears.
def snappy_dict(lev_gear=2, lev_shaft=1, extra_region=""):
    geo = "\n".join(f'  {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in
                    ("pinion","wheel","bore_pinion","bore_wheel"))
    surf = "\n".join(f'    {n} {{ level ({l} {l}); patchInfo {{ type wall; }} }}' for n, l in
                     (("pinion",lev_gear),("wheel",lev_gear),("bore_pinion",lev_shaft),("bore_wheel",lev_shaft)))
    return head % "snappyHexMeshDict" + f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geo}
}}
castellatedMeshControls
{{
  maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 10; maxLoadUnbalance 0.1;
  nCellsBetweenLevels 2; resolveFeatureAngle 30; allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces
  {{
{surf}
  }}
  refinementRegions {{ {extra_region} }}
  locationInMesh (-0.060 -0.060 -0.040);
}}
snapControls
{{
  nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
  nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls
{{
  relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
  nGrow 0; featureAngle 130; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
  maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
  nLayerIter 50;
}}
meshQualityControls
{{
  maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
  minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
  minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
  relaxed {{ maxNonOrtho 75; }}
}}
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snappy_dict())
r = subprocess.run("snappyHexMesh -overwrite > log.snappy 2>&1", shell=True)
print(r.returncode, open("log.snappy").read()[-1500:])

# -- cell 10 ------------------------------------------------------------------------
# 76 k cells, no errors. checkMesh and a look at the snapped gear patches:
r = subprocess.run("checkMesh > log.check 2>&1", shell=True)
txt = open("log.check").read()
print(txt[txt.find("Checking geometry"):][:1800])

# -- cell 11 ------------------------------------------------------------------------
# Volume 2564.5 cm³ vs box 2688 minus solids ≈ consistent. Let me look at the snapped patches before r
open("case.foam","w").close()
rd = pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
rd.patch_array_status = {k: ("patch/" in k or True) for k in rd.patch_array_names}
mb = rd.read(); bnd = mb["boundary"]
print(list(bnd.keys()))
pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
cols = dict(pinion="orange", wheel="steelblue", bore_pinion="gray", bore_wheel="gray")
for k, c in cols.items(): pl.add_mesh(bnd[k], color=c, show_edges=True, line_width=1)
pl.camera_position = [(170, -160, 140), (42, 0, 20), (0, 0, 1)]; pl.camera.zoom(1.4)
pl.screenshot("snap_l2.png"); display(Image("snap_l2.png"))

# -- cell 12 ------------------------------------------------------------------------
# Camera was in mm; the mesh is in metres. Re-aiming:
pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
for k, c in cols.items(): pl.add_mesh(bnd[k], color=c, show_edges=True, line_width=1)
pl.camera_position = [(0.170, -0.160, 0.140), (0.042, 0, 0.020), (0, 0, 1)]; pl.camera.zoom(1.4)
pl.screenshot("snap_l2.png"); display(Image("snap_l2.png"))

# -- cell 13 ------------------------------------------------------------------------
import time; time.sleep(120); print(open("log.snappy").read()[-700:])
