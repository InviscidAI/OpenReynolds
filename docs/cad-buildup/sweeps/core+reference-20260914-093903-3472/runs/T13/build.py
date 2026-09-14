"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll send that same cell through the tool.
import numpy as np, matplotlib.pyplot as plt, os

# --- given by the request ---
OD      = 0.040      # m, neck outside diameter
PITCH   = 0.002      # m
DEPTH   = 0.001      # m, thread depth
HTHREAD = 0.012      # m, threaded neck height
GAP     = 0.0002     # m, clearance on every flank

# --- chosen by me (not stated in the request) ---
CREST_FLAT = 0.0005  # m, axial width of the crest flat (contact land)
ROOT_FLAT  = 0.0005  # m, axial width of the root flat

r_c, r_r = OD/2, OD/2 - DEPTH
run = (PITCH - CREST_FLAT - ROOT_FLAT)/2
zA  = CREST_FLAT/2
A  = np.array([r_c, zA])
P1 = np.array([r_r, zA + run])
P2 = np.array([r_r, zA + run + ROOT_FLAT])
B  = np.array([r_c, zA + 2*run + ROOT_FLAT])

def outward_normal(p, q):
    d = (q - p)/np.linalg.norm(q - p)
    return np.array([d[1], -d[0]])
n1, n2, n3 = outward_normal(A,P1), outward_normal(P1,P2), outward_normal(P2,B)

def inter(p, dp, q, dq):
    t = np.linalg.solve(np.column_stack([dp, -dq]), q - p)[0]
    return p + t*dp
A1 = A + GAP*n1
Q1 = inter(A1, P1-A, P1+GAP*n2, P2-P1)
Q2 = inter(P1+GAP*n2, P2-P1, P2+GAP*n3, B-P2)
B3 = B + GAP*n3

print("flank angle from radial: %.2f deg" % np.degrees(np.arctan2(run, DEPTH)))
print("A ",A," P1",P1," P2",P2," B ",B)
print("A1",A1," Q1",Q1," Q2",Q2," B3",B3)
print("section axial span %.4f mm (pitch %.1f mm)" % (1e3*(B[1]-A[1]), 1e3*PITCH))
print("perp gap: flank1 %.4f mm, root %.4f mm, flank2 %.4f mm" % (
    1e3*np.dot(A1-A,n1), 1e3*np.dot(Q1-P1,n2), 1e3*np.dot(B3-B,n3)))

poly = np.array([A,P1,P2,B,B3,Q2,Q1,A1,A])
plt.figure(figsize=(5,5)); plt.plot(1e3*poly[:,0],1e3*poly[:,1],'o-')
plt.gca().set_aspect('equal'); plt.xlabel('r [mm]'); plt.ylabel('z [mm]'); plt.grid(True)
plt.title('leak-channel section'); plt.show()

# -- cell 2 -------------------------------------------------------------------------
# Section is exact: 0.2000 mm normal gap on both flanks and the root, 1.5 mm axial span inside a 2 mm 
import gmsh, math, numpy as np
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("leak")
g = gmsh.model.geo

TURNS = (HTHREAD - (B[1]-A[1]))/PITCH   # 5.25 -> total axial extent exactly 12 mm
print("turns", TURNS, "advance mm", 1e3*TURNS*PITCH, "total extent mm", 1e3*(TURNS*PITCH + (B[1]-A[1])))

pts = {}
def P(rz):
    k = (round(rz[0],9), round(rz[1],9))
    if k not in pts: pts[k] = g.addPoint(rz[0], 0.0, rz[1])
    return pts[k]
def L(a,b): return g.addLine(P(a),P(b))

# quad1 flank1 band, quad2 root band, quad3 flank2 band
l_neck1, l_t1, l_lid1, l_sealA = L(A,P1), L(P1,Q1), L(Q1,A1), L(A1,A)
l_neckR, l_t2, l_lidR          = L(P1,P2), L(P2,Q2), L(Q2,Q1)
l_neck2, l_sealB, l_lid2       = L(P2,B), L(B,B3), L(B3,Q2)

s1 = g.addPlaneSurface([g.addCurveLoop([l_neck1, l_t1, l_lid1, l_sealA])])
s2 = g.addPlaneSurface([g.addCurveLoop([l_neckR, l_t2, l_lidR, -l_t1])])
s3 = g.addPlaneSurface([g.addCurveLoop([l_neck2, l_sealB, l_lid2, -l_t2])])

NT, N1, N2 = 3, 5, 3          # nodes across gap / along flank / along root  (coarse)
for c in (l_t1, l_t2, l_sealA, l_sealB): g.mesh.setTransfiniteCurve(c, NT)
for c in (l_neck1, l_lid1, l_neck2, l_lid2): g.mesh.setTransfiniteCurve(c, N1)
for c in (l_neckR, l_lidR): g.mesh.setTransfiniteCurve(c, N2)
for s in (s1,s2,s3):
    g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2, s)
g.synchronize()

# edge -> patch, in the loop order used above (needed to label swept lateral faces)
edge_patch = {s1:[('neck_flank',(A,P1)),(None,(P1,Q1)),('lid_flank',(Q1,A1)),('lid_flank',(A1,A))],
              s2:[('neck_flank',(P1,P2)),(None,(P2,Q2)),('lid_flank',(Q2,Q1)),(None,(Q1,P1))],
              s3:[('neck_flank',(P2,B)),('lid_flank',(B,B3)),('lid_flank',(B3,Q2)),(None,(Q2,P2))]}
print("surfaces", s1,s2,s3)

# -- cell 3 -------------------------------------------------------------------------
# Twist-extrude all three quads in one call so shared internal curves are swept once; inspect the retu
ANG = math.pi/2
NLAY = 8            # layers per quarter turn (coarse first look)
dz   = PITCH*ANG/(2*math.pi)
out = g.twist([(2,s1),(2,s2),(2,s3)], 0,0,dz, 0,0,0, 0,0,1, ANG, [NLAY], [], True)
g.synchronize()
for i,o in enumerate(out): print(i, o)

# -- cell 4 -------------------------------------------------------------------------
# Shared laterals are reused (tags 23, 45 appear twice), and each block is `[top, volume, lat0..lat3]`
def swept_mid(e, ang, dzc, th0, z0):
    """centre of the surface swept from edge e=(p,q) over [th0,th0+ang], lifted from z0"""
    m = 0.5*(e[0]+e[1]); th = th0+ang/2
    return np.array([m[0]*math.cos(th), m[0]*math.sin(th), m[1]+z0+dzc/2])

def check_block(blk, s, ang, dzc, th0, z0):
    ok = True
    for i,(pname,e) in enumerate(edge_patch[s]):
        tag = blk[2+i][1]
        c = np.array(gmsh.model.occ.getCenterOfMass(2, tag)) if False else None
        c = np.array(gmsh.model.getValue(2, tag, [0.5,0.5]))  # param mid (extruded surf: u,v in [0,1])
        d = np.linalg.norm(c - swept_mid(e, ang, dzc, th0, z0))
        ok &= d < 1e-5
        print("  s%d lat%d tag%3d %-11s dist %.2e" % (s, i, tag, pname, d))
    return ok

blocks = {s1:out[0:6], s2:out[6:12], s3:out[12:18]}
print(all(check_block(blocks[s], s, ANG, dz, 0.0, 0.0) for s in (s1,s2,s3)))

# -- cell 5 -------------------------------------------------------------------------
for i,(pname,e) in enumerate(edge_patch[s1]):
    tag = out[2+i][1]
    print(tag, pname, np.array(gmsh.model.getValue(2, tag, [0.5,0.5])), swept_mid(e, ANG, dz, 0.0, 0.0))

# -- cell 6 -------------------------------------------------------------------------
# x, y match the half-angle swept midpoint exactly for every edge — the ordering is confirmed (the z d
def check_block(blk, s, ang, th0):
    ok = True
    for i,(pname,e) in enumerate(edge_patch[s]):
        tag = blk[2+i][1]
        c = np.array(gmsh.model.getValue(2, tag, [0.5,0.5]))
        exp = swept_mid(e, ang, 0.0, th0, 0.0)
        ok &= np.linalg.norm(c[:2]-exp[:2]) < 1e-9
    return ok
assert all(check_block(blocks[s], s, ANG, 0.0) for s in (s1,s2,s3)), "lateral ordering unverified"

patch_surfs = {'neck_flank':[], 'lid_flank':[]}
def record(blk, s):
    for i,(pname,e) in enumerate(edge_patch[s]):
        if pname: patch_surfs[pname].append(blk[2+i][1])
for s in (s1,s2,s3): record(blocks[s], s)

NCHUNK = round(2*math.pi*TURNS/ANG)
cur = {s1: out[0][1], s2: out[6][1], s3: out[12][1]}   # top surfaces after chunk 1
for k in range(1, NCHUNK):
    o = g.twist([(2,cur[s1]),(2,cur[s2]),(2,cur[s3])], 0,0,dz, 0,0,0, 0,0,1, ANG, [NLAY], [], True)
    g.synchronize()
    blk = {s1:o[0:6], s2:o[6:12], s3:o[12:18]}
    assert all(check_block(blk[s], s, ANG, ANG*k) for s in (s1,s2,s3)), f"ordering chunk {k}"
    for s in (s1,s2,s3): record(blk[s], s)
    cur = {s: o[j][1] for s,j in ((s1,0),(s2,6),(s3,12))}
print("chunks", NCHUNK, "  lateral surfs:", {k:len(v) for k,v in patch_surfs.items()})
print("top surfaces", cur)
bb = gmsh.model.getBoundingBox(-1,-1); print("bbox mm", [round(1e3*x,4) for x in bb])

# -- cell 7 -------------------------------------------------------------------------
# Argument order bug: `geo.twist` takes the axis **point** first and the translation second, so I buil
gmsh.clear(); gmsh.model.add("leak"); g = gmsh.model.geo
pts = {}
def P(rz):
    k = (round(rz[0],9), round(rz[1],9))
    if k not in pts: pts[k] = g.addPoint(rz[0], 0.0, rz[1])
    return pts[k]
def L(a,b): return g.addLine(P(a),P(b))

l_neck1, l_t1, l_lid1, l_sealA = L(A,P1), L(P1,Q1), L(Q1,A1), L(A1,A)
l_neckR, l_t2, l_lidR          = L(P1,P2), L(P2,Q2), L(Q2,Q1)
l_neck2, l_sealB, l_lid2       = L(P2,B), L(B,B3), L(B3,Q2)
s1 = g.addPlaneSurface([g.addCurveLoop([l_neck1, l_t1, l_lid1, l_sealA])])
s2 = g.addPlaneSurface([g.addCurveLoop([l_neckR, l_t2, l_lidR, -l_t1])])
s3 = g.addPlaneSurface([g.addCurveLoop([l_neck2, l_sealB, l_lid2, -l_t2])])
NT, N1, N2 = 3, 5, 3
for c in (l_t1, l_t2, l_sealA, l_sealB): g.mesh.setTransfiniteCurve(c, NT)
for c in (l_neck1, l_lid1, l_neck2, l_lid2): g.mesh.setTransfiniteCurve(c, N1)
for c in (l_neckR, l_lidR): g.mesh.setTransfiniteCurve(c, N2)
for s in (s1,s2,s3):
    g.mesh.setTransfiniteSurface(s); g.mesh.setRecombine(2, s)
g.synchronize()
edge_patch = {s1:[('neck_flank',(A,P1)),(None,(P1,Q1)),('lid_flank',(Q1,A1)),('lid_flank',(A1,A))],
              s2:[('neck_flank',(P1,P2)),(None,(P2,Q2)),('lid_flank',(Q2,Q1)),(None,(Q1,P1))],
              s3:[('neck_flank',(P2,B)),('lid_flank',(B,B3)),('lid_flank',(B3,Q2)),(None,(Q2,P2))]}

ANG, NLAY = math.pi/2, 8
dz = PITCH*ANG/(2*math.pi)
NCHUNK = round(2*math.pi*TURNS/ANG)
patch_surfs = {'neck_flank':[], 'lid_flank':[]}
bottom = [s1,s2,s3]
cur, vols = {s1:s1, s2:s2, s3:s3}, []
for k in range(NCHUNK):
    o = g.twist([(2,cur[s1]),(2,cur[s2]),(2,cur[s3])], 0,0,0, 0,0,dz, 0,0,1, ANG, [NLAY], [], True)
    g.synchronize()
    blk = {s1:o[0:6], s2:o[6:12], s3:o[12:18]}
    for s in (s1,s2,s3):
        assert check_block(blk[s], s, ANG, ANG*k), f"ordering chunk {k}"
        for i,(pname,e) in enumerate(edge_patch[s]):
            if pname: patch_surfs[pname].append(blk[s][2+i][1])
        vols.append(blk[s][1][1])
    cur = {s: o[j][1] for s,j in ((s1,0),(s2,6),(s3,12))}
top = [cur[s] for s in (s1,s2,s3)]
bb = gmsh.model.getBoundingBox(-1,-1)
print("bbox mm", [round(1e3*x,4) for x in bb], " z extent %.4f mm" % (1e3*(bb[5]-bb[2])))
print("vols", len(vols), {k:len(v) for k,v in patch_surfs.items()})

# -- cell 8 -------------------------------------------------------------------------
# Exactly 12.000 mm of thread height. Now physical groups (inner end = top, the jar-mouth side; outer 
for name, surfs in [('neck_flank', patch_surfs['neck_flank']),
                    ('lid_flank',  patch_surfs['lid_flank']),
                    ('inner_end',  top),        # z = 12.25 mm, jar-mouth side
                    ('outer_end',  bottom)]:    # z = 0.25 mm, open air
    gmsh.model.addPhysicalGroup(2, surfs, name=name)
gmsh.model.addPhysicalGroup(3, vols, name="internal")
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.generate(2)
et = {2:'tri', 3:'quad'}
for d in (2,):
    ts, tags, _ = gmsh.model.mesh.getElements(d)
    print(d, [(et.get(t,t), sum(len(x) for x in tags[i:i+1])) for i,t in enumerate(ts)])
print("nodes", len(gmsh.model.mesh.getNodes()[0]))

# -- cell 9 -------------------------------------------------------------------------
# Surface mesh is all quads. Now an analytic point-in-channel test (unwrap the helix, test the section
from matplotlib.path import Path as MPath
sec_path = MPath(np.array([A,P1,P2,B,B3,Q2,Q1,A1]))
TWOPI = 2*math.pi
def in_fluid(p):
    x,y,z = p
    r = math.hypot(x,y); th = math.atan2(y,x)
    for m in range(-1, int(TURNS)+2):
        thu = th + TWOPI*m
        if -1e-12 <= thu <= TWOPI*TURNS + 1e-12:
            if sec_path.contains_point((r, z - PITCH*thu/TWOPI)):
                return True
    return False

tests = [((0.0191,0,0.001), True), ((0.0195,0,0.001), False), ((0.0205,0,0.001), False),
         ((0.0191,0,0.0),   False),                      # below the start cap
         ((0.0191*math.cos(0.5),0.0191*math.sin(0.5),0.001+PITCH*0.5/TWOPI), True)]
print([ (in_fluid(np.array(p)), e) for p,e in tests])

# -- cell 10 ------------------------------------------------------------------------
# Inside test agrees on all five probes. Now export one STL per patch to `constant/triSurface`, with e
import os
os.makedirs("constant/triSurface", exist_ok=True)
nt, nc, _ = gmsh.model.mesh.getNodes()
XYZ = {t: np.array(nc[3*i:3*i+3]) for i,t in enumerate(nt)}

def tris_of_patch(name):
    dim_tags = gmsh.model.getEntitiesForPhysicalGroup(2, {g_.split('|')[0]:0 for g_ in []} or
               [pg[1] for pg in gmsh.model.getPhysicalGroups(2)
                if gmsh.model.getPhysicalName(2, pg[1]) == name][0])
    tris = []
    for s in dim_tags:
        ts, tags, nds = gmsh.model.mesh.getElements(2, s)
        for t, nl in zip(ts, nds):
            k = 4 if t == 3 else 3
            for i in range(0, len(nl), k):
                q = [XYZ[n] for n in nl[i:i+k]]
                tris += ([[q[0],q[1],q[2]], [q[0],q[2],q[3]]] if k == 4 else [[q[0],q[1],q[2]]])
    return tris

def write_stl(name, tris, eps=1e-5):
    flipped = 0
    with open(f"constant/triSurface/{name}.stl", "w") as f:
        f.write(f"solid {name}\n")
        for a,b,c in tris:
            n = np.cross(b-a, c-a); ln = np.linalg.norm(n); n = n/ln
            if in_fluid((a+b+c)/3.0 + eps*n): a, c, n, flipped = c, a, -n, flipped+1
            f.write(" facet normal %g %g %g\n  outer loop\n" % tuple(n))
            for v in (a,b,c): f.write("   vertex %.9g %.9g %.9g\n" % tuple(v))
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")
    return len(tris), flipped

for name in ('neck_flank','lid_flank','inner_end','outer_end'):
    print(name, write_stl(name, tris_of_patch(name)))

# -- cell 11 ------------------------------------------------------------------------
# Per-triangle orientation was applied, so let me verify closure and winding of the union of the four 
from collections import Counter
alltris = sum((tris_of_patch(n) for n in ('neck_flank','lid_flank','inner_end','outer_end')), [])
# re-apply the same orientation rule used when writing
ed = Counter(); area = 0.0
for a,b,c in alltris:
    n = np.cross(b-a, c-a); area += 0.5*np.linalg.norm(n); n = n/np.linalg.norm(n)
    if in_fluid((a+b+c)/3.0 + 1e-5*n): a, c = c, a
    for u,v in ((a,b),(b,c),(c,a)):
        ed[(tuple(np.round(u,10)), tuple(np.round(v,10)))] += 1
free = sum(1 for (u,v),k in ed.items() if (v,u) not in ed)
same = sum(k-1 for k in ed.values() if k > 1)
print("triangles", len(alltris), " free edges", free, " same-direction dup edges", same,
      " wetted+cap area mm^2 %.2f" % (1e6*area))

# -- cell 12 ------------------------------------------------------------------------
# 696 free edges = 696 same-direction duplicates, so some triangles are still wound the wrong way. Loc
bad = []
for name in ('neck_flank','lid_flank','inner_end','outer_end'):
    for a,b,c in tris_of_patch(name):
        n = np.cross(b-a,c-a); n/=np.linalg.norm(n)
        cen = (a+b+c)/3.0
        if in_fluid(cen + 1e-5*n): a,c,n = c,a,-n
        # a correctly oriented facet must have fluid just inside it
        if not in_fluid(cen - 1e-5*n): bad.append((name, cen, n))
print(len(bad), Counter(b[0] for b in bad))
for name,cen,n in bad[:5]:
    print(name, "r=%.5f th=%.3f z=%.5f" % (math.hypot(*cen[:2]), math.atan2(cen[1],cen[0]), cen[2]), n)

# -- cell 13 ------------------------------------------------------------------------
# Faceting was the real issue, not the winding: at 8 layers per quarter turn the chord sagitta at r=20
import subprocess
def build_model(nlay, nt=3, n1=5, n2=3, ang=math.pi/2):
    gmsh.clear(); gmsh.model.add("leak"); gg = gmsh.model.geo
    pp = {}
    def P(rz):
        k = (round(rz[0],9), round(rz[1],9))
        if k not in pp: pp[k] = gg.addPoint(rz[0], 0.0, rz[1])
        return pp[k]
    def L(a,b): return gg.addLine(P(a),P(b))
    e_n1,e_t1,e_l1,e_sA = L(A,P1), L(P1,Q1), L(Q1,A1), L(A1,A)
    e_nR,e_t2,e_lR      = L(P1,P2), L(P2,Q2), L(Q2,Q1)
    e_n2,e_sB,e_l2      = L(P2,B), L(B,B3), L(B3,Q2)
    a1 = gg.addPlaneSurface([gg.addCurveLoop([e_n1,e_t1,e_l1,e_sA])])
    a2 = gg.addPlaneSurface([gg.addCurveLoop([e_nR,e_t2,e_lR,-e_t1])])
    a3 = gg.addPlaneSurface([gg.addCurveLoop([e_n2,e_sB,e_l2,-e_t2])])
    for c in (e_t1,e_t2,e_sA,e_sB): gg.mesh.setTransfiniteCurve(c, nt)
    for c in (e_n1,e_l1,e_n2,e_l2): gg.mesh.setTransfiniteCurve(c, n1)
    for c in (e_nR,e_lR):           gg.mesh.setTransfiniteCurve(c, n2)
    for s in (a1,a2,a3): gg.mesh.setTransfiniteSurface(s); gg.mesh.setRecombine(2, s)
    gg.synchronize()
    ep = {a1:[('neck_flank',(A,P1)),(None,(P1,Q1)),('lid_flank',(Q1,A1)),('lid_flank',(A1,A))],
          a2:[('neck_flank',(P1,P2)),(None,(P2,Q2)),('lid_flank',(Q2,Q1)),(None,(Q1,P1))],
          a3:[('neck_flank',(P2,B)),('lid_flank',(B,B3)),('lid_flank',(B3,Q2)),(None,(Q2,P2))]}
    dzc = PITCH*ang/TWOPI; nchunk = round(TWOPI*TURNS/ang)
    ps = {'neck_flank':[], 'lid_flank':[]}; vs = []; cur = {a1:a1,a2:a2,a3:a3}
    for k in range(nchunk):
        o = gg.twist([(2,cur[a1]),(2,cur[a2]),(2,cur[a3])], 0,0,0, 0,0,dzc, 0,0,1, ang, [nlay], [], True)
        gg.synchronize()
        blk = {a1:o[0:6], a2:o[6:12], a3:o[12:18]}
        for s in (a1,a2,a3):
            for i,(pn,e) in enumerate(ep[s]):
                m = 0.5*(e[0]+e[1]); th = ang*k+ang/2
                assert np.linalg.norm(np.array(gmsh.model.getValue(2, blk[s][2+i][1], [0.5,0.5]))[:2]
                                      - np.array([m[0]*math.cos(th), m[0]*math.sin(th)])) < 1e-9
                if pn: ps[pn].append(blk[s][2+i][1])
            vs.append(blk[s][1][1])
        cur = {s:o[j][1] for s,j in ((a1,0),(a2,6),(a3,12))}
    for nm, sl in [('neck_flank',ps['neck_flank']), ('lid_flank',ps['lid_flank']),
                   ('inner_end',[cur[s] for s in (a1,a2,a3)]), ('outer_end',[a1,a2,a3])]:
        gmsh.model.addPhysicalGroup(2, sl, name=nm)
    gmsh.model.addPhysicalGroup(3, vs, name="internal")

build_model(32)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.generate(3)
ts, tags, _ = gmsh.model.mesh.getElements(3)
print("3D elems", {t:len(tags[i]) for i,t in enumerate(ts)})
gmsh.write("leak.msh")
bb = gmsh.model.getBoundingBox(-1,-1); print("bbox mm", [round(1e3*x,4) for x in bb])

# -- cell 14 ------------------------------------------------------------------------
# Now the OpenFOAM case: msh v2.2, a minimal system dict set, `gmshToFoam`, `checkMesh`.
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("leak.msh")
os.makedirs("system", exist_ok=True)
hdr = lambda cls,obj: ("FoamFile{version 2.0; format ascii; class %s; object %s;}\n" % (cls,obj))
open("system/controlDict","w").write(hdr("dictionary","controlDict") +
  "application icoFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
  "deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 6;\n"
  "writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable true;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes") +
  "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;}\n"
  "divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;}\n"
  "interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution") + "solvers{} \n")
r = subprocess.run(["gmshToFoam","leak.msh"], capture_output=True, text=True)
print(r.returncode, r.stdout[-1500:], r.stderr[-800:])

# -- cell 15 ------------------------------------------------------------------------
print(open("constant/polyMesh/boundary").read()[-1200:])

# -- cell 16 ------------------------------------------------------------------------
# All four patches populated (16 168 boundary faces, `defaultFaces` empty and dropped). Run checkMesh.
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-2600:])

# -- cell 17 ------------------------------------------------------------------------
# checkMesh: **Mesh OK** — 13 440 hexes, max aspect 10.3, max skewness 0.99, max non-orthogonality 28.
nt, nc, _ = gmsh.model.mesh.getNodes()
XYZ = {t: np.array(nc[3*i:3*i+3]) for i,t in enumerate(nt)}
for name in ('neck_flank','lid_flank','inner_end','outer_end'):
    print(name, write_stl(name, tris_of_patch(name), eps=2e-5))

alltris = sum((tris_of_patch(n) for n in ('neck_flank','lid_flank','inner_end','outer_end')), [])
ed = Counter()
for a,b,c in alltris:
    n = np.cross(b-a,c-a); n/=np.linalg.norm(n)
    if in_fluid((a+b+c)/3.0 + 2e-5*n): a,c = c,a
    for u,v in ((a,b),(b,c),(c,a)): ed[(tuple(np.round(u,10)),tuple(np.round(v,10)))] += 1
print("free edges", sum(1 for (u,v) in ed if (v,u) not in ed),
      " same-dir dups", sum(k-1 for k in ed.values() if k>1))

# -- cell 18 ------------------------------------------------------------------------
# Union of the four patch STLs is watertight (0 free edges) and consistently wound. Now look at the th
import pyvista as pv
pv.start_xvfb() if hasattr(pv,'start_xvfb') else None
pl = pv.Plotter(off_screen=True, window_size=(1100,520), shape=(1,2))
cols = dict(neck_flank='steelblue', lid_flank='indianred', inner_end='lime', outer_end='gold')
for j,(cam,zoom) in enumerate([((1,-1.6,0.9), 1.0), ((1,-0.25,0.15), 3.6)]):
    pl.subplot(0,j)
    for n,c in cols.items():
        m = pv.read(f"constant/triSurface/{n}.stl")
        pl.add_mesh(m, color=c, show_edges=(j==1), line_width=0.4, label=n)
    pl.camera_position = [tuple(20*np.array(cam)*0.001), (0,0,0.006), (0,0,1)]
    pl.camera.zoom(zoom)
pl.add_legend(); img = pl.screenshot(return_img=True)
import matplotlib.pyplot as plt
f = plt.figure(figsize=(13,6)); plt.imshow(img); plt.axis('off'); f

# -- cell 19 ------------------------------------------------------------------------
# Camera was inside the part. Frame it automatically instead.
pl = pv.Plotter(off_screen=True, window_size=(1200,560), shape=(1,2))
for j in range(2):
    pl.subplot(0,j)
    for n,c in cols.items():
        pl.add_mesh(pv.read(f"constant/triSurface/{n}.stl"), color=c,
                    label=n, show_edges=False)
    pl.view_isometric() if j==0 else pl.view_xz()
    pl.reset_camera(); pl.camera.zoom(1.3 if j==0 else 1.5)
pl.add_legend(size=(0.25,0.22))
f = plt.figure(figsize=(14,6.5)); plt.imshow(pl.screenshot(return_img=True)); plt.axis('off'); f

# -- cell 20 ------------------------------------------------------------------------
# Shape is right: a 5.25-turn ribbon, neck side inboard, lid side outboard. Now measure the requested 
neck = pv.read("constant/triSurface/neck_flank.stl")
lid  = pv.read("constant/triSurface/lid_flank.stl")
rn = np.hypot(neck.points[:,0], neck.points[:,1])
print("neck OD  %.4f mm  (asked 40)" % (2e3*rn.max()))
print("root dia %.4f mm -> thread depth %.4f mm (asked 1.0)" % (2e3*rn.min(), 1e3*(rn.max()-rn.min())))
print("z extent %.4f mm (asked 12 of neck height)" % (1e3*(neck.points[:,2].max()-neck.points[:,2].min())))

# pitch: unwrapped helical rise of the crest edge (points at r = OD/2)
sel = neck.points[rn > rn.max()-1e-7]
th = np.unwrap(np.sort(np.arctan2(sel[:,1], sel[:,0])))
crest = sel[np.argsort(np.arctan2(sel[:,1], sel[:,0]))]
thu = np.unwrap(np.arctan2(crest[:,1], crest[:,0]))
# two crest lines (corners A and B); split by residual
zres = crest[:,2] - PITCH*thu/TWOPI
for lab, m in (("A", zres < np.median(zres)), ("B", zres >= np.median(zres))):
    p = np.polyfit(thu[m], crest[m,2], 1)
    print("crest line %s: pitch %.5f mm" % (lab, 1e3*p[0]*TWOPI))

# gap: distance from neck facet centres to the lid surface
cc = neck.cell_centers().points
d = np.array([np.linalg.norm(cc[i]-lid.points[lid.find_closest_point(cc[i])]) for i in range(0,len(cc),97)])
print("neck->lid distance  min %.4f  mean %.4f  max %.4f mm (asked 0.2)" % (1e3*d.min(),1e3*d.mean(),1e3*d.max()))

xy = np.array([A,P1,P2,B,B3,Q2,Q1,A1]); x,y = xy[:,0], xy[:,1]
Asec = 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
rbar = np.mean([np.mean(xy[:,0])])  # rough; use polygon centroid
cx = (np.dot(x+np.roll(x,-1), x*np.roll(y,-1)-np.roll(x,-1)*y))/(6*Asec)
print("section area %.5f mm^2, centroid r %.4f mm" % (1e6*Asec, 1e3*abs(cx)))
print("Pappus volume %.4e m^3 vs mesh 3.20026e-07" % (Asec*TWOPI*TURNS*abs(cx)))

# -- cell 21 ------------------------------------------------------------------------
cc_pd = pv.PolyData(neck.cell_centers().points)
d = np.abs(cc_pd.compute_implicit_distance(lid)["implicit_distance"])
print("neck->lid surface distance: min %.4f mean %.4f max %.4f mm (asked 0.2)"
      % (1e3*d.min(), 1e3*d.mean(), 1e3*d.max()))
th_all = np.arctan2(neck.points[:,1], neck.points[:,0])
m = (np.abs(th_all) < 1e-6) & (rn > rn.max()-1e-9)
z = np.sort(neck.points[m,2])
print("crest z at theta=0 [mm]", np.round(1e3*z,4))
print("turn-to-turn rise [mm]", np.round(1e3*np.diff(z),4), "(asked pitch 2.0)")

# -- cell 22 ------------------------------------------------------------------------
print("neck->lid distance percentiles [mm]",
      np.round(1e3*np.percentile(d,[5,25,50,75,95,100]),4))
far = d[np.linalg.norm(neck.cell_centers().points[:,None,:] -
        np.array([[0,0,-9]]) , axis=2)[:,0] > 0]  # keep all; filter below instead
# facets away from the crest-corner seal strips: those whose distance is set by the opposing flank
print("median of the upper half (flank-to-flank) %.4f mm" % (1e3*np.median(d[d>0.15])))

m = (np.abs(th_all) < 1e-4) & (rn > rn.max()-1e-9)
z = np.sort(neck.points[m,2]); print("crest z near theta=0 [mm]", np.round(1e3*z,4))
print("rise between successive crest lines [mm]", np.round(1e3*np.diff(z),4))
res = (neck.points[rn>rn.max()-1e-9,2] - PITCH*np.arctan2(neck.points[rn>rn.max()-1e-9,1],
        neck.points[rn>rn.max()-1e-9,0])/TWOPI) % PITCH
print("crest helical residual clusters [mm]", np.round(1e3*np.unique(np.round(res,7)),4))

# -- cell 23 ------------------------------------------------------------------------
# Gap measures 0.1999–0.2040 mm over the 25th–100th percentile (the lower 5 % are facets next to the c
build_model(64, nt=5, n1=7, n2=4)      # 4 cells across the 0.2 mm gap, ~0.5 mm along the helix
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.generate(3)
ts, tags, _ = gmsh.model.mesh.getElements(3)
print("hexes", {int(t):len(tags[i]) for i,t in enumerate(ts)})
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.write("leak.msh")
nt_, nc_, _ = gmsh.model.mesh.getNodes()
XYZ = {t: np.array(nc_[3*i:3*i+3]) for i,t in enumerate(nt_)}
for name in ('neck_flank','lid_flank','inner_end','outer_end'):
    print(name, write_stl(name, tris_of_patch(name), eps=2e-5))

# -- cell 24 ------------------------------------------------------------------------
import shutil
shutil.rmtree("constant/polyMesh", ignore_errors=True)
r1 = subprocess.run(["gmshToFoam","leak.msh"], capture_output=True, text=True)
r2 = subprocess.run(["checkMesh"], capture_output=True, text=True)
print([l for l in r1.stdout.splitlines() if "Patch" in l or "undefined" in l])
print(r2.stdout[r2.stdout.find("Mesh stats"):][:400])
print(r2.stdout[r2.stdout.find("Checking geometry"):])

# -- cell 25 ------------------------------------------------------------------------
b = open("constant/polyMesh/boundary").read()
print([l.strip() for l in b.splitlines() if "nFaces" in l or "}" not in l and l.strip() and l.strip()[0].isalpha()][:12])
alltris = sum((tris_of_patch(n) for n in ('neck_flank','lid_flank','inner_end','outer_end')), [])
ed = Counter()
for a_,b_,c_ in alltris:
    n = np.cross(b_-a_,c_-a_); n/=np.linalg.norm(n)
    if in_fluid((a_+b_+c_)/3.0 + 2e-5*n): a_,c_ = c_,a_
    for u,v in ((a_,b_),(b_,c_),(c_,a_)): ed[(tuple(np.round(u,10)),tuple(np.round(v,10)))] += 1
print("tris", len(alltris), "free edges", sum(1 for (u,v) in ed if (v,u) not in ed),
      "same-dir dups", sum(k-1 for k in ed.values() if k>1))

# -- cell 26 ------------------------------------------------------------------------
import re
print(re.findall(r"^\s{4}(\w+)$", b, re.M), [int(x) for x in re.findall(r"nFaces\s+(\d+)", b)])
msh = pv.read("leak.msh") if False else None
pl = pv.Plotter(off_screen=True, window_size=(1000,500))
for n,c in cols.items():
    pl.add_mesh(pv.read(f"constant/triSurface/{n}.stl").clip('y'), color=c, show_edges=True, line_width=0.3)
pl.view_vector((0.2,-1,0.25)); pl.reset_camera(); pl.camera.zoom(3.2)
f = plt.figure(figsize=(12,6)); plt.imshow(pl.screenshot(return_img=True)); plt.axis('off'); f
