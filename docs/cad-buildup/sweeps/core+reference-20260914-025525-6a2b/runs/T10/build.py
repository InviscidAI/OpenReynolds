"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I'll start by establishing the aerofoil section geometry (parametric, in mm) and checking it numeric
import numpy as np, math, os, subprocess
print(os.getcwd(), os.listdir('.'))

# ---- given parameters (from request) ----
R_HUB   = 45.0      # mm
R_TIP   = 90.0      # mm  (blade span 45 mm)
N_BLADE = 28
CHORD   = 22.0      # mm
THICK   = 0.06      # 6% max thickness
STAG_ROOT = 30.0    # deg
TWIST     = 35.0    # deg extra at tip
PITCH = 2*math.pi/N_BLADE
# ---- assumption (not given): camber ----
CAMB_M = 0.04       # 4% max camber
CAMB_P = 0.40       # at 40% chord

def naca(n=81, m=CAMB_M, p=CAMB_P, t=THICK):
    """closed NACA-4 style section, unit chord, x in [0,1]; returns (N,2) loop"""
    beta = np.linspace(0, math.pi, n)
    x = 0.5*(1-np.cos(beta))                     # cosine clustering
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
    dy = np.where(x < p, 2*m/p**2*(p-x),        2*m/(1-p)**2*(p-x))
    th = np.arctan(dy)
    xu, yu = x-yt*np.sin(th), yc+yt*np.cos(th)
    xl, yl = x+yt*np.sin(th), yc-yt*np.cos(th)
    up = np.stack([xu, yu], 1)
    lo = np.stack([xl, yl], 1)[::-1]
    return np.vstack([up, lo[1:-1]])             # open loop (closed by wire)

sec = naca()
print("pitch deg", math.degrees(PITCH), " pitch@hub mm", PITCH*R_HUB, " @tip", PITCH*R_TIP)
print("max thickness/chord", (naca()[:,1].max()-0) , "chk t_max:",
      max(np.interp(xx,[0,1],[0,0]) for xx in [0]) )
# measure thickness directly
n=81; up=sec[:n]; lo=sec[n-1:][::-1]
print("t_max measured =", round(float(np.max(up[:,1]-np.interp(up[:,0], np.r_[lo[:,0],1.0], np.r_[lo[:,1],0.0]))),4))
print("camber max =", round(float(np.max(0.5*(up[:,1]+np.interp(up[:,0], np.r_[lo[:,0],1.0], np.r_[lo[:,1],0.0])))),4))

# -- cell 2 -------------------------------------------------------------------------
# The `loft` wrapper expects faces; my sections are non-planar wires on cylinders, so I use `Solid.mak
import build123d as bd
from build123d import Vector, Axis

X_IN  = -(0.5*CHORD*math.cos(math.radians(STAG_ROOT)) + CHORD)   # one chord upstream of blade
X_OUT = -X_IN                                                    # one chord downstream

def stagger_deg(r):
    return STAG_ROOT + TWIST*(r-R_HUB)/(R_TIP-R_HUB)

def section_3d(r, npts=81):
    """aerofoil section at radius r wrapped on the cylinder r, stacked mid-chord at x=0, theta=0"""
    s = naca(npts)
    xc = (s[:,0]-0.5)*CHORD
    yc = s[:,1]*CHORD
    g  = math.radians(stagger_deg(r))
    xa = xc*math.cos(g) - yc*math.sin(g)     # axial
    ta = xc*math.sin(g) + yc*math.cos(g)     # tangential arc length
    th = ta/r
    return [Vector(float(xa[i]), r*math.cos(th[i]), r*math.sin(th[i])) for i in range(len(th))]

R_LO, R_HI = R_HUB-2.0, R_TIP+2.0            # extended past hub/shroud for clean cutting
radii = np.linspace(R_LO, R_HI, 9)
wires = [bd.Wire(bd.Spline(section_3d(float(r)), periodic=True)) for r in radii]
blade = bd.Solid.make_loft(wires, ruled=False)
print("blade volume mm^3", round(blade.volume,1))
print("bbox", blade.bounding_box())
print("X_IN,X_OUT", round(X_IN,3), round(X_OUT,3))

# -- cell 3 -------------------------------------------------------------------------
# Blade x-extent (−16.2) is far outside the expected ±9.7 — likely spline overshoot at the sharp trail
for r in [43.0, 67.5, 92.0]:
    pts=np.array([[p.X,p.Y,p.Z] for p in section_3d(r)])
    w=bd.Wire(bd.Spline(section_3d(r), periodic=True))
    print(r, "pts x", pts[:,0].min().round(3), pts[:,0].max().round(3), " wire bbox x",
          round(w.bounding_box().min.X,3), round(w.bounding_box().max.X,3), " len", round(w.length,3))

# -- cell 4 -------------------------------------------------------------------------
# Wires are exact; the loft bbox is probably OCC's inflated estimate. Let me tessellate the solid and 
v,f = blade.tessellate(0.05)
P = np.array([[p.X,p.Y,p.Z] for p in v])
r = np.hypot(P[:,1],P[:,2])
print("tess x", P[:,0].min().round(3), P[:,0].max().round(3), " r", r.min().round(3), r.max().round(3))
print("ntri", len(f))

# -- cell 5 -------------------------------------------------------------------------
# Ruled loft on periodic splines fails; I'll rebuild each section as two splines (upper, lower) meetin
def naca_sides(n=61, m=CAMB_M, p=CAMB_P, t=THICK):
    """upper & lower surfaces LE->TE, unit chord, sharp closed TE"""
    beta = np.linspace(0, math.pi, n); x = 0.5*(1-np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yt = yt - x*yt[-1]                                  # close the TE exactly
    yc = np.where(x < p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
    dy = np.where(x < p, 2*m/p**2*(p-x),        2*m/(1-p)**2*(p-x))
    th = np.arctan(dy)
    up = np.stack([x-yt*np.sin(th), yc+yt*np.cos(th)], 1)
    lo = np.stack([x+yt*np.sin(th), yc-yt*np.cos(th)], 1)
    return up, lo

def wrap(pts2, r):
    """(x_chord,y_chord) unit-chord section coords -> 3D points on cylinder radius r"""
    g = math.radians(stagger_deg(r))
    xc = (pts2[:,0]-0.5)*CHORD; yc = pts2[:,1]*CHORD
    xa = xc*math.cos(g) - yc*math.sin(g)
    ta = xc*math.sin(g) + yc*math.cos(g)
    th = ta/r
    return [Vector(float(xa[i]), r*math.cos(th[i]), r*math.sin(th[i])) for i in range(len(th))]

def section_wire(r):
    up, lo = naca_sides()
    return bd.Wire([bd.Spline(wrap(up, r)), bd.Spline(wrap(lo[::-1], r))])

radii = np.linspace(R_LO, R_HI, 15)
wires = [section_wire(float(r)) for r in radii]
blade = bd.Solid.make_loft(wires, ruled=True)
print("volume mm^3", round(blade.volume,2), "faces", len(blade.faces()))
v,f = blade.tessellate(0.2); P=np.array([[q.X,q.Y,q.Z] for q in v]); rr=np.hypot(P[:,1],P[:,2])
print("x", P[:,0].min().round(3), P[:,0].max().round(3), "r", rr.min().round(3), rr.max().round(3), "ntri", len(f))

# -- cell 6 -------------------------------------------------------------------------
up,lo = naca_sides(401)
A2 = np.trapezoid(up[:,1],up[:,0]) - np.trapezoid(lo[:,1],lo[:,0])
print("section area unit-chord", round(A2,5), "-> mm^2", round(A2*CHORD**2,3))
print("area*span =", round(A2*CHORD**2*(R_HI-R_LO),1), " loft volume", round(blade.volume,1))
print("t_max/c =", round(float(np.max(up[:,1]-lo[:,1])),4),
      " camber_max/c =", round(float(np.max(0.5*(up[:,1]+lo[:,1]))),4))

# -- cell 7 -------------------------------------------------------------------------
print("valid?", blade.is_valid)
v,f = blade.tessellate(0.05)
P=np.array([[q.X,q.Y,q.Z] for q in v]); T=np.array(f)
a,b,c = P[T[:,0]],P[T[:,1]],P[T[:,2]]
print("tess volume", round(abs(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6.0),1),
      " occ volume", round(blade.volume,1), "ntri",len(T))
print("wire len r=67.5:", round(section_wire(67.5).length,3), " perimeter est", round(2.04*CHORD,2))

# -- cell 8 -------------------------------------------------------------------------
# Blade solid is invalid. Switching to polyline sections for a robust ruled loft, and checking validit
def section_poly(r, n=61):
    up, lo = naca_sides(n)
    pts2 = np.vstack([up, lo[::-1][1:-1]])          # closed loop, sharp LE/TE
    p3 = wrap(pts2, r)
    return bd.Wire(bd.Polyline(p3 + [p3[0]]))

radii = np.linspace(R_LO, R_HI, 9)
blade = bd.Solid.make_loft([section_poly(float(r)) for r in radii], ruled=True)
print("valid?", blade.is_valid, "volume", round(blade.volume,1), "faces", len(blade.faces()))

# -- cell 9 -------------------------------------------------------------------------
# Volume still short of estimate — let me just look at the 2D loop and the wrapped section to see what
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
up,lo = naca_sides(61); pts2=np.vstack([up, lo[::-1][1:-1]]); loop=np.vstack([pts2,pts2[0]])
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(loop[:,0],loop[:,1],'-o',ms=2); ax[0].set_aspect('equal'); ax[0].set_title("unit section")
for r in [45,67.5,90]:
    p=np.array([[q.X,q.Y,q.Z] for q in wrap(np.vstack([pts2,pts2[0]]), r)])
    th=np.arctan2(p[:,2],p[:,1]); ax[1].plot(p[:,0], r*th, label=f"r={r}")
ax[1].set_aspect('equal'); ax[1].legend(); ax[1].set_title("x vs r*theta")
# shoelace area of unit loop
A=0.5*abs(np.sum(loop[:-1,0]*loop[1:,1]-loop[1:,0]*loop[:-1,1]))
print("shoelace area unit chord", round(A,5), "-> mm^2", round(A*CHORD**2,2), " *49 =", round(A*CHORD**2*49,1))
plt.savefig("sec.png",dpi=90)

# -- cell 10 ------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("sec.png"))

# -- cell 11 ------------------------------------------------------------------------
# Sections are right; volume debate aside, time to get something on disk. Building the sector and cutt
PITCH_DEG = 360.0/N_BLADE
prof = bd.Plane.XZ * bd.Rectangle(X_OUT-X_IN, R_TIP-R_HUB)   # placeholder plane, fix below
prof = bd.Face(bd.Wire(bd.Polyline([Vector(X_IN,R_HUB,0),Vector(X_OUT,R_HUB,0),
                                    Vector(X_OUT,R_TIP,0),Vector(X_IN,R_TIP,0),Vector(X_IN,R_HUB,0)])))
sector = bd.revolve(prof, axis=Axis.X, revolution_arc=PITCH_DEG)
print("sector vol", round(sector.volume,1), "expected",
      round(math.radians(PITCH_DEG)/2*(R_TIP**2-R_HUB**2)*(X_OUT-X_IN),1), "faces", len(sector.faces()))
bladeA = blade
bladeB = blade.rotate(Axis.X, PITCH_DEG)
fluid = sector - bladeA - bladeB
print("fluid vol", round(fluid.volume,1), "valid", fluid.is_valid, "solids", len(fluid.solids()), "faces", len(fluid.faces()))
