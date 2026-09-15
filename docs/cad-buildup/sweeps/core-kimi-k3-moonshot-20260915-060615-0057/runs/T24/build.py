"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# `is_valid` is a property here. Re-running the full build cell with that fixed so the accepted script
import build123d as bd
import numpy as np, math

# --- parameters (build123d unitless: we build in mm, scale to m at export/mesh) ---
R   = 60.0      # vessel inner radius, mm
H   = 200.0     # vessel height, mm
R_OUT = 20.0    # outlet radius, mm
DUCT_W, DUCT_H = 20.0, 30.0   # duct width (tangential) x height, mm
DUCT_L = 50.0   # duct length along its own axis, mm
Z_DUCT = 40.0   # duct centerline height above floor (assumption: '40 mm above floor' = centerline)
ANG = 15.0      # degrees off the local tangent
OVERLAP = 30.0  # how far the duct box pokes into the vessel so the union is clean

a = math.radians(ANG)

# vessel gas volume: cylinder with the outlet hole cut through the top
vessel = bd.Pos(0, 0, 0) * bd.Cylinder(radius=R, height=H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
hole = bd.Pos(0, 0, H - 5) * bd.Cylinder(radius=R_OUT, height=10, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
vessel = vessel - hole

# ducts: box in a local frame, x = duct axis (outward), z = global z
ducts = []
duct_frames = []
for i in range(4):
    th = math.radians(i * 90.0)
    t_hat = (-math.sin(th), math.cos(th), 0.0)   # local tangent (swirl sense +)
    r_hat = ( math.cos(th), math.sin(th), 0.0)   # local outward radial
    # duct outward axis: 15 deg off tangent, toward the radial direction
    u = tuple(math.cos(a)*t_hat[k] + math.sin(a)*r_hat[k] for k in range(3))
    P = (R*math.cos(th), R*math.sin(th), Z_DUCT)          # point on wall
    C = tuple(P[k] + u[k]*(DUCT_L - OVERLAP)/2 for k in range(3))  # box centre
    Lbox = DUCT_L + OVERLAP
    pln = bd.Plane(origin=C, x_dir=u, z_dir=(0, 0, 1))
    box = pln * bd.Box(Lbox, DUCT_W, DUCT_H)
    ducts.append(box)
    duct_frames.append(dict(i=i, th=th, u=np.array(u), P=np.array(P), C=np.array(C), Lbox=Lbox, pln=pln))

fluid = vessel
for d in ducts:
    fluid = fluid + d

print("solid valid:", fluid.is_valid, " volume(mm^3):", fluid.volume)
print("bbox:", fluid.bounding_box())
print("n faces:", len(fluid.faces()))
for f in fluid.faces():
    c = f.center()
    n = f.normal_at()
    print(f"{str(f.geom_type):12s} area={f.area:10.1f}  c=({c.X:7.2f},{c.Y:7.2f},{c.Z:7.2f})  n=({n.X:5.2f},{n.Y:5.2f},{n.Z:5.2f})")

# -- cell 2 -------------------------------------------------------------------------
# Let me check the installed API for making a planar face from a wire:
import subprocess, inspect
print(subprocess.run(["grep", "-n", "-i", "make_face\|make from\|def Face\|class Face", ".reference/b123d_api.md"], capture_output=True, text=True).stdout[:2000])
import build123d as bd
print([m for m in dir(bd.Face) if 'make' in m.lower() or 'from' in m.lower()])
print(inspect.signature(bd.Face.make_brep_surface) if hasattr(bd.Face,'make_brep_surface') else '')

# -- cell 3 -------------------------------------------------------------------------
# `bd.make_face(edges)` is the function. Re-running the build cell with that:
import build123d as bd
import numpy as np, math

# --- parameters, METRES (OpenFOAM reads raw numbers as metres) ---
R   = 0.060      # vessel inner radius
H   = 0.200      # vessel height
R_OUT = 0.020    # outlet radius
DUCT_W, DUCT_H = 0.020, 0.030   # duct width (tangential) x height
DUCT_L = 0.050   # duct length along its own axis, from the wall point outward
Z_DUCT = 0.040   # duct centerline height above floor (assumption: '40 mm above floor' = centerline)
ANG = 15.0       # degrees off the local tangent, toward the radial direction

a = math.radians(ANG)

# vessel gas volume: cylinder, outlet hole cut from exactly the top plane upward
vessel = bd.Pos(0, 0, 0) * bd.Cylinder(radius=R, height=H, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
hole = bd.Pos(0, 0, H) * bd.Cylinder(radius=R_OUT, height=0.01, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
vessel = vessel - hole

# ducts: box from wall point P outward along axis u (inner end exactly at P -> clean miter by the cylinder)
ducts = []
duct_frames = []
for i in range(4):
    th = math.radians(i * 90.0)
    t_hat = np.array([-math.sin(th), math.cos(th), 0.0])   # local tangent (swirl sense)
    r_hat = np.array([ math.cos(th), math.sin(th), 0.0])   # local outward radial
    u = math.cos(a)*t_hat + math.sin(a)*r_hat              # duct outward axis, 15 deg off tangent
    y = np.cross([0, 0, 1], u)                             # duct width direction (horizontal)
    P = np.array([R*math.cos(th), R*math.sin(th), Z_DUCT]) # anchor point on wall, duct centerline
    C = P + u*(DUCT_L/2)
    pln = bd.Plane(origin=tuple(C), x_dir=tuple(u), z_dir=(0, 0, 1))
    box = pln * bd.Box(DUCT_L, DUCT_W, DUCT_H)
    ducts.append(box)
    duct_frames.append(dict(i=i, u=u, y=y, P=P))

fluid = vessel
for d in ducts:
    fluid = fluid + d

# outlet opening: disk face at z = H
outlet_face = bd.Pos(0, 0, H) * bd.make_face(bd.Circle(R_OUT).edges())

print("fluid valid:", fluid.is_valid, " volume(m^3):", fluid.volume)
print("outlet face area (expect %.6f):" % (math.pi*R_OUT**2), outlet_face.area)
print("n faces:", len(fluid.faces()))
for f in fluid.faces():
    c = f.center(); n = f.normal_at()
    print(f"{str(f.geom_type):12s} area={f.area:9.6f}  c=({c.X:7.4f},{c.Y:7.4f},{c.Z:7.4f})  n=({n.X:5.2f},{n.Y:5.2f},{n.Z:5.2f})")
