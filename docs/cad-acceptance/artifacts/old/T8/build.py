#!/usr/bin/env python3
"""Duct with embedded copper block, two conformal regions (fluid, solid) for CHT.

All dimensions authored in mm, converted to metres before any point is created.
"""
import gmsh

mm = 1e-3

# duct
L = 200 * mm      # length, x
W = 30  * mm      # width, y
H = 30  * mm      # height, z

# copper block set into the floor, centred halfway along
BL = 40 * mm      # block length (x)
BW = 30 * mm      # block width (y) -- spans full duct width
BT = 10 * mm      # block thickness (z), sits on floor z=0
BX0 = L/2 - BL/2  # start x of block

gmsh.initialize()
gmsh.model.add("duct_cht")
occ = gmsh.model.occ

duct  = occ.addBox(0, 0, 0, L, W, H)
block = occ.addBox(BX0, 0, 0, BL, BW, BT)

out, _ = occ.fragment([(3, duct)], [(3, block)])
occ.synchronize()

vols = gmsh.model.getEntities(3)
print("volumes:", vols)

def bbox(dim, tag):
    return occ.getBoundingBox(dim, tag)

tol = 1e-7

fluid_vol = None
solid_vol = None
for d, t in vols:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox(d, t)
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
    vol_est = dx*dy*dz
    # the small block volume is BL*BW*BT ; identify by volume size
    if abs(vol_est - BL*BW*BT) < 1e-9:
        solid_vol = t
    else:
        fluid_vol = t

print("fluid vol tag", fluid_vol, "solid vol tag", solid_vol)
assert fluid_vol is not None and solid_vol is not None

gmsh.model.addPhysicalGroup(3, [fluid_vol], name="fluid")
gmsh.model.addPhysicalGroup(3, [solid_vol], name="solid")

# classify surfaces
def near(a, b, t=1e-6):
    return abs(a-b) < t

inlet, outlet = [], []
fluidWall, solidWall = [], []
interface = []

for d, t in gmsh.model.getEntities(2):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox(d, t)
    up_fluid = t in [s[1] for s in gmsh.model.getBoundary([(3, fluid_vol)], oriented=False)]
    up_solid = t in [s[1] for s in gmsh.model.getBoundary([(3, solid_vol)], oriented=False)]

    if near(xmin, 0) and near(xmax, 0):
        inlet.append(t)
    elif near(xmin, L) and near(xmax, L):
        outlet.append(t)
    elif up_fluid and up_solid:
        interface.append(t)
    elif up_solid:
        solidWall.append(t)
    elif up_fluid:
        fluidWall.append(t)
    else:
        print("WARNING: unclassified surface", t, xmin, ymin, zmin, xmax, ymax, zmax)

print("inlet", inlet, "outlet", outlet)
print("fluidWall", fluidWall, "solidWall", solidWall, "interface(auto)", interface)

gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
gmsh.model.addPhysicalGroup(2, fluidWall, name="fluidWall")
gmsh.model.addPhysicalGroup(2, solidWall, name="solidWall")
# interface faces are left unnamed on purpose: splitMeshRegions builds the
# fluid_to_solid / solid_to_fluid coupled patches itself from the cellZones.

# mesh size -- coarse first pass
csize = 4 * mm
gmsh.option.setNumber("Mesh.MeshSizeMin", csize)
gmsh.option.setNumber("Mesh.MeshSizeMax", csize)

gmsh.model.mesh.generate(3)
gmsh.write("duct_cht.msh")
gmsh.finalize()
