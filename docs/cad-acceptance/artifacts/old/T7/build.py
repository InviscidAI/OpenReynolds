import numpy as np

# ---- geometry in mm ----
Lx, Ly, Lz = 100.0, 60.0, 40.0   # domain size
cube = 20.0                       # heater/cooler cube edge

# grid lines (mm)
xs = [0.0, cube, Lx-cube, Lx]     # 0,20,80,100
ys = [0.0, (Ly-cube)/2.0, (Ly-cube)/2.0+cube, Ly]  # 0,20,40,60
zs = [0.0, cube, Lz]              # 0,20,40

# cell counts per segment, target ~5mm cells
def ncells(seg_len):
    return max(1, round(seg_len/5.0))

nx = [ncells(xs[i+1]-xs[i]) for i in range(3)]
ny = [ncells(ys[i+1]-ys[i]) for i in range(3)]
nz = [ncells(zs[i+1]-zs[i]) for i in range(2)]

# heater block index (i,j,k), cooler block index
heater_ijk = (0,1,0)   # x:0-20 (near x=0 end), y: middle, z:0-20 (near floor)
cooler_ijk = (2,1,1)   # x:80-100 (other end), y: middle, z:20-40 (near ceiling)

# vertices: index(i,j,k) i=0..3,j=0..3,k=0..2 -> convert mm->m
def vidx(i,j,k):
    return i + j*4 + k*16

verts = []
for k in range(3):
    for j in range(4):
        for i in range(4):
            verts.append((xs[i]/1000.0, ys[j]/1000.0, zs[k]/1000.0))

assert len(verts) == 48

blocks = []
zone_of = {}
zone_of[heater_ijk] = "heater"
zone_of[cooler_ijk] = "cooler"

# boundary face collections, named per side of the box
faces_x0 = []   # x = 0            -> "endA" (heater end)
faces_x1 = []   # x = Lx           -> "endB" (cooler end)
faces_y0 = []   # y = 0            -> "sideA"
faces_y1 = []   # y = Ly           -> "sideB"
faces_z0 = []   # z = 0 (floor)    -> "floor"
faces_z1 = []   # z = Lz (ceiling) -> "ceiling"

for i in range(3):
    for j in range(3):
        for k in range(2):
            v0 = vidx(i,   j,   k)
            v1 = vidx(i+1, j,   k)
            v2 = vidx(i+1, j+1, k)
            v3 = vidx(i,   j+1, k)
            v4 = vidx(i,   j,   k+1)
            v5 = vidx(i+1, j,   k+1)
            v6 = vidx(i+1, j+1, k+1)
            v7 = vidx(i,   j+1, k+1)

            zone = zone_of.get((i,j,k), None)
            blocks.append((v0,v1,v2,v3,v4,v5,v6,v7, zone, nx[i], ny[j], nz[k]))

            if i == 0:
                faces_x0.append((v0,v4,v7,v3))
            if i == 2:
                faces_x1.append((v1,v2,v6,v5))
            if j == 0:
                faces_y0.append((v0,v1,v5,v4))
            if j == 2:
                faces_y1.append((v3,v7,v6,v2))
            if k == 0:
                faces_z0.append((v0,v3,v2,v1))
            if k == 1:
                faces_z1.append((v4,v5,v6,v7))

# ---- write blockMeshDict ----
lines = []
lines.append("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n")
lines.append("convertToMeters 1.0;\n")

lines.append("vertices\n(\n")
for (x,y,z) in verts:
    lines.append(f"    ({x:.6f} {y:.6f} {z:.6f})\n")
lines.append(");\n")

lines.append("blocks\n(\n")
for b in blocks:
    v0,v1,v2,v3,v4,v5,v6,v7, zone, cx, cy, cz = b
    zonestr = f"{zone} " if zone else ""
    lines.append(f"    hex ({v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}) {zonestr}({cx} {cy} {cz}) simpleGrading (1 1 1)\n")
lines.append(");\n")

lines.append("edges\n(\n);\n")

def facelist(name, faces):
    s = f"    {name}\n    {{\n        type wall;\n        faces\n        (\n"
    for f in faces:
        s += f"            ({f[0]} {f[1]} {f[2]} {f[3]})\n"
    s += "        );\n    }\n"
    return s

lines.append("boundary\n(\n")
lines.append(facelist("endA",    faces_x0))
lines.append(facelist("endB",    faces_x1))
lines.append(facelist("sideA",   faces_y0))
lines.append(facelist("sideB",   faces_y1))
lines.append(facelist("floor",   faces_z0))
lines.append(facelist("ceiling", faces_z1))
lines.append(");\n")

lines.append("mergePatchPairs\n(\n);\n")

with open("system/blockMeshDict", "w") as f:
    f.writelines(lines)

print("Wrote blockMeshDict")
print("nx,ny,nz per segment:", nx, ny, nz)
print("Total cells estimate:", sum(b[9]*b[10]*b[11] for b in blocks))
print("heater block ijk:", heater_ijk, "-> x[0,20] y[20,40] z[0,20] mm")
print("cooler block ijk:", cooler_ijk, "-> x[80,100] y[20,40] z[20,40] mm")
