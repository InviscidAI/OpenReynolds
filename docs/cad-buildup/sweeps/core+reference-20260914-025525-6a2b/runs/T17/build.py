"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
import build123d as bd, numpy as np, os, subprocess
os.makedirs("constant/triSurface", exist_ok=True)

L,W,H = 300.,120.,80.      # plenum length, width, height (given)
D_IN   = 70.               # throttle inlet dia (given)
L_STUB = 40.               # ASSUMED throttle stub length
D_R    = 38.               # runner dia (given)
R_BEND = 60.               # centreline bend radius (given)
STAND  = 90.               # port face distance from plenum wall (given)
ZC     = H/2               # ASSUMED runner axis height (mid plenum)
XS     = [L/8 + i*L/4 for i in range(4)]   # evenly spaced along length

plenum = bd.Box(L,W,H).moved(bd.Pos(L/2,0,H/2))
stub   = bd.Pos(-L_STUB,0,ZC)*bd.Rot(0,90,0)*bd.Cylinder(D_IN/2, L_STUB+5, align=(bd.Align.CENTER,bd.Align.CENTER,bd.Align.MIN))
print(XS)
print(plenum.bounding_box(), stub.bounding_box())

# -- cell 2 -------------------------------------------------------------------------
# Runner path: straight from y=50 (inside the plenum, for a clean fuse) out to y=90, then a 90° arc of
def runner(x, s):   # s=+1 or -1 side
    y0, y1, y2 = s*50., s*(W/2 + (STAND-R_BEND)), s*(W/2+STAND)
    z2 = ZC - R_BEND
    p0=(x,y0,ZC); p1=(x,y1,ZC); pm=(x, s*(abs(y1)+R_BEND*np.sin(np.pi/4)), z2+R_BEND*np.cos(np.pi/4)); p2=(x,y2,z2)
    path = bd.Wire([bd.Edge.make_line(p0,p1), bd.Edge.make_three_point_arc(p1,pm,p2)])
    sec  = bd.Plane(origin=p0, z_dir=(0,s*1.,0))*bd.Circle(D_R/2)
    return bd.sweep(sec, path), p2

r0,e0 = runner(XS[0], 1)
print(e0, r0.bounding_box(), r0.volume)

# -- cell 3 -------------------------------------------------------------------------
runners=[]; ports=[]
for x in XS:
    for s in (1.,-1.):
        r,p = runner(x,s); runners.append(r); ports.append(p)
solid = plenum + stub
for r in runners: solid = solid + r
print(len(solid.solids()), solid.volume/1e3, "cm^3")
print(solid.bounding_box())

# -- cell 4 -------------------------------------------------------------------------
fs = solid.faces()
from collections import Counter
print(len(fs), Counter(f.geom_type for f in fs))
for f in fs:
    if f.geom_type==bd.GeomType.PLANE:
        print(f.geom_type, [round(v,2) for v in f.center()], round(f.area,1), [round(v,2) for v in f.normal_at()])

# -- cell 5 -------------------------------------------------------------------------
from OCP.BRepAdaptor import BRepAdaptor_Surface
for f in fs:
    if f.geom_type!=bd.GeomType.PLANE:
        a=BRepAdaptor_Surface(f.wrapped)
        if f.geom_type==bd.GeomType.CYLINDER: print("CYL r=",round(a.Cylinder().Radius(),3),[round(v,1) for v in f.center()])
        else: print("TOR R=",round(a.Torus().MajorRadius(),3),"r=",round(a.Torus().MinorRadius(),3),[round(v,1) for v in f.center()])

# -- cell 6 -------------------------------------------------------------------------
# All measured against the request: runner r=19 (Ø38 ✓), bend major R=60 ✓, inlet r=35 (Ø70 ✓), port f
groups={"inlet":[], "plenumWalls":[], "runnerWalls":[]}
for i in range(8): groups[f"port{i+1}"]=[]
portmap={(round(p[0],1),round(p[1],1)):i for i,p in enumerate(ports)}
for f in fs:
    c=f.center(); a=BRepAdaptor_Surface(f.wrapped)
    if f.geom_type==bd.GeomType.PLANE:
        key=(round(c.X,1),round(c.Y,1))
        if key in portmap and abs(c.Z+20)<1e-6: groups[f"port{portmap[key]+1}"].append(f)
        elif abs(c.X+L_STUB)<1e-6: groups["inlet"].append(f)
        else: groups["plenumWalls"].append(f)
    elif f.geom_type==bd.GeomType.CYLINDER and a.Cylinder().Radius()>30: groups["plenumWalls"].append(f)
    else: groups["runnerWalls"].append(f)
print({k:(len(v),round(sum(f.area for f in v),1)) for k,v in groups.items()})

# -- cell 7 -------------------------------------------------------------------------
for name,flist in groups.items():
    comp = bd.scale(bd.Compound(children=[bd.Face(f.wrapped) for f in flist]), 0.001)
    bd.export_stl(comp, f"constant/triSurface/{name}.stl", tolerance=2e-4, angular_tolerance=0.2, ascii_format=True)
r=subprocess.run(["surfaceCheck","constant/triSurface/plenumWalls.stl"],capture_output=True,text=True)
print(sorted(os.listdir("constant/triSurface")))
print("\n".join(l for l in r.stdout.splitlines() if "Bounding" in l or "triangles" in l))

# -- cell 8 -------------------------------------------------------------------------
import pyvista as pv
pv.start_xvfb() if False else None
pl=pv.Plotter(off_screen=True, window_size=(900,650))
cols={"inlet":"red","plenumWalls":"lightgray","runnerWalls":"steelblue"}
for f in sorted(os.listdir("constant/triSurface")):
    n=f[:-4]; pl.add_mesh(pv.read("constant/triSurface/"+f), color=cols.get(n,"orange"), opacity=0.55 if n=="plenumWalls" else 1.0, show_edges=False)
pl.camera_position=[(0.6,-0.6,0.5),(0.13,0,0.02),(0,0,1)]
pl.screenshot("geom.png"); print("ok")

# -- cell 9 -------------------------------------------------------------------------
from IPython.display import Image, display
display(Image("geom.png"))

# -- cell 10 ------------------------------------------------------------------------
# Shape confirmed. Now the background block (10 mm cells) and a coarse snappy setup.
os.makedirs("system",exist_ok=True)
hdr=lambda cls,obj:f"FoamFile{{version 2.0;format ascii;class {cls};object {obj};}}\n"
open("system/controlDict","w").write(hdr("dictionary","controlDict")+
 "application snappyHexMesh;startFrom startTime;startTime 0;stopAt endTime;endTime 1;deltaT 1;writeControl timeStep;writeInterval 1;\n")
open("system/fvSchemes","w").write(hdr("dictionary","fvSchemes")+"gradSchemes{default Gauss linear;}divSchemes{default none;}laplacianSchemes{default Gauss linear corrected;}\n")
open("system/fvSolution","w").write(hdr("dictionary","fvSolution")+"solvers{}\n")
CS=0.010
x0,x1=-0.06,0.32; y0,y1=-0.19,0.19; z0,z1=-0.04,0.10
nx,ny,nz=[int(round((b-a)/CS)) for a,b in ((x0,x1),(y0,y1),(z0,z1))]
verts=[(x,y,z) for z in (z0,z1) for y in (y0,y1) for x in (x0,x1)]
order=[0,1,3,2,4,5,7,6]
open("system/blockMeshDict","w").write(hdr("dictionary","blockMeshDict")+"scale 1;\nvertices(\n"+
 "".join(f"({verts[i][0]} {verts[i][1]} {verts[i][2]})\n" for i in order)+
 f");\nblocks(hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));\nedges();\nboundary(bg{{type patch;faces((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(0 4 7 3));}});\n")
print(nx,ny,nz,nx*ny*nz)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).stdout[-400:])

# -- cell 11 ------------------------------------------------------------------------
bm = hdr("dictionary","blockMeshDict")+"scale 1;\nvertices\n(\n"+ \
 "".join("    (%g %g %g)\n"%verts[i] for i in order)+ \
 ");\n\nblocks\n(\n    hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1)\n);\n\nedges();\n\nboundary\n(\n    bg\n    {\n        type patch;\n        faces\n        (\n            (0 3 2 1)\n            (4 5 6 7)\n            (0 1 5 4)\n            (2 3 7 6)\n            (1 2 6 5)\n            (0 4 7 3)\n        );\n    }\n);\n"%(nx,ny,nz)
open("system/blockMeshDict","w").write(bm)
r=subprocess.run(["blockMesh"],capture_output=True,text=True); print(r.stdout[-500:], r.returncode)

# -- cell 12 ------------------------------------------------------------------------
# Background mesh in place. Now snappyHexMeshDict with one geometry entry per patch STL.
names=["inlet","plenumWalls","runnerWalls"]+[f"port{i+1}" for i in range(8)]
geom="\n".join(f'    {n}.stl {{ type triSurfaceMesh; name {n}; }}' for n in names)
surfs="\n".join(f'        {n} {{ level (0 0); patchInfo {{ type {"patch" if n!="plenumWalls" and n!="runnerWalls" else "wall"}; }} }}' for n in names)
snap=hdr("dictionary","snappyHexMeshDict")+f"""
castellatedMesh true; snap true; addLayers false;
geometry {{
{geom}
}};
castellatedMeshControls
{{
    maxLocalCells 2000000; maxGlobalCells 5000000; minRefinementCells 0; nCellsBetweenLevels 2;
    resolveFeatureAngle 45; allowFreeStandingZoneFaces true; maxLoadUnbalance 0.1;
    features ();
    refinementSurfaces
    {{
{surfs}
    }}
    refinementRegions {{}}
    locationInMesh (0.15 0.0 0.045);
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }}
addLayersControls {{ relativeSizes true; layers {{}} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }}
meshQualityControls {{ #includeEtc "caseDicts/meshQualityDict" }}
writeFlags (); mergeTolerance 1e-6;
"""
open("system/snappyHexMeshDict","w").write(snap)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode); print(r.stdout[-1500:])

# -- cell 13 ------------------------------------------------------------------------
print(r.stderr[-2000:]); print("----"); print(r.stdout[-300:])

# -- cell 14 ------------------------------------------------------------------------
snap2=snap.replace('meshQualityControls {{ #includeEtc "caseDicts/meshQualityDict" }}'.replace("{{","{").replace("}}","}"),
 'meshQualityControls { #includeEtc "caseDicts/meshQualityDict" \n nSmoothScale 4; errorReduction 0.75; }')
open("system/snappyHexMeshDict","w").write(snap2)
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stderr[-500:]); print(r.stdout[-800:])

# -- cell 15 ------------------------------------------------------------------------
c=subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print("\n".join(l for l in c.splitlines() if any(k in l for k in ["cells:","patch","Mesh OK","***","Patches","name"])) [:2000])

# -- cell 16 ------------------------------------------------------------------------
print("\n".join(c.splitlines()[20:50]))

# -- cell 17 ------------------------------------------------------------------------
open("case.foam","w").write("")
rd=pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0)
m=rd.read(); b=m["boundary"]
pl=pv.Plotter(off_screen=True,window_size=(900,650))
for k in b.keys(): pl.add_mesh(b[k], color=("red" if k=="inlet" else "orange" if k.startswith("port") else "lightgray" if k=="plenumWalls" else "steelblue"), show_edges=True, opacity=0.5 if k=="plenumWalls" else 1)
pl.camera_position=[(0.6,-0.6,0.5),(0.13,0,0.02),(0,0,1)]; pl.screenshot("mesh0.png")
display(Image("mesh0.png")); print(list(b.keys()))

# -- cell 18 ------------------------------------------------------------------------
# Topology is right and all 11 patches exist. Now the refined version: 6 mm background with level-1 re
CS2=0.006
nx,ny,nz=[int(round((b-a)/CS2)) for a,b in ((x0,x1),(y0,y1),(z0,z1))]
open("system/blockMeshDict","w").write(hdr("dictionary","blockMeshDict")+"scale 1;\nvertices\n(\n"+
 "".join("    (%g %g %g)\n"%verts[i] for i in order)+
 ");\n\nblocks\n(\n    hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1)\n);\n\nedges();\n\nboundary\n(\n    bg\n    {\n        type patch;\n        faces\n        (\n            (0 3 2 1)\n            (4 5 6 7)\n            (0 1 5 4)\n            (2 3 7 6)\n            (1 2 6 5)\n            (0 4 7 3)\n        );\n    }\n);\n"%(nx,ny,nz))
lvl={n:("(1 2)" if n in("runnerWalls",) or n.startswith("port") else "(1 1)") for n in names}
surfs2="\n".join(f'        {n} {{ level {lvl[n]}; patchInfo {{ type {"wall" if n in ("plenumWalls","runnerWalls") else "patch"}; }} }}' for n in names)
open("system/snappyHexMeshDict","w").write(snap2.replace(surfs,surfs2))
print(nx,ny,nz,nx*ny*nz)
print(subprocess.run(["blockMesh"],capture_output=True,text=True).returncode)

# -- cell 19 ------------------------------------------------------------------------
r=subprocess.run(["snappyHexMesh","-overwrite"],capture_output=True,text=True)
print(r.returncode, r.stderr[-300:])
print("\n".join(l for l in r.stdout.splitlines() if "Cells per refinement" in l or l.strip().startswith(("0\t","1\t","2\t")) or "Finished meshing" in l))

# -- cell 20 ------------------------------------------------------------------------
c=subprocess.run(["checkMesh"],capture_output=True,text=True).stdout
print("\n".join(c.splitlines()[26:60]))
print([l for l in c.splitlines() if "***" in l or "Mesh OK" in l or "Max skew" in l or "non-orthogonality" in l])

# -- cell 21 ------------------------------------------------------------------------
rd=pv.OpenFOAMReader("case.foam"); rd.set_active_time_value(0.0); m=rd.read(); b=m["boundary"]
print("mesh volume m^3", m["internalMesh"].compute_cell_sizes()["Volume"].sum(), "CAD", solid.volume*1e-9)
for k in b.keys(): print(k, round(b[k].area,6))
print("CAD port area m^2", 1134.1e-6, "inlet", 3848.5e-6)

# -- cell 22 ------------------------------------------------------------------------
for k in b.keys():
    if k.startswith("port"): print(k, [round(v,4) for v in b[k].center])
pl=pv.Plotter(off_screen=True,window_size=(950,680))
pl.add_mesh(m["internalMesh"].slice(normal="z",origin=(0,0,0.04)), color="lightblue", show_edges=True)
for k in b.keys():
    if k!="plenumWalls": pl.add_mesh(b[k], color=("red" if k=="inlet" else "orange" if k.startswith("port") else "steelblue"), opacity=0.35)
pl.camera_position=[(0.15,0.0,0.8),(0.13,0,0.02),(0,1,0)]; pl.screenshot("mesh1.png"); display(Image("mesh1.png"))

# -- cell 23 ------------------------------------------------------------------------
# The previous cell defined `fsm` in a cell that errored, so I re-emit the whole shared-tessellation s
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
solid_m = bd.scale(solid, 0.001)                      # single shape, metres
BRepMesh_IncrementalMesh(solid_m.wrapped, 2e-4, False, 0.2, True)   # ONE tessellation, shared edges
fsm = solid_m.faces()
def classify(f):
    c=f.center(); a=BRepAdaptor_Surface(f.wrapped)
    if f.geom_type==bd.GeomType.PLANE:
        if abs(c.Z+0.020)<1e-6 and abs(abs(c.Y)-0.150)<1e-6:
            return f"port{portmap[(round(c.X*1000,1), round(c.Y*1000,1))]+1}"
        if abs(c.X+L_STUB/1000)<1e-6: return "inlet"
        return "plenumWalls"
    if f.geom_type==bd.GeomType.CYLINDER and a.Cylinder().Radius()>0.030: return "plenumWalls"
    return "runnerWalls"
gm={}
for f in fsm: gm.setdefault(classify(f),[]).append(f)
print({k:len(v) for k,v in gm.items()})

# -- cell 24 ------------------------------------------------------------------------
# Now write each patch STL directly from that shared triangulation, honouring face orientation so all 
def write_stl(path, name, faces):
    with open(path,"w") as fh:
        fh.write(f"solid {name}\n")
        for f in faces:
            loc=TopLoc_Location(); tri=BRep_Tool.Triangulation_s(f.wrapped, loc); trsf=loc.Transformation()
            rev = f.wrapped.Orientation()==TopAbs_REVERSED
            P=[tri.Node(i+1).Transformed(trsf) for i in range(tri.NbNodes())]
            for t in range(1,tri.NbTriangles()+1):
                i1,i2,i3=tri.Triangle(t).Get()
                if rev: i2,i3=i3,i2
                a,b_,c_=P[i1-1],P[i2-1],P[i3-1]
                u=np.array([b_.X()-a.X(),b_.Y()-a.Y(),b_.Z()-a.Z()]); v=np.array([c_.X()-a.X(),c_.Y()-a.Y(),c_.Z()-a.Z()])
                n=np.cross(u,v); nn=np.linalg.norm(n); n = n/nn if nn>0 else n
                fh.write("facet normal %g %g %g\n outer loop\n"%tuple(n))
                for p in (a,b_,c_): fh.write("  vertex %.9g %.9g %.9g\n"%(p.X(),p.Y(),p.Z()))
                fh.write(" endloop\nendfacet\n")
        fh.write(f"endsolid {name}\n")
for n,fl in gm.items(): write_stl(f"constant/triSurface/{n}.stl", n, fl)
out=subprocess.run(["surfaceCheck","constant/triSurface/runnerWalls.stl"],capture_output=True,text=True).stdout
print("\n".join(l for l in out.splitlines() if "riangle" in l or "Bounding" in l))
