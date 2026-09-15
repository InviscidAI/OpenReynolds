"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# Plan: structured 2D mesh via blockMesh in metres with three hex blocks (bend as half-annulus with ar
bdm = """/*--------------------------------*- C++ -*----------------------------------*\
  U-bend 2D duct, metres
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
scale 1;

vertices
(
    (0      0      0)   // 0
    (0      0.01   0)   // 1
    (0      0.03   0)   // 2
    (0      0.04   0)   // 3
    (0      0      0.005)  // 4
    (0      0.01   0.005)  // 5
    (0      0.03   0.005)  // 6
    (0      0.04   0.005)  // 7
    (0.12   0      0)      // 8
    (0.12   0.01   0)      // 9
    (0.12   0      0.005)  // 10
    (0.12   0.01   0.005)  // 11
    (0.12   0.03   0)      // 12
    (0.12   0.04   0)      // 13
    (0.12   0.03   0.005)  // 14
    (0.12   0.04   0.005)  // 15
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (40 4 1) simpleGrading (1 1 1)
    hex (0 8 9 1 4 10 11 5) (40 4 1) simpleGrading (1 1 1)
    hex (2 12 13 3 6 14 15 7) (40 4 1) simpleGrading (1 1 1)
);

edges
(
    arc 1 2 (-0.01 0.02 0)
    arc 5 6 (-0.01 0.02 0.005)
    arc 3 0 (-0.02 0.02 0)
    arc 7 4 (-0.02 0.02 0.005)
);

boundary
(
    inlet
    {
        type patch;
        faces ((8 9 11 10));
    }
    outlet
    {
        type patch;
        faces ((12 13 15 14));
    }
    walls
    {
        type wall;
        faces
        (
            (0 3 7 4)
            (1 2 6 5)
            (0 8 10 4)
            (1 9 11 5)
            (2 12 14 6)
            (3 13 15 7)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 1 2 3)
            (4 7 6 5)
            (0 8 9 1)
            (4 5 11 10)
            (2 12 13 3)
            (6 7 15 14)
        );
    }
);
"""
import os, subprocess
os.makedirs("system", exist_ok=True)
open("system/blockMeshDict","w").write(bdm)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-300:])

# -- cell 2 -------------------------------------------------------------------------
# blockMesh errored; let me see the full message.
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("STDOUT:\n", r.stdout)
print("STDERR:\n", r.stderr)

# -- cell 3 -------------------------------------------------------------------------
# It just needs a controlDict. Adding a minimal one for a potentialFoam-style setup (geometry desk onl
open("system/controlDict","w").write("""FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1000;
deltaT 1;
writeControl timeStep;
writeInterval 1000;
purgeWrite 0;
writeFormat ascii;
writePrecision 6;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
""")
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-700:])
print("STDERR:", r.stderr[-300:])

# -- cell 4 -------------------------------------------------------------------------
# The bend block ordering points inward; reversing it (swap vertex pairs so the hex winds the other wa
bdm = bdm.replace("hex (0 1 2 3 4 5 6 7) (40 4 1)", "hex (1 0 3 2 5 4 7 6) (40 4 1)")
open("system/blockMeshDict","w").write(bdm)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print(r.stdout[-500:])
print("STDERR:", r.stderr[-300:])

# -- cell 5 -------------------------------------------------------------------------
# Orientation of that curved quad is clockwise; fully reversing the quad order (and matching z-face):
import re
b2 = open("system/blockMeshDict").read()
b2 = re.sub(r"hex \([0-9 ]+\) \(40 4 1\)", "hex (3 2 1 0 7 6 5 4) (40 4 1)", b2, count=1)
open("system/blockMeshDict","w").write(b2)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("STDERR:", r.stderr[-400:])

# -- cell 6 -------------------------------------------------------------------------
# A 180° arc span is awkward for the checker; I'll split the bend into two 90° blocks with mid-bend ve
z = 0.005
dict_text = f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
scale 1;
vertices
(
    (0 0 0)          // 0: lower leg left, outer
    (0 0.01 0)       // 1: lower leg left, inner
    (0 0.03 0)       // 2: upper leg left, inner
    (0 0.04 0)       // 3: upper leg left, outer
    (0 0 {z})
    (0 0.01 {z})
    (0 0.03 {z})
    (0 0.04 {z})
    (0.12 0 0)       // 8
    (0.12 0.01 0)    // 9
    (0.12 0 {z})     // 10
    (0.12 0.01 {z})  // 11
    (0.12 0.03 0)    // 12
    (0.12 0.04 0)    // 13
    (0.12 0.03 {z})  // 14
    (0.12 0.04 {z})  // 15
    (-0.02 0.02 0)   // 16 mid-bend outer
    (-0.01 0.02 0)   // 17 mid-bend inner
    (-0.02 0.02 {z}) // 18
    (-0.01 0.02 {z}) // 19
);
blocks
(
    hex (0 1 17 16 4 5 19 18) (20 4 1) simpleGrading (1 1 1)   // bend lower half
    hex (2 17 16 3 6 19 18 7) (20 4 1) simpleGrading (1 1 1)   // bend upper half
    hex (0 8 9 1 4 10 11 5) (40 4 1) simpleGrading (1 1 1)     // lower leg
    hex (2 12 13 3 6 14 15 7) (40 4 1) simpleGrading (1 1 1)   // upper leg
);
edges
(
    arc 1 17 (-0.007071 0.012929 0)
    arc 0 16 (-0.014142 0.005858 0)
    arc 17 2 (-0.007071 0.027071 0)
    arc 16 3 (-0.014142 0.034142 0)
    arc 5 19 (-0.007071 0.012929 {z})
    arc 4 18 (-0.014142 0.005858 {z})
    arc 19 6 (-0.007071 0.027071 {z})
    arc 18 7 (-0.014142 0.034142 {z})
);
boundary
(
    inlet
    {{
        type patch;
        faces ((8 9 11 10));
    }}
    outlet
    {{
        type patch;
        faces ((12 13 15 14));
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 16 18 4)
            (16 3 7 18)
            (1 17 19 5)
            (17 2 6 19)
            (0 8 10 4)
            (1 9 11 5)
            (2 12 14 6)
            (3 13 15 7)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 1 17 16)
            (4 18 19 5)
            (2 17 16 3)
            (6 7 18 19)
            (0 8 9 1)
            (4 5 11 10)
            (2 12 13 3)
            (6 7 15 14)
        );
    }}
);
"""
open("system/blockMeshDict","w").write(dict_text)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("stderr:", r.stderr[-400:])
print(r.stdout[-400:])

# -- cell 7 -------------------------------------------------------------------------
# Upper bend quad winds the wrong way; flip it:
d = open("system/blockMeshDict").read()
d = d.replace("hex (2 17 16 3 6 19 18 7)", "hex (2 3 16 17 6 7 18 19)")
# boundary faces for that block need the same vertex pairing; walls/empty faces unchanged indices
open("system/blockMeshDict","w").write(d)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("stderr:", r.stderr[-300:])
print(r.stdout[-350:])

# -- cell 8 -------------------------------------------------------------------------
# I had the subdivision directions swapped on the bend blocks (radial vs angular). nI should be radial
d = open("system/blockMeshDict").read()
d = d.replace("hex (0 1 17 16 4 5 19 18) (20 4 1)", "hex (0 1 17 16 4 5 19 18) (4 20 1)")
d = d.replace("hex (2 3 16 17 6 7 18 19) (20 4 1)", "hex (2 3 16 17 6 7 18 19) (4 20 1)")
open("system/blockMeshDict","w").write(d)
r = subprocess.run(["blockMesh"], capture_output=True, text=True)
print("stderr:", r.stderr[-300:])
print(r.stdout[-300:])

# -- cell 9 -------------------------------------------------------------------------
# checkMesh output got truncated before the verdict; let me print its end, and draw the mesh directly 
r = subprocess.run(["checkMesh"], capture_output=True, text=True)
print(r.stdout[-1400:])
print("exit:", r.returncode)
