#!/bin/bash
# What the image promises, each checked by doing it. Run as the last build step.
# $1, $2 and $3 are the gmsh, build123d and OCP versions the build pinned.
set -eo pipefail
GMSH_VERSION="$1"
BUILD123D_VERSION="$2"
OCP_VERSION="$3"
SHIM=/usr/local/lib/libfoamnotadmin.so
source /etc/foamd/shellinit.sh  # before -u: OpenFOAM's bashrc reads unset variables
set -u

echo "== OpenFOAM and HiSA"
test "$WM_PROJECT_VERSION" = v2512
blockMesh -help > /dev/null
hisa -help > /dev/null 2>&1

echo "== shell init: the helpers reach a child bash, a script and a login shell"
env -i PATH="$PATH" bash -c 'test "$(bash -c "type -t restore0Dir" 2>/dev/null)" != function' \
    || { echo "control failed: a bare child already had the helpers"; exit 1; }
test "$(env -i PATH="$PATH" BASH_ENV=/etc/foamd/shellinit.sh bash -c 'type -t restore0Dir')" = function
printf '#!/bin/bash\ntype -t runApplication\n' > /tmp/gate.sh
test "$(env -i PATH="$PATH" BASH_ENV=/etc/foamd/shellinit.sh bash /tmp/gate.sh)" = function
test "$(env -i PATH="$PATH" bash -lc 'type -t runParallel')" = function

echo "== the #codeStream shim: loads silently, and is what lets root compile"
nm -D "$SHIM" | grep " T _ZN4Foam15isAdministratorEv" > /dev/null
nm -D "$FOAM_LIBBIN/libOpenFOAM.so" | grep " T _ZN4Foam15isAdministratorEv" > /dev/null
test -z "$(LD_PRELOAD=$SHIM /bin/echo probe 2>&1 >/dev/null)"
test -z "$(LD_PRELOAD=$SHIM python3 -V 2>&1 >/dev/null)"
rm -rf /tmp/gate-case && cp -r "$FOAM_TUTORIALS/basic/potentialFoam/cylinder" /tmp/gate-case && cd /tmp/gate-case
if blockMesh > bare.log 2>&1; then echo "the root guard did not fire without the shim"; exit 1; fi
grep -q administrator bare.log
rm -rf constant/polyMesh dynamicCode
LD_PRELOAD=$SHIM blockMesh > shim.log 2>&1
test -s constant/polyMesh/points
cd / && rm -rf /tmp/gate-case

echo "== one gmsh, with OpenCASCADE, CLI and module the same build"
cli="$(gmsh -info 2>&1 | sed -n 's/^Version *: *//p' | head -1)"
mod="$(python3 -c 'import gmsh; print(gmsh.GMSH_API_VERSION)')"
test "$cli" = "$GMSH_VERSION" && test "$mod" = "$GMSH_VERSION"
gmsh -info | grep OpenCASCADE > /dev/null
! dpkg -s gmsh > /dev/null 2>&1

echo "== CAD: the pinned build123d and OCP, a solid built, exported, re-imported, meshed"
python3 - "$BUILD123D_VERSION" "$OCP_VERSION" <<'PY'
import sys
from importlib.metadata import version

import gmsh
import build123d
from build123d import Cylinder, export_step

# The versions, not just that an import worked: `build123d>=0.9` floated to 0.13.0 and
# took the CAD desk's patch export with it, and OCP is the half that actually broke.
# b123d_api.md names the build123d version, so a bump here is a bump there.
assert build123d.__version__ == sys.argv[1], f"build123d {build123d.__version__}"
assert version("cadquery-ocp-novtk") == sys.argv[2], version("cadquery-ocp-novtk")
export_step(Cylinder(5, 40) - Cylinder(4, 40), "/tmp/gate.step")
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
shapes = gmsh.model.occ.importShapes("/tmp/gate.step"); gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
tets = len(gmsh.model.mesh.getElementsByType(4)[0]); gmsh.finalize()
assert [s for s in shapes if s[0] == 3] and tets > 100, "no solid, or no tetrahedra"
print("cad ok:", tets, "tets")
PY

echo "== renders and animations, offline"
python3 - <<'PY'
import numpy, pyvista
pyvista.OFF_SCREEN = True
plotter = pyvista.Plotter(off_screen=True, window_size=(64, 64))
plotter.add_mesh(pyvista.Sphere()); image = plotter.screenshot(return_img=True); plotter.close()
assert image.std() > 0, "the render is blank"
import imageio.v2 as iio
frames = [(numpy.arange(64 * 64 * 3).reshape(64, 64, 3) // (i + 1) % 255).astype("uint8") for i in range(6)]
iio.mimwrite("/tmp/gate.gif", frames, fps=8, loop=0); iio.mimwrite("/tmp/gate.mp4", frames, fps=8)
assert open("/tmp/gate.gif", "rb").read(6) in (b"GIF87a", b"GIF89a")
assert open("/tmp/gate.mp4", "rb").read(8)[4:] == b"ftyp"
print("render and encoders ok")
PY
ffmpeg -version > /dev/null && ffprobe -version > /dev/null
command -v pdftoppm pdftotext > /dev/null

echo "== the kernel runs a cell"
python3 - <<'PY'
from jupyter_client.manager import KernelManager
out = []
km = KernelManager(kernel_name="python3"); km.start_kernel(extra_arguments=["--HistoryManager.enabled=False"])
kc = km.client(); kc.start_channels(); kc.wait_for_ready(timeout=60)
kc.execute_interactive("print(6 * 7)", timeout=60, output_hook=out.append)
kc.stop_channels(); km.shutdown_kernel(now=True)
text = "".join(m["content"].get("text", "") for m in out if m["msg_type"] == "stream")
assert text.strip() == "42", repr(text)
print("kernel ok")
PY

# --- stress analysis: nothing installed, so what is gated is that it is really here ---
# `solidDisplacementFoam` is a standard ESI solver and has been in this image since the
# first build; the base is a full distribution. Nothing in this repository had ever
# named it, so "the image ships a solid-mechanics solver" was an assumption, and the
# gate makes it a measurement. `-help` alone would not: the thermal half of this solver
# has no worked example in the distribution (its one tutorial, plateHole, ships
# `thermalStress no`), so what is exercised here is the path a user's case takes.
#
# The case is the constrained bar of benchmarks/thermal_stress/README.md, where the
# answer is exact: a bar held between frictionless rigid ends, resting on a
# frictionless floor, free on top, conducting dT across its height. With ex_xx = 0,
# sigma_yy = 0 and ex_zz = 0 put into the solver's own constitutive law, sigma is
# diag(s, 0, s) with s = -E*alpha*T, so the solver's own `Max sigmaEq` line --
# sqrt(1.5*magSqr(dev(sigma))) -- is exactly E*alpha*dT in plane stress and
# E*alpha*dT/(1-nu) in plane strain. One number per run, no field parsing, and the two
# runs are 43% apart, so the `planeStress` switch cannot be read the wrong way round
# and pass. T is written as a RISE: this solver subtracts no reference temperature
# anywhere, so the stress-free state of the body is T = 0.
echo "== stress analysis: solidDisplacementFoam, on a thermal stress it gets exactly"
solidDisplacementFoam -help > /dev/null 2>&1
solidEquilibriumDisplacementFoam -help > /dev/null 2>&1
PLATE_HOLE="$FOAM_TUTORIALS/stressAnalysis/solidDisplacementFoam/plateHole"
test -d "$PLATE_HOLE"
grep -qE '^thermalStress\s+no;' "$PLATE_HOLE/constant/thermalProperties"
rm -rf /tmp/gate-stress && mkdir -p /tmp/gate-stress/base/{0,constant,system}
cd /tmp/gate-stress/base
cat > system/blockMeshDict <<'FOAM'
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
    (0 0 0) (0.04 0 0) (0.04 0.02 0) (0 0.02 0)
    (0 0 0.001) (0.04 0 0.001) (0.04 0.02 0.001) (0 0.02 0.001)
);
blocks ( hex (0 1 2 3 4 5 6 7) (40 20 1) simpleGrading (1 1 1) );
boundary
(
    end_left     { type symmetryPlane; faces ((0 4 7 3)); }
    end_right    { type symmetryPlane; faces ((1 2 6 5)); }
    hot          { type wall;          faces ((0 1 5 4)); }
    cold         { type wall;          faces ((3 7 6 2)); }
    frontAndBack { type empty;         faces ((0 3 2 1) (4 5 6 7)); }
);
FOAM
cat > 0/D <<'FOAM'
FoamFile { version 2.0; format ascii; class volVectorField; object D; }
dimensions [0 1 0 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    end_left  { type symmetryPlane; }
    end_right { type symmetryPlane; }
    hot       { type fixedValue; value uniform (0 0 0); }
    cold      { type tractionDisplacement; traction uniform (0 0 0);
                pressure uniform 0; value uniform (0 0 0); }
    frontAndBack { type empty; }
}
FOAM
cat > 0/T <<'FOAM'
FoamFile { version 2.0; format ascii; class volScalarField; object T; }
dimensions [0 0 0 1 0 0 0];
internalField uniform 0;
boundaryField
{
    end_left  { type symmetryPlane; }
    end_right { type symmetryPlane; }
    hot       { type fixedValue; value uniform 100; }
    cold      { type fixedValue; value uniform 0; }
    frontAndBack { type empty; }
}
FOAM
cat > constant/thermalProperties <<'FOAM'
FoamFile { version 2.0; format ascii; class dictionary; object thermalProperties; }
C     { type uniform; value 434; }
k     { type uniform; value 60.5; }
alpha { type uniform; value 1.1e-05; }
thermalStress yes;
FOAM
cat > system/controlDict <<'FOAM'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application solidDisplacementFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 60; deltaT 2;
writeControl timeStep; writeInterval 30; purgeWrite 0;
writeFormat ascii; writePrecision 10; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable true;
FOAM
cat > system/fvSchemes <<'FOAM'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
d2dt2Schemes { default steadyState; }
ddtSchemes   { default Euler; }
gradSchemes  { default leastSquares; grad(D) leastSquares; grad(T) leastSquares; }
divSchemes   { default none; div(sigmaD) Gauss linear; }
laplacianSchemes { default none; laplacian(DD,D) Gauss linear corrected;
                   laplacian(DT,T) Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default none; }
FOAM
cat > system/fvSolution <<'FOAM'
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers { "(D|T)" { solver GAMG; tolerance 1e-10; relTol 0;
                    smoother GaussSeidel; nCellsInCoarsestLevel 20; } }
stressAnalysis { compactNormalStress yes; nCorrectors 3; D 1e-09; }
FOAM
for state in yes no; do
    rm -rf "/tmp/gate-stress/$state"
    cp -r /tmp/gate-stress/base "/tmp/gate-stress/$state"
    cd "/tmp/gate-stress/$state"
    cat > constant/mechanicalProperties <<FOAM
FoamFile { version 2.0; format ascii; class dictionary; object mechanicalProperties; }
rho { type uniform; value 7854; }
nu  { type uniform; value 0.3; }
E   { type uniform; value 2e+11; }
planeStress $state;
FOAM
    LD_PRELOAD=$SHIM blockMesh > log.blockMesh 2>&1
    LD_PRELOAD=$SHIM solidDisplacementFoam > log.solve 2>&1
    grep -q "^End" log.solve
done
python3 - <<'PY'
# E*alpha*dT with the dictionaries' own numbers: 2e11 * 1.1e-5 * 100.
peak = 2.0e11 * 1.1e-05 * 100.0
nu = 0.3
want = {"yes": peak, "no": peak / (1.0 - nu)}
for state, expected in want.items():
    lines = [line for line in open(f"/tmp/gate-stress/{state}/log.solve")
             if "Max sigmaEq" in line]
    assert lines, f"planeStress {state}: the solver printed no `Max sigmaEq` line"
    got = float(lines[-1].split("=")[1])
    error = abs(got - expected) / expected
    label = "plane stress" if state == "yes" else "plane strain"
    assert error < 5e-3, f"{label}: max sigmaEq {got:.6g} Pa against an exact {expected:.6g} Pa ({error:.2%})"
    print(f"thermal stress ok: {label}, max sigmaEq {got / 1e6:.4f} MPa "
          f"against an exact {expected / 1e6:.4f} MPa ({error:.2%})")
# The two states differ by 1/(1-nu) = 1.4286, so neither number can be reached by
# reading the planeStress switch the wrong way round.
assert abs(want["no"] / want["yes"] - 1.0 / (1.0 - nu)) < 1e-12
PY
cd / && rm -rf /tmp/gate-stress
# --- end stress analysis ----------------------------------------------------------

echo "== installs are refused, other pip use passes"
if pip install anything 2> /dev/null; then echo "pip install was not refused"; exit 1; fi
pip --version > /dev/null
rm -rf /root/.cache /root/.ipython /root/.fltk /root/.local
echo "workspace image ok"
