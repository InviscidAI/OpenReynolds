#!/bin/bash
# What the image promises, each checked by doing it. Run as the last build step.
# $1 is the gmsh version the build pinned.
set -eo pipefail
GMSH_VERSION="$1"
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

echo "== CAD: a solid built, exported to STEP, re-imported and meshed"
python3 - <<'PY'
import gmsh
from build123d import Cylinder, export_step
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

echo "== installs are refused, other pip use passes"
if pip install anything 2> /dev/null; then echo "pip install was not refused"; exit 1; fi
pip --version > /dev/null
rm -rf /root/.cache /root/.ipython /root/.fltk /root/.local
echo "workspace image ok"
