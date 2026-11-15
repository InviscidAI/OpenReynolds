#!/bin/bash
# HiSA (gitlab.com/hisa/hisa), built into OpenFOAM's site directories, which every
# shell that sources the OpenFOAM bashrc already has on PATH and LD_LIBRARY_PATH.
# A sparse checkout: upstream examples/ is 1.9 GB of meshes, and only the ONERA M6
# case is kept, as a working density-based dictionary set.
source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || true
set -euo pipefail
TAG=validated-v2512
COMMIT=271b44919d2b107eee05566dd92eb735e9bb08e9
test -n "$FOAM_SITE_APPBIN"
git clone --branch "$TAG" --depth 1 --filter=blob:none --sparse https://gitlab.com/hisa/hisa.git /opt/hisa
cd /opt/hisa
git sparse-checkout set src applications examples/oneraM6/simulation
test "$(git rev-parse HEAD)" = "$COMMIT"
# The longest path in the tree: proof the checkout materialised all of it.
test -f src/finiteVolume/boundaryConditions/characteristic/characteristicVelocityInletOutlet/characteristicVelocityInletOutletVelocityFvPatchVectorField.C
export FOAM_USER_APPBIN="$FOAM_SITE_APPBIN" FOAM_USER_LIBBIN="$FOAM_SITE_LIBBIN" WM_NCOMPPROCS="$(nproc)"
mkdir -p "$FOAM_SITE_APPBIN" "$FOAM_SITE_LIBBIN"
./Allwmake
test -x "$FOAM_SITE_APPBIN/hisa"
test -e "$FOAM_SITE_LIBBIN/libhisaFiniteVolume.so"
if ldd "$FOAM_SITE_APPBIN/hisa" | grep "not found"; then exit 1; fi
rm -rf /opt/hisa/.git
find /opt/hisa -type d \( -name lnInclude -o -name "linux64*" \) -prune -exec rm -rf {} +
