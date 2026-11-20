# The Reynolds workspace: OpenFOAM ESI v2512 and everything the agent's prompt and
# toolbox expect beside it. `openreynolds --local` runs this image in Docker; the
# hosted workspace runs it too, with one private layer on top (its job supervisor).
#
# Published as ghcr.io/inviscidai/reynolds-workspace:2512 by
# .github/workflows/workspace-image.yml. Build it yourself with
#
#     docker build -t reynolds-workspace:local docker/workspace
#     OPENREYNOLDS_IMAGE=reynolds-workspace:local openreynolds --local
#
# Every layer that installs something is followed by a check that the thing works,
# not that the install exited 0: a success marker that cannot fail is not one.

# Pinned by digest: a tag alone is not reproducible. Re-resolve and bump on purpose.
FROM opencfd/openfoam-default:2512@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- Python, headless rendering, and the X libraries gmsh and OCP link against -----
# The wheels' libgmsh.so and OCP's modules link the X stack even when no window is
# ever opened; the list is `ldd ... | grep "not found"` on this base, not a guess.
# ffmpeg alone without recommends: they are VDPAU drivers and a speech model.
RUN apt-get update && apt-get install -y \
        python3 python3-pip python3-venv git poppler-utils \
        libgl1 libosmesa6 libglu1-mesa libxcursor1 libxfixes3 libxft2 libxinerama1 \
        libxrender1 libfontconfig1 libxext6 libxi6 \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pyvista renders off-screen through VTK built against OSMesa, which is not on PyPI:
# install the stack, drop the vtk pyvista pulled in, install vtk-osmesa in its place.
# Empty means "whatever PyPI serves at build time", which is what this line did for its
# whole life and is the same shape as the `build123d>=0.9` floor that took the CAD
# desk down. `check.sh` prints what it resolved either way and asserts each one that
# is given, so the pins are a measurement rather than a guess.
ARG NUMPY_VERSION=
ARG MATPLOTLIB_VERSION=
ARG PANDAS_VERSION=
ARG PYVISTA_VERSION=
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy${NUMPY_VERSION:+==$NUMPY_VERSION}" \
        "matplotlib${MATPLOTLIB_VERSION:+==$MATPLOTLIB_VERSION}" \
        "pandas${PANDAS_VERSION:+==$PANDAS_VERSION}" \
        "pyvista${PYVISTA_VERSION:+==$PYVISTA_VERSION}" \
    && python3 -m pip uninstall -y --break-system-packages vtk \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        --extra-index-url https://wheels.vtk.org vtk-osmesa
ENV PYOPENGL_PLATFORM=osmesa

# --- HiSA: a density-based solver for compressible, shock-capturing cases ----------
# Pinned to the validated-v2512 release commit; the sources stay in /opt/hisa.
COPY build-hisa.sh /tmp/build-hisa.sh
RUN bash /tmp/build-hisa.sh && rm /tmp/build-hisa.sh

# --- animations: imageio with its bundled ffmpeg, and no fetching at write time ----
# Quoted: an unquoted `>=` in a shell is a redirection, and the floor never reaches pip.
RUN python3 -m pip install --no-cache-dir --break-system-packages 'imageio[ffmpeg]>=2.31'
ENV IMAGEIO_NO_INTERNET=1

# --- CAD: one gmsh (CLI and module from the same pinned wheel) and build123d -------
# The wheel's `gmsh` script says `#!/usr/bin/env python`, a name Ubuntu does not
# have; pointed at python3 it is the real CLI, the same build as the module.
#
# build123d is pinned exactly for the reason the base is: a floor alone is not
# reproducible. `build123d>=0.9` resolved to 0.9.x when this layer was written and to
# 0.13.0 by November, whose OCP dropped `TopTools_IndexedMapOfShape` and changed what
# `Bnd_Box.Get()` returns -- 18 of the suite's tests with it, and the CAD desk's
# patch-export path among them. 0.11.1 is the version b123d_api.md is checked against.
# OCP is pinned beside it because OCP is what actually broke and build123d 0.11.1 only
# bounds it to `>=7.9,<8.0`, three releases wide.
ARG GMSH_VERSION=4.15.2
ARG BUILD123D_VERSION=0.11.1
ARG OCP_VERSION=7.9.3.1.1
RUN python3 -m pip install --no-cache-dir --break-system-packages "gmsh==${GMSH_VERSION}" \
    && sed -i '1s|.*|#!/usr/bin/python3|' /usr/local/bin/gmsh && chmod +x /usr/local/bin/gmsh \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        "build123d==${BUILD123D_VERSION}" "cadquery-ocp-novtk==${OCP_VERSION}"

# --- the agent's Python kernel ---------------------------------------------------
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        'jupyter_client>=8' 'ipykernel>=6' 'pyzmq>=25'

# --- OpenFOAM's helpers in every bash ----------------------------------------------
# Login shells read /etc/profile.d; any other bash reads it through BASH_ENV, which
# whoever runs the container may set to /etc/foamd/shellinit.sh.
COPY shellinit.sh /etc/foamd/shellinit.sh
RUN echo 'source /etc/foamd/shellinit.sh' > /etc/profile.d/foamd.sh

# --- #codeStream as root -----------------------------------------------------------
# OpenFOAM refuses to compile code at run time as root (Foam::isAdministrator). A
# container that runs as root (Docker Desktop on Windows; the hosted workspace)
# preloads this library, which answers "no"; one that runs as the person needs none.
RUN echo 'int _ZN4Foam15isAdministratorEv(void) { return 0; }' > /tmp/shim.c \
    && gcc -O2 -fPIC -shared -nostdlib -o /usr/local/lib/libfoamnotadmin.so /tmp/shim.c \
    && rm /tmp/shim.c

# --- no installs at run time -------------------------------------------------------
# The workspace has no network. pip and apt fail fast with a pointer to what is
# already here, instead of hanging on it. Last, so it fences none of the above.
COPY install-guard.sh /usr/local/bin/reynolds-install-guard
RUN chmod +x /usr/local/bin/reynolds-install-guard \
    && for t in pip pip3 apt apt-get; do ln -sf /usr/local/bin/reynolds-install-guard /usr/local/bin/$t; done

# --- the checks --------------------------------------------------------------------
COPY check.sh /tmp/check.sh
RUN bash /tmp/check.sh "${GMSH_VERSION}" "${BUILD123D_VERSION}" "${OCP_VERSION}" \
        "${NUMPY_VERSION}" "${MATPLOTLIB_VERSION}" "${PANDAS_VERSION}" "${PYVISTA_VERSION}" \
    && rm -rf /tmp/check.sh /tmp/gate*
