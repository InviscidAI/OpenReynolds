#!/usr/bin/env python3
"""Run NVIDIA Isaac Sim itself -- not just the `isaaclab` wheel -- on a Modal GPU.

    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_isaacsim.py::vulkan
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_isaacsim.py::probe
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_isaacsim.py::demo

Scaffolding, like `modal_run.py` beside it, and it follows that file's rules: every
remote function is a one-shot `@app.function` with an explicit `timeout`, there is no
`@app.cls`, no warm pool, no `min_containers` and no volume, so a job starts, does one
thing, returns bytes and the container exits.

## Why this file is separate from `modal_run.py`

`modal_run.py` runs the live backend on an image that is four pip packages and about
2 GB. This one has to install Isaac Sim, which is a different kind of object: the
`isaacsim[all,extscache]` wheel set is roughly **7.5 GiB of wheels** and unpacks to
rather more, almost all of it the two `extscache` packages that carry the Omniverse Kit
runtime (`isaacsim-extscache-kit` alone is 5.2 GiB). Putting that in the benchmark image
would make every fluid measurement wait on an Isaac Sim pull it does not use.

## The three facts that decide whether this works at all

**Isaac Sim is cp312, manylinux_2_35, x86_64-or-aarch64 only.** Read off
`https://pypi.nvidia.com/isaacsim/` rather than remembered: version 6.2.0.0 publishes
exactly `isaacsim-6.2.0.0-cp312-none-manylinux_2_35_x86_64.whl` and the aarch64 and
win_amd64 siblings. So the base image must be Python 3.12 with glibc >= 2.35, which
Debian bookworm (2.36) satisfies and which is already the Python `modal_run.py` pins.
Version 5.1.0.0 and 5.0.0.0 are cp311 and 4.5.0.0 is cp310, so a downgrade is also a
Python downgrade.

**The install prompts for a EULA unless told not to.** `OMNI_KIT_ACCEPT_EULA` set to
`YES` suppresses it; without it the first `import isaacsim` blocks on stdin, which in a
container means the job hangs until the timeout rather than failing.

**Isaac Sim's own requirements page says "GPUs without RT Cores (A100, H100) are not
supported".** That is the opposite of what the fluid wants: the lattice is pure compute
and takes the H100, and Isaac Sim's renderer wants RT cores and does not care about
HBM. A10G (Ampere, RT cores) and L40S (Ada, RT cores) are the two shapes here that
satisfy both, which is also why probing is cheap -- an A10G is a small fraction of an
H100's rate.

## Why `vulkan` exists, and the wrong answer it gives

Isaac Sim's RTX renderer is Vulkan, and a CUDA container is not automatically a Vulkan
container: the NVIDIA container runtime mounts the graphics half of the driver only when
`NVIDIA_DRIVER_CAPABILITIES` asks for it. So `vulkan` runs first, on a 300 MB image, and
costs a minute.

**Read its output carefully, because the obvious reading of it is wrong.** On Modal it
reports that the NVIDIA ICD is present at `/etc/vulkan/icd.d/nvidia_icd.json`, that
every library it names resolves, that `libGLX_nvidia.so.0` exports
`vk_icdGetInstanceProcAddr`, `vk_icdNegotiateLoaderICDInterfaceVersion` and
`vkCreateInstance` -- and that `vulkaninfo` nevertheless enumerates only `llvmpipe`,
Mesa's software rasteriser, having failed with

    ERROR: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get
    'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0

The same failure appears on Debian bookworm with loader 1.3.239 and on Ubuntu 24.04 with
loader 1.3.275, on A10G and on L40S, with `/proc/driver/nvidia` reduced to a single
`params` file and the kernel reported as `4.19.0-gvisor`.

The conclusion that invites -- no GPU rendering on Modal -- is false, and this file drew
it for an afternoon. **Kit does not use the system Vulkan loader.** It brings its own,
and its own finds the device. From an actual run of `demo` on an A10G:

    | Driver Version: 620.70.06     | Graphics API: Vulkan
    | GPU | Name          | Active | GPU Memory | Vendor-ID |
    | 0   | NVIDIA A10G   | Yes: 0 | 23028   MB | 10de      |

So `vulkaninfo` is a fact about `vulkaninfo`, and the only test that settles whether
Isaac Sim can render here is starting Isaac Sim and looking at the pixels. The
`render_stddev` and `render_unique_colours` that `demo` reports exist so that "it
rendered" is a measurement rather than an impression -- the first green run of it
returned a 108 KB PNG that was a smooth grey gradient with no geometry in it at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal


def _local_repo() -> Path:
    """The checkout as seen from the machine `modal run` was typed on.

    Same reasoning as `modal_run.py::_local_repo`: this module is imported twice, once
    here at `benchmarks/live/modal_isaacsim.py` and once inside the container where
    Modal has moved the entrypoint to `/root/` and `parents[2]` would raise.
    """
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) > 2 else Path("/repo")


REPO = _local_repo()
RESULTS = Path(__file__).resolve().parent / "results" / "isaacsim"

ISAACSIM = "6.2.0.0"
"""The version pinned here, and the newest on `pypi.nvidia.com` as of 2026-11-23.

Pinned rather than floating for the reason `modal_run.py` pins Warp and Newton: the
whole point of this file is to report which Warp, Newton and MuJoCo Isaac Sim brings
with it, and that answer is meaningless without the version it is an answer about. The
install page's own warning is that these wheels "pin some packages, including Newton,
Warp, and MuJoCo, to specific versions", which is exactly the collision this repository
needs to know the shape of.
"""

TORCH = "torch==2.11.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
"""Installed before Isaac Sim, in the order and at the version the install page gives.

Isaac Sim does not declare a torch pin itself, so whatever is installed first is what
its tensor APIs hand back, and letting pip resolve a CPU-only torch into an image whose
entire purpose is a GPU is the sort of thing that is only discovered three layers later.
"""

# The X, GL and audio libraries Kit's extensions dlopen even with no window. A headless
# Kit still loads the UI extensions; it just never presents a surface, so the .so files
# have to be there. `vulkan-tools` is diagnosis rather than runtime -- `vulkaninfo` is
# how the `vulkan` entrypoint answers its question.
APT = [
    "libgl1", "libglu1-mesa", "libglx-mesa0", "libegl1", "libgles2",
    "libxrandr2", "libxinerama1", "libxcursor1", "libxi6", "libxt6",
    "libxrender1", "libxkbcommon-x11-0", "libsm6", "libice6",
    "libfreetype6", "libfontconfig1", "libgomp1", "libatomic1",
    "libvulkan1", "vulkan-tools", "libnvidia-egl-wayland1",
    "ca-certificates", "curl",
]

# Kit writes caches, shader binaries and logs under $HOME, and it writes a lot of them.
# All of it is pointed at /tmp so that a read-only or absent home directory is not a
# first-run failure, and so that nothing survives the container -- which is the same
# discipline `modal_run.py` applies to the Warp kernel cache.
KIT_ENV = {
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "ACCEPT_EULA": "Y",
    "PRIVACY_CONSENT": "Y",
    "OMNI_KIT_ALLOW_ROOT": "1",
    "WARP_CACHE_PATH": "/tmp/warp-cache",
    "XDG_CACHE_HOME": "/tmp/cache",
    "HOME": "/root",
    "PYTHONUNBUFFERED": "1",
    # Asks the NVIDIA container runtime for the graphics half of the driver as well as
    # the compute half. Whether Modal honours it is precisely what `vulkan` measures;
    # it is set here because it costs nothing and is the documented knob.
    "NVIDIA_DRIVER_CAPABILITIES": "all",
}

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("vulkan-tools", "libvulkan1", "libgl1", "pciutils", "binutils")
    .env(KIT_ENV)
)
"""Small and fast: just enough to ask the driver what it exposes."""

UBUNTU = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"
"""The base the requirements page names, kept only to compare against bookworm.

This existed to test a hypothesis that turned out to be wrong, and it is kept because
the negative result is worth being able to reproduce. The hypothesis was that the system
Vulkan loader's refusal of the NVIDIA ICD was a version mismatch -- bookworm ships loader
**1.3.239** from 2023 against a **610.57.04** driver -- and that Ubuntu 24.04, which the
requirements page lists as supported and which carries loader **1.3.275**, would
negotiate with it.

It does not. The failure is identical on both, and identical on A10G and L40S. What it
actually indicates is the sandbox: `/proc/driver/nvidia` holds only `params`, with none
of the `gpus/*/information` tree the driver's graphics path reads, and the kernel
reports as `4.19.0-gvisor`. The device nodes present are the compute set --
`nvidiactl`, `nvidia<N>`, `nvidia-uvm`, `nvidia-uvm-tools` -- with no `nvidia-modeset`.

None of which stops Isaac Sim, because Kit brings its own loader and gets the device
anyway. See the module docstring. The image this file actually uses is `debian_slim`.
"""

ubuntu_probe_image = (
    modal.Image.from_registry(UBUNTU, add_python="3.12")
    .apt_install("vulkan-tools", "libvulkan1", "libgl1", "pciutils", "binutils")
    .env(KIT_ENV)
)

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*APT)
    .pip_install(TORCH, index_url=TORCH_INDEX)
    .pip_install(
        f"isaacsim[all,extscache]=={ISAACSIM}",
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("pillow>=10", "numpy>=1.24")
    .env(KIT_ENV)
)
"""Everything expensive, and nothing that changes. Both images below derive from it, so
the 7.5 GiB layer is built once and neither the test dependencies nor the repository
mount can invalidate it."""

# Not `copy=True`, unlike `modal_run.py`. There the repository is baked in so that a
# benchmark image is a single immutable thing; here the layer under it takes five minutes
# to build and a runtime mount means editing `openreynolds/live/` never touches it.
# Nothing in this file needs the repository at image build time.
_REPO_MOUNT = dict(
    remote_path="/repo",
    ignore=[
        "studies/**", ".git/**", "**/__pycache__/**", "*.log", "launcher/**",
        ".pytest_cache/**", "benchmarks/live/results/**", "docker/**",
    ],
)

image = _base.add_local_dir(str(REPO), **_REPO_MOUNT)

# `tests/conftest.py` imports `anthropic`, `rich` and four `openreynolds` modules that
# are nothing to do with the live backend, and pytest loads it for anything under
# `tests/`. So running the live tests needs the package's own runtime dependencies, which
# are listed here from `pyproject.toml` rather than installed with `pip install -e /repo`
# -- an editable install would also run the build backend and re-resolve numpy and torch
# inside an image whose numpy and torch are the ones Isaac Sim pinned.
TEST_DEPS = [
    "pytest>=8.0.0", "pytest-asyncio>=0.24",
    "anthropic>=1.0,<2", "openai>=1.50,<4", "httpx2[socks]>=2.0.0",
    "rich>=13.0.0", "click>=8.1.0", "textual>=8.0", "mcp>=2.2,<3",
    "matplotlib>=3.7", "pandas>=2.0",
]

tests_image = _base.pip_install(*TEST_DEPS).add_local_dir(str(REPO), **_REPO_MOUNT)

app = modal.App("openreynolds-isaacsim")


def _gpu_name() -> str:
    import subprocess

    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return done.stdout.strip().splitlines()[0]
    except Exception as exc:
        return f"no nvidia-smi: {exc}"


# -- 1. does this container have a Vulkan device at all ---------------------------------


def _vulkan_report() -> dict:
    """Everything that decides whether Kit's RTX renderer can start.

    Three independent signals, because any one of them alone is ambiguous. The ICD JSON
    files are what the Vulkan loader reads to find a driver; `libnvidia-glcore` and
    friends are the graphics half of the driver mount; `vulkaninfo` is the only one that
    proves a device actually enumerates.
    """
    import glob
    import os
    import subprocess

    out: dict = {"gpu": _gpu_name()}
    out["icd_files"] = sorted(glob.glob("/usr/share/vulkan/icd.d/*")) + \
        sorted(glob.glob("/etc/vulkan/icd.d/*"))
    for name in out["icd_files"]:
        try:
            out[f"icd::{os.path.basename(name)}"] = Path(name).read_text()[:400]
        except Exception as exc:
            out[f"icd::{os.path.basename(name)}"] = f"unreadable: {exc}"
    # Globbed wide rather than by name. The first version of this listed the seven
    # libraries a working Vulkan stack is made of and found five of them, which looked
    # like a near miss; listing everything the runtime actually mounted is the only way
    # to tell a near miss from a different mount set entirely.
    libs = []
    for pattern in ("libnvidia-*.so*", "libGLX_nvidia.so*", "libEGL_nvidia.so*",
                    "libnvoptix.so*", "libcuda.so*", "libvulkan*.so*"):
        for root in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64",
                     "/usr/lib/x86_64-linux-gnu/nvidia/current"):
            libs += sorted(glob.glob(f"{root}/{pattern}"))
    out["driver_libs"] = sorted(set(libs))
    out["driver_capabilities"] = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "<unset>")
    # The last place the failure can live. The ICD exports every entry point the loader
    # asks for and all its dependencies resolve, so `vk_icdGetInstanceProcAddr` returning
    # NULL for `vkCreateInstance` means the driver declined to initialise, and the driver
    # initialises by opening device nodes and reading `/proc/driver/nvidia`. CUDA needs
    # `/dev/nvidiactl`, `/dev/nvidia0` and `/dev/nvidia-uvm`; the graphics side also wants
    # `/dev/nvidia-modeset` and the `/proc/driver/nvidia/gpus/*` tree, and a sandbox that
    # proxies only the compute ioctls will have the first set and not the second.
    out["dev_nvidia"] = sorted(glob.glob("/dev/nvidia*"))
    out["proc_driver_nvidia"] = sorted(glob.glob("/proc/driver/nvidia/**"))
    for path in ("/proc/version", "/sys/devices/virtual/dmi/id/product_name"):
        try:
            out[f"read::{path}"] = Path(path).read_text()[:300]
        except Exception as exc:
            out[f"read::{path}"] = f"unreadable: {exc}"
    out["egl"] = _egl_report()
    # `ldd` on the ICD itself. The loader's complaint is that it cannot find
    # `vkCreateInstance` in `libGLX_nvidia.so.0`, and there are two ways that happens:
    # the file is the GLX-only build with no Vulkan entry points, or it is the right file
    # and one of its own dependencies is missing so it never loaded. `ldd` separates
    # them, and nothing else does.
    # `objdump -T` then settles whether the entry point is exported at all, which is the
    # difference between the wrong file and a loader too old to negotiate an interface
    # version with the right one.
    for cmd in (["ldd", "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"],
                ["sh", "-c", "objdump -T /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 "
                             "| grep -i 'vk_icd\\|vkCreateInstance'"],
                ["sh", "-c", "dpkg -l 2>/dev/null | grep -i vulkan"],
                ["vulkaninfo", "--summary"], ["ldd", "--version"],
                ["cat", "/etc/os-release"]):
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out[" ".join(cmd)] = (done.stdout + done.stderr)[:4000]
        except Exception as exc:
            out[" ".join(cmd)] = f"failed: {exc}"
    return out


def _egl_report() -> dict:
    """Try to make a headless GPU rendering context the other way, through EGL.

    Worth the twenty lines because it separates two very different conclusions. If
    Vulkan is refused but EGL gives a hardware context, then the container does have a
    usable graphics path and only Kit's particular renderer is out, so a GL-based
    renderer could still produce an image. If EGL is refused the same way, the container
    has no GPU graphics at all and nothing that renders on the device will work here --
    which is a statement about the sandbox rather than about Isaac Sim, and it is the one
    that decides whether a different base image could ever help.

    Done with ctypes against `libEGL.so.1` rather than with a package, because every
    Python EGL binding would have to be installed and then believed, and
    `eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)` plus `eglInitialize` is the whole
    test. `EGL_EXT_platform_device` is the extension that offers a display with no X
    server and no DRM master, which is exactly the headless container case.
    """
    import ctypes

    out: dict = {}
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except Exception as exc:
        return {"libEGL": f"not loadable: {exc}"}
    EGL_NO_DISPLAY = ctypes.c_void_p(0)
    EGL_EXTENSIONS = 0x3055
    EGL_VENDOR = 0x3053
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    try:
        exts = egl.eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)
        out["client_extensions"] = (exts or b"").decode()
    except Exception as exc:
        out["client_extensions"] = f"FAILED: {exc}"
    # eglQueryDevicesEXT, the device-platform entry point, reached through
    # eglGetProcAddress because it is an extension and not in the .so's export table.
    try:
        egl.eglGetProcAddress.restype = ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        addr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
        out["has_eglQueryDevicesEXT"] = bool(addr)
        if addr:
            proto = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_int))
            query = proto(addr)
            devices = (ctypes.c_void_p * 16)()
            count = ctypes.c_int(0)
            ok = query(16, devices, ctypes.byref(count))
            out["eglQueryDevicesEXT"] = {"returned": bool(ok), "devices": count.value}
    except Exception as exc:
        out["eglQueryDevicesEXT"] = f"FAILED: {exc}"
    # And the default display, which is what a GL renderer would ask for first.
    try:
        egl.eglGetDisplay.restype = ctypes.c_void_p
        egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
        display = egl.eglGetDisplay(ctypes.c_void_p(0))  # EGL_DEFAULT_DISPLAY
        out["default_display"] = hex(display or 0)
        if display:
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
            ok = egl.eglInitialize(ctypes.c_void_p(display), ctypes.byref(major),
                                   ctypes.byref(minor))
            out["eglInitialize"] = {"ok": bool(ok),
                                    "version": f"{major.value}.{minor.value}"}
            if ok:
                vendor = egl.eglQueryString(ctypes.c_void_p(display), EGL_VENDOR)
                out["vendor"] = (vendor or b"").decode()
        egl.eglGetError.restype = ctypes.c_int
        out["eglGetError"] = hex(egl.eglGetError())
    except Exception as exc:
        out["eglInitialize"] = f"FAILED: {exc}"
    return out


@app.function(image=probe_image, gpu="A10G", timeout=600)
def vulkan_a10g() -> dict:
    return _vulkan_report()


@app.function(image=probe_image, gpu="L40S", timeout=600)
def vulkan_l40s() -> dict:
    return _vulkan_report()


@app.function(image=ubuntu_probe_image, gpu="A10G", timeout=600)
def vulkan_ubuntu_a10g() -> dict:
    return _vulkan_report()


@app.function(image=ubuntu_probe_image, gpu="L40S", timeout=600)
def vulkan_ubuntu_l40s() -> dict:
    return _vulkan_report()


# -- 2. what Isaac Sim is, and what it brings with it -----------------------------------


def _probe() -> dict:
    """Version-report Isaac Sim without starting it, then start it and report again.

    Split deliberately. If `SimulationApp({"headless": True})` is going to fail, the
    versions of the bundled Warp, Newton and MuJoCo are still the answer to half the
    question this file was written for, and they are readable from the installed wheels
    before Kit touches a GPU. So everything that can be learned statically is collected
    first and returned whatever happens afterwards.
    """
    import importlib
    import importlib.metadata as md
    import json
    import os
    import time
    import traceback

    sys.path.insert(0, "/repo")
    out: dict = {"gpu": _gpu_name(), "python": sys.version.split()[0]}

    def emit(key: str, value) -> None:
        """Record a finding and print it immediately, on one line, as JSON.

        Belt and braces, and the braces are the point. The second run of this probe got
        all the way to `[17.147s] app ready` and then the container went away, and Modal
        reported `InternalFailure: Server has lost track of input` -- a lost container,
        which returns no value at all. Everything learned in those seventeen seconds was
        lost with it. Printing each finding as it is made puts it in the job's log, which
        the local side is already saving to a file, so a crash costs one finding rather
        than all of them.
        """
        out[key] = value
        try:
            print("PROBE " + json.dumps({key: value}, default=str), flush=True)
        except Exception as exc:
            print(f"PROBE {{\"{key}\": \"unserialisable: {exc}\"}}", flush=True)

    emit("gpu", out["gpu"])

    # Every distribution Isaac Sim's dependency closure pinned, which is the real answer
    # to "can warp and newton live in here". Read from installed metadata rather than by
    # importing, so that a package that imports slowly or not at all still reports.
    wanted = ["isaacsim", "isaacsim-kernel", "isaacsim-core", "isaacsim-replicator",
              "isaacsim-extscache-kit", "isaacsim-extscache-physics",
              "warp-lang", "newton", "newton-physics", "mujoco", "mujoco-warp",
              "torch", "numpy", "usd-core", "isaaclab"]
    versions: dict = {}
    for name in wanted:
        try:
            versions[name] = md.version(name)
        except Exception:
            versions[name] = None
    emit("versions", versions)
    emit("all_isaacsim_dists", sorted(
        f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
        if (d.metadata["Name"] or "").lower().startswith("isaacsim")))

    # -- order matters here, and getting it wrong is fatal rather than slow -------------
    #
    # Nothing above imports anything. Everything below imports in exactly one order:
    # `isaacsim` first, then `SimulationApp`, then Warp, Newton and this repository.
    #
    # The first version of this function read the versions, then imported `warp`,
    # `newton` and `pxr` to see whether they coexist, and then started Kit. It never
    # reached the end: the process died at the C level with
    #
    #     python: symbol lookup error: .../omni.physics.stageupdate.../
    #     libomni.physics.stageupdate.plugin.so: undefined symbol:
    #     _ZN35pxrInternal_v0_25_11__pxrReserved__9pxr_boost6python9converter8registry...
    #
    # preceded by about forty `Could not load the dynamic library ... libusd_tf.so:
    # undefined symbol: _ZTIN35pxrInternal_v0_25_11__pxrReserved__...` from Kit's own
    # extscache.
    #
    # That is not Isaac Sim being broken. It is **two different builds of OpenUSD in one
    # process**. Kit's extscache ships USD in the `pxrInternal_v0_25_11` namespace, and
    # the pip `usd-core` wheel that Newton depends on ships `pxrInternal_v0_26_8`, and
    # both install libraries whose SONAME is plain `libusd_tf.so`. Whichever is dlopened
    # first wins the SONAME for the whole process, so `import pxr` or `import newton`
    # before Kit starts hands Kit the wrong USD and every symbol it looks for is in the
    # other namespace. The schema registry says the same thing one level up: "Cannot set
    # alias 'ParticleFieldSphericalHarmonicsAttributeAPI' under 'UsdSchemaBase', because
    # it is already set to 'UsdVol...'".
    #
    # So Kit goes first, unconditionally, and after that `pxr` resolves to Kit's own USD
    # and the pip one is simply never loaded. `import isaacsim` before
    # `isaacsim.simulation_app` is the documented form and is what registers the
    # extension search paths.
    started = time.time()
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        import isaacsim  # noqa: F401  -- registers extension paths; must precede the app
        from isaacsim.simulation_app import SimulationApp

        # Bound to a name and then never used again, deliberately. Nothing calls
        # `close()` any more (see the end of this function), but the reference has to
        # stay alive for the rest of the call: let the only handle to the app go out of
        # scope and Kit is free to be collected under the probe that is interrogating it.
        sim_app = SimulationApp({"headless": True})  # noqa: F841  -- keeps Kit alive
        emit("simulationapp_seconds", round(time.time() - started, 1))
        emit("simulationapp", "started")
    except Exception:
        emit("simulationapp", traceback.format_exc()[-6000:])
        emit("simulationapp_seconds", round(time.time() - started, 1))
        return out

    try:
        import carb

        settings = carb.settings.get_settings()
        emit("kit_version", settings.get("/app/version"))
        emit("renderer", settings.get("/renderer/active"))
    except Exception as exc:
        emit("kit_settings", f"FAILED: {exc}")

    # Warp is safe to import in process: it links CUDA and nothing from USD, so it
    # cannot take a SONAME Kit wants.
    for name in ("warp", "usdrt", "pxr"):
        try:
            mod = importlib.import_module(name)
            emit(f"import::{name}", getattr(mod, "__version__",
                                            "imported, no __version__"))
            emit(f"file::{name}", getattr(mod, "__file__", "?"))
        except Exception as exc:
            emit(f"import::{name}", f"FAILED: {type(exc).__name__}: {exc}")
    try:
        from pxr import Usd

        emit("usd_version_in_kit", str(Usd.GetVersion()))
    except Exception as exc:
        emit("usd_version_in_kit", f"FAILED: {exc}")

    try:
        import warp as wp

        wp.init()
        emit("warp_cuda_devices", wp.get_cuda_device_count())
        emit("warp_device", str(wp.get_preferred_device()))
    except Exception:
        emit("warp_init", traceback.format_exc()[-2000:])

    # The physics tensor and prim APIs the integration would use, discovered by import
    # rather than asserted from documentation, with their real signatures.
    _api_signatures(emit)

    # Does this repository's extension import inside a live Kit. Asked here because the
    # brief asks `probe` for it, and asked in a subprocess because `openreynolds.live`
    # imports Newton, Newton imports `pxr`, and a USD SONAME clash is a fatal loader
    # error rather than an exception. `coexistence` runs this and two more besides.
    emit("openreynolds_live_import", _subproc(_COEXIST_OPENREYNOLDS))

    # -- and deliberately no `sim_app.close()` ----------------------------------------
    #
    # The third run of this probe emitted every finding including the 688-extension
    # registry dump and then the job died with `InternalFailure: Server has lost track
    # of input`, with Modal's own gRPC health checks still ticking in the log. The only
    # statement between the last successful emit and the loss was `sim_app.close()`.
    # Kit's shutdown tears down 688 extensions, a USD stage and a renderer, and when it
    # wedges it takes the container's input handler with it; the documented symptom of
    # the same thing elsewhere is that `SimulationApp.close` "terminates the kernel" in
    # a notebook.
    #
    # Nothing here needs the process to survive its own exit. The container is one-shot
    # by design -- the rule this file inherited from `modal_run.py` -- so the cheapest
    # correct shutdown is to return the result and let Modal reap the container with
    # Kit still loaded. Not closing is not a leak; it is declining to run a teardown
    # whose only possible outcome is losing the answer.

    return out


_COEXIST_NEWTON = """
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
import isaacsim
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({"headless": True})
import newton, warp as wp
from pxr import Usd
print("NEWTON", newton.__version__, newton.__file__)
print("WARP", wp.config.version)
print("USD", Usd.GetVersion())
print("MODEL_BUILDER", hasattr(newton, "ModelBuilder"))
print("OK", flush=True)
os._exit(0)
"""

_COEXIST_OPENREYNOLDS = """
import os, sys
sys.path.insert(0, "/repo")
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
import isaacsim
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({"headless": True})
from openreynolds.live import world, couple
print("WORLD", world.__name__, "COUPLE", couple.__name__)
print("HAS_FLUIDWORLD", hasattr(world, "FluidWorld"))
print("OK", flush=True)
os._exit(0)
"""

_NEWTON_ALONE = """
import newton, warp as wp
from pxr import Usd
print("NEWTON", newton.__version__)
print("WARP", wp.config.version)
print("USD", Usd.GetVersion(), Usd.__file__)
print("OK")
"""


def _subproc(script: str, timeout: int = 600) -> dict:
    """Run a snippet in a fresh interpreter and report how it died, if it did."""
    import subprocess

    try:
        done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=timeout, cwd="/tmp")
    except subprocess.TimeoutExpired:
        return {"returncode": "timeout", "stdout": "", "stderr": ""}
    # Kit writes thousands of startup lines to stdout; only the lines this script
    # printed matter, and the tail of stderr is where a loader error lands.
    keep = [ln for ln in done.stdout.splitlines()
            if ln.startswith(("NEWTON", "WARP", "USD", "OK", "MODEL_BUILDER",
                              "WORLD", "HAS_FLUIDWORLD"))]
    return {"returncode": done.returncode, "stdout": keep,
            "stderr_tail": done.stderr[-2500:]}


def _api_signatures(emit) -> None:
    """Signatures of the read/write-per-step surface, taken from the live modules.

    Takes `emit` and returns nothing, rather than returning a dictionary, so that each
    module's answer is printed before the next module is imported. One of these imports
    can take the process down with it -- `isaacsim.core.experimental.prims` and
    `omni.physics.tensors` both pull C extensions that link USD, and a SONAME clash there
    is a `symbol lookup error` from the loader rather than an exception -- and when that
    happens the point is to know which import did it.

    Written as a loop over (module, attribute, members) rather than as prose, because a
    signature quoted from a release note is a claim and a signature from
    `inspect.signature` on the installed package is a fact. `openreynolds/live/isaac.py`
    takes the same line about `DirectRLEnv`.
    """
    import importlib
    import inspect

    # Read off the installed package, not off the documentation. The published docs for
    # these two extensions are in the middle of a move: `isaacsim.core.prims` is served
    # under `py/source/deprecated/` and the API reference page for `omni.physics.tensors`
    # renders empty, so several of the signatures that matter here are simply not on the
    # web. They are in the wheel, and the wheel is in this container.
    targets = [
        ("isaacsim.core.prims", "RigidPrim",
         ["__init__", "get_world_poses", "set_world_poses", "get_velocities",
          "set_velocities", "get_linear_velocities", "get_angular_velocities",
          "apply_forces", "apply_forces_and_torques_at_pos", "get_masses",
          "get_coms", "get_net_contact_forces"]),
        ("isaacsim.core.prims", "Articulation",
         ["__init__", "get_world_poses", "get_velocities", "set_joint_efforts",
          "get_jacobians", "get_mass_matrices"]),
        # The replacement path. `isaacsim.core.simulation_manager.set_backend` is marked
        # deprecated with "the core experimental API relies solely on Warp", which if
        # true is the zero-copy answer, so the experimental prims are asked directly.
        ("isaacsim.core.experimental.prims", "RigidPrim",
         ["__init__", "get_world_poses", "get_velocities", "apply_forces_and_torques",
          "apply_forces_and_torques_at_pos", "get_masses"]),
        ("isaacsim.core.simulation_manager", "SimulationManager",
         ["get_backend", "set_backend", "get_physics_sim_view", "get_device"]),
        ("omni.physics.tensors", None, []),
        ("omni.physics.tensors", "RigidBodyView",
         ["get_transforms", "get_velocities", "apply_forces_and_torques_at_pos",
          "get_masses"]),
        ("omni.physics.tensors", "SimulationView",
         ["create_rigid_body_view", "create_articulation_view", "set_subspace_roots"]),
        # The in-memory drawing path, which is the only candidate for a per-frame field
        # that does not go through disk.
        ("isaacsim.util.debug_draw", None, []),
        ("isaacsim.util.debug_draw._debug_draw", None, []),
    ]
    for mod_name, cls_name, members in targets:
        emit(f"api_attempt::{mod_name}.{cls_name}", "importing")
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            emit(f"api::{mod_name}", f"FAILED: {type(exc).__name__}: {exc}")
            continue
        if cls_name is None:
            emit(f"api::{mod_name}", {
                "file": getattr(mod, "__file__", "?"),
                "public": sorted(n for n in dir(mod) if not n.startswith("_"))[:80],
            })
            continue
        try:
            cls = getattr(mod, cls_name)
        except Exception as exc:
            emit(f"api::{mod_name}.{cls_name}", f"missing: {exc}")
            continue
        sigs: dict = {"__mro__": [f"{c.__module__}.{c.__name__}" for c in
                                  getattr(cls, "__mro__", [])][:6]}
        try:
            sigs["__file__"] = inspect.getfile(cls)
        except Exception:
            pass
        for name in members or sorted(n for n in dir(cls) if not n.startswith("_")):
            fn = getattr(cls, name, None)
            if fn is None:
                sigs[name] = "absent"
                continue
            try:
                sigs[name] = f"{name}{inspect.signature(fn)}"
            except Exception as exc:
                sigs[name] = f"unsignable: {type(exc).__name__}: {exc}"
        emit(f"api::{mod_name}.{cls_name}", sigs)
    # Which Kit extensions are present at all, filtered to the ones a field renderer or a
    # physics-tensor bridge would need. The registry is the ground truth for "is there a
    # volume extension in this build".
    try:
        import omni.kit.app

        mgr = omni.kit.app.get_app().get_extension_manager()
        names = sorted({e.get("name", "") for e in mgr.get_extensions()})
        emit("extensions_total", len(names))
        emit("extensions_of_interest", [
            n for n in names
            if any(k in n for k in ("debug_draw", "volume", "vdb", "warp", "tensors",
                                    "replicator", "ui.scene", "physx", "newton",
                                    "pointcloud", "particle"))
        ])
    except Exception as exc:
        emit("extensions", f"FAILED: {type(exc).__name__}: {exc}")


@app.function(image=image, timeout=900)
def deps() -> dict:
    """What the install actually resolved to, on no GPU at all.

    Separate from `probe` and deliberately cheap: it imports nothing, so it cannot crash
    the way the first probe did, and it needs no accelerator, so the answer to "what did
    `isaacsim[all,extscache]` pin" costs CPU seconds. It is also the only place that can
    report the version of a package whose import would poison the process, which is the
    whole point after the USD collision above.
    """
    import importlib.metadata as md

    rows = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name:
            rows[name] = dist.version
    interesting = {k: v for k, v in sorted(rows.items())
                   if k.startswith(("isaacsim", "omni", "usd", "warp", "newton",
                                    "mujoco", "torch", "nvidia", "pxr"))
                   or k in ("numpy", "usd-core", "pillow", "gymnasium")}
    out = {"count": len(rows), "interesting": interesting}
    # Which distribution owns `pxr`, which is the collision in one line.
    try:
        out["pxr_owners"] = sorted({
            d.metadata["Name"] for d in md.distributions()
            for f in (d.files or [])
            if str(f).startswith("pxr/") and str(f).endswith("__init__.py")})
    except Exception as exc:
        out["pxr_owners"] = f"FAILED: {exc}"
    return out


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768)
def probe_a10g() -> dict:
    return _probe()


@app.function(image=image, gpu="L40S", timeout=3600, memory=32768)
def probe_l40s() -> dict:
    return _probe()


@app.function(image=image, gpu="A10G", timeout=1800, memory=32768)
def coexist() -> dict:
    """Can Newton, and this repository's extension, be imported in a live Kit process.

    Three subprocesses rather than three imports, because the failure mode here is a
    `symbol lookup error` from the dynamic loader rather than an `ImportError`: two
    different builds of OpenUSD both install a `libusd_tf.so`, whichever is dlopened
    first wins the SONAME, and the loser's symbols are simply absent. That kills the
    interpreter before any `except` can run. A subprocess turns it into a return code
    and a captured stderr, so the answer survives whichever way it goes. Each script
    ends in `os._exit(0)` for the same reason `_probe` does not close its app.
    """
    return {
        "gpu": _gpu_name(),
        "newton_alone": _subproc(_NEWTON_ALONE),
        "kit_then_newton": _subproc(_COEXIST_NEWTON),
        "kit_then_openreynolds": _subproc(_COEXIST_OPENREYNOLDS),
    }


# -- 3. the minimal scene, stepped and rendered -----------------------------------------


def _demo(steps: int = 100, width: int = 1280, height: int = 720) -> dict:
    """Ground plane, a cube dropped onto it, `steps` physics steps, and a rendered PNG.

    The PNG comes from a Replicator render product rather than from a viewport capture.
    Both exist, and the viewport path is the one every tutorial shows, but a headless Kit
    has no viewport window, so `omni.kit.viewport.utility.capture_viewport_to_file` is
    capturing a surface that may not have been created. `rep.create.render_product` makes
    an offscreen product bound to a camera that is explicitly ours, and the `rgb`
    annotator hands back a numpy array in process, which is also the only version of
    this that can return bytes rather than write into a container filesystem nobody will
    look at.

    The pose readback uses `isaacsim.core.prims.RigidPrim`, which is the tensor-API
    wrapper over PhysX: `get_world_poses()` and `get_velocities()` read, and
    `apply_forces_and_torques_at_pos()` writes. Those three are the whole surface a
    fluid coupling needs, so the demo exercises all three rather than only dropping a
    box: a constant sideways force goes on for the second half of the run, and the
    recorded trajectory is what shows it arrived.
    """
    import os
    import time
    import traceback

    out: dict = {"gpu": _gpu_name(), "steps": steps}
    blobs: dict[str, bytes] = {}
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    started = time.time()
    # The EGL question, asked here rather than in `vulkan` because the small probe image
    # has no `libEGL.so.1` to load and this one does. It is the same question one API
    # over: if the container can make a hardware GL context when it cannot make a Vulkan
    # one, then a GL renderer could still produce an image and only Kit's RTX path is
    # lost. If EGL fails the same way, the sandbox has no GPU graphics at all.
    out["egl"] = _egl_report()
    try:
        import isaacsim  # noqa: F401  -- registers extension paths; must precede the app
        from isaacsim.simulation_app import SimulationApp

        # Held for the life of the call, not used again. Same reason as in `_probe`: the
        # reference is what keeps Kit from being collected, and nothing closes it.
        sim_app = SimulationApp({  # noqa: F841  -- keeps Kit alive
            "headless": True,
            "width": width,
            "height": height,
            # RaytracedLighting rather than PathTracing: one sample per pixel is enough
            # for a screenshot of a grey box and path tracing would want hundreds of
            # accumulation frames, each one a render.
            "renderer": "RaytracedLighting",
        })
        out["startup_seconds"] = round(time.time() - started, 1)
    except Exception:
        out["error"] = traceback.format_exc()
        return {"result": out, "blobs": blobs}

    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, UsdGeom, UsdLux

        import omni.replicator.core as rep

        # `device="cuda"` is not decoration. Without it `World` builds a CPU physics
        # pipeline, and then every tensor view it hands out is on the host however it is
        # labelled: a run with the Warp frontend selected returned real `wp.array` and
        # `wp.indexedarray` objects whose `.device` was `cpu` and whose `.ptr` was
        # `None`. The frontend chooses the array type; the pipeline chooses where the
        # memory lives, and for a coupling to a GPU fluid only the second one matters.
        world = World(stage_units_in_meters=1.0, device="cuda")
        world.scene.add(GroundPlane(prim_path="/World/ground", size=50.0))

        # Three bodies rather than one. The extra two cost nothing -- they are the same
        # `RigidPrim` view with three paths instead of one, which is how a real coupling
        # would read a scene anyway -- and they turn the picture from "a box" into
        # something with depth, contact and a reason to have been rendered at all.
        spec = [
            ("/World/cube_a", [0.0, 0.0, 1.5], 0.40, [0.90, 0.35, 0.15]),
            ("/World/cube_b", [0.7, 0.45, 2.3], 0.30, [0.25, 0.55, 0.85]),
            ("/World/cube_c", [-0.55, 0.3, 3.0], 0.25, [0.35, 0.72, 0.40]),
        ]
        for path, position, size, colour in spec:
            world.scene.add(DynamicCuboid(
                prim_path=path, name=path.rsplit("/", 1)[-1],
                position=np.array(position), size=size,
                color=np.array(colour), mass=1.0))

        # A light rig, because an unlit RTX render is a black rectangle and "it must look
        # good" is part of the brief. A key light at an angle gives the boxes a lit face,
        # a shaded face and a cast shadow on the plane, which is what makes a render read
        # as three dimensional; the dome fills the shadows so the dark faces stay legible
        # rather than going to black.
        stage = world.stage
        distant = UsdLux.DistantLight.Define(stage, "/World/sun")
        distant.CreateIntensityAttr(3600.0)
        distant.CreateAngleAttr(0.9)
        UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(-52.0, 22.0, 18.0))
        dome = UsdLux.DomeLight.Define(stage, "/World/sky")
        # 300 rather than 900. At 900 the dome alone lifts the ground plane to very near
        # white, the key light has nothing left to add, and the result is a correct
        # render of a washed out scene -- the boxes read as pastel stickers with faint
        # shadows. Dropping the fill and raising the key puts the contrast back into the
        # faces and the shadows, which is the whole difference between a picture that
        # proves the renderer ran and one worth looking at.
        dome.CreateIntensityAttr(300.0)

        # -- the camera, and why it is Replicator's and not a hand-built one ------------
        #
        # The first version of this demo defined a `UsdGeom.Camera` at an explicit
        # translate and `rotateXYZ`, and it rendered a 108 KB PNG of a smooth grey
        # gradient with no geometry in it at all. The renderer was not the problem: Kit's
        # own startup table in that same run reads
        #
        #     | GPU | Name          | Active | GPU Memory |
        #     | 0   | NVIDIA A10G   | Yes: 0 | 23028   MB |
        #
        # with `Graphics API: Vulkan`, so RTX had the device and was drawing something.
        # The frame was empty because aiming a camera by Euler angles is a calculation
        # with no feedback, and a camera that misses is indistinguishable in the output
        # from a renderer that failed. `rep.create.camera(position=..., look_at=...)`
        # states the intent instead of the arithmetic, so the only way to get an empty
        # frame now is for there to be nothing there.
        # Close enough that the subject fills the frame. The first framing put the
        # camera 6 m out with a 26 mm lens and the boxes occupied about a fifteenth of
        # the picture, which is a thumbnail of a render rather than a render.
        camera = rep.create.camera(position=(2.3, -2.7, 1.45),
                                   look_at=(0.05, 0.05, 0.22), focal_length=33.0)
        product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])

        world.reset()

        # The read/write handle. Constructed after `world.reset()` because the physics
        # tensor view does not exist until the simulation view does.
        view = RigidPrim(prim_paths_expr="/World/cube_*", name="cubes")
        out["rigidprim_class"] = f"{type(view).__module__}.{type(view).__name__}"
        out["bodies_in_view"] = int(view.count)

        # A gentle, brief push rather than a hard one. The first run used 12 N for fifty
        # steps and sent the box to x = 1.68 m travelling at 4.5 m/s, which is a fine
        # demonstration of `apply_forces_and_torques_at_pos` and a poor one of anything
        # being in shot. The point of the write is that it visibly arrives, so it is
        # sized to nudge the boxes across the frame and stop.
        # Torch tensors, not numpy, and not by preference. Asking `World` for
        # `device="cuda"` makes Isaac Sim change the frontend out from under you:
        #
        #     [isaacsim.core.simulation_manager] Changing backend from 'numpy' to
        #     'torch' since NumPy cannot be used with GPU piplines
        #
        # after which handing `apply_forces_and_torques_at_pos` a numpy array fails deep
        # inside the wrapper with `'numpy.ndarray' object has no attribute 'to'`, because
        # `move_data` calls `.to(device)` on whatever it is given. So the inputs follow
        # the backend the manager actually chose rather than the one that was asked for.
        import torch

        device = str(SimulationManager.get_device())
        out["sim_device"] = device
        out["sim_backend"] = str(SimulationManager.get_backend())
        push = torch.tensor([[2.5, 0.0, 0.0], [-1.8, 0.0, 0.0], [0.0, -2.0, 0.0]],
                            dtype=torch.float32, device=device)
        torque = torch.tensor([[0.0, 0.0, 0.45], [0.0, 0.0, -0.35], [0.0, 0.0, 0.30]],
                              dtype=torch.float32, device=device)
        track = []
        for i in range(steps):
            if steps // 2 <= i < steps // 2 + 20:
                # The write half of the coupling surface, exercised on purpose: a
                # sideways force and a yaw torque per body, applied in the world frame
                # every step. PhysX clears external forces each step, so this has to be
                # reapplied rather than set once -- which is the same contract Newton's
                # `State.clear_forces()` imposes in `couple.NewtonCoupling`.
                view.apply_forces_and_torques_at_pos(
                    forces=push, torques=torque, is_global=True)
            # `render=True` rather than `False`. Physics does not need it, but the RTX
            # scene does: Hydra syncs the stage on a rendered frame, and a renderer that
            # has never been given one has nothing to draw when the annotator is finally
            # read.
            world.step(render=True)
            if i % 10 == 0 or i == steps - 1:
                pos, quat = view.get_world_poses()
                vel = view.get_velocities()
                track.append({
                    "step": i,
                    "pos": [[round(float(v), 4) for v in row] for row in _host(pos)],
                    "quat_wxyz": [round(float(v), 4) for v in _host(quat)[0]],
                    "lin_vel": [round(float(v), 4) for v in _host(vel)[0][:3]],
                    "ang_vel": [round(float(v), 4) for v in _host(vel)[0][3:]],
                })
        out["track"] = track
        out["final_pose"] = track[-1]

        out["physics_dt"] = float(world.get_physics_dt())
        out["render_dt"] = float(world.get_rendering_dt())

        # -- the render ---------------------------------------------------------------
        #
        # The render product and annotator were attached before `world.reset()`, so by
        # here the renderer has already drawn every one of the `steps` frames above. All
        # that is left is to let the accumulation settle on the final pose: RTX is
        # temporally accumulated, so a frame read immediately after motion stops still
        # carries the noise of the frames before it.
        # -- getting a frame out, and why this is a loop -------------------------------
        #
        # `world.render()` is not a synchronisation point. Twenty-four calls to it
        # produced a good 138 KB render once and an entirely empty annotator twice more
        # on the same code path, with the physics loop finishing in 2.7 s instead of
        # 105 s -- which is what it looks like when the renderer is simply never asked.
        # `annotator.get_data()` then returns a zero-length array and Pillow raises
        # `ValueError: cannot write empty image`, several frames downstream of the
        # actual problem.
        #
        # `rep.orchestrator.step()` is Replicator's own sync: it drives the graph, the
        # render product and the annotator together, and `wait_until_complete()` blocks
        # until the frame is really there. Even so it is wrapped in a bounded retry and
        # the result is measured rather than assumed, because an asynchronous renderer
        # that has returned an empty buffer twice has earned the check.
        arr = np.empty(0)
        for attempt in range(8):
            rep.orchestrator.step(rt_subframes=8)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception:
                pass
            arr = np.asarray(annotator.get_data())
            if arr.size and float(arr[..., :3].astype("float32").std()) > 1.0:
                out["render_attempts"] = attempt + 1
                break
        out["render_shape"] = list(arr.shape)
        if arr.size:
            rgb = arr[..., :3].astype("float32")
            out["render_nonzero_fraction"] = float((rgb > 8).mean())
            # The number that would have caught the empty first render. A frame of sky
            # and nothing else has almost no spread; a frame with lit geometry, a cast
            # shadow and a horizon has a lot. Reported rather than asserted, because the
            # honest form of "the picture came out" is a measurement of the picture.
            out["render_stddev"] = round(float(rgb.std()), 2)
            out["render_unique_colours"] = int(
                len(np.unique(rgb.reshape(-1, 3)[::97], axis=0)))
        if arr.size:
            import io

            from PIL import Image

            if arr.ndim == 3 and arr.shape[2] == 4:
                img = Image.fromarray(arr[..., :3].astype("uint8"), "RGB")
            else:
                img = Image.fromarray(arr.astype("uint8"))
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            blobs["isaacsim_cubes.png"] = buf.getvalue()
            out["png_bytes"] = len(blobs["isaacsim_cubes.png"])
        else:
            # Said rather than raised. An empty annotator after eight orchestrator steps
            # is a finding about the renderer, and burying it under a Pillow traceback
            # about an empty image loses it.
            out["render_error"] = "annotator returned no data after 8 orchestrator steps"


        # What is actually on the stage, so that an empty frame is a diagnosis rather
        # than a mystery. This is the list the first failed render needed and did not
        # have.
        out["stage_prims"] = [p.GetPath().pathString for p in stage.Traverse()
                              if p.GetTypeName() in
                              ("Mesh", "Cube", "Camera", "DistantLight", "DomeLight",
                               "Plane", "Xform")][:40]

        # Last, and deliberately last. An earlier version asked the zero-copy question
        # before the render and got `render_shape: [0]` back -- an empty annotator and a
        # physics loop that finished in three seconds instead of a hundred. Probing the
        # tensor API swaps the simulation backend and tears the existing simulation view
        # down, and the render product does not survive it. The image is the deliverable,
        # so the image is captured first and the introspection that disturbs the stage
        # happens afterwards, where the worst it can cost is itself.
        out["zero_copy"] = _zero_copy_report(view)
        out["ok"] = True
    except Exception:
        out["error"] = traceback.format_exc()
        out["ok"] = False
    finally:
        # No `sim_app.close()`, for the reason spelled out at the end of `_probe`: Kit's
        # teardown is what loses the container, and a one-shot container does not need a
        # clean shutdown to return its bytes.
        out["total_seconds"] = round(time.time() - started, 1)
    return {"result": out, "blobs": blobs}


def _host(value):
    """Whatever the active frontend handed back, as a numpy array on the host.

    The tensor API returns `np.ndarray`, `torch.Tensor`, `wp.array` or
    `wp.indexedarray` depending on the frontend, and once the pipeline is on CUDA the
    last two have no host storage for `np.asarray` to find. `.numpy()` is the one method
    all four agree on.
    """
    import numpy as np

    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            return np.asarray(value.cpu())
        except Exception:
            pass
    return np.asarray(value)


def _zero_copy_report(view) -> dict:
    """Ask the live PhysX tensor view what memory it hands out.

    `omni.physics.tensors` can be asked for `numpy`, `torch` or `warp` backed tensors;
    which one a wrapper like `RigidPrim` actually uses is a property of the wrapper and
    of `SimulationManager`'s configured backend, not of the docs. The two things that
    decide whether the fluid can read PhysX state without a host round trip are whether
    the returned object is a device tensor and whether its data pointer is in device
    memory, so both are reported rather than inferred.
    """
    out: dict = {}
    try:
        pos, _ = view.get_world_poses()
        out["get_world_poses_type"] = f"{type(pos).__module__}.{type(pos).__name__}"
        out["device"] = str(getattr(pos, "device", "n/a"))
        ptr = getattr(pos, "data_ptr", None)
        out["data_ptr"] = hex(ptr()) if callable(ptr) else None
    except Exception as exc:
        out["poses"] = f"FAILED: {exc}"
    try:
        from isaacsim.core.simulation_manager import SimulationManager

        out["backend"] = str(SimulationManager.get_backend())
        out["physics_sim_view"] = str(type(SimulationManager.get_physics_sim_view()))
    except Exception as exc:
        out["simulation_manager"] = f"FAILED: {type(exc).__name__}: {exc}"
    # The wrapped tensor API underneath, which is where a warp-backed view would come
    # from if the wrapper's own one is torch.
    try:
        import omni.physics.tensors as tensors

        out["tensors_module"] = sorted(n for n in dir(tensors)
                                       if not n.startswith("_"))[:40]
        out["has_create_simulation_view"] = hasattr(tensors, "create_simulation_view")
        import inspect

        if hasattr(tensors, "create_simulation_view"):
            out["create_simulation_view_sig"] = str(
                inspect.signature(tensors.create_simulation_view))
    except Exception as exc:
        out["omni.physics.tensors"] = f"FAILED: {type(exc).__name__}: {exc}"
    # -- the decisive test: ask the tensor API for a warp frontend directly -------------
    #
    # Everything above describes the default, which is numpy on the host and therefore a
    # round trip per step. The question that matters for a Warp fluid is not what the
    # default is but whether a device-resident Warp view is available at all, and
    # `omni.physics.tensors` answers it in one call: `create_simulation_view` takes a
    # frontend name, the module exports `frontend_np` and `frontend_warp` beside each
    # other, and the signature read off the installed package is
    # `create_simulation_view(frontend_name, stage_id=-1, backend='physx')`.
    #
    # What is checked is the only thing that distinguishes a view from a copy: the
    # returned array's device, and whether its pointer stays the same across two calls
    # and across a simulation step. A fresh allocation per call is a copy however it is
    # labelled; a stable device pointer that the physics engine writes through is not.
    # -- the decisive test: the same bodies, read through the Warp frontend -----------
    #
    # Everything above describes the default, which is numpy on the host and therefore a
    # device-to-host copy per step. The question for a Warp fluid is not what the default
    # is but whether a device-resident Warp view is available at all.
    #
    # `tensors.create_simulation_view("warp")` is the obvious call and it does not work
    # here: with a `World` already running it raises `Failed to create simulation view
    # with backend 'physx'`, because one view already exists and the physics backend will
    # not hand out a second. The supported route is to change the frontend on the
    # manager and build a new view through it, which is what
    # `SimulationManager.set_backend` is for.
    #
    # What is checked is the only thing that distinguishes a view from a copy: the
    # array's device, and whether its pointer survives a second call. A fresh allocation
    # per call is a copy however it is labelled.
    try:
        import warp as wp
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager

        SimulationManager.set_backend("warp")
        out["backend_after_set"] = str(SimulationManager.get_backend())
        out["device_after_set"] = str(SimulationManager.get_device())
        # Regex, not glob. `isaacsim.core.prims` takes `prim_paths_expr` as a regular
        # expression, so `cube_*` means "cube" followed by any number of underscores and
        # matches nothing; `cube_.*` is the one that finds three bodies.
        warp_view = RigidPrim(prim_paths_expr="/World/cube_.*", name="cubes_warp")
        out["warp_view_count"] = int(warp_view.count)
        first_pos, first_quat = warp_view.get_world_poses(clone=False)
        out["warp_poses"] = {
            "type": f"{type(first_pos).__module__}.{type(first_pos).__name__}",
            "dtype": str(getattr(first_pos, "dtype", None)),
            "shape": str(getattr(first_pos, "shape", None)),
            "device": str(getattr(first_pos, "device", None)),
            "ptr": hex(int(first_pos.ptr)) if getattr(first_pos, "ptr", None) else None,
            "quat_type": f"{type(first_quat).__name__}",
        }
        out["is_warp_array"] = isinstance(
            first_pos, (wp.array, getattr(wp, "indexedarray", wp.array)))
        out["on_cuda"] = str(getattr(first_pos, "device", "")).startswith("cuda")
        second_pos, _ = warp_view.get_world_poses(clone=False)
        out["ptr_stable_across_calls"] = (
            getattr(first_pos, "ptr", 0) == getattr(second_pos, "ptr", -1))
        vel = warp_view.get_velocities(clone=False)
        out["warp_velocities"] = {
            "type": f"{type(vel).__module__}.{type(vel).__name__}",
            "shape": str(getattr(vel, "shape", None)),
            "device": str(getattr(vel, "device", None)),
        }
        # And the write half, through the same warp-backed view, with warp arrays in.
        warp_view.apply_forces_and_torques_at_pos(
            forces=wp.array([[0.5, 0.0, 0.0]] * int(warp_view.count),
                            dtype=wp.float32,
                            device=str(SimulationManager.get_device())),
            is_global=True)
        out["warp_apply_forces"] = "accepted a wp.array"
    except Exception as exc:
        import traceback

        out["warp_frontend"] = f"FAILED: {type(exc).__name__}: {exc}"
        out["warp_frontend_trace"] = traceback.format_exc()[-1800:]
    # And the other route to the same place: the experimental prims wrapper, whose own
    # annotations say `get_world_poses(...) -> tuple[wp.array, wp.array]` rather than the
    # deprecated wrapper's `np.ndarray | torch.Tensor | wp.indexedarray`.
    try:
        from isaacsim.core.experimental.prims import RigidPrim as XRigidPrim

        xview = XRigidPrim("/World/cube_.*")
        xpos, xquat = xview.get_world_poses()
        out["experimental_prims"] = {
            "pos_type": f"{type(xpos).__module__}.{type(xpos).__name__}",
            "pos_device": str(getattr(xpos, "device", None)),
            "pos_shape": str(getattr(xpos, "shape", None)),
            "pos_ptr": hex(int(xpos.ptr)) if getattr(xpos, "ptr", None) else None,
            "quat_shape": str(getattr(xquat, "shape", None)),
        }
    except Exception as exc:
        out["experimental_prims"] = f"FAILED: {type(exc).__name__}: {exc}"
    return out


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768)
def demo_a10g(steps: int = 100) -> dict:
    return _demo(steps)


@app.function(image=image, gpu="L40S", timeout=3600, memory=32768)
def demo_l40s(steps: int = 100) -> dict:
    return _demo(steps)


# -- 4. does the extension RUN on Kit's own Warp, not just import -----------------------


def _cases() -> dict:
    """The live test suite and two graded free-surface cases, inside a running Kit.

    The point is narrow and worth stating, because an earlier session already answered
    the near-miss version of it. `openreynolds/live/` is known to work across Warp
    1.17-1.19 and Newton 1.6-1.8 with no shim, and the Isaac Lab virtualenv -- Warp
    1.17.0, Newton 1.6.1 -- ran 52 of 52 of `tests/test_live_cases.py`. **That is a
    different install from this one.** Kit does not use the pip `warp-lang` at all; it
    imports `omni.warp.core`, its own extension, which carries its own build of the Warp
    kernel cache and its own CUDA toolchain resolution. Whether the extension's kernels
    compile and run *there* is a question only this container can answer, and the whole
    reason the app started before any of this runs is that Kit has to win the `libusd`
    SONAME before Newton is imported.

    So: Kit first, then pytest in this very interpreter rather than in a subprocess,
    because a subprocess would get the pip Warp back and measure the thing that was
    already known.
    """
    import io
    import json
    import os
    import sys
    import time
    import traceback
    from contextlib import redirect_stderr, redirect_stdout

    out: dict = {"gpu": _gpu_name()}

    def emit(key: str, value) -> None:
        out[key] = value
        try:
            print("CASES " + json.dumps({key: value}, default=str), flush=True)
        except Exception as exc:
            print(f'CASES {{"{key}": "unserialisable: {exc}"}}', flush=True)

    emit("gpu", out["gpu"])
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    started = time.time()
    try:
        import isaacsim  # noqa: F401  -- registers extension paths; must precede the app
        from isaacsim.simulation_app import SimulationApp

        sim_app = SimulationApp({"headless": True})  # noqa: F841  -- keeps Kit alive
        emit("kit_startup_seconds", round(time.time() - started, 1))
    except Exception:
        emit("kit", traceback.format_exc()[-4000:])
        return out

    sys.path.insert(0, "/repo")
    # And in the environment too, not only in `sys.path`.
    # `test_both_array_layouts_give_the_same_field` is the one test in the suite that
    # re-execs `sys.executable -c ...` in a subprocess, because the array layout is baked
    # into the generated kernels at import and the only way to compare two layouts is two
    # interpreters. That child inherits the environment and not the parent's `sys.path`,
    # so with `/repo` only in `sys.path` it fails with `ModuleNotFoundError: No module
    # named 'openreynolds'` -- which looks exactly like a broken extension and is in fact
    # a broken harness. Measured: 62 passed, 1 failed, and the one was this.
    os.environ["PYTHONPATH"] = "/repo" + os.pathsep + os.environ.get("PYTHONPATH", "")
    import warp as wp

    # The two facts that make the rest of this meaningful. If `file::warp` is not under
    # `isaacsim/extscache/omni.warp.core`, the run proved nothing: it measured the pip
    # Warp and the answer was already known.
    emit("warp_version", wp.config.version)
    emit("warp_file", wp.__file__)
    try:
        wp.init()
        emit("warp_cuda_devices", wp.get_cuda_device_count())
    except Exception:
        emit("warp_init", traceback.format_exc()[-2000:])

    # -- the suite ---------------------------------------------------------------------
    import pytest

    buffer = io.StringIO()
    suite_started = time.time()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            code = pytest.main([
                "/repo/tests/test_live_cases.py",
                "/repo/tests/test_live_freesurface.py",
                "-q", "--no-header", "-p", "no:cacheprovider",
                "--rootdir=/repo",
            ])
        emit("pytest_returncode", int(code))
    except Exception:
        emit("pytest_returncode", "raised")
        emit("pytest_exception", traceback.format_exc()[-4000:])
    emit("pytest_seconds", round(time.time() - suite_started, 1))
    text = buffer.getvalue()
    # The tail carries the summary line and, when something failed, the assertion. Kept
    # whole rather than parsed into a count, because a count is the one part of a test
    # report that is easy to get right and useless when it is wrong.
    emit("pytest_tail", text[-6000:])
    emit("pytest_summary", [ln for ln in text.splitlines()
                            if "passed" in ln or "failed" in ln or "error" in ln][-6:])

    # -- the two graded cases ----------------------------------------------------------
    #
    # Run after the suite and each in its own `try`, so that a kernel that will not
    # compile on Kit's Warp names itself instead of taking the whole job down. The brief
    # is explicit that a failure here is to be reported, not worked around.
    import tempfile

    from openreynolds.live import cases as case_module

    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    emit("device", device)
    for label, settings in (("hydrostatic", {"cells": 20}),
                            ("sloshing", {"cells": 80, "periods": 1.5})):
        case_started = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = case_module.run_case(label, out_dir=tmp, device=device,
                                              progress=None, **settings)
            emit(f"case::{label}", {
                "passed": bool(result.get("passed")),
                "seconds": round(time.time() - case_started, 1),
                "settings": settings,
                "checks": [
                    {"name": c.get("name"), "measured": c.get("measured"),
                     "target": c.get("target"), "passed": c.get("passed"),
                     "note": (c.get("note") or "")[:200]}
                    for c in result.get("checks", [])
                ],
                "report": [str(line) for line in result.get("report", [])][:40],
            })
        except Exception:
            emit(f"case::{label}", {
                "passed": False,
                "seconds": round(time.time() - case_started, 1),
                "error": traceback.format_exc()[-5000:],
            })
    return out


@app.function(image=tests_image, gpu="L40S", timeout=5400, memory=32768)
def cases_l40s() -> dict:
    return _cases()


@app.function(image=tests_image, gpu="A10G", timeout=5400, memory=32768)
def cases_a10g() -> dict:
    return _cases()


# -- 5. the free surface, solved by our kernels and rendered by RTX ---------------------


def _freesurface_demo(cells: int = 144, frames: int = 44, seconds: float = 0.52,
                      width: int = 1280, height: int = 720) -> dict:
    """A dam break in a closed tank, solved by `FreeSurfaceWorld`, rendered by Isaac Sim.

    This is the join the whole file exists to find: our Warp kernels advancing a free
    surface, and Kit's RTX renderer drawing it, in one process, on one GPU, with one copy
    of Warp between them -- Kit's own `omni.warp.core`, not the pip `warp-lang`.

    ## What is drawn, and why a heightfield

    `FreeSurfaceWorld.surface_height(axis=2)` returns the depth of water over each
    column of the lattice, in metres, sub-cell accurate because a partly filled
    interface cell contributes its fill fraction, and `nan` where a column is dry. That
    is exactly a heightfield, so the water is **one `UsdGeom.Mesh` whose `points` are
    rewritten every frame** and whose topology never changes.

    The alternatives were considered and are worse here. A `UsdGeomPointInstancer` over
    the interface cells draws the surface as beads and needs tens of thousands of
    instances to stop looking like beads. A `UsdVol` OpenVDB volume is the only way to
    draw the interior, and the USD schema animates it by time-sampling a `filePath`, one
    `.vdb` per frame, so a live loop would be writing files to a container filesystem and
    reading them back. Marching cubes over `world.fill` would give a true isosurface with
    overturning waves, which a heightfield cannot represent -- but it rebuilds the
    topology every frame and nothing in this repository needs it yet.

    So: a heightfield, stated plainly, with the limitation stated too. **A heightfield
    cannot show a wave that curls over itself.** It shows one depth per column. For a dam
    break against a wall that is most of the motion and all of the readable part.

    ## The two things that make it look like water rather than a graph

    **Vertex normals, computed from the height field.** Without a `normals` primvar the
    renderer uses face normals and 9,000 quads read as 9,000 facets. The normal of a
    heightfield is `(-dh/dx, -dh/dy, 1)` normalised, which numpy gradients give for the
    whole grid in two lines, and with it the surface shades smoothly.

    **Dry columns are pushed below the floor, not flattened to it.** `surface_height` is
    `nan` on a dry column. Setting those to zero puts a sheet of water in the same plane
    as the tank floor, which z-fights and reads as a wet film over the entire bed. They
    go to a small negative height instead, inside the floor slab, where they are simply
    not seen -- which is what a dry bed looks like.
    """
    import io
    import json
    import os
    import sys
    import time
    import traceback

    out: dict = {"gpu": _gpu_name(), "cells": cells, "frames": frames}
    blobs: dict[str, bytes] = {}

    def emit(key: str, value) -> None:
        out[key] = value
        try:
            print("FSDEMO " + json.dumps({key: value}, default=str), flush=True)
        except Exception as exc:
            print(f'FSDEMO {{"{key}": "unserialisable: {exc}"}}', flush=True)

    emit("gpu", out["gpu"])
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    started = time.time()
    try:
        import isaacsim  # noqa: F401  -- registers extension paths; must precede the app
        from isaacsim.simulation_app import SimulationApp

        sim_app = SimulationApp({  # noqa: F841  -- keeps Kit alive
            "headless": True, "width": width, "height": height,
            "renderer": "RaytracedLighting",
        })
        emit("kit_startup_seconds", round(time.time() - started, 1))
    except Exception:
        emit("error", traceback.format_exc()[-4000:])
        return {"result": out, "blobs": blobs}

    sys.path.insert(0, "/repo")
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        import warp as wp
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        emit("warp_version", wp.config.version)
        emit("warp_file", wp.__file__)

        # -- the solver ----------------------------------------------------------------
        #
        # Built through `cases._free_surface_world`, the repository's own helper, rather
        # than by constructing `FreeSurfaceWorld` here. The Froude scaling, the tau, the
        # "a zero in `size` means thin and thin means periodic" rule and the gravity sign
        # are all decisions that live in `cases.py` with the argument for each, and a
        # demo that re-derived them would be a second opinion nobody asked for. This tank
        # is fully three dimensional, so no axis is thin and that rule does not fire.
        from openreynolds.live.cases import _free_surface_world

        tank = (0.36, 0.16, 0.16)
        column_height = 0.12
        fluid, scaling = _free_surface_world(
            cells=cells, size=tank, depth=0.06, faces="wall",
            collision="trt", smagorinsky=0.0, device="cuda:0", gravity=9.81)
        fluid.initialise_column((0.0, 0.0, 0.0), (0.12, tank[1], column_height))
        emit("lattice_shape", list(fluid.shape))
        emit("dx", fluid.dx)
        emit("dt", scaling.dt)
        emit("initial_water_volume_m3", float(fluid.water_volume()))

        # How many solver steps between rendered frames, from the physical time asked
        # for rather than from a guessed stride. `scaling.dt` is seconds per lattice
        # step, so this is arithmetic and not a tuning knob.
        total_steps = max(int(seconds / scaling.dt), frames)
        stride = max(total_steps // frames, 1)
        emit("steps_per_frame", stride)
        emit("total_steps", stride * frames)
        emit("physical_seconds", round(stride * frames * scaling.dt, 4))

        # -- the stage -----------------------------------------------------------------
        stage = omni.usd.get_context().get_stage()
        # Metres and Z up, stated rather than assumed. Kit's own default stage is
        # centimetres and Y up; `_demo` never had to care because
        # `World(stage_units_in_meters=1.0)` sets both for you, and this demo has no
        # `World`. A tank authored in metres on a centimetre stage is a tank 100 times
        # too small for every length that does not scale with it -- of which the camera's
        # clipping range is one. See the camera below.
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world_prim = UsdGeom.Xform.Define(stage, "/World")  # noqa: F841

        nx, ny, _ = fluid.shape
        xs = (np.arange(nx) + 0.5) * fluid.dx
        ys = (np.arange(ny) + 0.5) * fluid.dx
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

        water = UsdGeom.Mesh.Define(stage, "/World/water")
        # Topology once. Only `points` and `normals` are rewritten per frame, which is
        # the whole reason a heightfield is the cheap choice: the renderer never has to
        # rebuild an acceleration structure for changed connectivity.
        counts, indices = [], []
        for i in range(nx - 1):
            for j in range(ny - 1):
                counts.append(4)
                indices += [i * ny + j, (i + 1) * ny + j,
                            (i + 1) * ny + j + 1, i * ny + j + 1]
        water.CreateFaceVertexCountsAttr(counts)
        water.CreateFaceVertexIndicesAttr(indices)
        water.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        water.CreateDoubleSidedAttr(True)
        emit("mesh_vertices", nx * ny)
        emit("mesh_faces", len(counts))

        def water_material() -> UsdShade.Material:
            """Blue, glossy, slightly translucent. Water, not a surface plot."""
            material = UsdShade.Material.Define(stage, "/World/looks/water")
            shader = UsdShade.Shader.Define(stage, "/World/looks/water/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.06, 0.26, 0.45))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.88)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")
            return material

        UsdShade.MaterialBindingAPI(water).Bind(water_material())

        # The tank: a floor slab and three low walls, visual only -- the solver's walls
        # are its boundary conditions and owe nothing to this geometry. The slab has real
        # thickness so that the dry columns, which sit just below z = 0, are inside it.
        floor = UsdGeom.Cube.Define(stage, "/World/floor")
        floor.CreateSizeAttr(1.0)
        floor_xform = UsdGeom.Xformable(floor.GetPrim())
        floor_xform.AddTranslateOp().Set(
            Gf.Vec3d(tank[0] / 2, tank[1] / 2, -0.012))
        floor_xform.AddScaleOp().Set(Gf.Vec3f(tank[0] * 1.06, tank[1] * 1.06, 0.024))
        # The Material first and the Shader second, which is not a style point. Defining
        # `/World/looks/floor/surface` before `/World/looks/floor` creates the shader
        # under an implicit typeless parent, and typing that parent as a Material
        # afterwards left the binding inert: the bed rendered as the default white, which
        # under a dome light is a blown-out tabletop and is where most of the frame went.
        floor_material = UsdShade.Material.Define(stage, "/World/looks/floor")
        floor_shader = UsdShade.Shader.Define(stage, "/World/looks/floor/surface")
        floor_shader.CreateIdAttr("UsdPreviewSurface")
        # A mid grey, not a pale one. The bed is most of the frame once the water has
        # spread, and a near-white bed under a bright dome clips and takes the surge's
        # shadow with it.
        floor_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.22, 0.215, 0.205))
        floor_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        floor_material.CreateSurfaceOutput().ConnectToSource(
            floor_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(floor).Bind(floor_material)

        sun = UsdLux.DistantLight.Define(stage, "/World/sun")
        # 2600, down from 4200. The bed is a horizontal face and the key light hits it
        # near normal, so at 4200 a 0.40 grey diffuse still saturated to white: the
        # material was bound and correct, and the exposure was wrong. The symptom reads
        # as an unbound material, which is the wrong thing to go and fix.
        sun.CreateIntensityAttr(2600.0)
        sun.CreateAngleAttr(0.8)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(-50.0, 20.0, 20.0))
        sky = UsdLux.DomeLight.Define(stage, "/World/sky")
        # 240, not 420. At 420 the dome alone takes the pale floor to very near white,
        # the key light has nothing left to add and the water loses the specular that is
        # most of what makes it read as water rather than as blue plastic.
        sky.CreateIntensityAttr(300.0)

        # -- tick Kit before asking it to draw ------------------------------------
        #
        # The first version of this demo authored the whole scene, stepped the solver,
        # and rendered 44 frames in 137.9 s -- every one of them a single flat colour,
        # `render_stddev` 0.0 and exactly one distinct colour in the frame. The solver
        # was fine in that run: 168.5 MLUPS, mass drift -1.7e-2, zero fluid cells
        # touching gas, and a trace showing the column collapsing and the bed wetting
        # from 0.34 to 1.00.
        #
        # What was missing is that **nothing ticked the application**. `_demo` never hit
        # this because it drives `isaacsim.core.api.World`, whose `step(render=True)` and
        # `render()` call `SimulationApp.update()` underneath; this demo has no PhysX
        # scene and no `World`, so the only thing moving was Replicator's graph. USD
        # authoring does not reach Hydra by itself -- the stage's change notices are
        # processed on an app update -- so the renderer was faithfully drawing the empty
        # scene it had been given.
        for _ in range(8):
            sim_app.update()

        # `clipping_range` is the whole reason the second attempt rendered a frame of
        # one single colour, mean RGB (191.6, 191.6, 191.6) and `render_stddev` exactly
        # 0.0 -- not black, so the lights worked and the renderer drew its background,
        # with nothing at all in front of it.
        #
        # The default near plane is **1.0 stage unit**. This tank is 0.36 m across and
        # the camera sits about 0.59 m from what it is aimed at, so in metres the entire
        # scene lies nearer than the near plane and every triangle is clipped before it
        # is shaded. A benchtop-scale scene in SI units is exactly the case the default
        # is wrong for, and it fails silently: a clipped scene and an empty scene produce
        # the same picture.
        # Raised and pulled back from the first framing, which put the camera 0.59 m out
        # at a shallow angle: the un-collapsed column filled the left of the frame, the
        # dry bed ran off to the right with nothing on it, and half the picture was sky.
        # From here the whole tank is in shot and the surge crosses it.
        camera = rep.create.camera(
            position=(0.62, -0.46, 0.30), look_at=(0.17, 0.08, 0.045),
            focal_length=32.0, clipping_range=(0.01, 100.0))
        product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])

        points_attr = water.GetPointsAttr()
        normals_attr = water.CreateNormalsAttr()
        water.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        def push_surface() -> float:
            """Read the solver's surface onto the mesh. Returns the wet-column fraction."""
            depth = fluid.surface_height(axis=2)
            wet = np.isfinite(depth)
            # Dry columns go below the floor slab rather than to zero. See the docstring.
            z = np.where(wet, np.nan_to_num(depth), -0.006).astype(np.float64)
            pts = np.stack([grid_x, grid_y, z], axis=-1).reshape(-1, 3)
            points_attr.Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            # Vertex normals from the height gradient: n = (-dh/dx, -dh/dy, 1).
            dzdx, dzdy = np.gradient(z, fluid.dx, fluid.dx)
            normals = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            normals_attr.Set(
                Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3).astype(np.float32)))
            return float(wet.mean())

        def grab() -> "np.ndarray":
            """One rendered frame, through Replicator's own sync. See `_demo`."""
            for _ in range(6):
                sim_app.update()
                rep.orchestrator.step(rt_subframes=6)
                try:
                    rep.orchestrator.wait_until_complete()
                except Exception:
                    pass
                arr = np.asarray(annotator.get_data())
                if arr.size and float(arr[..., :3].astype("float32").std()) > 1.0:
                    return arr
            return np.asarray(annotator.get_data())

        # -- the loop ------------------------------------------------------------------
        captured: list = []
        trace: list = []
        solver_seconds = 0.0
        render_seconds = 0.0
        push_surface()
        for frame in range(frames):
            mark = time.time()
            for _ in range(stride):
                fluid.step()
            wp.synchronize()
            solver_seconds += time.time() - mark
            wet = push_surface()
            # Two updates between authoring the new points and rendering them, so the
            # frame that comes back is this step's surface and not the previous one's.
            sim_app.update()
            sim_app.update()
            mark = time.time()
            arr = grab()
            render_seconds += time.time() - mark
            if not arr.size:
                emit("render_error", f"empty annotator at frame {frame}")
                break
            captured.append(arr[..., :3].astype("uint8").copy())
            depth = fluid.surface_height(axis=2)
            trace.append({
                "frame": frame,
                "t": round((frame + 1) * stride * scaling.dt, 4),
                "wet_fraction": round(wet, 4),
                "max_depth_m": round(float(np.nanmax(depth)), 5),
                "volume_m3": round(float(fluid.water_volume()), 9),
            })
        emit("trace", trace)
        emit("solver_seconds", round(solver_seconds, 1))
        emit("render_seconds", round(render_seconds, 1))
        if solver_seconds > 0:
            emit("solver_mlups", round(
                stride * frames * float(np.prod(fluid.shape)) / solver_seconds / 1e6, 1))
        try:
            emit("mass_drift", {k: float(v) for k, v in fluid.mass_drift().items()})
        except Exception as exc:
            emit("mass_drift", f"FAILED: {exc}")
        emit("fluid_touching_gas", int(fluid.fluid_touching_gas()))
        emit("final_water_volume_m3", float(fluid.water_volume()))

        # -- the artifacts -------------------------------------------------------------
        if not captured:
            emit("render_error", "no frames captured")
            return {"result": out, "blobs": blobs}

        from PIL import Image

        # -- which frame is the hero -----------------------------------------------
        #
        # The first rule here was "the frame with the greatest maximum depth", and it
        # chose frame 0 every time -- correctly, because the deepest water in a dam break
        # is the column before it is released. A picture of the initial condition is not
        # a picture of the solver working.
        #
        # What makes the picture is the surge front still crossing a dry bed, so the rule
        # is the last frame before the bed is essentially wet. `wet_fraction` is already
        # in the trace, it runs 0.34 -> 1.00 over the collapse, and the moment it reaches
        # 1 is the moment the interesting part is over.
        hero_index = 0
        for row in trace:
            if row["wet_fraction"] < 0.95:
                hero_index = row["frame"]
        hero_index = min(hero_index + 1, len(captured) - 1)
        hero = captured[hero_index]
        emit("hero_frame", hero_index)
        emit("render_shape", list(hero.shape))
        rgb = hero.astype("float32")
        emit("render_stddev", round(float(rgb.std()), 2))
        # The mean channel values, because a flat frame that is black means no light
        # reached the camera and a flat frame that is grey means the renderer drew its
        # background and there was no geometry in front of it. Those are different bugs
        # and `stddev == 0` alone does not separate them.
        emit("render_mean_rgb", [round(float(v), 1) for v in rgb.reshape(-1, 3).mean(0)])
        emit("render_unique_colours",
             int(len(np.unique(rgb.reshape(-1, 3)[::97], axis=0))))
        emit("render_nonzero_fraction", float((rgb > 8).mean()))

        buf = io.BytesIO()
        Image.fromarray(hero, "RGB").save(buf, format="PNG", optimize=True)
        blobs["freesurface_hero.png"] = buf.getvalue()
        emit("hero_png_bytes", len(blobs["freesurface_hero.png"]))

        # The GIF at half resolution. 44 frames of 1280x720 is about 120 MB of PNG and
        # Modal has to pickle the return value; 640x360 keeps the whole payload inside
        # the 16 MB an artifact is allowed and is still legible at the size anyone will
        # view it.
        small = [Image.fromarray(f, "RGB").resize((width // 2, height // 2),
                                                  Image.LANCZOS).quantize(
                     colors=128, method=Image.MEDIANCUT) for f in captured]
        buf = io.BytesIO()
        small[0].save(buf, format="GIF", save_all=True, append_images=small[1:],
                      duration=70, loop=0, optimize=True)
        blobs["freesurface.gif"] = buf.getvalue()
        emit("gif_bytes", len(blobs["freesurface.gif"]))
        emit("gif_frames", len(small))

        # Four stills across the run, so the sequence survives even if the GIF does not
        # travel well.
        for pick in sorted({0, len(captured) // 4, len(captured) // 2,
                            len(captured) - 1}):
            buf = io.BytesIO()
            Image.fromarray(captured[pick], "RGB").save(buf, format="PNG",
                                                        optimize=True)
            blobs[f"freesurface_{pick:03d}.png"] = buf.getvalue()
        emit("ok", True)
    except Exception:
        emit("error", traceback.format_exc()[-6000:])
        emit("ok", False)
    finally:
        # No `sim_app.close()`. See the end of `_probe`.
        out["total_seconds"] = round(time.time() - started, 1)
    return {"result": out, "blobs": blobs}


@app.function(image=image, gpu="L40S", timeout=5400, memory=32768)
def freesurface_l40s(cells: int = 144, frames: int = 44) -> dict:
    return _freesurface_demo(cells=cells, frames=frames)


@app.function(image=image, gpu="A10G", timeout=5400, memory=32768)
def freesurface_a10g(cells: int = 144, frames: int = 44) -> dict:
    return _freesurface_demo(cells=cells, frames=frames)


# -- 6. a submarine: PhysX-free, Newton-coupled, RTX-rendered ---------------------------


def _submarine(cells_per_d: int = 0, reynolds: float = 100.0, frames: int = 44,
               width: int = 1280, height: int = 720) -> dict:
    """A self-propelled capsule hull in water, two-way coupled to Newton, rendered by RTX.

    This answers the question at the top of the unknown list: **can a rigid body be
    two-way coupled to our solver inside a live Kit process.** Not PhysX -- Newton, which
    is what `openreynolds/live/couple.py` couples to and what Isaac Sim 6.2 now ships as
    a physics backend of its own.

    ## Why a submarine and not a boat

    `DECISIONS_LIVE.md` D9. The momentum-exchange wrench on a body sitting in a fluid
    that carries a body force is not yet derived, and the symptom is severe: a fully
    submerged box where the answer is exactly `+rho V g` reads -0.48 times it. Anything
    floating at a waterline goes through that path.

    A fully submerged body does not. Gravity goes to the **body**, through Newton, and
    never into the fluid, so the fluid carries no body force at all; the displaced weight
    arrives separately and analytically as `NewtonCoupling(buoyancy=True)` summing the
    voxelised volume. That is the path M3's `underwater` case validates, where a capsule
    of zero length is a sphere and reproduces Schiller-Naumann to +1.1%.

    So this demo is deliberately on `FluidWorld` and not `FreeSurfaceWorld`, and the hull
    stays well below any surface because there is no surface.

    ## What is measured, so that this is not only a picture

    The hull is **neutrally buoyant** -- its density is water's -- so weight and buoyancy
    cancel exactly and the only forces left are the thruster and the fluid. A constant
    thrust `T` is applied along the body's own axis. At terminal speed the body stops
    accelerating, which means the fluid is pushing back with exactly `T`, so

        Cd = 2 T / (rho V^2 A)

    with `V` the measured terminal speed and `A = pi r^2` the frontal area, the same
    definition `cases.run_underwater` grades against Schiller-Naumann.

    **And it is cross-checked against the solver's own wrench.** `world.wrench(body)`
    returns the surface force the momentum exchange actually summed, and at steady state
    its axial component must equal `-T`. Those two numbers are computed by completely
    different routes -- one from the body's kinematics, one from the populations crossing
    the boundary links -- so their ratio is a direct test that the coupling is wired up
    in both directions. A one-way coupling would still fly the submarine and would fail
    this.

    ## The manoeuvre

    Thrust is applied along the **body** axis, rotated by the body's own orientation each
    step, so a pitched hull thrusts downward as well as forward. Part way through, a
    short pitch moment is applied through `control.joint_f`, the nose goes down, and the
    submarine dives under its own thrust. That is the part a policy would care about and
    it is the part that shows the coupling is live: the wake tilts with the hull.
    """
    import io
    import json
    import math
    import os
    import sys
    import time
    import traceback

    out: dict = {"gpu": _gpu_name(), "cells_per_d": cells_per_d,
                 "reynolds_target": reynolds, "frames": frames}
    blobs: dict[str, bytes] = {}

    def emit(key: str, value) -> None:
        out[key] = value
        try:
            print("SUB " + json.dumps({key: value}, default=str), flush=True)
        except Exception as exc:
            print(f'SUB {{"{key}": "unserialisable: {exc}"}}', flush=True)

    emit("gpu", out["gpu"])
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    started = time.time()
    try:
        import isaacsim  # noqa: F401  -- registers extension paths; must precede the app
        from isaacsim.simulation_app import SimulationApp

        sim_app = SimulationApp({  # noqa: F841  -- keeps Kit alive
            "headless": True, "width": width, "height": height,
            "renderer": "RaytracedLighting",
        })
        emit("kit_startup_seconds", round(time.time() - started, 1))
    except Exception:
        emit("error", traceback.format_exc()[-4000:])
        return {"result": out, "blobs": blobs}

    sys.path.insert(0, "/repo")
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        import warp as wp
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        import newton
        from openreynolds.live.couple import NewtonCoupling
        from openreynolds.live.units import Scaling
        from openreynolds.live.world import Domain, FluidWorld

        emit("warp_version", wp.config.version)
        emit("newton_version", newton.__version__)

        # -- the hull and the water ----------------------------------------------------
        #
        # Every dimension and the scaling come from `cases.run_underwater`, which is the
        # validated setup: a 10 mm capsule in water at `u_lattice = 0.04`, the Reynolds
        # scaling built by `Scaling.for_reynolds` on the diameter. The fineness ratio of
        # 4 is that case's elongated hull, the one it reports rather than grades.
        diameter = 0.01
        radius = diameter / 2.0
        length_ratio = 3.0
        half_height = length_ratio * radius
        # The resolution is chosen by the repository, not here. Asking for 16 cells per
        # diameter at Re 100 is **refused**, and rightly:
        #
        #   UnphysicalScaling: tau_plus is 0.51920, within 0.02 of the 1/2 at which the
        #   collision stops relaxing anything
        #
        # `cases._cells_for` exists because the first GPU run of the sphere case hit the
        # same wall at the same number, and its docstring says so. It returns 20 here.
        # A fixed resolution across a range of Reynolds numbers is the thing that has no
        # business being hard-coded, so `--cells-per-d` defaults to 0 meaning "ask".
        from openreynolds.live.cases import _cells_for

        cells_per_d = int(cells_per_d) if cells_per_d else _cells_for(reynolds, 0.04)
        emit("cells_per_d", cells_per_d)
        dx = diameter / cells_per_d
        scaling = Scaling.for_reynolds(dx, diameter, reynolds, "water",
                                       u_lattice=0.04, scheme="trt")
        speed_target = scaling.reference_velocity
        water = scaling.fluid

        # A box the submarine would leave in a couple of seconds, with the moving window
        # on so that it does not. `follow=0` re-centres the lattice on body 0 whenever it
        # has drifted past `follow_slack` of the box, which is exactly the case the
        # window is correct for: a body that swims in a straight line is not driven by
        # its own wake. `run_falling_sphere` records the case where that is false.
        # -- no moving window, and that is a measurement too --------------------------
        #
        # `follow=0` is the validated way to keep a travelling body in frame and it is
        # what `cases.run_underwater` and `run_falling_sphere` use. It did not survive
        # here, and the evidence is unambiguous. With `follow=0, follow_slack=0.15` and
        # the hull started 35 mm off centre, the window shifted once on the first step
        # and the voxelisation then **collapsed**: `solid_cells` went from 23,184 to a
        # range of 0 to 2,346, `world.wrench(0)` returned exactly 0.0 on every frame,
        # and the submarine accelerated in a perfect straight line at
        #
        #     measured dv per frame  3.756e-04 m/s
        #     T / m                  3.7545e-04 m/s
        #
        # which is a body in vacuum to four significant figures. The fluid was still
        # being solved; it simply had nothing in it.
        #
        # This is reported rather than worked around in `world.py`, which this file is
        # not allowed to touch. The workaround here is to not need the window: a domain
        # twice as long in x, the hull started a quarter of the way along, and nothing
        # shifting. That costs 15.4 M nodes instead of 7.7 M and buys a run where the
        # body stays voxelised from the first step to the last.
        size = (24.0 * diameter, 8.0 * diameter, 10.0 * diameter)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                        faces={"-x": "outlet", "+x": "outlet", "-y": "slip",
                               "+y": "slip", "-z": "slip", "+z": "slip"})
        world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                           substeps=1, collision="trt", smagorinsky=0.0,
                           device="cuda:0", follow=None)
        emit("lattice_shape", list(world.shape))
        emit("nodes", int(world.nodes))
        emit("dx", world.dx)
        emit("dt", float(world.dt))

        # -- the Newton body -----------------------------------------------------------
        #
        # Neutrally buoyant on purpose: with the shape's density set to the water's,
        # `NewtonCoupling(buoyancy=True)` adds `rho V g` upward and Newton's gravity pulls
        # `rho_s V g` down, and they cancel to the precision of the voxelised volume. What
        # is left is thrust and fluid, which is what the measurement below needs.
        start = (3.0 * diameter, size[1] / 2.0, size[2] / 2.0)
        # A capsule's axis is its local z, which `cases.run_underwater` states and
        # measured; a quarter turn about y takes z to x so the hull lies along the
        # direction it swims.
        turn = wp.quat(0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))

        def build(density: float, gravity: float = -9.81):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            # `add_body` already makes the free joint; do not add one.
            hull_body = builder.add_body(
                xform=wp.transform(wp.vec3(*start), wp.quat_identity()), label="hull")
            cfg = builder.default_shape_cfg.copy()
            cfg.density = density
            cfg.has_shape_collision = False
            builder.add_shape_capsule(
                hull_body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), turn),
                radius=radius, half_height=half_height, cfg=cfg)
            built = builder.finalize()
            # Gravity BEFORE the solver is constructed, not after. `SolverMuJoCo`
            # converts the Newton model into a MuJoCo model in its constructor, so a
            # `set_gravity` that lands afterwards updates one of them and not the other.
            # Doing it in the wrong order sent the hull to 59 m/s within 98 steps and to
            # 1e8 m by the next frame -- a divergence that looks like a coupling blow-up
            # and is a model the solver never saw.
            built.set_gravity((0.0, 0.0, gravity))
            engine = newton.solvers.SolverMuJoCo(
                built, disable_contacts=True, solver="newton",
                integrator="implicitfast", iterations=10)
            return built, engine

        # -- gravity and buoyancy are both OFF, and that is a measurement ------------
        #
        # The intent was a neutrally buoyant hull: give it the water's density so that
        # `m = rho_w * V`, keep `NewtonCoupling(buoyancy=True)` so D9's analytic
        # displaced-weight path is exercised, and let thrust and fluid do the rest. Two
        # runs measured why that cannot work at this scale.
        #
        # **Run one, density = rho_water.** The hull *rose* at 15 mm/s and climbed 130 mm
        # while travelling 6.5 mm forward. Newton takes the mass from the analytic
        # capsule, 2.87979e-6 m3; the coupling takes the displaced weight from the
        # voxelised hull, 23,184 cells of (0.5 mm)^3 = 2.898e-6 m3. They differ by
        # **0.632%**, and
        #
        #     weight                                2.8198e-05 N  =  5,104 x the thrust
        #     net buoyancy from that 0.632%         1.7801e-04 N  =     32 x the thrust
        #     volume error the thrust could absorb      0.0196%
        #
        # **Run two, density calibrated to the voxel count.** Setting the density so that
        # `m = rho_w * V_voxel` at the starting pose drove the residual from 1.78e-4 N to
        # **9.7e-10 N**, and it still did not hold, because the voxelisation is a
        # function of where the body sits relative to the grid. Measured over the run:
        #
        #     solid cells          22,992 to 23,192, a swing of 200
        #     buoyancy swing       2.4481e-04 N  =  **44 x the thrust**
        #
        # and the hull sank out of frame on the mean deficit.
        #
        # So neutral buoyancy here is the cancellation of two forces five thousand times
        # larger than the one the demo is about, to a precision the voxelisation is two
        # orders of magnitude short of. **A body that is neutrally buoyant by definition
        # has no net weight, and the exact way to say that is for neither force to
        # exist.** Gravity goes to zero and `buoyancy=False`, which is not an evasion: it
        # is numerically identical to the ideal being modelled and it removes a 44x noise
        # source from a measurement of a 5.5e-6 N force.
        #
        # **What this costs, said plainly: D9's buoyancy path is NOT exercised by this
        # demo.** `cases.run_underwater` and `run_falling_sphere` still cover it. What is
        # exercised here is the momentum-exchange wrench and the pose feedback, which is
        # what "two-way coupled" names and what the balance check below tests.
        #
        # The fix that would let a submarine float properly is in `couple.py` and is
        # written up in the notes: let the caller pass the displaced volume instead of
        # having it recounted from the voxels every step.
        model, engine = build(water.rho, gravity=0.0)
        world.initialise()
        link = NewtonCoupling(model, engine, world, dt=float(world.dt), buoyancy=False)
        emit("gravity", [0.0, 0.0, 0.0])
        emit("buoyancy_enabled", False)
        analytic = math.pi * radius * radius * 2.0 * half_height             + 4.0 / 3.0 * math.pi * radius ** 3
        emit("analytic_volume_m3", analytic)
        emit("hull_mass_kg", float(model.body_mass.numpy()[0]))
        # NOT `world.solid_cells(0)` here (before the probe above there was nothing to
        # read). The lattice has no idea where the body is
        # until a step has pushed the poses and voxelised, so read before the first
        # `link.step()` it is 0 -- which is precisely the reading that
        # `world.solid_cells`' own docstring warns means "a body too small for its
        # lattice", and a 10 mm capsule on a 0.5 mm lattice is nothing of the kind. The
        # real count is emitted after the first step below, where it is a diagnostic
        # rather than an artifact of when it was taken.
        emit("solid_cells_before_any_step", int(world.solid_cells(0)))

        volume = math.pi * radius * radius * 2.0 * half_height \
            + 4.0 / 3.0 * math.pi * radius ** 3
        area = math.pi * radius * radius
        emit("hull_volume_m3", volume)
        emit("frontal_area_m2", area)

        # The thrust that should hold the target Reynolds number, from a guessed Cd. It
        # is a guess and it does not have to be right: whatever speed it settles at is
        # measured, and the Cd reported below is computed from that speed and this
        # thrust, not from the guess.
        # 3.5, not 1.4. Run seven measured Cd near 5.8 at Re 43 for this fineness-4
        # hull, which is several times a sphere's and is the right way round at this
        # Reynolds number -- `cases.run_underwater` says the same of its elongated hull.
        # A thrust sized for Cd 1.4 therefore settled at Re 49 and was still climbing
        # after twelve seconds. This aims the terminal speed at Re 100.
        cd_guess = 3.5
        thrust = 0.5 * water.rho * speed_target ** 2 * area * cd_guess
        emit("thrust_N", thrust)
        emit("target_speed_mps", speed_target)

        # -- the stage -----------------------------------------------------------------
        stage = omni.usd.get_context().get_stage()
        # Metres and Z up, for the reason the free-surface demo records: Kit's own stage
        # is centimetres and Y up, and a 0.1 m scene on a centimetre stage is inside the
        # camera's default near plane.
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")

        def preview_material(path: str, colour, roughness: float, metallic: float = 0.0,
                             opacity: float = 1.0, emissive=None) -> UsdShade.Material:
            """Material first, Shader second. The free-surface demo records why."""
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*colour))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            if emissive is not None:
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*emissive))
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")
            return material

        # The hull, drawn as a USD capsule rather than as the voxelisation. `solid_cells`
        # is the right thing to look at when asking whether the lattice can see the body
        # -- and it is reported above -- but a 16-cells-per-diameter voxel blob is a
        # picture of the discretisation, not of the submarine. The analytic capsule is
        # the same shape Newton was given, placed at the pose Newton computed.
        hull = UsdGeom.Capsule.Define(stage, "/World/hull")
        hull.CreateRadiusAttr(radius)
        hull.CreateHeightAttr(2.0 * half_height)
        hull.CreateAxisAttr("X")
        hull_xform = UsdGeom.Xformable(hull.GetPrim())
        hull_op = hull_xform.AddTransformOp()
        UsdShade.MaterialBindingAPI(hull).Bind(
            # metallic 0.0, not 0.65. A metallic surface has no diffuse term and shows
            # the environment instead, so a 65% metallic orange hull under a deep blue
            # dome rendered **dark olive green** -- a correct render of a mirror in blue
            # water, and not the submarine anyone was looking for.
            preview_material("/World/looks/hull", (0.95, 0.52, 0.11), 0.38, 0.0))

        # -- the wake ------------------------------------------------------------------
        #
        # A `UsdGeomPointInstancer` over the lattice cells the body has actually stirred.
        # Section 7 of the notes ranked this as the practical option for a field that has
        # to update every frame: positions are an in-memory attribute write, there is no
        # disk round trip the way a `UsdVol` OpenVDB sequence needs, and it scales to the
        # tens of thousands of points a decimated 3.9 M-node lattice produces.
        #
        # It is a decimated field and says so: every second node in each direction, and
        # only where the speed is above a fraction of the body's own. That is not an
        # isosurface of anything, it is a cloud whose density is where the momentum went,
        # which is what a wake looks like and is honest about being.
        wake = UsdGeom.PointInstancer.Define(stage, "/World/wake")
        droplet = UsdGeom.Sphere.Define(stage, "/World/wake/droplet")
        droplet.CreateRadiusAttr(world.dx * 1.0)
        UsdShade.MaterialBindingAPI(droplet).Bind(
            preview_material("/World/looks/wake", (0.20, 0.78, 0.92), 0.22,
                             opacity=0.30, emissive=(0.02, 0.14, 0.20)))
        wake.CreatePrototypesRel().SetTargets([droplet.GetPath()])
        wake_positions = wake.CreatePositionsAttr()
        wake_protos = wake.CreateProtoIndicesAttr()
        wake_scales = wake.CreateScalesAttr()
        # Per-instance colour, which the notes listed as unverified for this build. It is
        # tried here and the answer is in the picture: `primvars:displayColor` at `vertex`
        # interpolation on a PointInstancer is the documented way to colour instances
        # individually, and if it is ignored the wake comes out the flat material colour
        # above rather than graded by speed.
        wake_colour = UsdGeom.PrimvarsAPI(wake).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)

        # Deep water: a tinted dome is the whole background, because underwater there is
        # no horizon and no floor in shot. A key light from above stands in for the
        # surface and gives the hull a lit top and a shadowed belly.
        sun = UsdLux.DistantLight.Define(stage, "/World/sun")
        sun.CreateIntensityAttr(2400.0)
        sun.CreateAngleAttr(1.5)
        sun.CreateColorAttr(Gf.Vec3f(0.85, 0.95, 1.0))
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-62.0, 12.0, 20.0))
        sky = UsdLux.DomeLight.Define(stage, "/World/sky")
        sky.CreateIntensityAttr(900.0)
        sky.CreateColorAttr(Gf.Vec3f(0.05, 0.22, 0.34))

        # -- the camera, and the frame everything is drawn in --------------------------
        #
        # The second attempt built a `UsdGeom.Camera` by hand and moved it each frame to
        # track the hull, aiming it with `Gf.Matrix4d().SetLookAt(...).GetInverse()`. The
        # render came back a single flat colour, mean RGB (48.4, 138.0, 169.5) -- the
        # dome, and nothing in front of it.
        #
        # Rather than debug a hand-built camera against a `rep.create.render_product`
        # that takes a path string, this uses the call that demonstrably works in the
        # other two demos and **moves the scene instead of the camera**. Everything
        # drawn is offset by the hull's own x displacement, so the submarine stays put
        # horizontally while the wake streams past it. That is a change of reference
        # frame and nothing more -- and deliberately only in x, so the dive still reads
        # as a dive rather than as the world sliding upward.
        look_z = start[2]
        # Back and up from run seven, which sat 75 mm from a hull 40 mm long with the
        # wake drawn across a 240 mm box: the cloud filled the frame. The near-wake
        # window above is about 220 mm long, so the camera stands off far enough to hold
        # it and looks slightly down the length of it.
        camera = rep.create.camera(
            position=(start[0] + 0.055, start[1] - 0.165, look_z + 0.062),
            look_at=(start[0] - 0.052, start[1], look_z),
            focal_length=34.0, clipping_range=(0.005, 100.0))
        product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        for _ in range(8):
            sim_app.update()

        # -- the loop ------------------------------------------------------------------
        # Long enough to actually reach terminal speed, which run six did not: it was
        # still accelerating linearly when the dive began, so its "cruise speed" was a
        # number from the middle of a ramp and its Cd of 3.9 was meaningless. With drag
        # working, `v_term = sqrt(2T / (rho A Cd))` is 11 to 13 mm/s for Cd between 0.8
        # and 1.2, and the time constant `m v / T` is about 6 s, so 9 s of settling is
        # about one and a half of them.
        settle = int(7.0 / float(world.dt))
        dive_start = int(settle * 0.72)
        dive_stop = dive_start + int(0.35 / float(world.dt))
        total = settle + int(3.5 / float(world.dt))
        stride = max(total // frames, 1)
        total = stride * frames
        emit("total_steps", total)
        emit("steps_per_frame", stride)
        emit("dive_window_steps", [dive_start, dive_stop])
        emit("simulated_seconds", round(total * float(world.dt), 3))

        joint_f = np.zeros(6, dtype=np.float32)
        axial: list = []
        captured: list = []
        trace: list = []
        solver_seconds = 0.0
        render_seconds = 0.0

        def body_pose():
            """Position and orientation of the hull, as numpy, from Newton's own state.

            `link.state` is the public accessor and is the state `step()` will read
            next, which is what matters after the coupling swaps `state_0` and
            `state_1` at the end of every step.
            """
            q = link.state.body_q.numpy()[0]
            return np.asarray(q[:3], dtype=float), np.asarray(q[3:7], dtype=float)

        def rotate(quat, vector):
            """Rotate `vector` by the (x, y, z, w) quaternion. Warp and Newton's order."""
            x, y, z, w = [float(v) for v in quat]
            u = np.array([x, y, z])
            return (vector * (w * w - u.dot(u)) + 2.0 * u * u.dot(vector)
                    + 2.0 * w * np.cross(u, vector))

        def push_wake(body_position, track: float, track_z: float) -> int:
            """Decimate the velocity field onto the point instancer. Returns the count."""
            got = world.fields()
            u = got["u"][::3, ::3, ::3, :]
            speed = np.linalg.norm(u, axis=-1)
            # The threshold is a fraction of the body's own speed, so the cloud is "where
            # the hull has changed the water" at whatever speed it happens to be doing,
            # rather than a fixed number that is empty early and solid late.
            # 0.15 of the peak, not 0.33. At a third of the maximum the only cells left
            # are the ones against the hull, so the cloud came out as a shell wrapped
            # tight round the body and there was no trailing wake at all -- the thing
            # the picture is supposed to show. The near-wake box below is what keeps the
            # count manageable at a threshold this low; run seven had the low threshold
            # and no box, and drew the entire domain.
            reference = max(float(np.nanmax(speed)) * 0.15, 1e-9)
            xs, ys, zs = (world.node_coordinates(a) for a in range(3))
            gx, gy, gz = np.meshgrid(xs[::3], ys[::3], zs[::3], indexing="ij")
            # Only the near wake. Run seven drew every stirred cell in a 240 mm box and
            # the cloud filled the frame edge to edge with the hull buried inside it:
            # a correct picture of the whole field and a useless picture of a
            # submarine. This window is the eighteen diameters behind the nose and two
            # and a half either side, which is the part that reads as a wake.
            near = ((gx > body_position[0] - 18.0 * diameter)
                    & (gx < body_position[0] + 4.0 * diameter)
                    & (np.abs(gy - body_position[1]) < 2.5 * diameter)
                    & (np.abs(gz - body_position[2]) < 2.5 * diameter))
            keep = np.isfinite(speed) & (speed > reference) & near
            if not keep.any():
                wake_positions.Set(Vt.Vec3fArray.FromNumpy(
                    np.zeros((0, 3), dtype=np.float32)))
                wake_protos.Set(Vt.IntArray.FromNumpy(np.zeros(0, dtype=np.int32)))
                return 0
            points = np.stack([gx[keep], gy[keep], gz[keep]], axis=-1)
            magnitude = speed[keep]
            # Drop anything inside the hull's own radius: those are the cells the
            # voxelisation owns and drawing them puts the wake inside the submarine.
            # In WORLD coordinates, before the tracking shift below -- comparing shifted
            # points against an unshifted body position, which is what this did first
            # time round, masks a sphere of cells somewhere off to one side instead.
            offset = points - np.asarray(body_position)
            outside = np.linalg.norm(offset, axis=-1) > (radius * 1.25)
            points, magnitude = points[outside], magnitude[outside]
            points[:, 0] -= track          # the tracking frame; see the camera above
            points[:, 2] -= track_z
            if len(points) > 90000:
                pick = np.argsort(-magnitude)[:90000]
                points, magnitude = points[pick], magnitude[pick]
            wake_positions.Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            wake_protos.Set(Vt.IntArray.FromNumpy(
                np.zeros(len(points), dtype=np.int32)))
            # Bigger and brighter where the water is moving fastest, which is what makes
            # a point cloud read as a wake rather than as noise.
            level = np.clip(magnitude / max(float(magnitude.max()), 1e-12), 0.0, 1.0)
            scale = (0.55 + 1.15 * level).astype(np.float32)
            wake_scales.Set(Vt.Vec3fArray.FromNumpy(
                np.repeat(scale[:, None], 3, axis=1)))
            colours = np.stack([0.06 + 0.52 * level,
                                0.45 + 0.42 * level,
                                0.62 + 0.33 * level], axis=-1)
            wake_colour.Set(Vt.Vec3fArray.FromNumpy(colours.astype(np.float32)))
            return int(len(points))

        def grab():
            for _ in range(6):
                sim_app.update()
                rep.orchestrator.step(rt_subframes=6)
                try:
                    rep.orchestrator.wait_until_complete()
                except Exception:
                    pass
                arr = np.asarray(annotator.get_data())
                if arr.size:
                    sample = arr[..., :3].reshape(-1, 3)[::401]
                    # Distinct colours, not standard deviation: a flat tinted frame has
                    # a large whole-array deviation and exactly one colour.
                    if len(np.unique(sample, axis=0)) > 8:
                        return arr
            return np.asarray(annotator.get_data())

        for step in range(total):
            position, quat = body_pose()
            axis = rotate(quat, np.array([1.0, 0.0, 0.0]))
            joint_f[:] = 0.0
            joint_f[:3] = (axis * thrust).astype(np.float32)
            if dive_start <= step < dive_stop:
                # A short pitch-down pulse about the world y axis. The hull has no fins,
                # so nothing restores it: it keeps the attitude the pulse left it at and
                # the thrust, which follows the body axis, now has a downward component.
                # That is the dive, and it is driven entirely through Newton.
                # Positive about y pitches the nose DOWN: run six used -9.0e-7 and the
                # nose went UP through 90 degrees and the hull tumbled, which is both
                # the wrong sign and about six times too much. This is sized for roughly
                # twenty degrees.
                joint_f[4] = 4.0e-7
            link.control.joint_f.assign(joint_f)
            mark = time.time()
            link.step()
            solver_seconds += time.time() - mark
            if step % 4 == 0:
                # Every fourth step, not once a frame. The instantaneous wrench on a
                # body this small fluctuates hard -- run seven's per-frame samples of
                # |F_fluid| / (T - m a) ranged from 0.14 to 1.51 about a mean of 0.885
                # -- and one sample every 136 steps is not a measurement of a mean.
                # `cases.run_underwater` averages hundreds of samples for the same
                # reason.
                axial.append(float(world.wrench(0)[0][0]))
            if step == 0:
                emit("solid_cells", int(world.solid_cells(0)))
                emit("voxel_volume_m3", world.solid_cells(0) * world.dx ** 3)

            if (step + 1) % stride == 0:
                wp.synchronize()
                position, quat = body_pose()
                # `body_qd` is **linear first**, which Newton's own docstring states
                # and `isaac.py` records as checked by dropping a body and reading which
                # three components moved. This read was `[3:6]` first time round, which
                # would have reported the angular velocity as the swimming speed and put
                # a plausible-looking wrong number in every row of the table below.
                velocity = np.asarray(link.state.body_qd.numpy()[0][:3], dtype=float)
                spin = np.asarray(link.state.body_qd.numpy()[0][3:6], dtype=float)
                force, torque = world.wrench(0)
                track = float(position[0] - start[0])
                # Vertically the camera follows at 85%, so a 130 mm descent shows as
                # 20 mm of drift in a frame about 46 mm tall: visible as a dive, and
                # still in shot. Run four descended 128 mm and left the frame entirely,
                # which is why frames 26 and 43 of it are the dome and nothing else --
                # the renderer was fine and the submarine had gone.
                track_z = float(position[2] - look_z) * 0.85
                count = push_wake(position, track, track_z)
                nose = rotate(quat, np.array([1.0, 0.0, 0.0]))
                # The hull prim, at the pose Newton computed. `UsdGeom.Capsule` with
                # `axis="X"` is already along body x, so only the body rotation applies.
                rot = Gf.Rotation(Gf.Quatd(float(quat[3]), Gf.Vec3d(
                    float(quat[0]), float(quat[1]), float(quat[2]))))
                xform = Gf.Matrix4d().SetRotate(rot)
                xform.SetTranslateOnly(Gf.Vec3d(float(position[0] - track),
                                                float(position[1]),
                                                float(position[2] - track_z)))
                hull_op.Set(xform)
                sim_app.update()
                sim_app.update()
                mark = time.time()
                arr = grab()
                render_seconds += time.time() - mark
                if not arr.size:
                    emit("render_error", f"empty annotator at step {step}")
                    break
                captured.append(arr[..., :3].astype("uint8").copy())
                trace.append({
                    "step": step + 1,
                    "t": round((step + 1) * float(world.dt), 4),
                    "pos": [round(float(v), 5) for v in position],
                    "speed": round(float(np.linalg.norm(velocity)), 6),
                    "u_x": round(float(velocity[0]), 6),
                    "u_z": round(float(velocity[2]), 6),
                    "nose_z": round(float(nose[2]), 4),
                    "pitch_rate": round(float(spin[1]), 5),
                    "fluid_fx": float(force[0]),
                    "wake_points": count,
                    "solid_cells": int(world.solid_cells(0)),
                })
        emit("trace", trace)
        emit("solver_seconds", round(solver_seconds, 1))
        emit("render_seconds", round(render_seconds, 1))
        if solver_seconds > 0:
            emit("solver_mlups",
                 round(total * float(world.nodes) / solver_seconds / 1e6, 1))
        emit("window_shifts", int(world.shifts))
        # How much the voxelised volume moved about as the hull swam. This is the
        # residual the calibration above cannot remove: the lattice re-voxelises the
        # body at every new pose, and a cell gained or lost is 1.25e-10 m3 of displaced
        # water, which is 1.2e-6 N of buoyancy -- about a fifth of the thrust, per cell.
        cells = [row["solid_cells"] for row in trace]
        if cells:
            emit("solid_cells_range", [int(min(cells)), int(max(cells))])
            swing = (max(cells) - min(cells)) * world.dx ** 3 * water.rho * 9.81
            emit("buoyancy_swing_N", float(swing))
            emit("buoyancy_swing_over_thrust", round(float(swing / thrust), 3))

        # -- the numbers ---------------------------------------------------------------
        #
        # Measured over the stretch just before the dive, where the body is at terminal
        # speed and still swimming straight.
        cruise = [row for row in trace
                  if 0.45 * dive_start <= row["step"] <= dive_start]
        if cruise:
            speed = float(np.mean([row["speed"] for row in cruise]))
            lo = int(len(axial) * 0.45 * dive_start / max(total, 1))
            hi = max(lo + 1, int(len(axial) * dive_start / max(total, 1)))
            fluid_fx = float(np.mean(axial[lo:hi])) if axial else 0.0
            emit("axial_samples", len(axial[lo:hi]))
            emit("axial_stddev_N", float(np.std(axial[lo:hi])) if axial else 0.0)
            emit("cruise_speed_mps", round(speed, 6))
            emit("cruise_reynolds", round(speed * diameter / water.nu, 2))
            emit("cd_from_thrust", round(2.0 * thrust
                                         / (water.rho * speed * speed * area), 4))
            emit("fluid_axial_force_N", fluid_fx)
            # The cross-check. At terminal speed the fluid must be pushing back with
            # exactly the thrust, and these two numbers come from different routes: one
            # from the body's kinematics through Newton, one from the populations
            # crossing the boundary links. A one-way coupling flies the submarine and
            # fails this.
            emit("thrust_over_fluid_force", round(abs(fluid_fx) / thrust, 4))
            emit("cd_from_fluid_wrench",
                 round(2.0 * abs(fluid_fx) / (water.rho * speed * speed * area), 4))
        dived = [row for row in trace if row["step"] > dive_stop]
        if dived:
            emit("descent_rate_mps", round(float(np.mean(
                [row["u_z"] for row in dived])), 6))
            emit("final_nose_z", dived[-1]["nose_z"])
            emit("depth_change_m", round(dived[-1]["pos"][2] - trace[0]["pos"][2], 5))

        # -- the artifacts -------------------------------------------------------------
        if not captured:
            emit("render_error", "no frames captured")
            emit("ok", False)
            return {"result": out, "blobs": blobs}

        from PIL import Image

        # The hero is mid-dive: the hull pitched, the wake tilted behind it. That is the
        # frame that shows the coupling doing something a policy would care about.
        hero_index = min(len(captured) - 1,
                         max(0, int(dive_stop / max(stride, 1)) + 2))
        hero = captured[hero_index]
        emit("hero_frame", hero_index)
        emit("render_shape", list(hero.shape))
        rgb = hero.astype("float32")
        # **Per channel, then averaged** -- not `rgb.std()` over the whole array, which
        # is the trap the previous run walked into. A frame that is one flat teal,
        # R 48.4 / G 138.0 / B 169.5, has a whole-array standard deviation of 51.32
        # because most of that spread is the gap *between* the channel means, not any
        # variation within the picture. It read as a healthy render and was a blank one.
        # `render_unique_colours` caught it with the value 1, which is why that number
        # exists; this makes the headline number honest too.
        emit("render_stddev",
             round(float(np.mean([rgb[..., c].std() for c in range(3)])), 2))
        emit("render_stddev_whole_array", round(float(rgb.std()), 2))
        emit("render_mean_rgb", [round(float(v), 1) for v in rgb.reshape(-1, 3).mean(0)])
        emit("render_unique_colours",
             int(len(np.unique(rgb.reshape(-1, 3)[::97], axis=0))))

        buf = io.BytesIO()
        Image.fromarray(hero, "RGB").save(buf, format="PNG", optimize=True)
        blobs["submarine_hero.png"] = buf.getvalue()
        emit("hero_png_bytes", len(blobs["submarine_hero.png"]))

        small = [Image.fromarray(f, "RGB").resize((width // 2, height // 2),
                                                  Image.LANCZOS).quantize(
                     colors=128, method=Image.MEDIANCUT) for f in captured]
        buf = io.BytesIO()
        small[0].save(buf, format="GIF", save_all=True, append_images=small[1:],
                      duration=80, loop=0, optimize=True)
        blobs["submarine.gif"] = buf.getvalue()
        emit("gif_bytes", len(blobs["submarine.gif"]))
        emit("gif_frames", len(small))
        for pick in sorted({0, len(captured) // 3, hero_index, len(captured) - 1}):
            buf = io.BytesIO()
            Image.fromarray(captured[pick], "RGB").save(buf, format="PNG", optimize=True)
            blobs[f"submarine_{pick:03d}.png"] = buf.getvalue()
        emit("ok", True)
    except Exception:
        emit("error", traceback.format_exc()[-6000:])
        emit("ok", False)
    finally:
        # No `sim_app.close()`. See the end of `_probe`.
        out["total_seconds"] = round(time.time() - started, 1)
    return {"result": out, "blobs": blobs}


@app.function(image=image, gpu="L40S", timeout=5400, memory=32768)
def submarine_l40s(cells_per_d: int = 0, reynolds: float = 100.0,
                   frames: int = 44) -> dict:
    return _submarine(cells_per_d=cells_per_d, reynolds=reynolds, frames=frames)


@app.function(image=image, gpu="A10G", timeout=5400, memory=32768)
def submarine_a10g(cells_per_d: int = 0, reynolds: float = 100.0,
                   frames: int = 44) -> dict:
    return _submarine(cells_per_d=cells_per_d, reynolds=reynolds, frames=frames)


def _stagecheck() -> dict:
    """Which prim types and which clipping ranges actually draw, at submarine scale.

    A focused, solver-free diagnostic. The submarine demo rendered a frame of exactly
    one colour twice running -- once with a hand-built camera and once with
    `rep.create.camera`, both times mean RGB (48.4, 138.0, 169.5), the dome and nothing
    else. Two four-minute runs had already gone into guessing, and the remaining
    suspects were all cheap to separate: the prim types (an implicit `Capsule` and a
    `PointInstancer`, neither of which the other demos use), the scale (0.04 m against
    the free-surface demo's 0.36 m), and the near clipping plane (0.001, an order of
    magnitude tighter than anything that has worked here).

    So this builds the same lights and camera and renders one frame per variant, with no
    fluid at all. Forty seconds answers what three more full runs would have guessed at.
    """
    import json
    import os
    import sys
    import time
    import traceback

    out: dict = {"gpu": _gpu_name()}

    def emit(key, value):
        out[key] = value
        print("STAGE " + json.dumps({key: value}, default=str), flush=True)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    started = time.time()
    import isaacsim  # noqa: F401
    from isaacsim.simulation_app import SimulationApp

    sim_app = SimulationApp({"headless": True, "width": 640, "height": 360,  # noqa: F841
                             "renderer": "RaytracedLighting"})
    emit("kit_startup_seconds", round(time.time() - started, 1))
    sys.path.insert(0, "/repo")

    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2400.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-62.0, 12.0, 20.0))
    sky = UsdLux.DomeLight.Define(stage, "/World/sky")
    sky.CreateIntensityAttr(900.0)
    sky.CreateColorAttr(Gf.Vec3f(0.05, 0.22, 0.34))

    material = UsdShade.Material.Define(stage, "/World/looks/m")
    shader = UsdShade.Shader.Define(stage, "/World/looks/m/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.9, 0.5, 0.15))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    def colours(arr) -> int:
        if not arr.size:
            return 0
        return int(len(np.unique(arr[..., :3].reshape(-1, 3)[::401], axis=0)))

    def shoot(label, build, scale, near):
        """Build one variant at `scale` metres, render it, count the distinct colours."""
        root = f"/World/try_{label}"
        try:
            build(root, scale)
            camera = rep.create.camera(
                position=(1.2 * scale, -7.2 * scale, 1.9 * scale),
                look_at=(0.0, 0.0, 0.0), focal_length=34.0,
                clipping_range=(near, near * 20000.0))
            product = rep.create.render_product(camera, (640, 360))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([product])
            arr = np.empty(0)
            for _ in range(5):
                sim_app.update()
                rep.orchestrator.step(rt_subframes=6)
                try:
                    rep.orchestrator.wait_until_complete()
                except Exception:
                    pass
                arr = np.asarray(annot.get_data())
                if colours(arr) > 8:
                    break
            emit(f"{label}@{scale}m,near{near}", colours(arr))
            annot.detach([product])
            stage.RemovePrim(root)
        except Exception as exc:
            emit(f"{label}@{scale}m,near{near}", f"FAILED: {type(exc).__name__}: {exc}")

    def make_cube(root, scale):
        prim = UsdGeom.Cube.Define(stage, root)
        prim.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(prim.GetPrim())
        x.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def make_capsule(root, scale):
        prim = UsdGeom.Capsule.Define(stage, root)
        prim.CreateRadiusAttr(scale * 0.25)
        prim.CreateHeightAttr(scale * 1.5)
        prim.CreateAxisAttr("X")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def make_instancer(root, scale):
        inst = UsdGeom.PointInstancer.Define(stage, root)
        drop = UsdGeom.Sphere.Define(stage, root + "/drop")
        drop.CreateRadiusAttr(scale * 0.04)
        UsdShade.MaterialBindingAPI(drop).Bind(material)
        inst.CreatePrototypesRel().SetTargets([drop.GetPath()])
        grid = np.stack(np.meshgrid(*[np.linspace(-scale, scale, 12)] * 3,
                                    indexing="ij"), axis=-1).reshape(-1, 3)
        inst.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(grid.astype(np.float32)))
        inst.CreateProtoIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.zeros(len(grid), dtype=np.int32)))

    try:
        for scale in (0.04, 0.4):
            for near in (0.001, 0.01):
                shoot("cube", make_cube, scale, near)
                shoot("capsule", make_capsule, scale, near)
                shoot("instancer", make_instancer, scale, near)
    except Exception:
        emit("error", traceback.format_exc()[-3000:])
    return out


@app.function(image=image, gpu="L40S", timeout=1800, memory=32768)
def stagecheck() -> dict:
    return _stagecheck()


# -- the local side ---------------------------------------------------------------------

PROBES = {"a10g": probe_a10g, "l40s": probe_l40s}
DEMOS = {"a10g": demo_a10g, "l40s": demo_l40s}
VULKANS = {"a10g": vulkan_a10g, "l40s": vulkan_l40s,
           "ubuntu-a10g": vulkan_ubuntu_a10g, "ubuntu-l40s": vulkan_ubuntu_l40s}


def _save(name: str, payload: dict, blobs: dict | None = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    for blob_name, blob in (blobs or {}).items():
        (RESULTS / blob_name).write_bytes(blob)
        print(f"  wrote {RESULTS / blob_name} ({len(blob):,} bytes)")
    print(f"  wrote {path}")
    return path


@app.local_entrypoint()
def vulkan(gpu: str = "a10g") -> None:
    """Does this container get the graphics half of the NVIDIA driver. Run this first."""
    got = VULKANS[gpu.lower()].remote()
    for key in ("gpu", "driver_capabilities", "icd_files"):
        print(f"{key}: {got.get(key)}")
    print(f"driver_libs: {len(got.get('driver_libs', []))} found")
    for lib in got.get("driver_libs", []):
        print(f"    {lib}")
    for cmd in ("sh -c objdump -T /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 "
                "| grep -i 'vk_icd\\|vkCreateInstance'",
                "sh -c dpkg -l 2>/dev/null | grep -i vulkan"):
        if cmd in got:
            print(f"--- {cmd[:40]}\n{got[cmd]}")
    print(got.get("vulkaninfo --summary", "<no vulkaninfo>"))
    _save(f"vulkan-{gpu}", got)


@app.local_entrypoint()
def deplist() -> None:
    """What `isaacsim[all,extscache]` pinned, with no GPU and no imports."""
    got = deps.remote()
    print(f"{got['count']} distributions installed; the ones that matter:")
    for name, ver in got["interesting"].items():
        print(f"  {name:36s} {ver}")
    print(f"  pxr provided by: {got.get('pxr_owners')}")
    _save("deps", got)


@app.local_entrypoint()
def coexistence() -> None:
    """Does Newton survive beside a running Kit, and does our extension import there."""
    got = coexist.remote()
    print(f"gpu: {got['gpu']}")
    for name in ("newton_alone", "kit_then_newton", "kit_then_openreynolds"):
        row = got[name]
        print(f"\n--- {name}: returncode {row['returncode']}")
        for line in row["stdout"]:
            print(f"    {line}")
        if row["returncode"] not in (0, "0"):
            print(f"    stderr tail:\n{row['stderr_tail']}")
    _save("coexistence", got)


@app.local_entrypoint()
def probe(gpu: str = "a10g") -> None:
    """Isaac Sim's version, its bundled Warp and Newton, and whether Kit starts."""
    got = PROBES[gpu.lower()].remote()
    print(f"\ngpu: {got.get('gpu')}\npython: {got.get('python')}")
    for name, ver in sorted((got.get("versions") or {}).items()):
        print(f"  {name:28s} {ver}")
    for key in sorted(k for k in got if k.startswith("import::")):
        print(f"  {key:28s} {got[key]}")
    print(f"  warp cuda devices: {got.get('warp_cuda_devices')}")
    ore = got.get("openreynolds_live_import") or {}
    print(f"  openreynolds.live in Kit: returncode {ore.get('returncode')} "
          f"{ore.get('stdout')}")
    print(f"  SimulationApp: {got.get('simulationapp')} "
          f"in {got.get('simulationapp_seconds')} s")
    print(f"  kit version: {got.get('kit_version')}  renderer: {got.get('renderer')}")
    _save(f"probe-{gpu}", got)


@app.local_entrypoint()
def cases(gpu: str = "l40s") -> None:
    """Run the live suite and two graded free-surface cases on Kit's own Warp."""
    runner = {"l40s": cases_l40s, "a10g": cases_a10g}[gpu.lower()]
    got = runner.remote()
    print(f"\ngpu: {got.get('gpu')}")
    print(f"kit startup: {got.get('kit_startup_seconds')} s")
    print(f"warp in Kit: {got.get('warp_version')}")
    print(f"  from: {got.get('warp_file')}")
    print(f"  cuda devices: {got.get('warp_cuda_devices')}   device: {got.get('device')}")
    print(f"\npytest: returncode {got.get('pytest_returncode')} "
          f"in {got.get('pytest_seconds')} s")
    for line in got.get("pytest_summary", []):
        print(f"  {line}")
    if got.get("pytest_returncode") not in (0, "0"):
        print(got.get("pytest_tail", ""))
    for label in ("hydrostatic", "sloshing"):
        row = got.get(f"case::{label}")
        if not row:
            continue
        verdict = "PASSED" if row.get("passed") else "FAILED"
        print(f"\n{label}: {verdict} in {row.get('seconds')} s  {row.get('settings')}")
        if row.get("error"):
            print(row["error"])
            continue
        for check in row.get("checks", []):
            mark = {True: "ok", False: "FAIL", None: "--"}[check.get("passed")]
            print(f"  [{mark:>4}] {check['name']}: measured {check['measured']} "
                  f"against {check['target']}")
    _save(f"cases-{gpu}", got)
    ok = (got.get("pytest_returncode") in (0, "0")
          and all(got.get(f"case::{n}", {}).get("passed") for n in
                  ("hydrostatic", "sloshing")))
    if not ok:
        sys.exit(1)


@app.local_entrypoint()
def freesurface(gpu: str = "l40s", cells: int = 144, frames: int = 44) -> None:
    """Solve a dam break with our own kernels and render it with Isaac Sim's RTX."""
    runner = {"l40s": freesurface_l40s, "a10g": freesurface_a10g}[gpu.lower()]
    got = runner.remote(cells, frames)
    payload = got.get("result", got)
    blobs = got.get("blobs", {})
    print(f"\ngpu: {payload.get('gpu')}")
    print(f"warp in Kit: {payload.get('warp_version')}")
    print(f"lattice: {payload.get('lattice_shape')}  dx {payload.get('dx')} m  "
          f"dt {payload.get('dt')} s")
    print(f"mesh: {payload.get('mesh_vertices')} vertices, "
          f"{payload.get('mesh_faces')} quads")
    print(f"ran {payload.get('total_steps')} steps "
          f"({payload.get('physical_seconds')} s of water) in "
          f"{payload.get('solver_seconds')} s solver + "
          f"{payload.get('render_seconds')} s render; "
          f"{payload.get('solver_mlups')} MLUPS")
    print(f"volume: {payload.get('initial_water_volume_m3')} -> "
          f"{payload.get('final_water_volume_m3')} m3")
    print(f"mass drift: {payload.get('mass_drift')}")
    print(f"fluid cells touching gas: {payload.get('fluid_touching_gas')}")
    for row in (payload.get("trace") or [])[::8]:
        print(f"  t {row['t']:.4f} s   wet {row['wet_fraction']:.3f}   "
              f"max depth {row['max_depth_m']:.4f} m")
    print(f"hero frame {payload.get('hero_frame')}: {payload.get('render_shape')} "
          f"stddev {payload.get('render_stddev')} "
          f"colours {payload.get('render_unique_colours')}")
    if payload.get("error"):
        print(payload["error"])
    _save(f"freesurface-{gpu}", payload, blobs)
    if not payload.get("ok"):
        sys.exit(1)


@app.local_entrypoint()
def stage_check() -> None:
    """Which prim types draw at submarine scale, and at which near clipping plane."""
    got = stagecheck.remote()
    print(f"gpu: {got.get('gpu')}   kit {got.get('kit_startup_seconds')} s")
    print("distinct colours in the frame (1 means nothing but the dome):")
    for key, value in got.items():
        if "@" in key:
            print(f"  {key:34s} {value}")
    _save("stagecheck", got)


@app.local_entrypoint()
def submarine(gpu: str = "l40s", cells_per_d: int = 0, reynolds: float = 100.0,
              frames: int = 44) -> None:
    """A Newton-coupled submarine in our fluid, rendered by Isaac Sim's RTX."""
    runner = {"l40s": submarine_l40s, "a10g": submarine_a10g}[gpu.lower()]
    got = runner.remote(cells_per_d, reynolds, frames)
    payload = got.get("result", got)
    blobs = got.get("blobs", {})
    print(f"\ngpu: {payload.get('gpu')}")
    print(f"warp {payload.get('warp_version')}   newton {payload.get('newton_version')}")
    print(f"lattice {payload.get('lattice_shape')} = {payload.get('nodes')} nodes, "
          f"dx {payload.get('dx')} m, dt {payload.get('dt')} s")
    print(f"hull: {payload.get('hull_mass_kg')} kg, "
          f"{payload.get('hull_volume_m3')} m3, "
          f"{payload.get('solid_cells')} solid cells")
    print(f"thrust {payload.get('thrust_N')} N, "
          f"{payload.get('total_steps')} steps = "
          f"{payload.get('simulated_seconds')} s, "
          f"{payload.get('window_shifts')} window shifts")
    print(f"solver {payload.get('solver_seconds')} s "
          f"({payload.get('solver_mlups')} MLUPS), "
          f"render {payload.get('render_seconds')} s")
    print("\n-- measured --")
    print(f"  cruise speed        {payload.get('cruise_speed_mps')} m/s")
    print(f"  Reynolds            {payload.get('cruise_reynolds')}")
    print(f"  Cd from thrust      {payload.get('cd_from_thrust')}")
    print(f"  Cd from fluid wrench{payload.get('cd_from_fluid_wrench')}")
    print(f"  |fluid Fx| / thrust {payload.get('thrust_over_fluid_force')}  "
          f"(1.0 means the two-way coupling balances)")
    print(f"  descent rate        {payload.get('descent_rate_mps')} m/s")
    print(f"  depth change        {payload.get('depth_change_m')} m")
    print(f"  final nose z        {payload.get('final_nose_z')}")
    for row in (payload.get("trace") or [])[::6]:
        print(f"  t {row['t']:.3f}  pos {row['pos']}  |u| {row['speed']:.5f}  "
              f"uz {row['u_z']:+.5f}  nose_z {row['nose_z']:+.3f}  "
              f"wake {row['wake_points']}")
    print(f"\nhero frame {payload.get('hero_frame')}: {payload.get('render_shape')} "
          f"stddev {payload.get('render_stddev')} "
          f"colours {payload.get('render_unique_colours')} "
          f"mean {payload.get('render_mean_rgb')}")
    if payload.get("error"):
        print(payload["error"])
    _save(f"submarine-{gpu}", payload, blobs)
    if not payload.get("ok"):
        sys.exit(1)


@app.local_entrypoint()
def demo(gpu: str = "a10g", steps: int = 100) -> None:
    """The minimal scene: ground plane, cube, physics, forces, and a rendered PNG."""
    got = DEMOS[gpu.lower()].remote(steps)
    payload = got.get("result", got)
    blobs = got.get("blobs", {})
    print(f"\ngpu: {payload.get('gpu')}")
    print(f"startup: {payload.get('startup_seconds')} s   "
          f"total: {payload.get('total_seconds')} s")
    for row in payload.get("track", []):
        print(f"  step {row['step']:>4}  pos {row['pos']}")
    print(f"  render: {payload.get('render_shape')} "
          f"nonzero {payload.get('render_nonzero_fraction')} "
          f"stddev {payload.get('render_stddev')} "
          f"colours {payload.get('render_unique_colours')}")
    print(f"  stage: {payload.get('stage_prims')}")
    print(f"  zero copy: {json.dumps(payload.get('zero_copy', {}), indent=2, default=str)}")
    if payload.get("error"):
        print(payload["error"])
    _save(f"demo-{gpu}", payload, blobs)
    if not payload.get("ok"):
        sys.exit(1)
