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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*APT)
    .pip_install(TORCH, index_url=TORCH_INDEX)
    .pip_install(
        f"isaacsim[all,extscache]=={ISAACSIM}",
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("pillow>=10", "numpy>=1.24")
    .env(KIT_ENV)
    # Not `copy=True`, unlike `modal_run.py`. There the repository is baked in so that a
    # benchmark image is a single immutable thing; here the layer under it takes tens of
    # minutes to build and a runtime mount means editing `openreynolds/live/` never
    # touches it. Nothing in this file needs the repository at build time.
    .add_local_dir(
        str(REPO), remote_path="/repo",
        ignore=[
            "studies/**", ".git/**", "**/__pycache__/**", "*.log", "launcher/**",
            ".pytest_cache/**", "benchmarks/live/results/**", "docker/**",
        ],
    )
)

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

        sim_app = SimulationApp({"headless": True})
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

        sim_app = SimulationApp({
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
