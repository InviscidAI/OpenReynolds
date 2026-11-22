# Isaac Sim on Modal

What was established by running it, on 2026-11-23, against `isaacsim` 6.2.0.0 on Modal
A10G and L40S containers. The app that produced every number here is
`benchmarks/live/modal_isaacsim.py`; its entrypoints are `vulkan`, `deplist`, `probe`,
`coexistence` and `demo`.

Nothing in this file is quoted from NVIDIA's documentation unless it says so. Where the
documentation and the container disagreed, the container won and the disagreement is
recorded, because two of them mattered.

## The short version

Isaac Sim installs from `pypi.nvidia.com` into an ordinary Modal image, starts headless
in **17 to 47 seconds**, runs PhysX on the GPU, and **renders with RTX** -- the A10G is
an active Vulkan device inside Kit even though the container's own `vulkaninfo` says
there is no Vulkan device at all. Newton and `openreynolds.live` both import inside a
live Kit process. The one real constraint is that Kit's bundled Warp is **1.17.0**,
which is two minor versions below the **1.19.0** this repository pins.

## 1. The install

```python
modal.Image.debian_slim(python_version="3.12")
    .apt_install(... X, GL, Vulkan and font libraries ...)
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("isaacsim[all,extscache]==6.2.0.0",
                 extra_index_url="https://pypi.nvidia.com")
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", ...})
```

**Version.** 6.2.0.0 is the newest on `https://pypi.nvidia.com/isaacsim/`. The index
lists `isaacsim-6.2.0.0-cp312-none-manylinux_2_35_x86_64.whl` and its aarch64 and
win_amd64 siblings. 6.1.0.0, 6.0.1.0, 6.0.0.1 and 6.0.0.0 are also cp312; 5.1.0.0 and
5.0.0.0 are cp311; 4.5.0.0 and earlier are cp310. **Choosing an older Isaac Sim is
therefore also choosing an older Python.**

**Python and glibc.** cp312 and `manylinux_2_35`. Debian bookworm is glibc 2.36 and
ships Python 3.12, so `debian_slim(python_version="3.12")` satisfies both. This is the
same Python `benchmarks/live/modal_run.py` already pins.

**EULA.** `OMNI_KIT_ACCEPT_EULA=YES` in the image environment. Without it the first
`import isaacsim` prompts on stdin, which in a container is a job that hangs until its
timeout rather than one that fails.

**Size and time.** The wheels are **7,465 MiB** downloaded, read from the index with
HEAD requests rather than estimated. Almost all of it is the extension cache:

| package | MiB |
| --- | --- |
| isaacsim-extscache-kit | 5,243.5 |
| isaacsim-extscache-kit-sdk | 1,562.0 |
| isaacsim-extscache-physics | 246.7 |
| isaacsim-utils | 179.8 |
| isaacsim-replicator | 74.4 |
| isaacsim-kernel | 71.4 |
| everything else | ~87 |

**The image built in 304.84 s** (`im-JFAgqSC5xLEu3ipnp1AVp9`), once. Every later run
reused the layer and the incremental builds were 1 to 43 s. 24 `isaacsim-*`
distributions land in site-packages, 176 distributions in total.

**GPU.** Isaac Sim's requirements page says "GPUs without RT Cores (A100, H100) are not
supported", which is the opposite of what the fluid wants. A10G and L40S both have RT
cores and both work. Everything here was measured on an A10G.

**Memory.** The requirements page asks for 32 GB RAM, so the Kit functions pass
`memory=32768`. The containers reported 688 GB total, so this is a floor and not a
constraint.

## 2. Starting it headless

```python
import isaacsim                                   # must come first
from isaacsim.simulation_app import SimulationApp
sim_app = SimulationApp({"headless": True, "width": 1280, "height": 720,
                         "renderer": "RaytracedLighting"})
```

Startup: **17.1 s** in `probe`, **47.3 s** in `demo` (which also loads Replicator).
`carb.settings.get_settings().get("/app/version")` returns `6.2.0`. Kit launches itself
with `--no-window --allow-root --portable` and reports 688 registered extensions.

Two things about this are not optional and both cost a run to find out.

### `import isaacsim` must precede every other import

Not a style preference. Isaac Sim's dependency closure installs `usd-exchange` 3.0.1,
which provides a pip `pxr` built in the `pxrInternal_v0_26_8` namespace. Kit's extension
cache ships its own OpenUSD in `pxrInternal_v0_25_11`. **Both install libraries whose
SONAME is plain `libusd_tf.so`**, so whichever is dlopened first wins it for the whole
process.

Importing `newton` (which imports `pxr`) before Kit therefore does not produce an
`ImportError`. It produces, at Kit startup, about forty lines of

```
[omni.ext.plugin] Could not load the dynamic library from .../omni.usd.libs-.../bin/
libusd_tf.so. Error: ... undefined symbol:
_ZTIN35pxrInternal_v0_25_11__pxrReserved__9pxr_boost6python7objects21py_function_impl_baseE
```

and then kills the interpreter outright with

```
python: symbol lookup error: .../omni.physics.stageupdate.../
libomni.physics.stageupdate.plugin.so: undefined symbol:
_ZN35pxrInternal_v0_25_11__pxrReserved__9pxr_boost6python9converter8registry6lookupENS1_9type_infoE
```

A `symbol lookup error` from the dynamic loader is not catchable. There is no `except`
that runs. The schema registry says the same thing one level up: `Cannot set alias
'ParticleFieldSphericalHarmonicsAttributeAPI' under 'UsdSchemaBase', because it is
already set to 'UsdVol...'`.

Start Kit first and the problem disappears entirely -- see section 4.

### Do not call `SimulationApp.close()`

A run that emitted every finding it was asked for, including the 688-extension registry
dump, then died with Modal reporting `InternalFailure: Server has lost track of input`.
The only statement between the last successful output and the loss was `sim_app.close()`.
Kit's teardown unwinds 688 extensions, a USD stage and a renderer, and when it wedges it
takes the container's input handler with it.

A one-shot container does not need a clean shutdown. `modal_isaacsim.py` returns its
result and lets Modal reap the container with Kit still loaded, and subprocesses end in
`os._exit(0)`. This is the documented quirk elsewhere too: NVIDIA's own install page
notes that `SimulationApp.close` terminates the kernel in a notebook.

## 3. Rendering, and the misleading Vulkan probe

**It renders. The output is `benchmarks/live/results/isaacsim/isaacsim_cubes.png`.**

The container's own Vulkan stack says otherwise, and this is worth writing down because
the obvious reading of it is wrong and cost an afternoon.

`vulkan` reports that `/etc/vulkan/icd.d/nvidia_icd.json` exists and names
`libGLX_nvidia.so.0`; that every dependency of that library resolves, including
`libnvidia-glvkspirv`, `libnvidia-glcore`, `libnvidia-rtcore`, `libnvidia-gpucomp` and
`libnvoptix`; that `objdump -T` on it shows `vk_icdGetInstanceProcAddr`,
`vk_icdNegotiateLoaderICDInterfaceVersion` and `vkCreateInstance` all exported; and that
`vulkaninfo` nevertheless enumerates only `llvmpipe`, having failed with

```
ERROR: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get
'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
```

Identical on Debian bookworm (loader 1.3.239) and Ubuntu 24.04 (loader 1.3.275),
identical on A10G and L40S. The sandbox is visible underneath it: `/proc/driver/nvidia`
holds only `params` with none of the `gpus/*/information` tree, the kernel reports as
`4.19.0-gvisor`, and the device nodes are the compute set (`nvidiactl`, `nvidia<N>`,
`nvidia-uvm`, `nvidia-uvm-tools`) with no `nvidia-modeset`.

**None of which stops Kit, because Kit does not use the system loader.** From a real
`demo` run:

```
| Driver Version: 620.70.06     | Graphics API: Vulkan
| GPU | Name          | Active | GPU Memory | Vendor-ID |
| 0   | NVIDIA A10G   | Yes: 0 | 23028   MB | 10de      |
| OS: 12 (bookworm) debian, Kernel: 4.19.0-gvisor
```

EGL agrees: `eglInitialize` succeeds at version 1.5 with `vendor = NVIDIA` and
`eglQueryDevicesEXT` returns 2 devices. The only harmless leftover is one warning from
the MDL plugin, `[rtx.neuraylib.plugin] It appears that the "nvidia" kernel module is
not loaded`, which does not stop the render.

**So the only test that settles whether Isaac Sim can render in a container is starting
Isaac Sim and looking at the pixels.** The first green render here was a 108 KB PNG of a
smooth grey gradient with no geometry in it, which looks exactly like a renderer that
failed and was in fact a camera aimed by hand with `rotateXYZ` at nothing in particular.
`demo` therefore uses `rep.create.camera(position=..., look_at=...)` and reports
`render_stddev` and `render_unique_colours` beside the image, so that "it rendered" is a
measurement.

The capture path that works headless:

```python
camera  = rep.create.camera(position=(2.3, -2.7, 1.45), look_at=(0.05, 0.05, 0.22),
                            focal_length=33.0)
product = rep.create.render_product(camera, (1280, 720))
annot   = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([product])          # attach BEFORE world.reset()
...
for _ in range(steps): world.step(render=True)   # render=True, or Hydra never syncs
for _ in range(24):    world.render()            # let RTX accumulation settle
rgb = np.asarray(annot.get_data())               # (H, W, 4) uint8, in process
```

`omni.kit.viewport.utility.capture_viewport_to_file` is the path every tutorial shows
and is the wrong one here: a headless Kit has no viewport window to capture.

## 4. Living beside Warp and Newton

`coexistence` runs three scripts as subprocesses, because the failure mode is a fatal
loader error rather than an exception. All three returned 0.

| | Newton | Warp | OpenUSD |
| --- | --- | --- | --- |
| plain interpreter, no Kit | 1.6.1 | **1.19.0** | (0, 26, 8) from `usd-exchange` |
| inside a running Kit | 1.6.1 | **1.17.0** | (0, 25, 11) from Kit's extscache |

Inside Kit, `import warp` resolves to
`/usr/local/lib/python3.12/site-packages/isaacsim/extscache/omni.warp.core-1.17.0+lx64/warp/__init__.py`
-- Kit's own extension, not the pip `warp-lang` 1.19.0 that `pip show` reports.
`newton.ModelBuilder` is present, `warp.get_cuda_device_count()` is 1 and the preferred
device is `cuda:0`.

**`openreynolds.live` imports inside a live Kit process.** `world` and `couple` both
load and `FluidWorld` is present, with `/repo` on `sys.path`.

### The version gap, stated plainly

`benchmarks/live/modal_run.py` pins `warp-lang==1.19.0`, `newton==1.8.0` and
`mujoco-warp==4.0.1`. Isaac Sim 6.2.0.0 pins Warp **1.17.0** (inside Kit), Newton
**1.6.1** and mujoco-warp **3.12.0**. The install page's own warning is that these wheels
"pin some packages, including Newton, Warp, and MuJoCo, to specific versions", and that
is exactly what it does.

Nothing here has measured whether `openreynolds/live/` **runs** on Warp 1.17 and Newton
1.6; what is measured is that it **imports**. That is the next thing to find out and it
is cheap: run the existing cases inside a Kit process on this image.

### Isaac Sim 6.2 ships Newton as a physics backend

Unprompted finding from the 688-extension registry: `isaacsim.physics.newton`,
`isaacsim.physics.newton.tensors`, `isaacsim.physics.newton.ui`, `isaacsim.exp.full.newton`
and `omni.usd.schema.newton` are all present. Isaac Sim 6.2 can run Newton as its
physics engine rather than PhysX. Given that `openreynolds/live/couple.py` couples to
Newton already, this is probably a shorter path to a coupled scene than the PhysX one
below, and it is not a path this repository has looked at yet.

## 5. The rigid-body read/write surface

Signatures are `inspect.signature` output from the installed 6.2.0.0 package, not from
the documentation. The published docs are mid-move: `isaacsim.core.prims` is served
under `py/source/deprecated/` and the `omni.physics.tensors` API reference renders empty,
so several of these are not on the web at all.

### `isaacsim.core.prims.RigidPrim` -- the wrapper that works today

At `isaacsim/extsDeprecated/isaacsim.core.prims/.../impl/rigid_prim.py`. Despite the
directory name this is what `isaacsim.core.prims` imports.

```python
RigidPrim(prim_paths_expr: str | list[str], name: str = 'rigid_prim_view',
          positions=None, translations=None, orientations=None, scales=None,
          visibilities=None, reset_xform_properties: bool = True,
          masses=None, densities=None, linear_velocities=None, angular_velocities=None,
          track_contact_forces: bool = False, prepare_contact_sensors: bool = True,
          disable_stablization: bool = True,
          contact_filter_prim_paths_expr: list[str] | None = None,
          max_contact_count: int = 0) -> None

get_world_poses(indices=None, clone: bool = True, usd: bool = True)
    -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]
     | tuple[wp.indexedarray, wp.indexedarray]
set_world_poses(positions=None, orientations=None, indices=None, usd: bool = True) -> None
get_velocities(indices=None, clone: bool = True)        # (M, 6), linear then angular
set_velocities(velocities, indices=None) -> None
get_linear_velocities(indices=None, clone: bool = True)
get_angular_velocities(indices=None, clone: bool = True)
apply_forces(forces, indices=None, is_global: bool = True) -> None
apply_forces_and_torques_at_pos(forces=None, torques=None, positions=None,
                                indices=None, is_global: bool = True) -> None
get_masses(indices=None, clone: bool = True)
get_coms(indices=None, clone: bool = True)
get_net_contact_forces(indices=None, clone: bool = True, dt: float = 1.0)
```

Conventions, from the docstrings: quaternions are **scalar first, (w, x, y, z)**;
`positions` are world frame and `translations` are local; `is_global=True` means the
forces, torques and application positions are in the world frame; `clone=True` returns a
copy of the internal buffer and `clone=False` does not.

`prim_paths_expr` is a glob -- `"/World/cube_*"` gives one view over three bodies and
`view.count` is 3 -- which is the batched form a coupling wants.

**Forces must be reapplied every step.** PhysX clears external forces each step. This is
the same contract `openreynolds/live/couple.py` already works under, where
`newton.State.clear_forces()` is called between the fluid and the solver.

Exercised end to end in `demo`: three cubes dropped from 1.5 to 3.0 m, settling at
z = 0.20 for a 0.4 m cube (half the edge, correct), then a per-body force and yaw torque
applied for twenty steps and the resulting translation and rotation read back out.
`physics_dt` and `render_dt` are both 1/60.

### `isaacsim.core.experimental.prims.RigidPrim` -- the Warp-native replacement

At `isaacsim/exts/isaacsim.core.experimental.prims/.../impl/rigid_prim.py`. This is where
Isaac Sim is going, and the annotations are the reason it matters here:

```python
RigidPrim(paths: str | list[str], *, resolve_paths: bool = True,
          positions: list | np.ndarray | wp.array | None = None,
          translations=None, orientations=None, scales=None,
          reset_xform_op_properties: bool = False,
          masses=None, densities=None,
          contact_filter_paths: str | list[str] | None = None,
          max_contact_count: int = 0) -> None

get_world_poses(*, indices=None) -> tuple[wp.array, wp.array]
get_velocities(*, indices=None) -> tuple[wp.array, wp.array]
get_masses(*, indices=None, inverse: bool = False) -> wp.array
apply_forces_and_torques_at_pos(forces=None, torques=None, *, positions=None,
                                indices=None, local_frame: bool = False) -> None
```

Note the differences from the old wrapper: the returns are **`wp.array`, not a union**;
velocities come back as a `(linear, angular)` pair rather than one `(M, 6)` block; and
the frame flag inverts, `local_frame=False` where the old one had `is_global=True`.

`isaacsim.core.simulation_manager.SimulationManager` carries
`get_backend() -> str`, `set_backend(val: str) -> None`,
`get_physics_sim_view() -> Any` and `get_device() -> wp.Device`. The first two are
documented as deprecated since 1.8.0 with "no replacement is provided, as the core
experimental API relies solely on Warp".

## 6. Zero copy between PhysX and Warp

`omni.physics.tensors` is the layer underneath both wrappers. Read off the installed
package:

```python
create_simulation_view(frontend_name, stage_id=-1, backend='physx')
SimulationView.create_rigid_body_view(pattern)
SimulationView.create_articulation_view(pattern)
SimulationView.set_subspace_roots(pattern)
```

The module exports `frontend_np` and **`frontend_warp`** beside each other, and the
accepted frontend names are `"numpy"`, `"torch"` and `"warp"`. `omni.physics.tensors` is
the interface and `omni.physx.tensors` the only concrete implementation; both extensions
are registered.

**The default is not the one you want.** A default `World` plus
`isaacsim.core.prims.RigidPrim` reports `SimulationManager.get_backend() == "numpy"`, and
`get_world_poses()` returns a `numpy.ndarray` on `cpu` with no `data_ptr`. Every read is
a device-to-host copy, which for a per-step coupling to a GPU fluid is exactly the thing
to avoid.

The Warp frontend has to be asked for, either through
`tensors.create_simulation_view("warp")` or by using the experimental prims, whose
annotations return `wp.array` unconditionally.

<!-- MEASURED-ZERO-COPY -->

## 7. Visualising a field inside Isaac Sim

The extension registry in this build contains, among the 688:
`isaacsim.util.debug_draw`, `omni.ui.scene`, `omni.volume`,
`omni.vdb_timesample_editor`, `omni.usd.schema.usd_particle_field`,
`omni.kit.pointclouds`, `omni.warp.core`, `omni.replicator.core`.

The practical ranking, for a lattice field updated every frame on the GPU:

1. **`isaacsim.util.debug_draw`, for vectors and sparse samples.** In-memory, no disk,
   takes point and line lists with per-point colour and size. The module is present and
   exposes `bindings` and `impl`; note that `isaacsim.util.debug_draw._debug_draw` is
   **not** an importable submodule path in 6.2.0.0, which is what the older docs show.
   Best for streamlines, probe arrows and a decimated velocity field. Not for a dense
   volume.

2. **`UsdGeomPointInstancer`, for a decimated scalar field.** Positions update in memory
   through USD attribute writes, so no disk round trip, and it is the only option here
   that scales to hundreds of thousands of primitives. A 256^3 lattice is 16.7 M cells,
   so this means subsampling or thresholding to an isosurface shell rather than drawing
   the volume. Per-instance colour needs checking against the schema in this build; it
   was not verified here.

3. **`UsdVol` plus OpenVDB, for a true volume, and it costs a disk round trip.** The USD
   schema represents volume animation by time-sampling the `filePath` attribute of an
   `OpenVDBAsset`, one `.vdb` per frame. `omni.volume` and `omni.vdb_timesample_editor`
   are both present so Kit can display it, but the documented input is a file path and
   not a pointer. Right for an offline beauty render of a saved run, wrong for a live
   coupled loop.

4. **`omni.ui.scene` is not the tool.** It exists and is registered, but it is for
   manipulators and viewport helpers -- its own overview describes "3d manipulators and
   3d helpers" -- and there is nothing to suggest it handles a dense field.

The one that was not investigated and probably should be: `omni.usd.schema.usd_particle_field`
is in this build and its name is suggestive. It was found in the registry, not read.

## 8. Cost and discipline

Every function is a one-shot `@app.function` with an explicit `timeout`, matching
`benchmarks/live/modal_run.py`. No `@app.cls`, no warm pool, no `min_containers`, no
volume. Probing used A10G throughout; no H100 was touched.

`vulkan` runs on a separate 300 MB image so that the cheap question does not wait on the
expensive build, and `deps` runs on **no GPU at all**, because "what did the install
pin" is a question about pip metadata.

`add_local_dir` here is a runtime mount rather than `copy=True`, unlike `modal_run.py`.
The layer beneath it takes five minutes to build and nothing in this file needs the
repository at image build time, so editing `openreynolds/live/` must not invalidate it.

## 9. What is still unknown

- Whether `openreynolds/live/` **runs**, rather than merely imports, on Kit's Warp 1.17.0
  and Newton 1.6.1. It imports. That is all that was measured.
- Whether the `isaacsim.physics.newton` backend can drive our existing
  `couple.NewtonCoupling` directly. Probably the most valuable thing to try next.
- Per-instance colour on `UsdGeomPointInstancer` in this build.
- What `omni.usd.schema.usd_particle_field` is.
- Whether Replicator's render throughput is good enough to film a live coupled run, as
  opposed to taking one frame of a settled one.
