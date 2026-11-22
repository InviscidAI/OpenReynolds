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

### The capture path that works headless

```python
import omni.replicator.core as rep

camera  = rep.create.camera(position=(2.3, -2.7, 1.45), look_at=(0.05, 0.05, 0.22),
                            focal_length=33.0)
product = rep.create.render_product(camera, (1280, 720))
annot   = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([product])                      # attach BEFORE world.reset()
...
for _ in range(steps):
    world.step(render=True)                  # render=True, or Hydra never syncs

rep.orchestrator.step(rt_subframes=8)        # Replicator's own sync point
rep.orchestrator.wait_until_complete()
rgb = np.asarray(annot.get_data())           # (720, 1280, 4) uint8, in process
```

Three things in that are load-bearing and each one cost a run.

**`rep.create.camera(position=..., look_at=...)`, not a hand-placed `UsdGeom.Camera`.**
Aiming by `rotateXYZ` is arithmetic with no feedback, and a camera that misses produces
an image indistinguishable from a renderer that failed.

**`world.step(render=True)`.** With `render=False` the physics is identical and Hydra
never syncs the stage, so the renderer has nothing to draw.

**`rep.orchestrator.step()` plus `wait_until_complete()`, not `world.render()`.**
`world.render()` is not a synchronisation point. Twenty-four calls to it produced a good
138 KB render once and a completely empty annotator twice more on the same code path,
with the physics loop finishing in 2.7 s instead of 105 s. `annot.get_data()` then
returns a zero-length array and Pillow raises `ValueError: cannot write empty image`
several frames downstream of the real problem. `demo` wraps the orchestrator call in a
bounded retry and measures the result, and has not produced an empty frame since.

`omni.kit.viewport.utility.capture_viewport_to_file` is the path every tutorial shows
and is the wrong one here: a headless Kit has no viewport window to capture.

### The numbers from the final run

A10G, 100 physics steps at `physics_dt` 1/60 on the CUDA pipeline, three rigid bodies,
1280x720:

| | |
| --- | --- |
| Kit startup | 18.8 s |
| whole job, startup to PNG | 118.7 s |
| render shape | (720, 1280, 4) uint8 |
| orchestrator attempts needed | 1 |
| pixel standard deviation | 46.39 |
| distinct sampled colours | 1,417 |
| PNG | 202,822 bytes |

The image is `benchmarks/live/results/isaacsim/isaacsim_cubes.png`: three coloured
boxes at rest on a lit ground plane with cast shadows, after being dropped, settled and
pushed sideways.

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

### Measured: yes, with two settings that are easy to miss

Both have to be right, and each one silently degrades the other if it is not.

**1. `World(device="cuda")`.** Without it the physics pipeline is on the host, and then
every view it hands out is on the host *however it is labelled*. A run with the Warp
frontend selected and a default `World` returned genuine `wp.array` and
`wp.indexedarray` objects whose `.device` was `cpu` and whose `.ptr` was `None`.
**The frontend chooses the array type; the pipeline chooses where the memory lives.**

**2. `SimulationManager.set_backend("warp")`.** Asking for a CUDA `World` on its own
does not give you Warp, it gives you torch -- Isaac Sim switches the frontend itself and
says so:

```
[isaacsim.core.simulation_manager] Changing backend from 'numpy' to 'torch'
since NumPy cannot be used with GPU piplines
```

That is worth knowing for a second reason: after the switch, passing a numpy array to
`apply_forces_and_torques_at_pos` fails deep in the wrapper with `'numpy.ndarray' object
has no attribute 'to'`, because `move_data` calls `.to(device)` on whatever it is given.
**The inputs have to follow the frontend the manager actually chose, not the one you
asked for.**

With both set, measured on an A10G over three bodies:

| call | type | device | pointer |
| --- | --- | --- | --- |
| `isaacsim.core.prims.RigidPrim.get_world_poses(clone=False)` | `warp.types.indexedarray` (3, 3) | `cuda:0` | none (index-mapped) |
| `.get_velocities(clone=False)` | `warp.types.array` (3, 6) | `cuda:0` | - |
| `isaacsim.core.experimental.prims.RigidPrim.get_world_poses()` | `warp.types.array` (3, 3) | `cuda:0` | `0x302000e00` |

`SimulationManager.get_backend()` reads back `warp` and `get_device()` reads back
`cuda:0`. `apply_forces_and_torques_at_pos` accepts a `wp.array` on the device.

**The verdict: a Warp fluid can read PhysX rigid-body state without a host round trip.**
The experimental prims are the cleaner half of it -- a plain `wp.array` on `cuda:0`
carrying a real device pointer, which a Warp kernel can take directly.

Two honest caveats, both measured rather than assumed:

- The deprecated wrapper returns a `wp.indexedarray`, not a `wp.array`. That is an
  index-mapped view with no single base pointer, so it is zero-copy in the sense that
  matters -- no device-to-host transfer -- but it is not a flat buffer and a kernel has
  to take it as an `indexedarray`.
- The pointer was **not** stable across two consecutive `get_world_poses(clone=False)`
  calls. So treat each call as producing a handle to read from now, not a buffer to
  cache across steps.

What was *not* measured: whether a Warp kernel writing through that array is seen by
PhysX on the next step. Reading is established; writing through the array rather than
through `apply_forces_and_torques_at_pos` is not.

One dead end worth recording so it is not retried: `tensors.create_simulation_view("warp")`
raises `Exception: Failed to create simulation view with backend 'physx'` when a `World`
already exists, because the backend will not hand out a second view.
`SimulationManager.set_backend` is the supported route.

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

## 9. What this cost, and the runs it took

21 ephemeral `openreynolds-isaacsim` apps, all stopped. The image built once, for
304.84 s (`im-JFAgqSC5xLEu3ipnp1AVp9`); every run after that reused the layer.

The ones worth naming:

| app | what it established |
| --- | --- |
| `ap-pxMxpTwkTXJHLvK4Fv0EW3` | first `vulkan`: NVIDIA ICD present, only llvmpipe enumerates |
| `ap-1CnnXb1DvHRjnjmGDQesQg` | first `probe`: the USD SONAME collision, fatal |
| `ap-jYZPMB6ZhU4Ncwynlzvnco` | `deplist`: warp 1.19.0, newton 1.6.1, `pxr` from `usd-exchange` |
| `ap-mryUxy8hbT5FzOam6XKPXh` | `vulkan` on Ubuntu 24.04 and L40S: identical failure, hypothesis dead |
| `ap-cthLZSwXfawjSCNJLgL9tP` | `probe` reaching `app ready` in 17.1 s, then lost on `close()` |
| `ap-Tuxb2uOo8pyR18IsHDmhux` | `coexistence`: all three subprocesses returncode 0 |
| `ap-qfgsTPblLOb1hEaoL3uggJ` | first `demo`: physics correct, render a blank gradient |
| `ap-PJOefbbg05fRsxuTlg7HSR` | `demo` on the CUDA pipeline: the render, and `wp.array` on `cuda:0` |
| `ap-vDKz5hbIfFw3TOjewFrpaB` | `probe` end to end, including the `openreynolds.live` check |

Everything ran on A10G except two `vulkan` runs on L40S. No H100 was used. Adding up the
GPU container time across all of it gives roughly 45 to 60 minutes, which at A10G list
price is **order of one to two dollars**. That is arithmetic over counted runs, not a
billing figure read from the dashboard.

## 10. The extension RUNS inside Kit, and the free surface renders

Added 2026-11-23, after the sections above were written. Section 4 established that
`openreynolds.live` **imports** beside a running Kit. This section establishes that it
**runs** there, on Kit's own Warp, and that the free surface it computes can be drawn by
the RTX renderer in the same process.

Both entrypoints are in `benchmarks/live/modal_isaacsim.py`: `cases` and `freesurface`.

### 10.1 The suite, on Kit's bundled Warp

`cases` starts Kit headless, then runs pytest **in that same interpreter**. That is the
whole point. An earlier session had already run 52 of 52 of `tests/test_live_cases.py`
in the Isaac Lab virtualenv on Warp 1.17.0 and Newton 1.6.1, but a virtualenv is not
this: Kit never loads the pip `warp-lang`, it loads `omni.warp.core`, its own extension
with its own build. Confirmed from inside the run:

```
warp in Kit: 1.17.0
  from: /usr/local/lib/python3.12/site-packages/isaacsim/extscache/
        omni.warp.core-1.17.0+lx64/warp/__init__.py
```

On an L40S, Kit up in 20.7 s:

```
pytest: returncode 0 in 41.4 s
  63 passed, 1 warning in 35.41s
```

**63 passed, 0 failed.** No kernel failed to compile and no kernel gave a different
answer. Kit's Warp 1.17.0 needs no shim.

One failure on the way there was mine and not the extension's, and it is worth recording
because it looks exactly like a broken extension:
`test_both_array_layouts_give_the_same_field` re-execs `sys.executable -c ...` in a
subprocess, because the array layout is baked into the generated kernels at import and
two layouts need two interpreters. The child inherits the environment, not the parent's
`sys.path`, so with `/repo` only in `sys.path` it failed with
`ModuleNotFoundError: No module named 'openreynolds'` -- 62 passed, 1 failed. Setting
`PYTHONPATH=/repo` fixed it. **A harness bug and an extension bug are the same colour
in a test report.**

The image for this is `tests_image`, which is the Isaac Sim image plus the package's own
runtime dependencies. `tests/conftest.py` imports `anthropic`, `rich` and four
`openreynolds` modules unrelated to the live backend, and pytest loads it for anything
under `tests/`. They are installed from a list rather than with `pip install -e /repo`,
because an editable install would re-resolve numpy and torch inside an image whose numpy
and torch are the ones Isaac Sim pinned.

### 10.2 The two graded cases

Both run inside the live Kit process, on `cuda:0`, after the suite.

**`hydrostatic`, cells=20, 7.2 s: PASSED, 7 of 7 graded checks.**

| check | measured | target |
| --- | --- | --- |
| water volume at the start | 150.0 mL | 150.0000 mL +/- 2% |
| surface height vs compressible prediction | 59.8611 mm | 59.8530 mm +/- 2% of depth |
| surface flatness | 0.00090 | <= 2% of the depth |
| hydrostatic pressure gradient | 1.00263 | 1.0 +/- 10% (-81.5205 kg/m4 exact) |
| fluid cells touching gas cells | 0 | exactly 0 |
| mass drift | 5.086e-06 | <= 1e-3 of the initial mass |
| lattice finite | True | True |

**`sloshing`, cells=80, periods=1.5, 6.8 s: PASSED, 4 of 4 graded checks.**

| check | measured | target |
| --- | --- | --- |
| first-mode period | 0.418598 s | 0.41709 s +/- 5% |
| mass drift | 2.204e-06 | <= 1e-3 of the initial mass |
| fluid cells touching gas cells | 0 | exactly 0 |
| lattice finite | True | True |

The period is 1.0036x linear theory (`omega^2 = g k tanh(k h)`, k = pi/L). The
zero-crossing cross-check reads 0.392916 s and is reported, not graded, which is how
`cases.py` already treats it.

### 10.3 A bug this caught, which was fixed while the job was running

The first `cases` run failed `hydrostatic` outright:

```
File "/repo/openreynolds/live/cases.py", line 2940, in run_hydrostatic
  limit = FREE_SURFACE_REFERENCES["hydrostatic"]["tolerance"]
NameError: name 'FREE_SURFACE_REFERENCES' is not defined.
Did you mean: '_FREE_SURFACE_REFERENCES'?
```

Not a Warp or a Kit problem: a plain name error on an unconditional line, so
`run_hydrostatic` could not complete at any resolution on any device. `git log -S`
places it in commit `2cf6584`, and by the time the next run started it was already
corrected to `REFERENCES["hydrostatic"]` in the working tree. Nothing here fixed it.

**Worth recording as a property of these runs rather than as a bug report.**
`modal_isaacsim.py` mounts the working tree, not a commit, so a run measures whatever
was on disk when it was launched, and this tree is being edited by another contributor
while these jobs are in flight. Two runs an hour apart are not necessarily runs of the
same code. Every number in this section comes from the run whose app id is given below.

### 10.4 The free-surface demo

`freesurface` solves a three-dimensional dam break with `FreeSurfaceWorld` and renders it
with RTX, in one process, on one GPU, through one copy of Warp.

The tank is 0.36 x 0.16 x 0.16 m closed on all six faces, a 0.12 x 0.16 x 0.12 m column
of water released at t = 0, built through `cases._free_surface_world` so that the Froude
scaling, the tau and the gravity sign are the repository's own decisions and not a second
opinion. Lattice **144 x 64 x 64**, dx 2.5 mm, dt 131.69 us.

Measured on an L40S, 3,920 steps covering 0.5162 s of water:

| | |
| --- | --- |
| solver | 12.3 s, **188.7 MLUPS** |
| render, 40 frames at 1280x720 | 9.2 s |
| water volume | 2.35250e-3 -> 2.35020e-3 m3 |
| mass drift | -7.70e-3 of the initial mass over 1,010,757 conversions |
| fluid cells touching gas cells | **0** |
| hero frame | stddev 41.7, 1,962 distinct colours |

The bed wets as the surge crosses it, which is the run in one column:

| t (s) | wet fraction | max depth (m) |
| --- | --- | --- |
| 0.0129 | 0.347 | 0.1210 |
| 0.1162 | 0.592 | 0.1181 |
| 0.2194 | 0.968 | 0.1154 |
| 0.3227 | 1.000 | 0.1130 |
| 0.4259 | 1.000 | 0.1106 |

**Why a heightfield.** `surface_height(axis=2)` is per-column depth in metres, sub-cell
accurate, `nan` where dry -- which is exactly a heightfield. So the water is one
`UsdGeom.Mesh` of 9,216 vertices and 9,009 quads whose `points` and `normals` are
rewritten each frame and whose topology never changes, so the renderer never rebuilds an
acceleration structure. A `UsdGeomPointInstancer` over the interface cells draws the
surface as beads; a `UsdVol` OpenVDB volume is the only way to draw the interior and the
schema animates it by time-sampling a `filePath`, one `.vdb` per frame, so a live loop
would be writing files and reading them back. Marching cubes over `world.fill` would give
a true isosurface and rebuilds topology every frame.

**The limitation, stated: a heightfield cannot show a wave that curls over itself.** One
depth per column. For a dam break on a flat bed that is all the readable motion, and the
vertical cliff where the reservoir meets the bed is drawn as a near-vertical sheet, which
is correct and reads as a wall of water.

**And an artifact worth naming rather than cropping.** A few single-column spikes appear
at the tank's edges in the later frames, visible in `freesurface_039.png` as thin
spires on the back rim and at the right edge. They are one column whose summed fill
differs sharply from its neighbours, drawn at full height because a heightfield mesh
joins every column to the next with no notion of how far apart their depths may
reasonably be. They are a drawing artifact, not a solver one: `fluid_touching_gas` is 0
for the whole run and the mass drift is -7.7e-3, so the interface is closed and the
bookkeeping holds. A median filter over the height field before meshing would hide them,
and would also hide a real spike, so nothing filters them here.

Two details that decide whether it looks like water or like a surface plot: **vertex
normals** computed from the height gradient as `(-dh/dx, -dh/dy, 1)` normalised, without
which 9,009 quads read as 9,009 facets; and **dry columns pushed to z = -6 mm**, inside
the floor slab, rather than flattened to zero where they would z-fight with the bed and
read as a wet film over the whole tank.

### 10.5 Three ways a correct scene renders as nothing

All three happened here, in order, and all three produce a picture that looks like a
broken renderer.

**Nothing ticked the application.** First attempt: 44 frames in 137.9 s, every one a
single flat colour, `render_stddev` exactly 0.0, one distinct colour. The solver was
perfect in that same run -- 168.5 MLUPS, zero fluid cells touching gas. USD authoring
does not reach Hydra on its own; the stage's change notices are processed on an app
update. `_demo` never hit this because it drives `isaacsim.core.api.World`, whose
`step(render=True)` calls `SimulationApp.update()` underneath. A demo with no PhysX scene
has to call `sim_app.update()` itself.

**The scene was inside the near clip plane.** Second attempt, after adding the updates:
still one colour, but now `render_mean_rgb` was **(191.6, 191.6, 191.6)** rather than
black -- lit, drawing its background, nothing in front of it. The default camera near
plane is **1.0 stage unit**. This tank is 0.36 m across with the camera 0.59 m from what
it is aimed at, so in SI units the entire scene lies nearer than the near plane and every
triangle is clipped before it is shaded. `clipping_range=(0.01, 100.0)` fixes it. The
stage also has to be told `SetStageMetersPerUnit(stage, 1.0)` and
`SetStageUpAxis(stage, Z)`; Kit's default stage is centimetres and Y up, and `_demo`
never had to care because `World(stage_units_in_meters=1.0)` sets both.

**That is why `render_mean_rgb` is now emitted beside `render_stddev`.** A flat black
frame means no light reached the camera; a flat grey frame means the renderer drew its
background and there was no geometry in front of it. Those are different bugs and
`stddev == 0` alone does not separate them.

**The exposure, not the material.** The bed then rendered white, which looks exactly like
an unbound material -- and there was a real latent binding bug beside it, the floor's
`Shader` defined at `/World/looks/floor/surface` before its parent `Material` at
`/World/looks/floor`. Fixing the ordering changed nothing, because the material was
being applied and a 0.40 grey diffuse on a horizontal face under a 4,200-intensity key
light simply saturates. Lowering the key to 2,600 and the diffuse to 0.22 is what fixed
it. **The wrong thing went and got fixed first because the symptom named it.**

### 10.6 Artifacts

In `benchmarks/live/results/isaacsim/`:

| file | what |
| --- | --- |
| `freesurface_hero.png` | 1280x720, the surge crossing the dry bed |
| `freesurface.gif` | 40 frames at 640x360, the whole collapse |
| `freesurface_000.png`, `_010`, `_020`, `_039` | stills across the run |
| `freesurface-l40s.json` | every number above, including the per-frame trace |
| `cases-l40s.json` | the pytest result and both graded cases with all checks |

### 10.7 The runs behind this section

Five more ephemeral `openreynolds-isaacsim` apps, all stopped, all L40S. The numbers in
this section come from the last two.

| app | what it established |
| --- | --- |
| `ap-YPxen1OsjMJynqw61lHPjY` | `cases`: 63 passed, hydrostatic and sloshing both PASSED |
| `ap-zoFbrc0RIQa3bRHcpYz0dD` | `freesurface`: the hero, the GIF and the per-frame trace |

The three before them are the three failure modes in 10.5, in order: the flat frame with
no app updates, the flat grey frame inside the near clip plane, and the blown-out bed.

A `cases` job is about 2 minutes of L40S and a `freesurface` job about 2.5, so the five
together are roughly 12 minutes, **order of forty cents** at L40S list price. As in
section 9 that is arithmetic over counted runs and not a billing figure. Running
`freesurface` on an A10G instead is supported (`--gpu a10g`) and untested.


## 11. A submarine: a Newton body two-way coupled to our fluid, inside Kit

Added 2026-11-23. This closes the item that sat at the top of the unknown list: whether a
rigid body can be **two-way coupled** to `openreynolds/live/` inside a live Kit process.
It can. Everything below is from `benchmarks/live/modal_isaacsim.py::submarine`.

Deliberately on `FluidWorld` and not `FreeSurfaceWorld`, on the brief that D9's
momentum-exchange wrench was underived and a fully submerged box reading `-0.48 x rho V g`
blocked anything at a waterline.

**That premise was retracted while this was being built, and the retraction is the more
important news.** Commit `1096a13`, "NewtonCoupling pushes poses at construction; four
buoyancy findings retracted", found that `world.step()` syncs poses from the world's own
arrays and not from Newton, so a body that existed only in Newton sat at the origin for
every world-only step and voxelised into the corner of the domain. All four buoyancy
numbers came from that, including the -0.48. `4ae1c95` then measured that there is no
force defect: the momentum exchange gives buoyancy to 0.1%.

So a submarine was not, in the end, necessary to dodge a defect -- **a boat is no longer
blocked by that finding.** What a submarine still does, and what this section is for, is
answer the two-way-coupling question with the fewest moving parts.

### 11.1 The answer

**Yes, and here is the number.** A capsule hull, fineness 4, 10 mm diameter, is driven
along its own axis by a constant thrust through `control.joint_f`, and the fluid pushes
back through `world.wrench(0)`. Those two forces are computed by completely different
routes -- one from the body's kinematics in Newton, one from the populations crossing the
boundary links in our Warp kernels -- and at steady state they must cancel:

| | |
| --- | --- |
| **mean `|fluid Fx|` / thrust at cruise** | **0.815** |
| **momentum balance `|F_fluid| / (T - m a)`, 20 pre-dive frames** | **0.879** |
| axial wrench samples it is averaged over | 347, standard deviation 9.33e-6 N |
| hull voxelised into the lattice | 23,184 cells of (0.5 mm)^3, holding 23,075 to 23,200 |
| voxel volume against the analytic capsule | 2.898e-6 against 2.87979e-6 m3, **+0.63%** |
| Cd from the fluid wrench, late pre-dive | 4.41 at Re 64 |

A one-way coupling flies the submarine and fails both of those. The second row is the
better test: it does not assume terminal velocity, because `m a = T - F_fluid` holds at
every instant, and it is what caught the vacuum case in 11.3.

The run: lattice **480 x 160 x 200 = 15.36 M nodes**, dx 0.5 mm, dt 1.993 ms, 5,236
coupled steps covering 10.4 s, solver **514 MLUPS** on an L40S, render 9.2 s for 44
frames at 1280x720. The hull swims from x = 30 mm to x = 99 mm and reaches 9.1 mm/s;
the pitch pulse takes the nose from level to 45 degrees down and the hull descends at
1.7 mm/s on thrust vectoring alone.

**The Cd of 4.41 at Re 64 is reported, not graded.** `cases.run_underwater` grades a
*zero-length* capsule, which is a sphere with a published answer, and reports its
fineness-4 hull as higher than the sphere's on the same frontal area without grading it,
because no published coefficient for that shape has been read for this repository. The
same applies here and more so: this hull is self-propelled and still a little short of
terminal speed, so the number is a measurement of this run and not a claim about
capsules.

### 11.2 Three things that had to be got right, each found by being got wrong

**Neutral buoyancy is not achievable here, and that is a measurement.** The obvious
build gives the hull the water's density so `m = rho_w V`. It rose at 15 mm/s and climbed
130 mm while travelling 6.5 mm forward. Newton takes the mass from the **analytic**
capsule; `NewtonCoupling(buoyancy=True)` takes the displaced weight from the
**voxelised** one; they differ by 0.63%, and

```
weight                                2.8198e-05 N  =  5,104 x the thrust
net buoyancy from that 0.63%          1.7801e-04 N  =     32 x the thrust
volume error the thrust could absorb      0.0196%
```

Calibrating the density so that `m = rho_w V_voxel` at the start pose drove the residual
to **9.7e-10 N** and still did not hold, because the voxelisation is a function of where
the body sits relative to the grid. Measured over a run: `solid_cells` swung **22,992 to
23,192**, which is **2.4481e-04 N of buoyancy, 44 x the thrust**, and the hull sank out
of frame on the mean deficit.

So neutral buoyancy here is a cancellation of two forces five thousand times larger than
the one being measured, to a precision the voxelisation is two orders of magnitude short
of. The demo sets gravity to zero and `buoyancy=False`, which is numerically identical to
the ideal being modelled. **D9's buoyancy path is therefore not exercised by this demo;**
`cases.run_underwater` and `run_falling_sphere` still cover it.

**Gravity has to be set before the solver is built.** `SolverMuJoCo` converts the Newton
model into a MuJoCo model in its constructor, so a `set_gravity` landing afterwards
updates one and not the other. Doing it in the wrong order sent the hull to 59 m/s within
98 steps and to 1e8 m by the next frame. That reads like a coupling blow-up and is a
model the solver never saw.

**`body_qd` is linear first.** Newton's own docstring says so and `isaac.py` records it as
checked by dropping a body and reading which three components moved. Reading `[3:6]` for
the swimming speed would have put the angular velocity in every row of the table and
looked entirely plausible.

### 11.3 The moving window broke the voxelisation

`follow=0` is the validated way to keep a travelling body in frame and is what
`run_underwater` and `run_falling_sphere` use. **It did not survive here.** With
`follow=0, follow_slack=0.15` and the hull started 35 mm off centre, the window shifted
once on the first step and the voxelisation collapsed:

```
solid_cells            23,184  ->  range 0 to 2,346
world.wrench(0)[0]     exactly 0.0 on every frame
measured dv per frame  3.756e-04 m/s
T / m                  3.7545e-04 m/s
```

A body in vacuum to four significant figures. The fluid was still being solved; it simply
had nothing in it. With `follow=None` and a domain twice as long, `solid_cells` holds at
**23,092 to 23,192** across the whole run and the wrench is non-zero throughout.

**This is a bug report, not a workaround.** `benchmarks/` may not touch
`openreynolds/live/`, so the demo avoids the window rather than fixing it. What wants
looking at is `FluidWorld._maybe_shift`: it shifts `f`/`f_next`, updates `domain.origin`
and calls `_sync_bodies()`, and something in that sequence leaves `owner` unpopulated
afterwards. The cheap reproduction is any body started outside `follow_slack` of the
domain centre, which forces a large shift on step one.

### 11.4 What this demo would like changed in `openreynolds/live/`

Neither is needed for anything here to work; both would make a submarine easier to build.

1. **`FluidWorld._maybe_shift` and the voxelisation.** Section 11.3. This is the one that
   matters, because the moving window is how any travelling body is meant to be run and
   the failure is silent: a zero wrench reads as a coupling that was never wired up.
2. ~~Let `NewtonCoupling` take the displaced volume instead of recounting voxels.~~
   **Withdrawn after reading `couple.py` properly.** It already says why it uses the
   voxelised volume and not the analytic one: "using the analytic volume for the
   buoyancy and the voxelised surface for the drag does not balance and the error lands
   squarely on the quantity being graded". For a falling sphere, whose terminal velocity
   is the difference between two numbers of the same size, that is correct and an
   analytic override would break it.

   The measurement in 11.2 still stands -- for a *near-neutrally-buoyant* body the voxel
   quantisation is the dominant force error -- but the remedy is not an override. It is
   a finer lattice, or a sub-cell-weighted volume of the kind `lattice.submerged_volumes`
   now computes for the free surface, or accepting that neutral buoyancy is out of reach
   at this resolution and modelling it as the absence of both forces, which is what this
   demo does.

### 11.5 Drawing the fluid, and the renderer traps that were already written down

The wake is a **`UsdGeomPointInstancer`** over the decimated velocity field -- every third
node, where the speed is above 15% of the frame's peak, inside a box eighteen diameters
behind the hull. Section 7 ranked this as the practical option for a field that updates
every frame, and it is: positions are an in-memory attribute write with no disk round
trip, and per-instance scale and `primvars:displayColor` at `vertex` interpolation give
the speed map. The hull itself is an analytic `UsdGeom.Capsule` at the pose Newton
computed, not the voxelisation -- `solid_cells` is the right thing to *measure*, and a
16-cell-per-diameter voxel blob is a picture of the discretisation.

Three new traps, on top of the three in section 10.5:

**`render_stddev` over the whole array is fooled by a tinted background.** A frame that is
one flat teal, R 48.4 / G 138.0 / B 169.5, has a whole-array standard deviation of
**51.32**, because most of that spread is the gap between the channel means and not any
variation in the picture. It read as a healthy render and was a blank one;
`render_unique_colours` caught it with the value **1**. The guard now averages the
per-channel deviations, and the retry inside the frame grab counts distinct colours
rather than looking at a deviation.

**An empty annotator and a flat frame are different failures.** `render_unique_colours`
of **0** means the annotator returned nothing and the renderer was not ready;
**1** means it drew the background and there was no geometry in front of it. Separating
them turned a diagnostic run from ambiguous into decisive.

**A metallic surface has no diffuse term.** The hull was built at `metallic=0.65` and
rendered dark olive green: a correct render of an orange mirror under a deep blue dome.

A solver-free `stage_check` entrypoint exists because of all this. It renders a cube, a
capsule and a point instancer at 0.04 m and 0.4 m, at two near clipping planes, and
reports distinct colours for each. **All six prim-and-scale combinations draw**, which is
what proved the submarine's blank frames were never a renderer problem -- the submarine
had dived out of shot.

### 11.6 Which commit each number is from

This tree was being edited by another contributor throughout, including
`openreynolds/live/couple.py` and `lattice.py`, and `modal_isaacsim.py` mounts the
working tree rather than a commit.

**Every number in 11.1, and the artifacts, come from the last run, which started at
06:05 and therefore postdates `1096a13` (05:44) and `4ae1c95` (05:51) and predates
`531404a` (06:13, which touched only `cases.py` and `freesurface.py`).** It is current
against the coupling code as committed.

The earlier runs quoted in 11.2 and 11.3 -- the rising hull, the density calibration, the
44x buoyancy swing, the window collapse -- predate `1096a13`. Their subject is the
granularity of the voxelisation and the behaviour of `_maybe_shift`, neither of which
that commit touched, and the last run reproduces the stable-voxelisation half of it
(`solid_cells` 23,075 to 23,200 with the window off). The window finding has **not** been
re-measured against the current `couple.py`, and should be before anyone acts on it.

### 11.7 The runs, and what they cost

Twelve `submarine` runs plus one `stage_check`, all L40S, all stopped. The numbers above
are from the last.

| app | what it settled |
| --- | --- |
| `ap-Kha1Xuf3CLF5Ybhc4caUq0` | first coupled run: hull rose on the 0.63% voxel buoyancy |
| `ap-IfNwtbfPVzLhzZExEHH8kQ` | density calibrated, residual 9.7e-10 N, still sank on the 44x swing |
| `ap-xQ1hStFuROM2AHCXTZGv1H` | gravity set after the solver was built: divergence to 1e8 m |
| `ap-i5VvDMGkvAJnIGfp1l26XM` | gravity order fixed; the window had killed the voxelisation |
| `ap-OBvi4O4AnXWKibwzMgiEOy` | `stage_check`: all six prim-and-scale combinations draw |
| `ap-NIusmgDmcXcTjfT8tItgpm` | `follow=None`: wrench non-zero, `solid_cells` stable |
| `ap-Ukoy1RneOo3tEXKNjLnGFV` | the run every number in 11.1 comes from |

The four runs between the last two are framing and lighting: the metallic hull, the
wake threshold, the wake's hull mask, and three camera stand-offs picked by eye before
one was finally computed from the aperture. A `submarine` job is about 3.5 minutes of
L40S, so the thirteen together are roughly 45 minutes, **order of a dollar fifty** at
L40S list price. Arithmetic over counted runs, not a billing figure.

**On rebuilding the image:** none was needed when `openreynolds/live/lattice.py`
changed. This file mounts the repository with `add_local_dir` **without** `copy=True`,
so the working tree is uploaded at run time and is never baked into a cached layer.
Section 10.3 is the evidence: a `cases` run picked up a fix to `cases.py` minutes after
it was made, while the run before it had the bug.



### 11.8 Artifacts

In `benchmarks/live/results/isaacsim/`:

| file | what |
| --- | --- |
| `submarine_hero.png` | 1280x720, the hull pitched nose-down mid-dive with its wake |
| `submarine.gif` | 44 frames at 640x360, the whole run |
| `submarine_000.png`, `_014`, `_024`, `_043` | stills across the run |
| `submarine-l40s.json` | every number above and the per-frame trace |
| `stagecheck.json` | which prim types draw at which scale and near plane |

## 12. What is still unknown

- **`FluidWorld._maybe_shift` and the voxelisation collapse (11.3).** The most valuable
  thing on this list, because the moving window is how a travelling body is meant to run
  and the failure is silent. **Measured before `1096a13` and not since**; that commit
  changed when poses are pushed, which is adjacent enough that it should be re-run
  before anyone digs into `_maybe_shift`.
- Whether a Warp kernel **writing** through a PhysX tensor view is seen by PhysX on the
  next step. Reading at `cuda:0` with a real device pointer is established (section 6).
- Whether `isaacsim.physics.newton`, Isaac Sim 6.2's own Newton backend, can drive
  `couple.NewtonCoupling` directly, which would remove the separate Newton install.
- A **floating** body, which needs D9's body-force wrench derived first.
- Whether the free-surface solver and a coupled rigid body run together in one process.
  Each has now been done inside Kit; they have not been done at once.
- Per-instance `displayColor` on a `UsdGeomPointInstancer`: it is set by this demo and
  the wake does read as a speed map, but no run has isolated it against a flat-coloured
  control.
- Whether a marching-cubes isosurface over `world.fill` renders fast enough to replace
  the free-surface heightfield, which is what overturning waves would need.

