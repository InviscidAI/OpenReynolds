#!/usr/bin/env python3
"""Render a field over time into a directory of frames, on the instance.

The frames are the expensive half of an animation and they need the case data, so
they render here, next to it -- one PNG per write time, one camera and one colour
range for the whole sequence, so the only thing that moves between frames is the
flow.

The cheap half, turning those frames into a gif, an mp4 or a webp, does not run
here and cannot: this image has no encoder on it at all, on purpose. The frames
land in a directory named `*_frames/`, they mirror to the user's machine as they
are written, and **the encoding happens on the user's machine**, with whatever
encoder it has. Nothing here imports imageio or shells out to ffmpeg.

what the sidecar is for

`frames.json` is written beside the frames and rewritten after every one. It says
which container was asked for, at what frame rate, which frames are on disk and in
what order, how many of the expected total that is, and the colour limits they
were all drawn with. So the frames directory describes itself: the intended
animation does not live only in the command that started it, and a directory that
arrives home half-finished still says what it was going to be.

what this is built to survive

- **Run it as a job, not a synchronous command.** Rendering fifty frames takes
  minutes, and a synchronous `bash` call that long trips the exec ceiling and reads
  back as a failure even when it worked. `job_start` has no such limit.
      job_start: python3 .toolbox/animate.py /work/case --field vorticity
- **Each frame is flushed as it is written**, so the mirror can carry a
  half-finished animation home and the frames-so-far can be assembled into a
  partial gif while the rest still render -- which is exactly what someone asks
  for when they say "a gif of what you have so far". Each PNG is written under a
  dotted `.part` name and moved into place, so a frame that exists is a frame that
  is whole, and a job killed mid-screenshot leaves no half-written picture for the
  mirror to pick up.
- **Re-running is cheap.** A frame whose PNG is already on disk and non-empty is
  not drawn again, so this can be run every few minutes against a running solve
  and each run only pays for the write times that are new. `--force` redraws
  everything.
- **Colour limits are decided once and then carried.** They are recorded in
  `frames.json` and reused on the next run, because a limit recomputed halfway
  through a sequence makes the frames before and after it incomparable --
  the flicker that makes an animation unreadable. `--clim-from` chooses where the
  limits come from the first time; `last` is the default because a wake that is
  still developing has almost nothing in frame 0 and limits taken from it leave
  every later frame saturated. `--reclim` throws the recorded limits away and
  redraws.

Labels are burned into each frame -- field, simulation time, and Reynolds number
when it is known (`--reynolds`, or `--length` with the case's `nu` and inlet
velocity, or an `Re = ...` written in the case dictionaries).

    python3 animate.py /work/case --field vorticity
    python3 animate.py /work/case --field velocity --every 2 --clim -5 5
    python3 animate.py /work/case --field pressure --clim-from all --format mp4 --fps 24
    python3 animate.py /work/case --field k --streamlines --reynolds 1.65e4
    python3 animate.py /work/case --out /work/case/wake_frames --from 2.0 --to 8.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state

NORMALS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
VIEWUP = {"z": (0.0, 1.0, 0.0), "y": (0.0, 0.0, 1.0), "x": (0.0, 0.0, 1.0)}
"""Which way is up, per slice normal, so the streamwise axis stays horizontal."""

PLANE_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
"""Which world axes lie across and up the picture, per slice normal."""

VIEW_UP = {"x": (0, 0, 1), "y": (0, 0, 1), "z": (0, 1, 0)}

FIELD_ALIASES = {
    "velocity": "U",
    "speed": "U",
    "pressure": "p",
    "vorticity": "vorticity",
    "turbulent-kinetic-energy": "k",
    "eddy-viscosity": "nut",
}
"""Names a person uses for a field, mapped to what OpenFOAM wrote. Anything not
in here is passed through untouched: `--field alphaWater` has to keep working, and
a list of known names is not a list of permitted ones."""

VECTOR_FIELDS = ("U", "vorticity", "UMean", "wallShearStress")

DIVERGING = ("vorticity", "p", "p_rgh", "pMean")
"""Fields that live around zero, where a diverging map puts the sign in the colour
instead of hiding it in the middle of viridis."""

TURBULENCE = ("k", "omega", "epsilon", "nut", "nuTilda", "alphat")

CONTAINERS = {"gif": ".gif", "mp4": ".mp4", "webp": ".webp"}

SIDECAR = "frames.json"
SIDECAR_VERSION = 1

FRAMES_SUFFIX = "_frames"

DEFAULT_SEEDS = 200
"""Streamline seeds when `--streamlines` is given without a number."""

TINY = 1e-12


# -- fields ------------------------------------------------------------------------


def resolve_field(name: str) -> str:
    """The array name to look for, given what was asked for on the command line."""
    return FIELD_ALIASES.get(name, name)


def default_component(field: str) -> str:
    """Which part of a vector to colour by when nobody said.

    `vorticity` defaults to its z component rather than its magnitude: in a plane
    slice the sign is the whole point -- it is what separates the two rows of a
    von Karman street -- and a magnitude throws it away. Every other vector
    defaults to magnitude.
    """
    return "z" if field == "vorticity" else "mag"


def default_cmap(field: str) -> str:
    if field in DIVERGING:
        return "RdBu_r" if field == "vorticity" else "coolwarm"
    if field in TURBULENCE:
        return "inferno"
    return "viridis"


def scalar_name(field: str, component: str | None = None) -> str:
    """What `scalar_of` will call the array, worked out without opening the case.

    Used to fill the sidecar in on a resumed run that renders nothing and so never
    reads a mesh. It is a prediction from the known vector fields; a case with a
    vector field this file has never heard of is named as a scalar here and
    corrected the moment a frame is actually read.
    """
    component = component or default_component(field)
    if field in VECTOR_FIELDS:
        return f"{field}_mag" if component == "mag" else f"{field}_{component}"
    return field


def scalar_of(mesh, field: str, component: str | None = None):
    """A per-point scalar to colour by: a named field, or a component/magnitude of
    a vector. Vorticity is computed from U if the case did not write it.

    Returns the mesh (which may be a derived one) and the name of the array on it.
    """
    component = component or default_component(field)
    if field == "vorticity" and "vorticity" not in mesh.point_data:
        if "U" not in mesh.point_data:
            raise SystemExit("vorticity needs U, which is not in this case")
        mesh = mesh.compute_derivative(scalars="U", vorticity=True)
    if field not in mesh.point_data:
        raise SystemExit(
            f"field '{field}' is not in this case; available: "
            + ", ".join(sorted(mesh.point_data.keys()))
        )
    data = np.asarray(mesh.point_data[field])
    if data.ndim == 2 and data.shape[1] >= 3:
        if component == "mag":
            name = f"{field}_mag"
            mesh.point_data[name] = np.linalg.norm(data, axis=1)
        else:
            index = {"x": 0, "y": 1, "z": 2}[component]
            name = f"{field}_{component}"
            mesh.point_data[name] = data[:, index]
        return mesh, name
    return mesh, field


# -- which times, which frames -----------------------------------------------------


def select_times(times, *, every: int = 1, start: float | None = None,
                 end: float | None = None) -> list[float]:
    """The write times this run covers, in order.

    Filtering happens before the stride, so `--every 2` means every second time in
    the window asked for rather than every second time in the case.
    """
    chosen = [float(t) for t in times]
    if start is not None:
        chosen = [t for t in chosen if t >= start]
    if end is not None:
        chosen = [t for t in chosen if t <= end]
    return chosen[:: max(1, int(every))]


def frame_name(index: int) -> str:
    """Zero-padded, so the shell's glob order, the sidecar's order and the clock
    are the same order -- `frame_10.png` sorting before `frame_2.png` is how a
    finished animation ends up playing backwards in the middle."""
    return f"frame_{index:04d}.png"


def frame_ready(path: Path) -> bool:
    """A frame counts as done when it exists and has bytes in it. Zero-length is
    what a screenshot interrupted at exactly the wrong moment used to leave."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def plan_frames(out_dir: Path, times) -> list[dict]:
    """One row per frame this run intends to produce, in time order, each saying
    whether it is already on disk. Index comes from position in the selected
    times, so the same `--every`/`--from` window gives the same file the same name
    on every re-run and new write times append at the end."""
    plan = []
    for index, time_value in enumerate(times):
        path = Path(out_dir) / frame_name(index)
        plan.append({
            "index": index,
            "time": float(time_value),
            "name": path.name,
            "path": path,
            "ready": frame_ready(path),
        })
    return plan


def pending_frames(plan, *, force: bool = False) -> list[dict]:
    """The rows still to render: everything not already on disk, or everything."""
    return [row for row in plan if force or not row.get("ready")]


# -- colour limits -----------------------------------------------------------------


def percentile_limits(values, low: float = 2.0, high: float = 98.0):
    """A colour range from one frame's data, with the extremes trimmed.

    Raw min/max is dominated by a handful of cells at a corner or a wall, and the
    resulting range leaves the flow itself in the middle two colours of the map.
    """
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    return (float(np.percentile(data, low)), float(np.percentile(data, high)))


def merge_limits(pairs):
    """The widest range covering all of them, or None if there is nothing to cover.

    A degenerate range (a field that is uniform in the frame that was sampled, so
    low equals high) is widened rather than passed on: VTK draws a zero-width
    colour range as a single flat colour and the frame looks like a bug.
    """
    usable = []
    for pair in pairs:
        if pair is None:
            continue
        low, high = float(pair[0]), float(pair[1])
        if np.isfinite(low) and np.isfinite(high):
            usable.append((low, high))
    if not usable:
        return None
    low = min(pair[0] for pair in usable)
    high = max(pair[1] for pair in usable)
    if high <= low:
        spread = max(abs(low) * 1e-3, 1e-9)
        return (low - spread, low + spread)
    return (low, high)


def limit_times(times, mode: str) -> list[float]:
    """Which write times have to be read to decide the colour limits.

    `all` is the honest answer and costs a read per time; `last` is the cheap one
    that is usually right, because the strongest gradients in a developing flow
    are at the end of it.
    """
    ordered = [float(t) for t in times]
    if not ordered or mode == "explicit":
        return []
    if mode == "first":
        return ordered[:1]
    if mode == "last":
        return ordered[-1:]
    return ordered


def carried_limits(sidecar: dict | None, explicit=None, reclim: bool = False):
    """The colour limits to use without reading any field data: the ones given on
    the command line, else the ones a previous run recorded in the sidecar, else
    None, meaning they have to be sampled."""
    if explicit is not None:
        return (float(explicit[0]), float(explicit[1]))
    if reclim:
        return None
    recorded = (sidecar or {}).get("clim")
    if isinstance(recorded, (list, tuple)) and len(recorded) == 2:
        try:
            return (float(recorded[0]), float(recorded[1]))
        except (TypeError, ValueError):
            return None
    return None


# -- the sidecar -------------------------------------------------------------------


def output_name(frames_dir_name: str, container: str) -> str:
    """What the assembled animation should be called, from the frames directory's
    own name: `wake_frames/` -> `wake.gif`."""
    stem = frames_dir_name
    if stem.endswith(FRAMES_SUFFIX):
        stem = stem[: -len(FRAMES_SUFFIX)]
    return (stem or "animation") + CONTAINERS[container]


def contiguous_pattern(names) -> str | None:
    """`frame_%04d.png` when the ready frames are 0..n-1 with no holes, else None.

    Encoders that take a printf pattern stop at the first missing number, so
    offering the pattern when there is a hole would silently truncate the
    animation. A list of names is always in the sidecar; the pattern is a
    convenience that is only there when it is safe.
    """
    names = list(names)
    if not names:
        return None
    if names != [frame_name(index) for index in range(len(names))]:
        return None
    return "frame_%04d.png"


def stray_frames(out_dir: Path, plan) -> list[str]:
    """`frame_*.png` in the directory that this run's plan does not account for.

    An earlier run over a wider window leaves numbers behind that this one never
    reaches. They are harmless to a reader that follows the sidecar's frame list
    and fatal to one that follows the printf pattern, which walks the numbers on
    disk and would splice a frame of the old sequence onto the end of the new one.
    """
    planned = {row["name"] for row in plan}
    try:
        names = [path.name for path in Path(out_dir).glob("frame_*.png") if path.is_file()]
    except OSError:
        return []
    return sorted(name for name in names if name not in planned)


RENDER_IDENTITY = ("field", "component", "normal", "cmap", "streamlines", "size")
"""What the PNGs in a frames directory are a picture of.

Every one of these changes what a frame looks like, so a run whose settings differ
from the ones recorded in the sidecar is not a continuation of that sequence --
its frames cannot be mixed with the ones already there.
"""


def same_render(sidecar: dict | None, settings: dict) -> bool:
    """Whether the frames a previous run left are pictures of the same thing.

    The default frames directory is named after the case and nothing else, so
    `--field pressure` re-run over a vorticity sequence lands in the same
    directory. Without this the resume logic sees four finished PNGs, draws
    nothing, and rewrites the sidecar to say `pressure` over frames that are
    vorticity -- a directory that lies about what is in it.

    A key the sidecar does not carry is not a disagreement: an empty dict (no
    previous run, or one written before a setting was recorded) matches, so an
    existing sequence is never redrawn just because the sidecar is older.
    """
    recorded = sidecar or {}
    for key in RENDER_IDENTITY:
        if key not in recorded:
            continue
        before, now = recorded[key], settings.get(key)
        if key == "streamlines":
            before, now = int(before or 0), int(now or 0)
        elif key == "size":
            before = [int(v) for v in (before or [])]
            now = [int(v) for v in (now or [])]
        if before != now:
            return False
    return True


def build_sidecar(plan, *, frames_dir_name: str = "", container: str = "gif",
                  fps: float = 10.0, field: str = "", scalar: str = "",
                  clim=None, case: str = "", reynolds: float | None = None,
                  streamlines: int = 0, normal: str = "z", cmap: str = "",
                  component: str = "", window=(1000, 750), stray=()) -> dict:
    """The contents of `frames.json`: everything the encoding machine needs and
    nothing it has to guess."""
    ready = [row for row in plan if row.get("ready")]
    names = [row["name"] for row in ready]
    stray = sorted(stray)
    data = {
        "version": SIDECAR_VERSION,
        "container": container,
        "extension": CONTAINERS.get(container, ""),
        "output": output_name(frames_dir_name, container) if container in CONTAINERS else "",
        "fps": float(fps),
        "frames": names,
        "frame_times": [row["time"] for row in ready],
        "pattern": None if stray else contiguous_pattern(names),
        "stray": stray,
        "expected": len(plan),
        "complete": len(names),
        "partial": len(names) < len(plan),
        "field": field,
        "component": component,
        "scalar": scalar,
        "cmap": cmap,
        "normal": normal,
        "size": [int(window[0]), int(window[1])],
        "clim": [float(clim[0]), float(clim[1])] if clim is not None else None,
        "streamlines": int(streamlines),
        "case": case,
        "reynolds": float(reynolds) if reynolds is not None else None,
        "encoded_by": "the user's machine; this image has no encoder",
        "written_at": study_state.now_iso(),
    }
    if container == "gif":
        data["loop"] = 0
    return data


def write_sidecar(out_dir: Path, data: dict) -> Path:
    """Written to a temporary file and moved into place, for the same reason the
    phase table is: a reader that arrives mid-write must see the previous sidecar,
    not half of the next one."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SIDECAR
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_sidecar(out_dir: Path) -> dict:
    """The sidecar a previous run left, or an empty dict. Never raises: a sidecar
    that will not parse is a reason to write a new one, not to stop."""
    path = Path(out_dir) / SIDECAR
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# -- labels ------------------------------------------------------------------------


def frame_label(time_value: float, scalar: str, reynolds: float | None = None,
                case: str = "", note: str = "") -> str:
    """The text burned into a frame.

    Burned in rather than written to the side, because frames get pulled out of
    the directory one at a time and pasted into messages, and a picture of a wake
    with no time on it cannot be placed in the sequence it came from.
    """
    lines = []
    if case:
        lines.append(case)
    if scalar:
        lines.append(scalar)
    lines.append(f"t = {time_value:g} s")
    if reynolds:
        lines.append(f"Re = {reynolds:.3g}")
    if note:
        lines.append(note)
    return "\n".join(lines)


# -- the camera --------------------------------------------------------------------


def camera_setup(bounds, normal: str, window=(1000, 750), padding: float = 0.05) -> dict:
    """One parallel-projection camera for the whole sequence, from the slice's
    bounds.

    Computed once and set explicitly on every frame rather than re-fitted per
    frame: a fit that depends on what is in view drifts as the flow develops, and
    a wake that appears to zoom is a wake nobody can measure off the screen.
    """
    low = np.array([float(bounds[0]), float(bounds[2]), float(bounds[4])])
    high = np.array([float(bounds[1]), float(bounds[3]), float(bounds[5])])
    center = (low + high) / 2.0
    extent = high - low

    across, up_axis = PLANE_AXES[normal]
    half_width = max(extent[across] / 2.0, TINY)
    half_height = max(extent[up_axis] / 2.0, TINY)
    aspect = (window[0] / window[1]) if window[1] else 1.0
    # parallel_scale is half the visible height, so a wide slice is fitted by the
    # width it needs divided by the window's aspect.
    scale = max(half_height, half_width / aspect) * (1.0 + padding)

    axis = np.array(NORMALS[normal], dtype=float)
    distance = max(float(np.linalg.norm(extent)), TINY) * 2.0
    return {
        "position": tuple(float(v) for v in (center + axis * distance)),
        "focal_point": tuple(float(v) for v in center),
        "up": VIEW_UP[normal],
        "parallel_scale": float(scale),
    }


def seed_stride(n_points: int, count: int) -> int:
    """Take every Nth point of the slice as a streamline seed."""
    if count <= 0 or n_points <= 0:
        return 1
    return max(1, n_points // count)


# -- Reynolds number ---------------------------------------------------------------

DECLARED_RE = re.compile(
    r"\bRe(?:ynolds)?(?:\s*number)?\s*[=:]\s*([0-9][0-9_.]*(?:[eE][-+]?[0-9]+)?)"
)

NU_ENTRY = re.compile(
    r"^\s*nu\s+(?:nu\s+)?(?:\[[^\]]*\]\s*)?([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*;",
    re.MULTILINE,
)

UNIFORM_VECTOR = re.compile(
    r"uniform\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
)


def find_declared_reynolds(text: str) -> float | None:
    """A Reynolds number somebody wrote down, in a comment or an entry.

    Worth looking for before computing one: a case set up from a benchmark has the
    number it was set up to match written in it, and that is the number the
    picture should carry -- not one recomputed from a length scale this script
    guessed at.
    """
    match = DECLARED_RE.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def parse_nu(text: str) -> float | None:
    """Kinematic viscosity from transportProperties, old or new spelling."""
    match = NU_ENTRY.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def parse_reference_velocity(text: str) -> float | None:
    """The largest uniform vector magnitude in a `0/U`.

    Which is the inlet in every case shaped like a flow past something: the
    internal field and the walls are zero and the inlet is not. It is a heuristic
    and it is only used to label a picture.
    """
    best = 0.0
    for match in UNIFORM_VECTOR.finditer(text or ""):
        try:
            vector = [float(part) for part in match.groups()]
        except ValueError:
            continue
        best = max(best, float(np.linalg.norm(vector)))
    return best if best > 0 else None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def reynolds_from_case(case: Path, length: float | None = None) -> float | None:
    """A Reynolds number for the label, or None. Never raises and never guesses a
    length scale: without `--length` only a number already written in the case is
    returned."""
    case = Path(case)
    properties = [
        case / "constant" / "transportProperties",
        case / "constant" / "physicalProperties",
        case / "constant" / "momentumTransport",
        case / "system" / "controlDict",
    ]
    for path in properties:
        declared = find_declared_reynolds(_read(path))
        if declared:
            return declared

    if not length or length <= 0:
        return None
    viscosity = None
    for path in properties:
        viscosity = viscosity or parse_nu(_read(path))
    velocity = None
    for path in (case / "0" / "U", case / "0.orig" / "U", case / "0.org" / "U"):
        velocity = velocity or parse_reference_velocity(_read(path))
    if not viscosity or not velocity:
        return None
    return float(velocity) * float(length) / float(viscosity)


# -- rendering ---------------------------------------------------------------------


def _pyvista():
    """pyvista is imported here and not at the top of the file.

    Everything above is ordinary python -- time selection, resume, colour limits,
    the sidecar -- and it is the half that rots quietly, so it is tested on
    machines that have no VTK on them. Importing pyvista at module level would put
    the whole file out of their reach.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


def open_reader(case: Path):
    pv = _pyvista()
    case = Path(case)
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    reader.cell_to_point_creation = True
    return reader


def internal_mesh(block):
    """The internalMesh block, whatever nesting this case produced."""
    pv = _pyvista()

    def walk(item):
        if isinstance(item, pv.MultiBlock):
            keys = item.keys() or []
            if "internalMesh" in keys:
                return item["internalMesh"]
            for child in item:
                if child is None:
                    continue
                found = walk(child)
                if found is not None:
                    return found
            return None
        return item

    return walk(block)


def slice_at(mesh, normal: str):
    return mesh.slice(normal=NORMALS[normal], origin=np.array(mesh.center))


def streamlines_over(mesh, cut, count: int, vectors: str = "U"):
    """Streamlines seeded from the points of the slice, or None.

    Seeded from the cut rather than from a sphere in the middle of the domain so
    the lines lie in the plane being drawn. Any failure returns None: a frame
    without its streamlines is worth more than a sequence that stopped.
    """
    if count <= 0 or vectors not in mesh.point_data:
        return None
    pv = _pyvista()
    try:
        points = np.asarray(cut.points)
        if points.size == 0:
            return None
        seeds = pv.PolyData(points[:: seed_stride(len(points), count)])
        lines = mesh.streamlines_from_source(
            seeds, vectors=vectors, integration_direction="both"
        )
    except Exception:
        return None
    if lines is None or lines.n_points == 0:
        return None
    return lines


def render_frame(mesh, scalar: str, out: Path, *, normal: str = "z", clim=None,
                 cmap: str = "viridis", camera: dict | None = None, label: str = "",
                 streamlines: int = 0, window=(1000, 750)) -> Path:
    """One frame, written under a dotted temporary name and moved into place."""
    pv = _pyvista()
    cut = slice_at(mesh, normal)
    plotter = pv.Plotter(off_screen=True, window_size=list(window))
    plotter.add_mesh(cut, scalars=scalar, clim=clim, cmap=cmap, show_edges=False,
                     show_scalar_bar=False)
    # Under the picture, not across it: with a parallel projection filling the frame,
    # pyvista's default put the tick labels on top of the flow.
    plotter.add_scalar_bar(title=scalar, n_labels=5, vertical=False,
                           position_x=0.15, position_y=0.02, width=0.7, height=0.06,
                           title_font_size=14, label_font_size=12)
    lines = streamlines_over(mesh, cut, streamlines)
    if lines is not None:
        plotter.add_mesh(lines, color="black", line_width=1.0, opacity=0.5)
    plotter.enable_parallel_projection()
    if camera is not None:
        plotter.camera.position = camera["position"]
        plotter.camera.focal_point = camera["focal_point"]
        plotter.camera.up = camera["up"]
        plotter.camera.parallel_scale = camera["parallel_scale"]
    else:
        # With no up vector VTK picks one, and for a z-normal slice it picks +x --
        # every frame of a left-to-right flow comes out a quarter turn round, which
        # is believed rather than checked. The first frame's camera is then reused
        # for the rest, so one wrong frame is a wrong animation.
        plotter.view_vector(NORMALS[normal], viewup=VIEWUP.get(normal, (0.0, 1.0, 0.0)))
    if label:
        plotter.add_text(label, font_size=10, position="upper_left")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Dotted, and still `.png`: pyvista picks its writer off the extension and
    # refuses one it does not know, while the leading dot keeps the half-written
    # file out of the `frame_*.png` the mirror and the encoder look at.
    tmp = out.with_name(f".{out.stem}.part.png")
    plotter.screenshot(str(tmp))
    plotter.close()
    os.replace(tmp, out)
    return out


# -- state -------------------------------------------------------------------------


def _note_state(action, *args, **kwargs):
    """State writing must never take the render down with it."""
    try:
        return action(*args, **kwargs)
    except Exception as exc:  # a lost manifest row is not a lost animation
        print(f"  (study state not updated: {exc})", flush=True)
        return None


# -- the command line --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path)
    parser.add_argument("--field", default="vorticity",
                        help="Field to colour by: velocity, pressure, vorticity, k, "
                             "omega, epsilon, nut, or any array name in the case.")
    parser.add_argument("--component", default=None, choices=["mag", "x", "y", "z"],
                        help="Part of a vector field (default: z for vorticity, "
                             "magnitude otherwise).")
    parser.add_argument("--normal", default="z", choices=sorted(NORMALS))
    parser.add_argument("--every", type=int, default=1, help="Render every Nth write time.")
    parser.add_argument("--from", dest="start", type=float, default=None,
                        help="Skip times before this.")
    parser.add_argument("--to", dest="end", type=float, default=None,
                        help="Skip times after this.")
    parser.add_argument("--clim", type=float, nargs=2, default=None, help="Fixed colour range.")
    parser.add_argument("--clim-from", dest="clim_from", default=None,
                        choices=["first", "last", "all", "explicit"],
                        help="Where the colour limits come from (default: last).")
    parser.add_argument("--clim-percentile", dest="clim_percentile", type=float, nargs=2,
                        default=(2.0, 98.0), help="Percentiles trimmed when sampling limits.")
    parser.add_argument("--reclim", action="store_true",
                        help="Recompute the colour limits and redraw every frame.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Frames directory (default <case>_frames).")
    parser.add_argument("--format", dest="container", default="gif", choices=sorted(CONTAINERS),
                        help="Container the frames are meant to become, recorded in "
                             "frames.json for the machine that encodes them.")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate for that encode.")
    parser.add_argument("--streamlines", nargs="?", type=int, const=DEFAULT_SEEDS, default=0,
                        help="Overlay streamlines, optionally with a seed count.")
    parser.add_argument("--reynolds", type=float, default=None,
                        help="Reynolds number for the labels (default: read from the case).")
    parser.add_argument("--length", type=float, default=None,
                        help="Length scale, so Re can be computed from nu and the inlet U.")
    parser.add_argument("--cmap", default=None, help="Colour map (default: per field).")
    parser.add_argument("--size", type=int, nargs=2, default=(1000, 750), metavar=("W", "H"))
    parser.add_argument("--label", default="", help="Extra line burned into every frame.")
    parser.add_argument("--force", action="store_true",
                        help="Redraw frames that are already on disk.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    case = args.case
    out = args.out or case.parent / f"{case.name}{FRAMES_SUFFIX}"
    window = tuple(args.size)

    mode = args.clim_from or ("explicit" if args.clim else "last")
    if mode == "explicit" and not args.clim:
        raise SystemExit("--clim-from explicit needs --clim LOW HIGH")

    field = resolve_field(args.field)
    component = args.component or default_component(field)
    cmap = args.cmap or default_cmap(field)
    reynolds = args.reynolds
    if reynolds is None:
        reynolds = reynolds_from_case(case, args.length)

    reader = open_reader(case)
    times = select_times(reader.time_values, every=args.every, start=args.start, end=args.end)
    if len(times) < 2:
        raise SystemExit(f"need at least 2 write times to animate; found {len(times)}")

    settings = {
        "field": field, "component": component, "normal": args.normal, "cmap": cmap,
        "streamlines": int(args.streamlines), "size": [int(window[0]), int(window[1])],
    }
    previous = read_sidecar(out)
    stale = bool(previous) and not same_render(previous, settings)

    plan = plan_frames(out, times)
    todo = pending_frames(plan, force=args.force or args.reclim or stale)
    already = len(plan) - len(todo)

    clim = carried_limits({} if stale else previous, args.clim, args.reclim)
    scalar = scalar_name(field, component)
    if clim is None:
        pairs = []
        for time_value in limit_times(times, mode):
            reader.set_active_time_value(time_value)
            mesh, scalar = scalar_of(internal_mesh(reader.read()), field, component)
            pairs.append(percentile_limits(mesh.point_data[scalar], *args.clim_percentile))
        clim = merge_limits(pairs)
        source = f"sampled from the {mode} frame" if mode in ("first", "last") else \
            f"sampled from all {len(times)} frames"
    else:
        source = "given" if args.clim else "carried from " + SIDECAR

    def sidecar_now() -> dict:
        return build_sidecar(
            plan, frames_dir_name=out.name, container=args.container, fps=args.fps,
            field=field, scalar=scalar, clim=clim, case=case.name,
            reynolds=reynolds, streamlines=args.streamlines, normal=args.normal, cmap=cmap,
            component=component, window=window, stray=stray_frames(out, plan),
        )

    print(f"{len(plan)} frames -> {out}", flush=True)
    if stale:
        print(f"  the frames already here were drawn with different settings "
              f"({previous.get('scalar') or previous.get('field')}, "
              f"normal {previous.get('normal')}); redrawing all of them", flush=True)
    print(f"  {already} already on disk, {len(todo)} to render", flush=True)
    if clim is not None:
        print(f"  colour limits {clim[0]:.4g} .. {clim[1]:.4g}  ({source})", flush=True)
    else:
        print("  colour limits: nothing finite to sample; each frame scales itself",
              flush=True)
    if reynolds:
        print(f"  Re = {reynolds:.4g}", flush=True)
    write_sidecar(out, sidecar_now())

    _note_state(study_state.set_phase, "animate", "running", root=case, case=case.name,
                note=f"{field} -> {out.name}")

    camera = None
    try:
        for position, row in enumerate(todo, start=1):
            reader.set_active_time_value(row["time"])
            mesh, scalar = scalar_of(internal_mesh(reader.read()), field, component)
            if camera is None:
                camera = camera_setup(slice_at(mesh, args.normal).bounds, args.normal,
                                      window=window, padding=0.05)
            render_frame(
                mesh, scalar, row["path"], normal=args.normal, clim=clim, cmap=cmap,
                camera=camera, streamlines=args.streamlines, window=window,
                label=frame_label(row["time"], scalar, reynolds, case=case.name,
                                  note=args.label),
            )
            row["ready"] = frame_ready(row["path"])
            # Rewritten every frame: the directory is never in a state that
            # misreports how much of the animation is in it.
            write_sidecar(out, sidecar_now())
            print(f"  frame {position}/{len(todo)}  {row['name']}  t={row['time']:g}", flush=True)
    except BaseException as exc:
        write_sidecar(out, sidecar_now())
        _note_state(study_state.set_phase, "animate", "failed", root=case, case=case.name,
                    note=str(exc)[:200])
        raise

    data = sidecar_now()
    write_sidecar(out, data)
    _note_state(
        study_state.record, "animation", out, root=case, case=case.name,
        label=f"{data['scalar']} over time", field=field, scalar=data["scalar"],
        container=args.container, fps=args.fps, frames=data["complete"],
        expected=data["expected"], partial=data["partial"], clim=data["clim"],
        reynolds=reynolds, streamlines=bool(args.streamlines),
    )
    _note_state(study_state.set_phase, "animate",
                "running" if data["partial"] else "done", root=case, case=case.name,
                note=f"{data['complete']}/{data['expected']} frames in {out.name}")

    print(f"done: {data['complete']}/{data['expected']} frames in {out}", flush=True)
    if data["stray"]:
        print(f"  {len(data['stray'])} other frame(s) here are left over from an earlier "
              f"window; frames.json lists them under 'stray' and withholds the printf "
              f"pattern so they cannot be encoded into this sequence", flush=True)
    print(f"  {SIDECAR} names the container ({args.container}), the frame rate "
          f"({args.fps:g} fps) and the frame order; the encode runs on your machine",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
