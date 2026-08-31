#!/usr/bin/env python3
"""Pictures made to be shown, not just to be checked.

Everything else in this toolbox draws to answer a question -- is the mesh fit, did the
residuals fall, where is the wake -- and it draws accordingly: white background, default
colours, a scalar bar across the middle. That is right for a working render and wrong for
the one picture that ends up in a report, a post, or a case study. A study can be
excellent and still leave nothing anybody would look at twice.

So this is the other end. Same data, same case, no new solve: dark ground, lit surfaces,
a colour map chosen per quantity rather than per habit, and the three views that actually
show a flow --

- **vortices**    Q-criterion isosurface, coloured by speed. The one that reads as "this
                  is fluid dynamics" to somebody who does not read contour plots: tip
                  vortices off a propeller, the horseshoe round a sail, the shed street
                  behind a bluff body.
- **streamlines** tubes seeded upstream and integrated through the field, coloured by
                  speed. Shows where the flow goes, which a slice cannot.
- **surface**     the body itself under its own pressure field, which is what a person
                  recognises as the thing they uploaded.
- **slice**       a cut plane, styled the same way, for when the interior is the point.

and two ways to make them move, both of which work on a **steady** solution, because most
studies are steady and a still picture of a steady field is where the story usually stops:

- `--orbit N`     N frames of the camera swinging around the subject. The flow is fixed;
                  the eye moves. This is what makes a 3D result legible as 3D.
- `--sweep N`     N frames of a slice plane travelling through the domain.

Frames land in `<out>/<name>_frames/` with a `frames.json` beside them, which is the
contract `openreynolds video` and the harness's `Gallery` already read -- so the encoding
happens on the user's machine, where there is an encoder, and this image stays free of
one. Nothing here imports imageio or shells out to ffmpeg.

    python3 showcase.py /work/case --scene vortices
    python3 showcase.py /work/case --scene vortices --orbit 72 --fps 20
    python3 showcase.py /work/case --scene streamlines --seed-plane inlet
    python3 showcase.py /work/case --scene surface --patch hull
    python3 showcase.py /work/case --all            # one of each, plus an orbit

It degrades rather than fails: a scene that cannot be built says why and the others still
run, because half a gallery is worth more than a traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

BACKGROUND = "#0b0e12"
"""Near-black with a little blue in it. A pure black ground makes an isosurface look
cut out; this one lets the edges sit."""

MAPS = {
    "speed": "turbo",
    "p": "coolwarm",
    "T": "inferno",
    "vorticity": "plasma",
}
"""Colour map per quantity rather than one default everywhere. Speed reads best on a
full-range map, pressure wants a diverging one because it has a meaningful zero, and
temperature reads as heat on inferno to anybody who has seen a thermal camera."""

WINDOW = (1600, 1000)
FRAME_WINDOW = (1100, 720)
"""Frames are smaller than stills: an orbit is sixty of them and they become one file."""


# -- reading the case ------------------------------------------------------------------


def open_case(case: Path, time: float | None):
    """The case at one time, plus what times it has. Reconstructed data only.

    A decomposed case that has not been reconstructed reports a single time of zero, so
    a picture drawn from it is the initial condition with a confident label on it -- the
    same trap `web_mesh.py` had. If that is what we find and `processor0` exists, say so
    rather than drawing it."""
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    reader.cell_to_point_creation = True
    times = [float(t) for t in reader.time_values]
    if (case / "processor0").is_dir() and (not times or times == [0.0]):
        raise SystemExit(
            "this case is decomposed and not reconstructed: the only time available is 0, "
            "which is the initial condition. Run reconstructPar first."
        )
    chosen = times[-1] if times else 0.0
    if time is not None and times:
        chosen = min(times, key=lambda t: abs(t - time))
    if times:
        reader.set_active_time_value(chosen)
    return reader, reader.read(), chosen, times


def internal(block):
    """The internalMesh, whatever nesting this case produced."""
    if isinstance(block, pv.MultiBlock):
        keys = block.keys() or []
        if "internalMesh" in keys:
            return block["internalMesh"]
        for item in block:
            if item is None:
                continue
            found = internal(item)
            if found is not None:
                return found
        return None
    return block


def patches(block) -> dict:
    """`{patch name: surface}` for every boundary patch the reader exposed."""
    out: dict = {}

    def walk(node, name=""):
        if isinstance(node, pv.MultiBlock):
            for key in node.keys() or []:
                try:
                    walk(node[key], key)
                except (KeyError, TypeError):
                    continue
        elif node is not None and getattr(node, "n_cells", 0):
            out.setdefault(name, node)

    if isinstance(block, pv.MultiBlock) and "boundary" in (block.keys() or []):
        walk(block["boundary"])
    return out


def body_patch(block, wanted: str | None):
    """The patch that is most likely to be the thing the study is about.

    Named if named. Otherwise the smallest patch that is not obviously part of the
    domain box -- the same reasoning `first_look.py` uses to pick a region of interest,
    and for the same reason: the object is small and the tunnel is large."""
    found = patches(block)
    if not found:
        return None, ""
    if wanted:
        for name, surface in found.items():
            if name.lower() == wanted.lower():
                return surface, name
        raise SystemExit(f"no patch called {wanted!r}; this case has: {', '.join(sorted(found))}")
    box = ("inlet", "outlet", "top", "bottom", "side", "front", "back", "wall",
           "frontandback", "symmetry", "farfield", "atmosphere", "upperwall", "lowerwall")
    candidates = {n: s for n, s in found.items()
                  if not any(word in n.lower() for word in box)}
    pool = candidates or found
    name = min(pool, key=lambda n: pool[n].n_cells)
    return pool[name], name


def speed_of(mesh):
    """`|U|` as a point array, added if it is not there. Returns the array name."""
    for key in ("U", "u"):
        if key in mesh.point_data:
            data = np.asarray(mesh.point_data[key])
            if data.ndim > 1 and data.shape[1] >= 3:
                mesh["speed"] = np.linalg.norm(data[:, :3], axis=1)
                return "speed"
    for key in ("U", "u"):
        if key in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()
            return speed_of(mesh)
    return ""


# -- the look --------------------------------------------------------------------------


def stage(size=WINDOW) -> pv.Plotter:
    """A plotter dressed for showing rather than for checking."""
    plotter = pv.Plotter(off_screen=True, window_size=list(size), lighting="three lights")
    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing("ssaa")
    return plotter


def coloured(plotter, mesh, scalars: str | None, cmap: str, bar: str = "", **kw):
    """Add a mesh coloured by a named array, with its colour bar made in the same call.

    Both halves of this are the fix for one bug. A streamtube inherits every array the
    field had, and its *active* scalars are whichever OpenFOAM left active -- `p` here --
    so the array has to be named and made active explicitly. And the bar has to be built
    by the same `add_mesh` that owns the colours: added separately it binds to whatever
    mapper it likes, which produced a picture correctly coloured by speed under a bar
    labelled "speed m/s" reading -20.3 to 10.9, for a flow whose fastest point is 3.27.
    A speed cannot be negative, which is the only reason anyone noticed.
    """
    if not (scalars and scalars in mesh.point_data):
        return plotter.add_mesh(mesh, show_scalar_bar=False, **kw)
    # Strip every other array off first. Naming the scalars and setting them active was
    # not enough: a streamtube arrives carrying U, k, nut, omega, p, yPlus, Vorticity and
    # more, and the colour bar kept binding to `p` -- the picture was coloured correctly
    # by speed and captioned "speed m/s" over a bar reading -20.3 to 10.9, for a flow
    # whose fastest point is 3.27 m/s. With one array present there is nothing else for
    # anything to bind to.
    values = np.asarray(mesh.point_data[scalars]).copy()
    mesh.clear_data()
    mesh[scalars] = values
    mesh.set_active_scalars(scalars)
    if not bar:
        return plotter.add_mesh(mesh, scalars=scalars, cmap=cmap,
                                show_scalar_bar=False, **kw)
    kw.setdefault("clim", [float(np.nanmin(values)), float(np.nanmax(values))])
    actor = plotter.add_mesh(
        mesh, scalars=scalars, cmap=cmap, show_scalar_bar=True,
        scalar_bar_args=dict(
            title=" ", vertical=False, position_x=0.31, position_y=0.045,
            width=0.38, height=0.030, label_font_size=12,
            color="#dfe6ec", n_labels=5, fmt="%.3g",
        ), **kw)
    # The bar carries no title of its own: VTK lays a horizontal bar's title across its
    # own tick labels at these proportions. The quantity is named just above instead.
    plotter.add_text(bar, position=(0.31, 0.088), viewport=True,
                     font_size=10, color="#8fa0ad")
    return actor


def caption(plotter, text: str, sub: str = "") -> None:
    plotter.add_text(text, position=(0.032, 0.925), viewport=True,
                     font_size=15, color="#f2f6f9")
    if sub:
        plotter.add_text(sub, position=(0.032, 0.888), viewport=True,
                         font_size=10, color="#8fa0ad")


def save(plotter: pv.Plotter, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    return out


# -- the scenes ------------------------------------------------------------------------


RESAMPLE_BUDGET = 6_000_000
"""Points in the uniform grid a field is resampled onto before it is contoured.

This changes how the field is *drawn*, not what is in it. Marching cubes on a coarse
unstructured mesh produces facets the size of the cells -- a propeller's tip vortex came
out as a handful of chunky polygons, which reads as a broken renderer rather than as a
coarse solve. Interpolating the same field onto a fine regular lattice first and
contouring that gives a smooth surface of the same data.

It adds no information and it is not allowed to pretend otherwise: a resampled figure
says so in its caption. What it removes is an artefact of the *discretisation of the
picture*, which is not physics and should not be in the picture."""


def refined(mesh, budget: int = RESAMPLE_BUDGET):
    """The field on a uniform grid whose spacing follows the domain's own proportions.

    One spacing for all three axes, chosen so the whole box lands near `budget` points.
    A fixed count per axis would give a propeller -- 43 mm x 254 mm x 12 mm -- a lattice
    twenty times finer across the blade than along it."""
    lo = np.array(mesh.bounds[::2]); hi = np.array(mesh.bounds[1::2])
    span = np.maximum(hi - lo, 1e-12)
    step = float((span.prod() / max(budget, 1)) ** (1.0 / 3.0))
    dims = np.maximum(2, (span / step).astype(int) + 1)
    grid = pv.ImageData(dimensions=tuple(int(d) for d in dims),
                        spacing=(step, step, step), origin=tuple(lo))
    return grid.sample(mesh), int(np.prod(dims))


def smoothed(surface, iterations: int = 24):
    """Taubin smoothing: takes the stair-steps off an isosurface without shrinking it.

    Laplacian smoothing would also work and would slowly deflate the structure, which on
    a vortex core is exactly the dimension being looked at."""
    if surface.n_points == 0 or iterations <= 0:
        return surface
    try:
        return surface.smooth_taubin(n_iter=iterations, pass_band=0.05)
    except Exception:  # noqa: BLE001 - an older VTK; the unsmoothed surface is still true
        return surface


def q_field(mesh):
    """Q-criterion as a point array, or None if the velocity gradient cannot be had."""
    name = "U" if "U" in mesh.point_data else ("u" if "u" in mesh.point_data else "")
    if not name:
        return None
    try:
        derived = mesh.compute_derivative(scalars=name, qcriterion=True)
    except Exception:  # noqa: BLE001 - an older VTK, or a mesh it will not differentiate
        return None
    key = next((k for k in derived.point_data.keys() if "qcrit" in k.lower()), "")
    if not key:
        return None
    mesh["Q"] = np.asarray(derived.point_data[key])
    return mesh


def q_level(mesh, fraction: float) -> float:
    """Where to put the isosurface.

    A Q-criterion level has no natural value -- it is a threshold on a quantity whose
    scale is set by the flow -- so it is chosen as a high percentile of the positive
    values in this field. Too low and the whole domain fills with noise; too high and
    the picture is empty. The percentile is the knob worth exposing, not the raw level."""
    q = np.asarray(mesh["Q"])
    positive = q[q > 0]
    if positive.size == 0:
        return 0.0
    return float(np.percentile(positive, fraction))


def add_body(plotter, context) -> None:
    """The object, in the picture, behind whatever the scene is about.

    A Q-criterion isosurface floating in black space is not a picture of anything: the
    structures only mean something next to the thing that shed them, and without a body
    the camera has nothing sensible to frame on either."""
    if context is None:
        return
    # Stripped of its arrays before it goes in. The hull arrives carrying `p`, and even
    # added with a flat colour and no bar of its own it was what the scene's colour bar
    # bound to -- so a streamline picture coloured by speed carried a bar showing the
    # hull's pressure range. It is drawn as a solid object; it needs no data at all.
    body = context.copy()
    body.clear_data()
    plotter.add_mesh(body, color="#5a6672", smooth_shading=True,
                     specular=0.12, specular_power=8, show_scalar_bar=False)


def scene_vortices(mesh, out: Path, title: str, percentile: float = 99.0,
                   frames: int = 0, fps: float = 20.0, sweep: int = 0,
                   context=None, refine: int = 0, smooth: int = 24) -> list[Path]:
    """Q-criterion isosurface, coloured by speed, with the body behind it."""
    note = ""
    if q_field(mesh) is None:
        print("  vortices: no velocity gradient available in this case")
        return []
    # The threshold comes from the ORIGINAL cells and the resampling only changes how
    # the surface at that threshold is drawn. Taking the percentile on the resampled
    # field instead moves the threshold: the uniform grid fills the far field with
    # points where Q is ~0, which drags the distribution down -- on the propeller the
    # 96th percentile fell from 7,170 to 37.6 and the isosurface grew until it swallowed
    # the camera. Same number, smoother picture, is the whole point.
    level = q_level(mesh, percentile)
    if refine:
        mesh, points = refined(mesh, refine)
        if q_field(mesh) is None:
            print("  vortices: the resampled grid carries no velocity to differentiate")
            return []
        note = f"; field resampled to a {points:,}-point uniform grid for rendering"
    if level <= 0:
        print("  vortices: no positive Q anywhere -- nothing is rotating in this field")
        return []
    surface = smoothed(mesh.contour(isosurfaces=[level], scalars="Q"), smooth)
    if surface.n_points == 0:
        print(f"  vortices: the isosurface at Q={level:.4g} is empty; try a lower --percentile")
        return []
    speed = speed_of(mesh)
    if speed:
        surface = surface.sample(mesh)

    def draw(plotter):
        add_body(plotter, context)
        coloured(plotter, surface, speed, MAPS["speed"], bar="speed  m/s",
                 smooth_shading=True, specular=0.35, specular_power=18,
                 **({} if speed else {"color": "#7fd4ff"}))
        caption(plotter, title, f"Q-criterion isosurface at the {percentile:g}th percentile"
                                f" of positive Q  (Q = {level:.3g} s^-2){note}")

    # Framed on the body, not the isosurface: at a high percentile the surface is a few
    # scattered fragments and framing on those points the camera at empty space.
    return _still_or_frames(draw, context if context is not None else surface,
                            out, "vortices", frames, fps, sweep, mesh)


def scene_streamlines(mesh, out: Path, title: str, count: int = 900,
                      frames: int = 0, fps: float = 20.0, context=None) -> list[Path]:
    """Streamtubes seeded across the inflow, coloured by speed."""
    speed = speed_of(mesh)
    vectors = "U" if "U" in mesh.point_data else ("u" if "u" in mesh.point_data else "")
    if not vectors:
        print("  streamlines: no velocity field on the points")
        return []
    lo, hi = np.array(mesh.bounds[::2]), np.array(mesh.bounds[1::2])
    span = hi - lo
    # Seed on a disc a little inside the inlet face, sized to the cross-section: seeding
    # on the face itself puts half the seeds in the boundary cell and they go nowhere.
    centre = (lo + hi) / 2.0
    centre[0] = lo[0] + 0.04 * span[0]
    # Seeded to the size of the BODY, not the domain. A disc spanning the whole
    # cross-section puts most of its seeds in undisturbed freestream, and they integrate
    # into a solid curtain of straight parallel tubes at one colour -- 1200 of them
    # filling the frame and hiding the hull behind them. What is worth seeing is the
    # flow that goes near the thing, so the disc is a little wider than the thing.
    if context is not None:
        blo = np.array(context.bounds[::2]); bhi = np.array(context.bounds[1::2])
        radius = 1.7 * 0.5 * max(float((bhi - blo)[1]), float((bhi - blo)[2]))
        centre[1], centre[2] = ((blo + bhi) / 2.0)[1], ((blo + bhi) / 2.0)[2]
    else:
        radius = 0.42 * max(span[1], span[2])
    common = dict(vectors=vectors, source_center=tuple(centre),
                  source_radius=float(radius),
                  n_points=count, integration_direction="forward")
    reach = float(4.0 * span[0])
    lines = None
    # pyvista renamed this between versions -- `max_time` up to 0.47, `max_length` from
    # 0.48, and the old name raises rather than warning. The image's pyvista is not this
    # script's to choose, so try the current name and fall back to the old one.
    for key in ("max_length", "max_time"):
        try:
            lines = mesh.streamlines(**common, **{key: reach})
            break
        except TypeError:
            continue
        except Exception as exc:  # noqa: BLE001
            if "deprecat" in str(exc).lower():
                continue
            print(f"  streamlines: could not integrate ({type(exc).__name__}: {exc})")
            return []
    if lines is None:
        print("  streamlines: this pyvista accepts neither max_length nor max_time")
        return []
    if lines.n_points == 0:
        print("  streamlines: nothing integrated -- the seeds may be outside the mesh")
        return []
    tubes = lines.tube(radius=float(0.0016 * max(span)))

    def draw(plotter):
        add_body(plotter, context)
        coloured(plotter, tubes, speed, MAPS["speed"], bar="speed  m/s",
                 smooth_shading=True)
        caption(plotter, title, f"{count} streamlines seeded across the inflow")

    return _still_or_frames(draw, context if context is not None else tubes,
                            out, "streamlines", frames, fps, 0, mesh)


def scene_surface(block, out: Path, title: str, patch: str | None,
                  frames: int = 0, fps: float = 20.0) -> list[Path]:
    """The body under its own pressure field."""
    surface, name = body_patch(block, patch)
    if surface is None:
        print("  surface: this case exposes no boundary patches")
        return []
    field = "p" if "p" in surface.point_data else ("p" if "p" in surface.cell_data else "")
    if field and field not in surface.point_data:
        surface = surface.cell_data_to_point_data()

    def draw(plotter):
        coloured(plotter, surface, field, MAPS["p"], bar="pressure  p/rho  m2/s2",
                 smooth_shading=True, specular=0.4, specular_power=25,
                 **({} if field else {"color": "#c3ccd6"}))
        caption(plotter, title, f"patch '{name}', {surface.n_cells:,} faces")

    return _still_or_frames(draw, surface, out, "surface", frames, fps, 0, None)


def scene_slice(mesh, out: Path, title: str, field: str, normal: str,
                frames: int = 0, fps: float = 20.0, sweep: int = 0) -> list[Path]:
    """A cut plane, styled like the rest."""
    speed = speed_of(mesh)
    scalars = speed if field in ("speed", "U", "u") else field
    if scalars not in mesh.point_data:
        print(f"  slice: no field called {field!r}; this case has "
              f"{', '.join(sorted(mesh.point_data.keys()))}")
        return []
    axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[normal]
    cut = mesh.slice(normal=axis, origin=mesh.center)
    cmap = MAPS.get(field, MAPS["speed"])

    def draw(plotter):
        coloured(plotter, cut, scalars, cmap, bar=scalars)
        caption(plotter, title, f"{normal}-normal cut")

    return _still_or_frames(draw, cut, out, f"slice_{scalars}", frames, fps, sweep, mesh,
                            sweep_axis=axis, sweep_scalars=scalars, sweep_cmap=cmap,
                            face_axis=axis)


# -- stills, orbits and sweeps ---------------------------------------------------------


def _camera(plotter, target, angle: float, elevation: float = 22.0, zoom: float = 1.0):
    """Put the camera at `angle` degrees around the subject, looking at its centre."""
    lo = np.array(target.bounds[::2]); hi = np.array(target.bounds[1::2])
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) * 1.15 / max(zoom, 1e-6)
    theta, phi = math.radians(angle), math.radians(elevation)
    plotter.camera_position = [
        (centre[0] + radius * math.cos(theta) * math.cos(phi),
         centre[1] + radius * math.sin(theta) * math.cos(phi),
         centre[2] + radius * math.sin(phi)),
        tuple(centre),
        (0, 0, 1),
    ]


def _sidecar(directory: Path, name: str, fps: float, count: int, expected: int) -> None:
    """What the frames were rendered for, so the encoder on the other machine knows.

    Rewritten after every frame, so a directory that arrives home half-finished still
    describes the animation it was going to be."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "frames.json").write_text(json.dumps({
        "output": f"{name}.gif", "fps": fps, "frames": count, "expected": expected,
    }, indent=2) + "\n", encoding="utf-8")


def _face_on(plotter, subject, axis) -> None:
    """Look straight down a plane's normal, filling the frame with it.

    A cut plane seen from the default three-quarter angle is a parallelogram with most of
    its area foreshortened -- the tube bundle's thermal wakes read as a smear rather than
    as wakes. A slice is a flat thing and wants to be looked at flatly."""
    lo = np.array(subject.bounds[::2]); hi = np.array(subject.bounds[1::2])
    centre = (lo + hi) / 2.0
    span = hi - lo
    direction = np.array(axis, dtype=float)
    up = np.array((0.0, 0.0, 1.0) if abs(direction[2]) < 0.5 else (0.0, 1.0, 0.0))
    plotter.camera_position = [tuple(centre + direction * float(np.linalg.norm(span))),
                               tuple(centre), tuple(up)]
    plotter.enable_parallel_projection()
    # Under parallel projection the camera's distance does nothing to the zoom --
    # `parallel_scale` is the half-height of the view in world units, and leaving it
    # alone cropped the tube bundle out of its own frame. Sized so the whole plane fits
    # both ways: the across-frame extent has to be divided by the aspect ratio first,
    # or a wide plane is cut off at the sides while the height looks fine.
    across = np.array([s for s, d in zip(span, direction) if abs(d) < 0.5], dtype=float)
    if across.size >= 2:
        width, height = float(np.max(across)), float(np.min(across))
        size = plotter.window_size
        aspect = (size[0] / size[1]) if size and size[1] else 1.6
        plotter.camera.parallel_scale = 1.06 * max(height / 2.0, width / (2.0 * aspect))


def _still_or_frames(draw, subject, out: Path, name: str, frames: int, fps: float,
                     sweep: int, mesh, sweep_axis=None, sweep_scalars=None,
                     sweep_cmap=None, face_axis=None) -> list[Path]:
    if sweep and mesh is not None and sweep_axis is not None:
        return _sweep(out, name, sweep, fps, mesh, sweep_axis, sweep_scalars, sweep_cmap)
    if frames:
        return _orbit(draw, subject, out, name, frames, fps)
    plotter = stage()
    draw(plotter)
    if face_axis is not None:
        _face_on(plotter, subject, face_axis)
    else:
        _camera(plotter, subject, 235.0)
    return [save(plotter, out / f"{name}.png")]


def _orbit(draw, subject, out: Path, name: str, frames: int, fps: float) -> list[Path]:
    """The camera goes round; the flow stays still. Written frame by frame.

    A frame is written under a dotted `.part` name and moved into place, so a frame that
    exists is a frame that is whole -- the mirror can carry a half-finished animation
    home while the rest are still rendering, which is what makes a partial gif possible."""
    directory = out / f"{name}_frames"
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(frames):
        plotter = stage(FRAME_WINDOW)
        draw(plotter)
        _camera(plotter, subject, 360.0 * index / frames)
        target = directory / f"frame_{index:04d}.png"
        part = directory / f".{target.name}.part"
        plotter.screenshot(str(part))
        plotter.close()
        part.replace(target)
        written.append(target)
        _sidecar(directory, name, fps, len(written), frames)
        print(f"  {name} frame {index + 1}/{frames}", flush=True)
    return written


def _sweep(out: Path, name: str, frames: int, fps: float, mesh, axis,
           scalars, cmap) -> list[Path]:
    """A slice plane travelling through the domain, one frame per station."""
    directory = out / f"{name}_sweep_frames"
    directory.mkdir(parents=True, exist_ok=True)
    lo = np.array(mesh.bounds[::2]); hi = np.array(mesh.bounds[1::2])
    which = int(np.argmax(np.abs(np.array(axis))))
    # Inset from the faces: a plane exactly on the boundary cuts nothing.
    stations = np.linspace(lo[which] + 0.02 * (hi - lo)[which],
                           hi[which] - 0.02 * (hi - lo)[which], frames)
    written: list[Path] = []
    for index, station in enumerate(stations):
        origin = list((lo + hi) / 2.0)
        origin[which] = float(station)
        cut = mesh.slice(normal=axis, origin=origin)
        plotter = stage(FRAME_WINDOW)
        if cut.n_points:
            coloured(plotter, cut, scalars, cmap, bar=scalars,
                     clim=[float(np.nanmin(mesh[scalars])), float(np.nanmax(mesh[scalars]))])
        caption(plotter, name.replace("_", " "), f"station {station:.3g} m")
        _face_on(plotter, mesh, axis)
        target = directory / f"frame_{index:04d}.png"
        part = directory / f".{target.name}.part"
        plotter.screenshot(str(part))
        plotter.close()
        part.replace(target)
        written.append(target)
        _sidecar(directory, f"{name}_sweep", fps, len(written), frames)
        print(f"  {name} sweep {index + 1}/{frames}", flush=True)
    return written


# -- the command -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="defaults to <case>/renders")
    ap.add_argument("--scene", default="vortices",
                    choices=["vortices", "streamlines", "surface", "slice"])
    ap.add_argument("--all", action="store_true", help="one still of every scene, then an orbit")
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--field", default="speed", help="for --scene slice")
    ap.add_argument("--normal", default="z", choices=["x", "y", "z"])
    ap.add_argument("--patch", default=None, help="for --scene surface")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="where the Q isosurface sits, as a percentile of positive Q")
    ap.add_argument("--seeds", type=int, default=220,
                    help="streamline seeds. A few hundred reads as flow; a few thousand reads as a curtain.")
    ap.add_argument("--orbit", type=int, default=0, metavar="N", help="N frames, camera going round")
    ap.add_argument("--sweep", type=int, default=0, metavar="N", help="N frames, plane travelling")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--title", default="", help="what to caption the pictures with")
    ap.add_argument("--refine", type=int, default=0, metavar="POINTS",
                    help="resample the field onto a uniform grid of about this many "
                         "points before contouring. Smooths the depiction, adds no data, "
                         "and says so in the caption. 6000000 is a good starting value.")
    ap.add_argument("--smooth", type=int, default=24, metavar="N",
                    help="Taubin smoothing passes over the isosurface (0 to disable).")
    args = ap.parse_args(argv)

    _reader, block, chosen, times = open_case(args.case, args.time)
    mesh = internal(block)
    if mesh is None:
        raise SystemExit("no internalMesh in this case -- has it been meshed?")
    out = args.out or args.case / "renders"
    context, context_name = body_patch(block, args.patch)
    title = args.title or args.case.name
    sub = f"t = {chosen:g}" if times else "no written times"
    print(f"{title}: {mesh.n_cells:,} cells, {sub}"
          + (f", body patch '{context_name}'" if context is not None else ", no body patch found"),
          flush=True)

    written: list[Path] = []
    scenes = ["vortices", "streamlines", "surface", "slice"] if args.all else [args.scene]
    for scene in scenes:
        label = f"{title}  |  {scene}"
        orbit = args.orbit if (not args.all or scene == "vortices") else 0
        try:
            if scene == "vortices":
                written += scene_vortices(mesh, out, label, args.percentile, orbit,
                                          args.fps, args.sweep, context=context,
                                          refine=args.refine, smooth=args.smooth)
            elif scene == "streamlines":
                written += scene_streamlines(mesh, out, label, args.seeds, orbit,
                                             args.fps, context=context)
            elif scene == "surface":
                written += scene_surface(block, out, label, args.patch, orbit, args.fps)
            else:
                written += scene_slice(mesh, out, label, args.field, args.normal,
                                       orbit, args.fps, args.sweep)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - one scene failing is not the gallery failing
            print(f"  {scene}: {type(exc).__name__}: {exc}")

    for path in written[:12]:
        print(path)
    if len(written) > 12:
        print(f"... and {len(written) - 12} more")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
