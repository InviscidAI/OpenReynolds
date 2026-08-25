#!/usr/bin/env python3
"""Render a field over time into a directory of frames, on the instance.

The frames are the expensive half of an animation and they need the case data, so
they render here, next to it — one PNG per write time, a fixed camera so the only
thing that moves between frames is the flow. The cheap half, assembling the frames
into a gif, does not run here: the frames land in a directory named `*_frames/`, they
mirror to the user's machine as they are written, and the harness assembles them there
with the encoder it has. Nothing on this image needs ffmpeg or imageio, and a frame
that is on disk is already a frame the user has.

Two things this is built to survive, because a live animation of a running solve is
where the last few sessions came apart:

- **Run it as a job, not a synchronous command.** Rendering fifty frames takes minutes,
  and a synchronous `bash` call that long trips the exec ceiling and reads back as a
  failure even when it worked. `job_start` has no such limit.
      job_start: python3 .toolbox/animate.py /work/case --field vorticity
- **Each frame is flushed as it is written.** The directory grows one file at a time,
  so the mirror can carry a half-finished animation home and the harness can assemble
  the frames-so-far into a partial gif while the rest still render — which is exactly
  what someone asks for when they say "a gif of what you have so far."

    python3 animate.py /work/case --field vorticity
    python3 animate.py /work/case --field U --component mag --every 2 --clim -5 5
    python3 animate.py /work/case --out /work/case/wake_frames --from 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

NORMALS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def open_reader(case: Path):
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    reader.cell_to_point_creation = True
    return reader


def internal_mesh(block):
    """The internalMesh block, whatever nesting this case produced."""
    if isinstance(block, pv.MultiBlock):
        keys = block.keys() or []
        if "internalMesh" in keys:
            return block["internalMesh"]
        for item in block:
            if item is None:
                continue
            found = internal_mesh(item)
            if found is not None:
                return found
        return None
    return block


def scalar_of(mesh, field: str, component: str):
    """A per-point scalar to colour by: a named field, or a component/magnitude of a
    vector. Vorticity is computed if it was not written."""
    if field == "vorticity" and "vorticity" not in mesh.point_data:
        mesh = mesh.compute_derivative(scalars="U", vorticity=True)
        vec = mesh.point_data["vorticity"]
        mesh.point_data["vorticity_z"] = vec[:, 2]
        return mesh, "vorticity_z"
    if field not in mesh.point_data:
        raise SystemExit(
            f"field '{field}' is not in this case; available: "
            + ", ".join(sorted(mesh.point_data.keys()))
        )
    data = mesh.point_data[field]
    if data.ndim == 2 and data.shape[1] >= 3:
        if component == "mag":
            mesh.point_data[f"{field}_mag"] = np.linalg.norm(data, axis=1)
            return mesh, f"{field}_mag"
        idx = {"x": 0, "y": 1, "z": 2}[component]
        mesh.point_data[f"{field}_{component}"] = data[:, idx]
        return mesh, f"{field}_{component}"
    return mesh, field


def camera_bounds(mesh):
    """A fixed frame around the whole slice, so nothing drifts between frames."""
    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    return (xmin, xmax, ymin, ymax)


def render_frame(mesh, scalar, normal, clim, bounds, out: Path, title: str):
    origin = np.array(mesh.center)
    cut = mesh.slice(normal=NORMALS[normal], origin=origin)
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 750))
    plotter.add_mesh(cut, scalars=scalar, clim=clim, cmap="RdBu_r", show_edges=False)
    plotter.view_vector(NORMALS[normal])
    plotter.enable_parallel_projection()
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        plotter.camera.tight(padding=0.05)
    plotter.add_text(title, font_size=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", type=Path)
    ap.add_argument("--field", default="vorticity", help="Field to colour by (default vorticity_z).")
    ap.add_argument("--component", default="mag", choices=["mag", "x", "y", "z"])
    ap.add_argument("--normal", default="z", choices=list(NORMALS))
    ap.add_argument("--every", type=int, default=1, help="Render every Nth write time.")
    ap.add_argument("--from", dest="start", type=float, default=None, help="Skip times before this.")
    ap.add_argument("--clim", type=float, nargs=2, default=None, help="Fixed colour range.")
    ap.add_argument("--out", type=Path, default=None, help="Frames directory (default <case>_frames).")
    args = ap.parse_args()

    case = args.case
    out = args.out or case.parent / f"{case.name}_frames"
    reader = open_reader(case)
    times = list(reader.time_values)
    times = [t for t in times if args.start is None or t >= args.start]
    times = times[:: max(1, args.every)]
    if len(times) < 2:
        raise SystemExit(f"need at least 2 write times to animate; found {len(times)}")

    clim = tuple(args.clim) if args.clim else None
    bounds = None
    print(f"{len(times)} frames -> {out}", flush=True)
    for i, t in enumerate(times):
        reader.set_active_time_value(t)
        mesh = internal_mesh(reader.read())
        mesh, scalar = scalar_of(mesh, args.field, args.component)
        if clim is None:
            # Fix the colour range on the first frame so it does not flicker.
            values = mesh.point_data[scalar]
            clim = (float(np.percentile(values, 2)), float(np.percentile(values, 98)))
        if bounds is None:
            bounds = camera_bounds(mesh.slice(normal=NORMALS[args.normal], origin=np.array(mesh.center)))
        frame = out / f"frame_{i:04d}.png"
        render_frame(mesh, scalar, args.normal, clim, bounds, frame, f"t = {t:g}")
        # Flush the count so a watcher (and the mirror) sees the directory grow.
        print(f"  frame {i + 1}/{len(times)}  t={t:g}", flush=True)
    print(f"done: {len(times)} frames in {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
