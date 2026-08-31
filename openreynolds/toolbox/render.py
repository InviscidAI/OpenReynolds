#!/usr/bin/env python3
"""A few fixed pyvista scenes: mesh cuts and field slices, saved as PNGs.

Headless via OSMesa. Fixed cameras so two runs of the same case are a visual diff
rather than two unrelated pictures.

    python3 render.py /work/case                      # latest time, all default scenes
    python3 render.py /work/case --fields U p --normal z --time 500
    python3 render.py /work/case --scene mesh --out /work/case/renders
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

NORMALS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def open_case(case: Path, time: float | None):
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    times = list(reader.time_values)
    chosen = times[-1] if times else 0.0
    if time is not None and times:
        chosen = min(times, key=lambda t: abs(t - time))
    reader.set_active_time_value(chosen)
    reader.cell_to_point_creation = True
    return reader.read(), chosen, times


def internal_mesh(block):
    """The internalMesh block, whatever nesting this case produced."""
    if isinstance(block, pv.MultiBlock):
        if "internalMesh" in (block.keys() or []):
            return block["internalMesh"]
        for item in block:
            if item is None:
                continue
            found = internal_mesh(item)
            if found is not None:
                return found
        return None
    return block


def slice_at(mesh, normal: str):
    origin = np.array(mesh.center)
    return mesh.slice(normal=NORMALS[normal], origin=origin)


VIEWUP = {"z": (0.0, 1.0, 0.0), "y": (0.0, 0.0, 1.0), "x": (0.0, 0.0, 1.0)}
"""Which way is up, per slice normal.

Without it VTK picks, and for a z-normal slice it picks something collinear with the
view direction and falls back to an arbitrary up vector: a 28x16 domain came out drawn
as a portrait column with the flow running bottom to top and 60% of the canvas empty.
`results.py` and `animate.py` have carried this dict since they were written; this file
was the one that never got it, and every study that called `render.py --scene mesh`
paid for the omission by hand-writing its own pyvista script instead."""


def aim(plotter, normal: str) -> None:
    """Point the camera down the slice normal, with x across the page.

    `plotter.camera_position = "z"` reads as if it would do this and does not: the
    string form takes a view *plane* ("xy", "xz", "yz"), and a single axis letter is
    not one of them. A live study hit this, worked out the fix and applied it to its
    own copy under `/work/.toolbox/` -- which is the copy that gets overwritten from
    the distribution at the start of the next session, so the fix has been rediscovered
    since. Naming the direction as a vector says what is meant and survives the sync;
    naming the up vector with it is what keeps the streamwise axis horizontal.

    Parallel projection because these are slices: perspective on a flat cut makes the
    far side of the domain smaller than the near side, so a uniform mesh reads as graded.
    """
    plotter.view_vector(NORMALS[normal], viewup=VIEWUP.get(normal, (0.0, 1.0, 0.0)))
    plotter.enable_parallel_projection()


def frame(plotter, cut, zoom: float | None, bounds: tuple | None) -> None:
    """Fill the canvas with the part worth looking at.

    A close-up was impossible without this: asked for the mesh around a step, the tool
    drew the whole domain as a horizontal sliver with the step invisible, so the study
    wrote its own script twice. `bounds` crops to a region, `zoom` magnifies about the
    centre; neither is applied unless asked for, so the default picture is unchanged.
    """
    if bounds is not None:
        plotter.reset_camera(bounds=list(bounds))
    if zoom:
        plotter.camera.zoom(zoom)


def save(plotter: pv.Plotter, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    return out


def render_field(mesh, field: str, normal: str, out: Path, zoom: float | None = None,
                 bounds: tuple | None = None) -> Path | None:
    if field not in mesh.point_data and field not in mesh.cell_data:
        return None
    cut = slice_at(mesh, normal)
    data = cut.point_data.get(field, cut.cell_data.get(field))
    if data is None:
        return None
    scalars = field
    if data.ndim > 1 and data.shape[1] > 1:
        cut[f"|{field}|"] = np.linalg.norm(data, axis=1)
        scalars = f"|{field}|"

    plotter = pv.Plotter(off_screen=True, window_size=(1100, 800))
    plotter.add_mesh(cut, scalars=scalars, cmap="viridis", show_edges=False)
    plotter.add_scalar_bar(title=scalars, n_labels=5)
    aim(plotter, normal)
    frame(plotter, cut, zoom, bounds)
    plotter.add_text(f"{scalars} — {normal}-normal slice", font_size=10)
    return save(plotter, out)


def render_mesh(mesh, normal: str, out: Path, zoom: float | None = None,
                bounds: tuple | None = None) -> Path:
    cut = slice_at(mesh, normal)
    plotter = pv.Plotter(off_screen=True, window_size=(1100, 800))
    plotter.add_mesh(cut, color="white", show_edges=True, edge_color="black", line_width=0.4)
    aim(plotter, normal)
    frame(plotter, cut, zoom, bounds)
    label = f"mesh — {normal}-normal cut"
    if bounds is not None or zoom:
        label += " (close-up)"
    plotter.add_text(label, font_size=10)
    return save(plotter, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="defaults to <case>/renders")
    parser.add_argument("--fields", nargs="*", default=["U", "p"])
    parser.add_argument("--normal", default="z", choices=sorted(NORMALS))
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--scene", default="all", choices=["all", "mesh", "fields"])
    parser.add_argument("--zoom", type=float, default=None,
                        help="magnify about the centre, e.g. 4 for a close-up")
    parser.add_argument("--bounds", nargs=6, type=float, default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="frame on this region instead of the whole domain")
    args = parser.parse_args()
    bounds = tuple(args.bounds) if args.bounds else None

    block, chosen, times = open_case(args.case, args.time)
    mesh = internal_mesh(block)
    if mesh is None:
        raise SystemExit("no internalMesh in this case — has it been meshed?")

    out_dir = args.out or args.case / "renders"
    print(f"time {chosen:g}  (available: {', '.join(f'{t:g}' for t in times) or 'none'})")
    print(f"cells {mesh.n_cells:,}  points {mesh.n_points:,}")
    written = []

    suffix = "_closeup" if (bounds or args.zoom) else ""
    if args.scene in ("all", "mesh"):
        written.append(render_mesh(mesh, args.normal,
                                   out_dir / f"mesh_{args.normal}{suffix}.png",
                                   zoom=args.zoom, bounds=bounds))
    if args.scene in ("all", "fields"):
        for field in args.fields:
            path = render_field(
                mesh, field, args.normal,
                out_dir / f"{field}_{args.normal}_{chosen:g}{suffix}.png",
                zoom=args.zoom, bounds=bounds,
            )
            if path:
                written.append(path)
            else:
                print(f"  (no field named {field} at this time)")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
