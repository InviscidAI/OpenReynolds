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


def save(plotter: pv.Plotter, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    return out


def render_field(mesh, field: str, normal: str, out: Path) -> Path | None:
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
    plotter.camera_position = normal
    plotter.add_text(f"{scalars} — {normal}-normal slice", font_size=10)
    return save(plotter, out)


def render_mesh(mesh, normal: str, out: Path) -> Path:
    cut = slice_at(mesh, normal)
    plotter = pv.Plotter(off_screen=True, window_size=(1100, 800))
    plotter.add_mesh(cut, color="white", show_edges=True, edge_color="black", line_width=0.4)
    plotter.camera_position = normal
    plotter.add_text(f"mesh — {normal}-normal cut", font_size=10)
    return save(plotter, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="defaults to <case>/renders")
    parser.add_argument("--fields", nargs="*", default=["U", "p"])
    parser.add_argument("--normal", default="z", choices=sorted(NORMALS))
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--scene", default="all", choices=["all", "mesh", "fields"])
    args = parser.parse_args()

    block, chosen, times = open_case(args.case, args.time)
    mesh = internal_mesh(block)
    if mesh is None:
        raise SystemExit("no internalMesh in this case — has it been meshed?")

    out_dir = args.out or args.case / "renders"
    print(f"time {chosen:g}  (available: {', '.join(f'{t:g}' for t in times) or 'none'})")
    print(f"cells {mesh.n_cells:,}  points {mesh.n_points:,}")
    written = []

    if args.scene in ("all", "mesh"):
        written.append(render_mesh(mesh, args.normal, out_dir / f"mesh_{args.normal}.png"))
    if args.scene in ("all", "fields"):
        for field in args.fields:
            path = render_field(
                mesh, field, args.normal, out_dir / f"{field}_{args.normal}_{chosen:g}.png"
            )
            if path:
                written.append(path)
            else:
                print(f"  (no field named {field} at this time)")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
