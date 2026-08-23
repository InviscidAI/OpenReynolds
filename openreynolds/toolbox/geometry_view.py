#!/usr/bin/env python3
"""Surface geometry -> pictures of it, plus the numbers that describe it.

Meshing decisions are made against geometry nobody has looked at, and a surface that
is 1000x too large, inside-out, or full of holes looks exactly like a good one from
the shell. This draws it -- four fixed views, every facet edge visible -- and prints
what is measurable about it. It attaches no verdicts: whether 40 open edges matter
depends on where they are and what you are about to run.

    python3 geometry_view.py wing.stl
    python3 geometry_view.py constant/triSurface/            # every surface in a dir
    python3 geometry_view.py a.stl b.stl --per-part --out /work/case/renders
    python3 geometry_view.py wing.stl --check               # numbers only, no render

Reads anything VTK reads: .stl, .obj, .ply, .vtk, .vtp, .vtu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

SUFFIXES = (".stl", ".obj", ".ply", ".vtk", ".vtp", ".vtu")

VIEWS = (("iso", "isometric"), ("xy", "+z looking down"), ("xz", "+y"), ("yz", "+x"))

EDGE_BUDGET = 200_000
"""Above this many faces, every-edge rendering draws mud. Decimate and say so."""

COLOURS = (
    "lightsteelblue", "salmon", "darkseagreen", "khaki", "plum",
    "lightcoral", "paleturquoise", "sandybrown", "thistle", "yellowgreen",
)


def gather(paths: list[Path]) -> list[Path]:
    """Files given, or every surface inside a directory given."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in SUFFIXES))
        else:
            found.append(path)
    return found


def load(path: Path) -> pv.PolyData:
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh


def measure(mesh: pv.PolyData) -> dict:
    """Facts about one surface. No thresholds are applied to any of them."""
    bounds = np.asarray(mesh.bounds, dtype=float)
    extent = bounds[1::2] - bounds[0::2]

    facts: dict = {
        "points": mesh.n_points,
        "faces": mesh.n_cells,
        "bounds": bounds,
        "extent": extent,
        "area": float(mesh.area),
    }

    try:
        open_edges = mesh.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False,
        )
        facts["open_edges"] = int(open_edges.n_cells)
    except Exception as exc:  # a missing number is better than a lost report
        facts["open_edges"] = f"not computed ({exc})"

    try:
        weird = mesh.extract_feature_edges(
            boundary_edges=False, feature_edges=False,
            manifold_edges=False, non_manifold_edges=True,
        )
        facts["non_manifold_edges"] = int(weird.n_cells)
    except Exception:
        facts["non_manifold_edges"] = "not computed"

    if facts.get("open_edges") == 0:
        # Signed volume only means anything on a closed surface. On an open one it is
        # an arbitrary number, and reporting it would invite reading it as one.
        facts["volume"] = float(mesh.volume)

    try:
        cleaned = mesh.clean()
        facts["duplicate_points"] = int(mesh.n_points - cleaned.n_points)
    except Exception:
        facts["duplicate_points"] = "not computed"

    try:
        facts["bodies"] = int(len(mesh.split_bodies()))
    except Exception:
        facts["bodies"] = "not computed"

    try:
        sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sizes.cell_data["Area"], dtype=float)
        areas = areas[np.isfinite(areas)]
        if areas.size:
            facts["face_area_min"] = float(areas.min())
            facts["face_area_max"] = float(areas.max())
            facts["degenerate_faces"] = int((areas <= 0).sum())
            # Edge length of the equivalent equilateral triangle: the number a surface
            # refinement level is actually chosen against.
            facts["edge_typical"] = float(np.sqrt(4 * np.median(areas) / np.sqrt(3)))
    except Exception:
        pass

    return facts


def report(name: str, facts: dict) -> list[str]:
    bounds, extent = facts["bounds"], facts["extent"]
    lines = [
        f"{name}",
        f"  faces {facts['faces']:,}   points {facts['points']:,}   "
        f"bodies {facts.get('bodies', '?')}",
        f"  bbox x [{bounds[0]:.6g}, {bounds[1]:.6g}]  "
        f"y [{bounds[2]:.6g}, {bounds[3]:.6g}]  z [{bounds[4]:.6g}, {bounds[5]:.6g}]",
        f"  extent {extent[0]:.6g} x {extent[1]:.6g} x {extent[2]:.6g}  "
        f"(OpenFOAM reads these as metres)",
        f"  open edges {facts.get('open_edges')}   "
        f"non-manifold edges {facts.get('non_manifold_edges')}   "
        f"duplicate points {facts.get('duplicate_points')}",
    ]
    if "volume" in facts:
        lines.append(f"  enclosed volume {facts['volume']:.6g}   surface area {facts['area']:.6g}")
    else:
        lines.append(f"  surface area {facts['area']:.6g}   (not closed, so no volume)")
    if "face_area_min" in facts:
        lines.append(
            f"  face area {facts['face_area_min']:.3g} .. {facts['face_area_max']:.3g}   "
            f"typical edge {facts['edge_typical']:.3g}   "
            f"zero-area faces {facts['degenerate_faces']}"
        )
    return lines


PANEL_PIXELS = 700
"""Roughly how many pixels across one view gets, for deciding what can be seen."""

MIN_FACE_PIXELS = 20
"""Below this a facet is smaller than its own outline, and edges draw a black blob."""


def for_display(mesh: pv.PolyData) -> tuple[pv.PolyData, str]:
    """Every facet edge, unless there are so many that the picture becomes grey."""
    if mesh.n_cells <= EDGE_BUDGET:
        return mesh, ""
    keep = EDGE_BUDGET / mesh.n_cells
    try:
        reduced = mesh.triangulate().decimate_pro(1.0 - keep, preserve_topology=True)
    except Exception:
        return mesh, "edges hidden: too many faces"
    return reduced, f"decimated {mesh.n_cells:,}->{reduced.n_cells:,} faces for drawing"


def edges_would_read(mesh: pv.PolyData, part_span: float, scene_span: float) -> bool:
    """Whether drawing every facet edge shows structure or just fills it in black.

    A small part in a large scene gets few pixels, and its facets end up smaller than
    the lines around them -- which is how a sphere becomes a black dot. Per part
    rather than per file, because in one picture the domain box and the object in it
    are drawn at wildly different scales.
    """
    if not mesh.n_cells or scene_span <= 0:
        return mesh.n_cells <= EDGE_BUDGET
    share = min(1.0, part_span / scene_span)
    per_face = (share * PANEL_PIXELS) ** 2 / mesh.n_cells
    return per_face >= MIN_FACE_PIXELS


def diagonal(mesh: pv.PolyData) -> float:
    bounds = np.asarray(mesh.bounds, dtype=float)
    return float(np.linalg.norm(bounds[1::2] - bounds[0::2]))


def draw(parts: list[tuple[str, pv.PolyData]], out: Path, title: str) -> Path:
    """Four views of the same thing, labelled, with the scale drawn on.

    The biggest surface is almost always the one enclosing everything else -- a
    domain box, a tunnel, a room -- and drawn solid it hides exactly what you opened
    the picture to look at. So it goes translucent, and the caption says so, because
    translucency that is not explained reads as a property of the geometry.
    """
    plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1000), border=False)
    notes = set()

    spans = [diagonal(mesh) for _name, mesh in parts]
    scene_span = max(spans) if spans else 0.0
    biggest = max(range(len(parts)), key=lambda i: spans[i]) if len(parts) > 1 else -1
    if biggest >= 0:
        notes.add(f"{parts[biggest][0]} drawn translucent (it encloses the rest)")

    try:
        plotter.enable_depth_peeling(10)
    except Exception:
        pass

    for index, (position, label) in enumerate(VIEWS):
        plotter.subplot(index // 2, index % 2)
        # The translucent one last: VTK composites in the order it is given.
        order = [i for i in range(len(parts)) if i != biggest] + (
            [biggest] if biggest >= 0 else []
        )
        for slot in order:
            name, mesh = parts[slot]
            shown, caveat = for_display(mesh)
            if caveat:
                notes.add(f"{name}: {caveat}")
            show_edges = edges_would_read(shown, spans[slot], scene_span)
            if not show_edges and index == 0:
                notes.add(f"{name}: facets too small to outline here - see its own picture")
            plotter.add_mesh(
                shown,
                color=COLOURS[slot % len(COLOURS)],
                show_edges=show_edges,
                edge_color="black",
                line_width=0.3,
                opacity=0.25 if slot == biggest else 1.0,
                label=name,
            )
        # `%.3g` and not the default `%.1f`: on a 6 cm case every tick otherwise
        # reads 0.0, and the mm-versus-m question the picture exists to answer
        # becomes unanswerable from it.
        plotter.show_bounds(
            grid="back", location="outer", ticks="outside", fmt="%.3g", font_size=8,
            n_xlabels=3, n_ylabels=3, n_zlabels=3,
        )
        plotter.camera_position = position
        plotter.add_text(label, font_size=9, position="upper_left")

    plotter.subplot(1, 1)
    if len(parts) > 1:
        plotter.add_legend(bcolor="white", size=(0.3, min(0.4, 0.06 * len(parts))), loc="lower right")

    plotter.subplot(1, 0)
    caption = "\n".join([title, *sorted(notes)])
    plotter.add_text(caption, position="lower_left", font_size=8, color="dimgray")

    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="surface files, or a directory")
    parser.add_argument("--out", type=Path, default=None, help="defaults to ./renders")
    parser.add_argument("--per-part", action="store_true", help="also one picture per surface")
    parser.add_argument("--check", action="store_true", help="numbers only, draw nothing")
    args = parser.parse_args()

    files = gather(args.paths)
    if not files:
        raise SystemExit(f"no surface files in {', '.join(str(p) for p in args.paths)}")

    parts: list[tuple[str, pv.PolyData]] = []
    for path in files:
        if not path.exists():
            print(f"{path}: not found")
            continue
        try:
            mesh = load(path)
        except Exception as exc:
            print(f"{path}: could not be read ({exc})")
            continue
        parts.append((path.name, mesh))
        print("\n".join(report(str(path), measure(mesh))))

    if not parts:
        raise SystemExit("nothing could be read")

    if len(parts) > 1:
        combined = parts[0][1].copy()
        for _name, mesh in parts[1:]:
            combined = combined.merge(mesh)
        print("\n".join(report(f"all {len(parts)} surfaces together", measure(combined))))

    if args.check:
        return

    out_dir = args.out or Path("renders")
    written = [draw(parts, out_dir / "geometry.png", f"{len(parts)} surface(s)")]
    if args.per_part and len(parts) > 1:
        for name, mesh in parts:
            stem = Path(name).stem
            written.append(draw([(name, mesh)], out_dir / f"geometry_{stem}.png", name))

    print()
    for path in written:
        print(path)


if __name__ == "__main__":
    sys.exit(main())
