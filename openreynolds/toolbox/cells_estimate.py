#!/usr/bin/env python3
"""Rough snappyHexMesh cell-count prediction, before the build.

A snappy build costs twenty minutes to several hours. This reads the STL and the
dictionaries and takes seconds, which makes it the one place where cheap foresight
reliably saves a doomed hour: a refinement level implying hundreds of millions of
cells, or a mesh far too coarse to resolve the region you care about, is visible here.

The numbers are order-of-magnitude. Real snappy meshes land within roughly a factor of
two of this for ordinary geometries, and further out for thin or highly curved ones.

    python3 cells_estimate.py /work/case
    python3 cells_estimate.py /work/case --stl constant/triSurface/body.stl
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import numpy as np

NUM = r"[-+0-9.eE]+"
VERTICES = re.compile(r"vertices\s*\((.*?)\)\s*;", re.S)
POINT = re.compile(rf"\(\s*({NUM})\s+({NUM})\s+({NUM})\s*\)")
HEX_BLOCK = re.compile(
    rf"hex\s*\((?:\s*\d+){{8}}\s*\)\s*(?:\w+\s*)?\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", re.S
)
SCALE = re.compile(rf"(?:convertToMeters|scale)\s+({NUM})\s*;")
LEVEL_PAIR = re.compile(rf"level\s*\(\s*(\d+)\s+(\d+)\s*\)\s*;")
N_LAYERS = re.compile(r"nSurfaceLayers\s+(\d+)\s*;")
REFINEMENT_SURFACES = re.compile(r"refinementSurfaces\s*\{(.*?)\n\s{4}\}", re.S)


def read_dict(case: Path, name: str) -> str:
    path = case / "system" / name
    return path.read_text(errors="replace") if path.exists() else ""


def block_mesh_delta(text: str) -> tuple[float | None, float | None, tuple]:
    """Base cell size and domain volume from the background mesh."""
    vertices_match = VERTICES.search(text)
    blocks = HEX_BLOCK.findall(text)
    if not vertices_match or not blocks:
        return None, None, ()

    points = np.array(
        [[float(c) for c in point] for point in POINT.findall(vertices_match.group(1))]
    )
    if points.size == 0:
        return None, None, ()

    scale_match = SCALE.search(text)
    scale = float(scale_match.group(1)) if scale_match else 1.0
    points = points * scale

    extent = points.max(axis=0) - points.min(axis=0)
    nx, ny, nz = (int(v) for v in blocks[0])
    deltas = extent / np.array([max(nx, 1), max(ny, 1), max(nz, 1)])
    delta0 = float(np.mean(deltas))
    volume = float(np.prod(extent))
    return delta0, volume, tuple(round(float(v), 6) for v in extent)


def surface_levels(text: str) -> dict[str, int]:
    """Maximum refinement level per named surface in refinementSurfaces."""
    section = REFINEMENT_SURFACES.search(text)
    if not section:
        return {}
    levels: dict[str, int] = {}
    body = section.group(1)
    for block in re.finditer(r"(\w[\w.\-]*)\s*\{(.*?)\}", body, re.S):
        name, inner = block.groups()
        pair = LEVEL_PAIR.search(inner)
        if pair:
            levels[name] = max(int(pair.group(1)), int(pair.group(2)))
    return levels


def stl_area(path: Path) -> tuple[float, np.ndarray, int]:
    """Surface area, bounding box extent, and triangle count, ASCII or binary."""
    raw = path.read_bytes()
    triangles = _binary_triangles(raw)
    if triangles is None:
        triangles = _ascii_triangles(raw)
    if triangles is None or len(triangles) == 0:
        return 0.0, np.zeros(3), 0

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area = float(np.abs(np.linalg.norm(np.cross(b - a, c - a), axis=1)).sum() / 2.0)
    flat = triangles.reshape(-1, 3)
    return area, flat.max(axis=0) - flat.min(axis=0), len(triangles)


def _binary_triangles(raw: bytes):
    if len(raw) < 84:
        return None
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + count * 50:
        return None
    data = np.frombuffer(raw[84:], dtype=np.uint8).reshape(count, 50)
    floats = data[:, :48].copy().view("<f4").reshape(count, 4, 3)
    return floats[:, 1:, :].astype(np.float64)


def _ascii_triangles(raw: bytes):
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None
    values = re.findall(rf"vertex\s+({NUM})\s+({NUM})\s+({NUM})", text)
    if len(values) < 3:
        return None
    points = np.array(values, dtype=float)
    usable = len(points) - len(points) % 3
    return points[:usable].reshape(-1, 3, 3)


def estimate(delta0, volume, levels, area, n_layers):
    """Background cells, surface-refinement band, and layer cells."""
    background = volume / delta0**3 if (delta0 and volume) else 0.0

    surface = 0.0
    finest = max(levels.values()) if levels else 0
    if delta0 and area:
        # A refinement band roughly two cells thick at the refined size, per the
        # cell-count heuristic: area * 2^level / delta0^2.
        surface = area * (2**finest) / delta0**2

    layers = 0.0
    if delta0 and area and n_layers:
        delta_fine = delta0 / (2**finest) if finest else delta0
        layers = n_layers * area / delta_fine**2

    return {
        "background": background,
        "surface": surface,
        "layers": layers,
        "total": background + surface + layers,
        "finest_level": finest,
        "delta0": delta0,
        "delta_finest": (delta0 / 2**finest) if delta0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--stl", type=Path, default=None, help="defaults to every STL found")
    args = parser.parse_args()

    block_text = read_dict(args.case, "blockMeshDict")
    snappy_text = read_dict(args.case, "snappyHexMeshDict")
    if not block_text:
        print("no system/blockMeshDict — nothing to size the background mesh from")
    delta0, volume, extent = block_mesh_delta(block_text)

    stls = [args.stl] if args.stl else sorted((args.case / "constant" / "triSurface").glob("*.stl"))
    stls = [p if p.is_absolute() else args.case / p for p in stls]

    total_area = 0.0
    print("# geometry")
    if not stls:
        print("  no STL found under constant/triSurface")
    for path in stls:
        if not path.exists():
            print(f"  {path.name}: missing")
            continue
        area, size, count = stl_area(path)
        total_area += area
        print(
            f"  {path.name}: area {area:.4g} m^2, "
            f"bbox {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} m, {count:,} triangles"
        )
        if size.max() > 0 and size.max() > 100:
            print("    (bbox is large — worth checking whether this file is in millimetres)")

    levels = surface_levels(snappy_text)
    layers_match = N_LAYERS.search(snappy_text)
    n_layers = int(layers_match.group(1)) if layers_match else 0

    print("\n# background mesh")
    if delta0:
        print(f"  domain extent: {extent} m, volume {volume:.4g} m^3")
        print(f"  base cell size delta0 ~ {delta0:.4g} m")
    else:
        print("  could not read vertices/blocks from blockMeshDict")

    print("\n# snappy settings read")
    print(f"  refinementSurfaces levels: {levels or 'none found'}")
    print(f"  nSurfaceLayers: {n_layers}")

    if not delta0:
        return

    result = estimate(delta0, volume, levels, total_area, n_layers)
    print("\n# rough prediction")
    print(f"  finest level {result['finest_level']} -> cell size ~ {result['delta_finest']:.4g} m")
    print(f"  background            {result['background']:>15,.0f}")
    print(f"  surface refinement    {result['surface']:>15,.0f}")
    print(f"  layers                {result['layers']:>15,.0f}")
    print(f"  total                 {result['total']:>15,.0f}  cells (order of magnitude)")


if __name__ == "__main__":
    main()
